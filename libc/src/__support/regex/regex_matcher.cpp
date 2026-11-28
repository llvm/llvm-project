//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Brzozowski derivative evaluation and regular expression matching
/// implementation.
///
//===----------------------------------------------------------------------===//

#include "src/__support/regex/regex_matcher.h"

#include "hdr/regex_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/CPP/unique_ptr.h"
#include "src/__support/checked_ptr.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/regex/regex_ast.h"
#include "src/__support/regex/regex_expr_pool.h"

namespace LIBC_NAMESPACE_DECL {
namespace regex {

namespace {

// Computes the active AnchorKind bitmask at fencepost position pos in input.
uint8_t ctx_at(cpp::string_view input, size_t pos, int cflags, int eflags) {
  LIBC_ASSERT(pos <= input.size());
  uint8_t ctx = 0;
  if ((pos == 0 && !(eflags & REG_NOTBOL)) ||
      (pos > 0 && (cflags & REG_NEWLINE) && input[pos - 1] == '\n'))
    ctx |= ANCHOR_BOL;
  if ((pos == input.size() && !(eflags & REG_NOTEOL)) ||
      (pos < input.size() && (cflags & REG_NEWLINE) && input[pos] == '\n'))
    ctx |= ANCHOR_EOL;
  return ctx;
}

// One parent node whose derivative is still being computed.
// d_left is INVALID_EXPR_ID while the node's first child (left or body) is
// being walked.  When a Concat or Alt finishes its left child and must walk its
// right child next, d_left saves the left-hand term until the right child's
// result arrives and the two are joined with make_alt.
// The default constructor is trivial so make_unique_for_overwrite does not zero
// the heap array.
struct DerivFrame {
  ExprId id;
  ExprId d_left;

  DerivFrame() = default;
  constexpr explicit DerivFrame(ExprId expr_id)
      : id(expr_id), d_left(INVALID_EXPR_ID) {
    LIBC_ASSERT(expr_id != INVALID_EXPR_ID);
  }
};
static_assert(sizeof(DerivFrame) == 8);
static_assert(cpp::is_trivially_constructible<DerivFrame>::value);

// Stack of DerivFrames for derivative_impl().  The first INLINE_CAP frames live
// inside the object; deeper walks move to a heap array that doubles in size.
// BlockStore is not used because its pop_back() frees a block each time depth
// drops back across a block boundary, and a tree walk crosses that boundary
// repeatedly.  match() uses one DerivStack for every input byte, so a heap
// array allocated for one byte is reused for the rest.
class DerivStack {
  // 8 frames (64 bytes) keep match()'s own stack frame under 256 bytes.
  static constexpr size_t INLINE_CAP = 8;
  static constexpr size_t GROWTH_FACTOR = 2;
  static_assert(ExprPool::MAX_NODE_LIMIT <= (~size_t(0)) / GROWTH_FACTOR);

  DerivFrame inline_buf[INLINE_CAP];
  cpp::unique_ptr<DerivFrame[]> heap;
  size_t len = 0;
  size_t cap = INLINE_CAP;

  DerivFrame *data() { return heap ? heap.get() : inline_buf; }

  // Moves the frames to a heap array twice the size.  Because every pushed
  // frame has a smaller ExprId than its parent, the stack never needs more than
  // ExprPool::MAX_NODE_LIMIT frames.
  [[nodiscard]] bool grow() {
    LIBC_ASSERT(len == cap && cap < ExprPool::MAX_NODE_LIMIT);
    const size_t new_cap =
        cpp::min(cap * GROWTH_FACTOR, ExprPool::MAX_NODE_LIMIT);
    auto next = make_unique_for_overwrite<DerivFrame[]>(new_cap);
    if (!next)
      return false;
    const DerivFrame *src = data();
    for (size_t i = 0; i < len; ++i)
      next[i] = src[i];
    heap = next.take_unique();
    cap = new_cap;
    return true;
  }

public:
  DerivStack() = default;
  // The implicit move operations would move heap while copying len and cap,
  // leaving the source stack in an inconsistent state.
  DerivStack(DerivStack &&) = delete;
  DerivStack &operator=(DerivStack &&) = delete;

  bool empty() const { return len == 0; }
  void clear() { len = 0; }

  DerivFrame &top() {
    LIBC_ASSERT(len > 0);
    return data()[len - 1];
  }

  void pop() {
    LIBC_ASSERT(len > 0);
    --len;
  }

  [[nodiscard]] bool push(DerivFrame frame) {
    if (LIBC_UNLIKELY(len == cap) && !grow())
      return false;
    LIBC_ASSERT(len < cap);
    data()[len] = frame;
    ++len;
    return true;
  }
};

// Computes D_c(root) iteratively in post-order.  stack is passed in so callers
// looping over input bytes can reuse one heap buffer across steps instead of
// reallocating on every byte.
ErrorOr<ExprId> derivative_impl(ExprId root, char c, ExprPool &pool,
                                uint8_t ctx_before, DerivStack &stack) {
  if (!pool.is_valid_input(root))
    return Error(REG_BADPAT);

  // Reset len without freeing stack's heap buffer, discarding any frames left
  // by an earlier failed call.
  stack.clear();
  ExprId cur = root;
  ExprId res = INVALID_EXPR_ID;
  do {
    // Walk down first children until we reach a leaf.
    while (res == INVALID_EXPR_ID) {
      const Expr e = pool.get(cur);
      ExprId next_child = INVALID_EXPR_ID;
      // ExprPool interns children before their parent, so ids shrink on the way
      // down: the walk always reaches a leaf, and the stack never holds more
      // frames than the pool has nodes.
      switch (e.kind) {
      case ExprKind::EmptySet:
      case ExprKind::EmptyStr:
      case ExprKind::Anchor:
        res = EMPTY_SET_ID;
        break;
      case ExprKind::Literal:
        res = (c == e.ch) ? EMPTY_STR_ID : EMPTY_SET_ID;
        break;
      case ExprKind::ByteSet:
        res = pool.get_byte_set(e.byte_set_id).test(static_cast<uint8_t>(c))
                  ? EMPTY_STR_ID
                  : EMPTY_SET_ID;
        break;
      case ExprKind::Concat:
      case ExprKind::Alt:
        LIBC_ASSERT(e.bin.left < cur && e.bin.right < cur);
        next_child = e.bin.left;
        break;
      case ExprKind::Star:
        LIBC_ASSERT(e.body < cur);
        next_child = e.body;
        break;
      case ExprKind::Repeat:
        LIBC_ASSERT(e.rep.body < cur);
        next_child = e.rep.body;
        break;
      }
      if (next_child != INVALID_EXPR_ID) {
        if (!stack.push(DerivFrame(cur)))
          return Error(REG_ESPACE);
        cur = next_child;
      }
    }

    // Combine child results back up the stack until a parent needs its right
    // child walked (res is reset to INVALID_EXPR_ID) or the stack empties.
    while (!stack.empty() && res != INVALID_EXPR_ID) {
      DerivFrame &frame = stack.top();
      if (frame.d_left != INVALID_EXPR_ID) {
        // Both children of a binary Concat or Alt finished; union their terms.
        LIBC_ASSERT(pool.get(frame.id).kind == ExprKind::Concat ||
                    pool.get(frame.id).kind == ExprKind::Alt);
        ErrorOr<ExprId> combined = pool.make_alt(frame.d_left, res);
        if (!combined.has_value())
          return combined;
        res = *combined;
      } else {
        // First child finished; either combine now or save the left term and
        // walk the right child.
        const Expr pe = pool.get(frame.id);
        switch (pe.kind) {
        case ExprKind::Concat: {
          // Product Rule:
          // D_c(L . R) = (D_c(L) . R) | (nullable(L, ctx_before) ? D_c(R) : Ø)
          ErrorOr<ExprId> lhs = pool.make_concat(res, pe.bin.right);
          if (!lhs.has_value())
            return lhs;
          if (nullable(pe.bin.left, ctx_before, pool)) {
            LIBC_ASSERT(*lhs != INVALID_EXPR_ID);
            frame.d_left = *lhs;
            cur = pe.bin.right;
            res = INVALID_EXPR_ID;
            continue;
          }
          res = *lhs;
          break;
        }
        case ExprKind::Alt:
          // Sum Rule: D_c(L | R) = D_c(L) | D_c(R)
          frame.d_left = res;
          cur = pe.bin.right;
          res = INVALID_EXPR_ID;
          continue;
        case ExprKind::Star: {
          // Kleene Rule: D_c(R*) = D_c(R) . R*
          ErrorOr<ExprId> combined = pool.make_concat(res, frame.id);
          if (!combined.has_value())
            return combined;
          res = *combined;
          break;
        }
        case ExprKind::Repeat: {
          // Bounded Repetition Rule:
          // D_c(R{m,n}) = D_c(R) . R{nullable(R, ctx_before) ? 0 : max(0,m-1),
          //                          n-1}.
          // When R is nullable at ctx_before (e.g. via '^'), up to m - 1
          // leading copies may match empty before the copy consuming c.
          LIBC_ASSERT(pe.rep.max >= 1 && pe.rep.min <= pe.rep.max);
          if (res != EMPTY_SET_ID) {
            uint16_t next_min =
                (pe.rep.min > 1 && !nullable(pe.rep.body, ctx_before, pool))
                    ? static_cast<uint16_t>(pe.rep.min - 1)
                    : static_cast<uint16_t>(0);
            uint16_t next_max = (pe.rep.max == REPEAT_INF)
                                    ? REPEAT_INF
                                    : static_cast<uint16_t>(pe.rep.max - 1);
            ErrorOr<ExprId> tail =
                pool.make_repeat(pe.rep.body, next_min, next_max);
            if (!tail.has_value())
              return tail;
            ErrorOr<ExprId> combined = pool.make_concat(res, *tail);
            if (!combined.has_value())
              return combined;
            res = *combined;
          }
          break;
        }
        case ExprKind::EmptySet:
        case ExprKind::EmptyStr:
        case ExprKind::Literal:
        case ExprKind::Anchor:
        case ExprKind::ByteSet:
          LIBC_ASSERT(false);
          __builtin_unreachable();
        }
      }
      stack.pop();
    }
  } while (res == INVALID_EXPR_ID);
  LIBC_ASSERT(stack.empty() && res != INVALID_EXPR_ID);
  return res;
}

} // namespace

ErrorOr<ExprId> derivative(ExprId id, char c, ExprPool &pool,
                           uint8_t ctx_before) {
  DerivStack stack;
  return derivative_impl(id, c, pool, ctx_before, stack);
}

MatchResult match(ExprPool &pool, const ExprId pattern, cpp::string_view input,
                  int cflags, int eflags) {
  if (!pool.is_valid_input(pattern) || pattern == EMPTY_SET_ID)
    return MatchResult::NoMatch;
  uint8_t ctx = ctx_at(input, 0, cflags, eflags);
  if (nullable(pattern, ctx, pool))
    return MatchResult::Match;

  // Unanchored substring search: at each byte, step active partial matches via
  // derivative_impl() and union with pattern so a new match can begin at the
  // next fencepost (equivalent to matching .* . pattern without constructing a
  // prefix wildcard node).
  // One stack serves every byte, so a heap array allocated for one byte is
  // reused for the rest.
  DerivStack stack;
  ExprId state = pattern;
  for (size_t pos = 0; pos < input.size(); ++pos) {
    ErrorOr<ExprId> d = derivative_impl(state, input[pos], pool, ctx, stack);
    if (!d.has_value()) {
      LIBC_ASSERT(d.error() == REG_ESPACE);
      return MatchResult::OutOfMemory;
    }
    ctx = ctx_at(input, pos + 1, cflags, eflags);
    if (nullable(*d, ctx, pool) || nullable(pattern, ctx, pool))
      return MatchResult::Match;
    ErrorOr<ExprId> next = pool.make_alt(pattern, *d);
    if (!next.has_value()) {
      LIBC_ASSERT(next.error() == REG_ESPACE);
      return MatchResult::OutOfMemory;
    }
    state = *next;
  }

  return MatchResult::NoMatch;
}

} // namespace regex
} // namespace LIBC_NAMESPACE_DECL
