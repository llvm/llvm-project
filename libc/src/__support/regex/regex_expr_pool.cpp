//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Pool for Regular Expression AST nodes (Implementation).
///
//===----------------------------------------------------------------------===//

#include "src/__support/regex/regex_expr_pool.h"
#include "hdr/regex_macros.h"
#include "src/__support/CPP/algorithm.h"
#include "src/__support/CPP/new.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/CPP/scope.h"
#include "src/__support/CPP/utility.h"
#include "src/__support/alloc-checker.h"
#include "src/__support/hash.h"
#include "src/__support/macros/config.h"
#include "src/string/memory_utils/inline_memset.h"

namespace LIBC_NAMESPACE_DECL {
namespace regex {

namespace {

// Hash an Expr node for hash-consing.
uint32_t hash_expr(const Expr &e) {
  // Initialise HashState with a constant seed. The specific value (0x12345678)
  // is an arbitrary placeholder; HashState immediately mixes this seed with
  // high-entropy constants (derived from aHash) to produce a strong hash, while
  // the constant value guarantees deterministic hashing for hash-consing.
  internal::HashState hasher(0x12345678);
  struct HashKey {
    uint64_t kind;
    uint64_t payload;
  } key = {static_cast<uint64_t>(e.kind), 0};
  static_assert(sizeof(HashKey) == 2 * sizeof(uint64_t));
  switch (e.kind) {
  case ExprKind::Literal:
    key.payload = static_cast<uint8_t>(e.ch);
    break;
  case ExprKind::Anchor:
    key.payload = e.anchor_mask;
    break;
  case ExprKind::ByteSet:
    key.payload = e.byte_set_id;
    break;
  case ExprKind::Star:
    key.payload = e.body;
    break;
  case ExprKind::Repeat:
    key.payload = (static_cast<uint64_t>(e.rep.body) << 32) |
                  (static_cast<uint64_t>(e.rep.min) << 16) | e.rep.max;
    break;
  case ExprKind::Concat:
  case ExprKind::Alt:
    key.payload = (static_cast<uint64_t>(e.bin.left) << 32) | e.bin.right;
    break;
  default:
    break;
  }
  hasher.update(&key, sizeof(key));
  uint64_t h = hasher.finish();
  return static_cast<uint32_t>(h ^ (h >> 32));
}

// Bounded linear-probing range over a power-of-two open-addressing table.
class BucketProber {
  cpp::span<ExprId> table;
  size_t start_idx;

  constexpr size_t mask() const { return table.size() - 1; }

public:
  struct Sentinel {};

  struct Iterator {
    cpp::span<ExprId> table;
    size_t idx;
    size_t step;

    constexpr size_t mask() const { return table.size() - 1; }

    ExprId &operator*() const { return table[idx]; }
    Iterator &operator++() {
      idx = (idx + 1) & mask();
      ++step;
      return *this;
    }
    constexpr bool operator!=(Sentinel) const { return step != table.size(); }
  };

  BucketProber(cpp::span<ExprId> t, uint32_t hash)
      : table(t), start_idx(hash & (t.size() - 1)) {
    LIBC_ASSERT(!t.empty() && (t.size() & mask()) == 0);
  }

  Iterator begin() const { return {table, start_idx, 0}; }
  Sentinel end() const { return {}; }
};

} // namespace

// TODO: When cpp::unique_ptr and cpp::make_unique are added to
// src/__support/CPP/, replace the manual AllocChecker + new[]/delete[] calls in
// ensure_initialized(), ~ExprPool(), allocate_block(), make_byte_set(), and
// grow_hashtable() with cpp::make_unique<ExprId[]>(...) and
// cpp::make_unique<Block>(), eliminating explicit zeroing via inline_memset and
// the destructor cleanup loop.
ExprPool::~ExprPool() {
  delete[] byte_sets.data();
  delete[] hashtable.data();
  for (Block *blk : active_blocks())
    delete blk;
  delete[] blocks.data();
}

bool ExprPool::ensure_initialized() {
  if (block_count > 0)
    return true;
  AllocChecker ac;
  ExprId *raw_table = new (ac) ExprId[INITIAL_HASH_CAPACITY];
  if (!ac)
    return false;
  cpp::scope_exit table_guard([raw_table] { delete[] raw_table; });
  if (!allocate_block())
    return false;
  table_guard.release();
  hashtable = cpp::span<ExprId>(raw_table, INITIAL_HASH_CAPACITY);
  inline_memset(hashtable.data(), 0, hashtable.size_bytes());

  [[maybe_unused]] auto empty_set_id = intern(Expr::make_empty_set());
  [[maybe_unused]] auto empty_str_id = intern(Expr::make_empty_str());
  LIBC_ASSERT(empty_set_id.has_value() && *empty_set_id == EMPTY_SET_ID);
  LIBC_ASSERT(empty_str_id.has_value() && *empty_str_id == EMPTY_STR_ID);
  return true;
}

ExprId &ExprPool::find_bucket(cpp::span<ExprId> table, const Expr &e) const {
  for (ExprId &slot : BucketProber(table, e.hash)) {
    if (slot == INVALID_EXPR_ID)
      return slot;
    const Expr &cand = get(slot);
    if (cand.hash == e.hash && cand == e)
      return slot;
  }
  __builtin_unreachable();
}

bool ExprPool::allocate_block() {
  AllocChecker ac;
  if (block_count == blocks.size()) {
    size_t new_cap = blocks.empty() ? INITIAL_BLOCKS_CAPACITY
                                    : blocks.size() * GROWTH_FACTOR;
    Block **new_blocks = new (ac) Block *[new_cap];
    if (!ac)
      return false;
    Block **dst = new_blocks;
    for (Block *blk : active_blocks()) {
      *dst = blk;
      ++dst;
    }
    delete[] blocks.data();
    blocks = cpp::span<Block *>(new_blocks, new_cap);
  }
  Block *blk = new (ac) Block();
  if (!ac)
    return false;
  blocks[block_count] = blk;
  ++block_count;
  return true;
}

bool ExprPool::grow_hashtable() {
  size_t new_cap = hashtable.size() * GROWTH_FACTOR;
  AllocChecker ac;
  ExprId *raw_table = new (ac) ExprId[new_cap];
  if (!ac)
    return false;
  cpp::span<ExprId> new_table(raw_table, new_cap);
  inline_memset(new_table.data(), 0, new_table.size_bytes());

  for (ExprId id : hashtable)
    if (id != INVALID_EXPR_ID)
      find_bucket(new_table, get(id)) = id;

  delete[] hashtable.data();
  hashtable = new_table;
  return true;
}

ErrorOr<ExprId> ExprPool::intern(Expr e) {
  if (!ensure_initialized())
    return Error(REG_ESPACE);

  // Probe for an existing node with identical content.
  e.hash = hash_expr(e);
  ExprId *bucket = &find_bucket(hashtable, e);
  if (*bucket != INVALID_EXPR_ID)
    return *bucket;

  // Admission Control: Check the hard limit on AST nodes.
  if (node_count >= MAX_NODE_LIMIT)
    return Error(REG_ESPACE);

  if (node_count >= max_hashtable_nodes()) {
    if (!grow_hashtable())
      return Error(REG_ESPACE);
    bucket = &find_bucket(hashtable, e);
  }

  // Arena Allocation: If no matching node found, allocate a stable slot.
  if (node_count == block_count * NODES_PER_BLOCK) {
    if (!allocate_block())
      return Error(REG_ESPACE);
  }

  // Node Initialisation: Copy the structural definition into the arena.
  ++node_count;
  ExprId new_id = static_cast<ExprId>(node_count);
  slot_at(new_id) = e;
  *bucket = new_id;
  return new_id;
}

ErrorOr<ExprId> ExprPool::empty_set() {
  if (!ensure_initialized())
    return Error(REG_ESPACE);
  return EMPTY_SET_ID;
}
ErrorOr<ExprId> ExprPool::empty_str() {
  if (!ensure_initialized())
    return Error(REG_ESPACE);
  return EMPTY_STR_ID;
}
ErrorOr<ExprId> ExprPool::make_lit(char c) {
  return intern(Expr::make_literal(c));
}
ErrorOr<ExprId> ExprPool::make_anchor(uint8_t mask) {
  return intern(Expr::make_anchor(mask));
}

ErrorOr<ExprId> ExprPool::make_byte_set(const ByteSet &bs) {
  size_t count = 0;
  size_t last_bit = 0;
  for (size_t b = 0; b < BYTE_SET_BITS; ++b) {
    if (bs.test(b)) {
      ++count;
      last_bit = b;
    }
  }
  if (count == 0)
    return empty_set();
  if (count == 1)
    return make_lit(static_cast<char>(last_bit));

  uint32_t bs_id = 0;
  for (const ByteSet &existing : active_byte_sets()) {
    if (existing == bs)
      return intern(Expr::make_byte_set(bs_id));
    ++bs_id;
  }

  if (byte_set_count == byte_sets.size()) {
    size_t new_cap = byte_sets.empty() ? INITIAL_BYTE_SETS_CAPACITY
                                       : byte_sets.size() * GROWTH_FACTOR;
    AllocChecker ac;
    ByteSet *new_sets = new (ac) ByteSet[new_cap];
    if (!ac)
      return Error(REG_ESPACE);
    ByteSet *dst = new_sets;
    for (const ByteSet &existing : active_byte_sets()) {
      *dst = existing;
      ++dst;
    }
    delete[] byte_sets.data();
    byte_sets = cpp::span<ByteSet>(new_sets, new_cap);
  }

  ErrorOr<ExprId> id = intern(Expr::make_byte_set(bs_id));
  if (!id.has_value())
    return id;
  byte_sets[byte_set_count] = bs;
  ++byte_set_count;
  return id;
}

namespace {

// Multiplies a subexpression's repeat expansion factor by an enclosing repeat
// bound, saturating at MAX_REPEAT_EXPANSION + 1 so nested bounds checks never
// overflow uint16_t.
uint16_t scale_exp(uint16_t base, uint16_t factor) {
  if (factor <= 1)
    return base;
  uint32_t prod = static_cast<uint32_t>(base) * static_cast<uint32_t>(factor);
  return static_cast<uint16_t>(
      cpp::min<uint32_t>(prod, MAX_REPEAT_EXPANSION + 1));
}

} // namespace

ErrorOr<ExprId> ExprPool::make_star(ExprId body) {
  if (body == INVALID_EXPR_ID)
    return Error(REG_BADPAT);
  // Ø* = ε and ε* = ε
  if (body == EMPTY_SET_ID || body == EMPTY_STR_ID)
    return empty_str();
  // (R*)* = R* and (R{m,n})* = R* iff m <= 1 (since R^1 is in R{m,n} iff
  // m <= 1, whereas (a{2,3})* excludes "a").
  const Expr be = get(body);
  if (be.kind == ExprKind::Star)
    return body;
  if (be.kind == ExprKind::Repeat && be.rep.min <= 1)
    return make_star(be.rep.body);
  return intern(
      Expr::make_star(body, be.has_anchors, be.has_repeat, be.rep_expansion));
}

ErrorOr<ExprId> ExprPool::make_repeat(ExprId body, uint16_t min, uint16_t max) {
  if (body == INVALID_EXPR_ID || min > max || min == REPEAT_INF)
    return Error(REG_BADPAT);
  // Normalize R{0,0} = ε, ε{m,n} = ε, Ø{m,n}, R{1,1} = R, and
  // R{0,REPEAT_INF} = R*.
  if (max == 0 || body == EMPTY_STR_ID)
    return empty_str();
  if (body == EMPTY_SET_ID)
    return (min == 0) ? empty_str() : empty_set();
  if (min == 1 && max == 1)
    return body;
  if (min == 0 && max == REPEAT_INF)
    return make_star(body);
  const Expr be = get(body);
  if (be.kind == ExprKind::Star)
    return body;
  if (be.kind == ExprKind::Repeat && be.rep.min == 1 &&
      be.rep.max == REPEAT_INF && min == 1 && max == REPEAT_INF)
    return body;
  uint16_t factor = (max == REPEAT_INF) ? min : max;
  return intern(Expr::make_repeat(body, min, max, (min == 0) || be.nullable,
                                  be.has_anchors,
                                  scale_exp(be.rep_expansion, factor)));
}

ErrorOr<ExprId> ExprPool::make_concat(ExprId l, ExprId r) {
  if (l == INVALID_EXPR_ID || r == INVALID_EXPR_ID)
    return Error(REG_BADPAT);
  // Apply basic algebraic identities for concatenation:
  // Ø · R = R · Ø = Ø (Identity: null set)
  if (l == EMPTY_SET_ID || r == EMPTY_SET_ID)
    return empty_set();
  // ε · R = R · ε = R (Identity: empty string)
  if (l == EMPTY_STR_ID)
    return r;
  if (r == EMPTY_STR_ID)
    return l;
  // (A · B) · C = A · (B · C) (Right-associative canonical form)
  const Expr le = get(l);
  if (le.kind == ExprKind::Concat) {
    ErrorOr<ExprId> tail = make_concat(le.bin.right, r);
    if (!tail.has_value())
      return tail;
    return make_concat(le.bin.left, *tail);
  }
  const Expr re = get(r);
  return intern(Expr::make_concat(
      l, r, le.nullable && re.nullable, le.has_anchors || re.has_anchors,
      le.has_repeat || re.has_repeat,
      cpp::max(le.rep_expansion, re.rep_expansion)));
}

namespace {

// Coalesces R{m1,n1} | R{m2,n2} (and shared-prefix/shared-suffix Concat nodes)
// when their repetition intervals overlap or are adjacent.  Returns
// INVALID_EXPR_ID when the pair cannot be coalesced.
ErrorOr<ExprId> coalesce_repeat_pair(ExprId a, ExprId b, ExprPool &pool) {
  const Expr ae = pool.get(a);
  const Expr be = pool.get(b);
  if (!ae.has_repeat && !be.has_repeat)
    return INVALID_EXPR_ID;
  if (ae.kind == ExprKind::Concat && be.kind == ExprKind::Concat) {
    if (ae.bin.right == be.bin.right) {
      ErrorOr<ExprId> left =
          coalesce_repeat_pair(ae.bin.left, be.bin.left, pool);
      if (!left.has_value() || *left == INVALID_EXPR_ID)
        return left;
      return pool.make_concat(*left, ae.bin.right);
    }
    if (ae.bin.left == be.bin.left) {
      ErrorOr<ExprId> right =
          coalesce_repeat_pair(ae.bin.right, be.bin.right, pool);
      if (!right.has_value() || *right == INVALID_EXPR_ID)
        return right;
      return pool.make_concat(ae.bin.left, *right);
    }
  }
  // Handle Concat(X, S) | S and Concat(P, X) | P where R{0,0} normalized to
  // EmptyStr on one side.
  if (ae.kind == ExprKind::Concat && ae.bin.right == b) {
    ErrorOr<ExprId> left =
        coalesce_repeat_pair(ae.bin.left, EMPTY_STR_ID, pool);
    if (!left.has_value() || *left == INVALID_EXPR_ID)
      return left;
    return pool.make_concat(*left, b);
  }
  if (be.kind == ExprKind::Concat && be.bin.right == a) {
    ErrorOr<ExprId> left =
        coalesce_repeat_pair(EMPTY_STR_ID, be.bin.left, pool);
    if (!left.has_value() || *left == INVALID_EXPR_ID)
      return left;
    return pool.make_concat(*left, a);
  }
  if (ae.kind != ExprKind::Repeat && be.kind != ExprKind::Repeat)
    return INVALID_EXPR_ID;
  ExprId body = (ae.kind == ExprKind::Repeat) ? ae.rep.body : be.rep.body;
  struct RepeatInterval {
    uint16_t min;
    uint16_t max;
  };
  // Undo make_repeat normalizations: R -> [1,1], EmptyStr -> [0,0], and
  // Star(R) -> [0,REPEAT_INF].
  auto get_interval = [&](ExprId id,
                          const Expr &e) -> cpp::optional<RepeatInterval> {
    if (e.kind == ExprKind::Repeat && e.rep.body == body)
      return RepeatInterval{e.rep.min, e.rep.max};
    if (e.kind == ExprKind::Star && e.body == body)
      return RepeatInterval{0, REPEAT_INF};
    if (id == body)
      return RepeatInterval{1, 1};
    if (id == EMPTY_STR_ID)
      return RepeatInterval{0, 0};
    return cpp::nullopt;
  };
  cpp::optional<RepeatInterval> lhs = get_interval(a, ae);
  cpp::optional<RepeatInterval> rhs = get_interval(b, be);
  if (!lhs.has_value() || !rhs.has_value())
    return INVALID_EXPR_ID;
  // Integer intervals [lhs->min, lhs->max] and [rhs->min, rhs->max] coalesce
  // iff they overlap or are adjacent (later_min <= earlier_max + 1, e.g.
  // [2,3] U [4,5] = [2,5]).
  uint16_t later_min = cpp::max(lhs->min, rhs->min);
  uint16_t earlier_max = cpp::min(lhs->max, rhs->max);
  if (earlier_max != REPEAT_INF && later_min > earlier_max + 1)
    return INVALID_EXPR_ID;
  return pool.make_repeat(body, cpp::min(lhs->min, rhs->min),
                          cpp::max(lhs->max, rhs->max));
}

// Attempts to coalesce a single non-Alt branch `item` with any branch in
// `spine`, returning INVALID_EXPR_ID if no branch coalesces.
ErrorOr<ExprId> coalesce_item_into_spine(ExprId item, ExprId spine,
                                         ExprPool &pool) {
  const Expr item_e = pool.get(item);
  const Expr spine_e = pool.get(spine);
  if (!item_e.has_repeat && !spine_e.has_repeat)
    return INVALID_EXPR_ID;
  if (spine_e.kind != ExprKind::Alt)
    return coalesce_repeat_pair(item, spine, pool);
  ErrorOr<ExprId> head_c = coalesce_repeat_pair(item, spine_e.bin.left, pool);
  if (!head_c.has_value())
    return head_c;
  if (*head_c != INVALID_EXPR_ID)
    return pool.make_alt(*head_c, spine_e.bin.right);
  ErrorOr<ExprId> tail_c =
      coalesce_item_into_spine(item, spine_e.bin.right, pool);
  if (!tail_c.has_value() || *tail_c == INVALID_EXPR_ID)
    return tail_c;
  return pool.make_alt(spine_e.bin.left, *tail_c);
}

} // namespace

ErrorOr<ExprId> ExprPool::make_alt(ExprId l, ExprId r) {
  if (l == INVALID_EXPR_ID || r == INVALID_EXPR_ID)
    return Error(REG_BADPAT);
  // Apply basic algebraic identities for alternation:
  // Ø | R = R | Ø = R (Identity: null set)
  if (l == EMPTY_SET_ID)
    return r;
  if (r == EMPTY_SET_ID)
    return l;
  // R | R = R (Idempotency)
  if (l == r)
    return l;

  const Expr le = get(l);
  const Expr re = get(r);
  if (le.kind != ExprKind::Alt && re.kind != ExprKind::Alt) {
    ErrorOr<ExprId> coalesced = coalesce_repeat_pair(l, r, *this);
    if (!coalesced.has_value() || *coalesced != INVALID_EXPR_ID)
      return coalesced;
    if (l > r)
      cpp::swap(l, r);
    return intern(Expr::make_alt(l, r, le.nullable || re.nullable,
                                 le.has_anchors || re.has_anchors,
                                 le.has_repeat || re.has_repeat,
                                 cpp::max(le.rep_expansion, re.rep_expansion)));
  }

  if (le.has_repeat || re.has_repeat) {
    if (le.kind != ExprKind::Alt) {
      ErrorOr<ExprId> c = coalesce_item_into_spine(l, r, *this);
      if (!c.has_value() || *c != INVALID_EXPR_ID)
        return c;
    } else if (re.kind != ExprKind::Alt) {
      ErrorOr<ExprId> c = coalesce_item_into_spine(r, l, *this);
      if (!c.has_value() || *c != INVALID_EXPR_ID)
        return c;
    } else {
      ErrorOr<ExprId> c = coalesce_item_into_spine(le.bin.left, r, *this);
      if (!c.has_value())
        return c;
      if (*c != INVALID_EXPR_ID)
        return make_alt(le.bin.right, *c);
    }
  }

  // Merge two right-associated sorted Alt spines in ascending ExprId order so
  // multi-way alternations canonicalize to a single deduplicated tree and
  // derivative state spaces under '*' and '{m,n}' stay finite.
  ExprId hl = (le.kind == ExprKind::Alt) ? le.bin.left : l;
  ExprId tl = (le.kind == ExprKind::Alt) ? le.bin.right : EMPTY_SET_ID;
  ExprId hr = (re.kind == ExprKind::Alt) ? re.bin.left : r;
  ExprId tr = (re.kind == ExprKind::Alt) ? re.bin.right : EMPTY_SET_ID;
  ExprId head = hl;
  ErrorOr<ExprId> rest = EMPTY_SET_ID;
  if (hl == hr) {
    rest = make_alt(tl, tr);
  } else if (hl < hr) {
    rest = make_alt(tl, r);
  } else {
    head = hr;
    rest = make_alt(l, tr);
  }
  if (!rest.has_value())
    return rest;
  // Tail coalescing may produce an already-interned ExprId <= head; re-merge
  // when needed to preserve ascending ExprId order along the Alt spine.
  const Expr re_rest = get(*rest);
  ExprId rest_head = (re_rest.kind == ExprKind::Alt) ? re_rest.bin.left : *rest;
  if (head >= rest_head)
    return make_alt(head, *rest);
  const Expr he = get(head);
  return intern(
      Expr::make_alt(head, *rest, he.nullable || re_rest.nullable,
                     he.has_anchors || re_rest.has_anchors,
                     he.has_repeat || re_rest.has_repeat,
                     cpp::max(he.rep_expansion, re_rest.rep_expansion)));
}

} // namespace regex
} // namespace LIBC_NAMESPACE_DECL
