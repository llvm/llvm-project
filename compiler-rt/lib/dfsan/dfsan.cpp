//===-- dfsan.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DataFlowSanitizer.
//
// DataFlowSanitizer runtime.  This file defines the public interface to
// DataFlowSanitizer as well as the definition of certain runtime functions
// called automatically by the compiler (specifically the instrumentation pass
// in llvm/lib/Transforms/Instrumentation/DataFlowSanitizer.cpp).
//
// The public interface is defined in include/sanitizer/dfsan_interface.h whose
// functions are prefixed dfsan_ while the compiler interface functions are
// prefixed __dfsan_.
//===----------------------------------------------------------------------===//

#include "dfsan/dfsan.h"

#include "dfsan/dfsan_chained_origin_depot.h"
#include "dfsan/dfsan_flags.h"
#include "dfsan/dfsan_origin.h"
#include "dfsan/dfsan_thread.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_flag_parser.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_report_decorator.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

using namespace __dfsan;

Flags __dfsan::flags_data;

// The size of TLS variables. These constants must be kept in sync with the ones
// in DataFlowSanitizer.cpp.
static const int kDFsanArgTlsSize = 800;
static const int kDFsanRetvalTlsSize = 800;
static const int kDFsanArgOriginTlsSize = 800;

SANITIZER_INTERFACE_ATTRIBUTE THREADLOCAL u64
    __dfsan_retval_tls[kDFsanRetvalTlsSize / sizeof(u64)];
SANITIZER_INTERFACE_ATTRIBUTE THREADLOCAL u32 __dfsan_retval_origin_tls;
SANITIZER_INTERFACE_ATTRIBUTE THREADLOCAL u64
    __dfsan_arg_tls[kDFsanArgTlsSize / sizeof(u64)];
SANITIZER_INTERFACE_ATTRIBUTE THREADLOCAL u32
    __dfsan_arg_origin_tls[kDFsanArgOriginTlsSize / sizeof(u32)];

// Instrumented code may set this value in terms of -dfsan-track-origins.
// * undefined or 0: do not track origins.
// * 1: track origins at memory store operations.
// * 2: TODO: track origins at memory store operations and callsites.
extern "C" SANITIZER_WEAK_ATTRIBUTE const int __dfsan_track_origins;

int __dfsan_get_track_origins() {
  return &__dfsan_track_origins ? __dfsan_track_origins : 0;
}

// On Linux/x86_64, memory is laid out as follows:
//
// +--------------------+ 0x800000000000 (top of memory)
// | application memory |
// +--------------------+ 0x700000008000 (kAppAddr)
// |                    |
// |       unused       |
// |                    |
// +--------------------+ 0x300000000000 (kUnusedAddr)
// |       origin       |
// +--------------------+ 0x200000008000 (kOriginAddr)
// |       unused       |
// +--------------------+ 0x200000000000
// |   shadow memory    |
// +--------------------+ 0x100000008000 (kShadowAddr)
// |       unused       |
// +--------------------+ 0x000000010000
// | reserved by kernel |
// +--------------------+ 0x000000000000
//
// To derive a shadow memory address from an application memory address, bits
// 45-46 are cleared to bring the address into the range
// [0x100000008000,0x200000000000).  See the function shadow_for below.
//
//


extern "C" SANITIZER_INTERFACE_ATTRIBUTE
dfsan_label __dfsan_union_load(const dfsan_label *ls, uptr n) {
  dfsan_label label = ls[0];
  for (uptr i = 1; i != n; ++i)
    label |= ls[i];
  return label;
}

// Return the union of all the n labels from addr at the high 32 bit, and the
// origin of the first taint byte at the low 32 bit.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE u64
__dfsan_load_label_and_origin(const void *addr, uptr n) {
  dfsan_label label = 0;
  u64 ret = 0;
  uptr p = (uptr)addr;
  dfsan_label *s = shadow_for((void *)p);
  for (uptr i = 0; i < n; ++i) {
    dfsan_label l = s[i];
    if (!l)
      continue;
    label |= l;
    if (!ret)
      ret = *(dfsan_origin *)origin_for((void *)(p + i));
  }
  return ret | (u64)label << 32;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE
void __dfsan_unimplemented(char *fname) {
  if (flags().warn_unimplemented)
    Report("WARNING: DataFlowSanitizer: call to uninstrumented function %s\n",
           fname);
}

// Use '-mllvm -dfsan-debug-nonzero-labels' and break on this function
// to try to figure out where labels are being introduced in a nominally
// label-free program.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __dfsan_nonzero_label() {
  if (flags().warn_nonzero_labels)
    Report("WARNING: DataFlowSanitizer: saw nonzero label\n");
}

// Indirect call to an uninstrumented vararg function. We don't have a way of
// handling these at the moment.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE void
__dfsan_vararg_wrapper(const char *fname) {
  Report("FATAL: DataFlowSanitizer: unsupported indirect call to vararg "
         "function %s\n", fname);
  Die();
}

// Resolves the union of two labels.
SANITIZER_INTERFACE_ATTRIBUTE dfsan_label
dfsan_union(dfsan_label l1, dfsan_label l2) {
  return l1 | l2;
}

// Return the origin of the first taint byte in the size bytes from the address
// addr.
static dfsan_origin GetOriginIfTainted(uptr addr, uptr size) {
  for (uptr i = 0; i < size; ++i, ++addr) {
    dfsan_label *s = shadow_for((void *)addr);
    if (!is_shadow_addr_valid((uptr)s)) {
      // The current DFSan memory layout is not always correct. For example,
      // addresses (0, 0x10000) are mapped to (0, 0x10000). Before fixing the
      // issue, we ignore such addresses.
      continue;
    }
    if (*s)
      return *(dfsan_origin *)origin_for((void *)addr);
  }
  return 0;
}

// For platforms which support slow unwinder only, we need to restrict the store
// context size to 1, basically only storing the current pc, because the slow
// unwinder which is based on libunwind is not async signal safe and causes
// random freezes in forking applications as well as in signal handlers.
// DFSan supports only Linux. So we do not restrict the store context size.
#define GET_STORE_STACK_TRACE_PC_BP(pc, bp) \
  BufferedStackTrace stack;                 \
  stack.Unwind(pc, bp, nullptr, true, flags().store_context_size);

#define PRINT_CALLER_STACK_TRACE        \
  {                                     \
    GET_CALLER_PC_BP_SP;                \
    (void)sp;                           \
    GET_STORE_STACK_TRACE_PC_BP(pc, bp) \
    stack.Print();                      \
  }

// Return a chain with the previous ID id and the current stack.
// from_init = true if this is the first chain of an origin tracking path.
static u32 ChainOrigin(u32 id, StackTrace *stack, bool from_init = false) {
  // StackDepot is not async signal safe. Do not create new chains in a signal
  // handler.
  DFsanThread *t = GetCurrentThread();
  if (t && t->InSignalHandler())
    return id;

  // As an optimization the origin of an application byte is updated only when
  // its shadow is non-zero. Because we are only interested in the origins of
  // taint labels, it does not matter what origin a zero label has. This reduces
  // memory write cost. MSan does similar optimization. The following invariant
  // may not hold because of some bugs. We check the invariant to help debug.
  if (!from_init && id == 0 && flags().check_origin_invariant) {
    Printf("  DFSan found invalid origin invariant\n");
    PRINT_CALLER_STACK_TRACE
  }

  Origin o = Origin::FromRawId(id);
  stack->tag = StackTrace::TAG_UNKNOWN;
  Origin chained = Origin::CreateChainedOrigin(o, stack);
  return chained.raw_id();
}

static const uptr kOriginAlign = sizeof(dfsan_origin);
static const uptr kOriginAlignMask = ~(kOriginAlign - 1UL);

static uptr AlignUp(uptr u) {
  return (u + kOriginAlign - 1) & kOriginAlignMask;
}

static uptr AlignDown(uptr u) { return u & kOriginAlignMask; }

static void ChainAndWriteOriginIfTainted(uptr src, uptr size, uptr dst,
                                         StackTrace *stack) {
  dfsan_origin o = GetOriginIfTainted(src, size);
  if (o) {
    o = ChainOrigin(o, stack);
    *(dfsan_origin *)origin_for((void *)dst) = o;
  }
}

// Copy the origins of the size bytes from src to dst. The source and target
// memory ranges cannot be overlapped. This is used by memcpy. stack records the
// stack trace of the memcpy. When dst and src are not 4-byte aligned properly,
// origins at the unaligned address boundaries may be overwritten because four
// contiguous bytes share the same origin.
static void CopyOrigin(const void *dst, const void *src, uptr size,
                       StackTrace *stack) {
  uptr d = (uptr)dst;
  uptr beg = AlignDown(d);
  // Copy left unaligned origin if that memory is tainted.
  if (beg < d) {
    ChainAndWriteOriginIfTainted((uptr)src, beg + kOriginAlign - d, beg, stack);
    beg += kOriginAlign;
  }

  uptr end = AlignDown(d + size);
  // If both ends fall into the same 4-byte slot, we are done.
  if (end < beg)
    return;

  // Copy right unaligned origin if that memory is tainted.
  if (end < d + size)
    ChainAndWriteOriginIfTainted((uptr)src + (end - d), (d + size) - end, end,
                                 stack);

  if (beg >= end)
    return;

  // Align src up.
  uptr s = AlignUp((uptr)src);
  dfsan_origin *src_o = (dfsan_origin *)origin_for((void *)s);
  u32 *src_s = (u32 *)shadow_for((void *)s);
  dfsan_origin *src_end = (dfsan_origin *)origin_for((void *)(s + (end - beg)));
  dfsan_origin *dst_o = (dfsan_origin *)origin_for((void *)beg);
  dfsan_origin last_src_o = 0;
  dfsan_origin last_dst_o = 0;
  for (; src_o < src_end; ++src_o, ++src_s, ++dst_o) {
    if (!*src_s)
      continue;
    if (*src_o != last_src_o) {
      last_src_o = *src_o;
      last_dst_o = ChainOrigin(last_src_o, stack);
    }
    *dst_o = last_dst_o;
  }
}

// Copy the origins of the size bytes from src to dst. The source and target
// memory ranges may be overlapped. So the copy is done in a reverse order.
// This is used by memmove. stack records the stack trace of the memmove.
static void ReverseCopyOrigin(const void *dst, const void *src, uptr size,
                              StackTrace *stack) {
  uptr d = (uptr)dst;
  uptr end = AlignDown(d + size);

  // Copy right unaligned origin if that memory is tainted.
  if (end < d + size)
    ChainAndWriteOriginIfTainted((uptr)src + (end - d), (d + size) - end, end,
                                 stack);

  uptr beg = AlignDown(d);

  if (beg + kOriginAlign < end) {
    // Align src up.
    uptr s = AlignUp((uptr)src);
    dfsan_origin *src =
        (dfsan_origin *)origin_for((void *)(s + end - beg - kOriginAlign));
    u32 *src_s = (u32 *)shadow_for((void *)(s + end - beg - kOriginAlign));
    dfsan_origin *src_begin = (dfsan_origin *)origin_for((void *)s);
    dfsan_origin *dst =
        (dfsan_origin *)origin_for((void *)(end - kOriginAlign));
    dfsan_origin src_o = 0;
    dfsan_origin dst_o = 0;
    for (; src >= src_begin; --src, --src_s, --dst) {
      if (!*src_s)
        continue;
      if (*src != src_o) {
        src_o = *src;
        dst_o = ChainOrigin(src_o, stack);
      }
      *dst = dst_o;
    }
  }

  // Copy left unaligned origin if that memory is tainted.
  if (beg < d)
    ChainAndWriteOriginIfTainted((uptr)src, beg + kOriginAlign - d, beg, stack);
}

// Copy or move the origins of the len bytes from src to dst. The source and
// target memory ranges may or may not be overlapped. This is used by memory
// transfer operations. stack records the stack trace of the memory transfer
// operation.
static void MoveOrigin(const void *dst, const void *src, uptr size,
                       StackTrace *stack) {
  if (!has_valid_shadow_addr(dst) ||
      !has_valid_shadow_addr((void *)((uptr)dst + size)) ||
      !has_valid_shadow_addr(src) ||
      !has_valid_shadow_addr((void *)((uptr)src + size))) {
    return;
  }
  // If destination origin range overlaps with source origin range, move
  // origins by copying origins in a reverse order; otherwise, copy origins in
  // a normal order. The orders of origin transfer are consistent with the
  // orders of how memcpy and memmove transfer user data.
  uptr src_aligned_beg = reinterpret_cast<uptr>(src) & ~3UL;
  uptr src_aligned_end = (reinterpret_cast<uptr>(src) + size) & ~3UL;
  uptr dst_aligned_beg = reinterpret_cast<uptr>(dst) & ~3UL;
  if (dst_aligned_beg < src_aligned_end && dst_aligned_beg >= src_aligned_beg)
    return ReverseCopyOrigin(dst, src, size, stack);
  return CopyOrigin(dst, src, size, stack);
}

// Set the size bytes from the addres dst to be the origin value.
static void SetOrigin(const void *dst, uptr size, u32 origin) {
  if (size == 0)
    return;

  // Origin mapping is 4 bytes per 4 bytes of application memory.
  // Here we extend the range such that its left and right bounds are both
  // 4 byte aligned.
  uptr x = unaligned_origin_for((uptr)dst);
  uptr beg = AlignDown(x);
  uptr end = AlignUp(x + size);  // align up.
  u64 origin64 = ((u64)origin << 32) | origin;
  // This is like memset, but the value is 32-bit. We unroll by 2 to write
  // 64 bits at once. May want to unroll further to get 128-bit stores.
  if (beg & 7ULL) {
    if (*(u32 *)beg != origin)
      *(u32 *)beg = origin;
    beg += 4;
  }
  for (uptr addr = beg; addr < (end & ~7UL); addr += 8) {
    if (*(u64 *)addr == origin64)
      continue;
    *(u64 *)addr = origin64;
  }
  if (end & 7ULL)
    if (*(u32 *)(end - kOriginAlign) != origin)
      *(u32 *)(end - kOriginAlign) = origin;
}

static void WriteShadowInRange(dfsan_label label, uptr beg_shadow_addr,
                               uptr end_shadow_addr) {
  // TODO: After changing dfsan_label to 8bit, use internal_memset when label
  // is not 0.
  dfsan_label *labelp = (dfsan_label *)beg_shadow_addr;
  if (label) {
    for (; (uptr)labelp < end_shadow_addr; ++labelp) *labelp = label;
    return;
  }

  for (; (uptr)labelp < end_shadow_addr; ++labelp) {
    // Don't write the label if it is already the value we need it to be.
    // In a program where most addresses are not labeled, it is common that
    // a page of shadow memory is entirely zeroed.  The Linux copy-on-write
    // implementation will share all of the zeroed pages, making a copy of a
    // page when any value is written.  The un-sharing will happen even if
    // the value written does not change the value in memory.  Avoiding the
    // write when both |label| and |*labelp| are zero dramatically reduces
    // the amount of real memory used by large programs.
    if (!*labelp)
      continue;

    *labelp = 0;
  }
}

static void WriteShadowWithSize(dfsan_label label, uptr shadow_addr,
                                uptr size) {
  WriteShadowInRange(label, shadow_addr, shadow_addr + size * sizeof(label));
}

#define RET_CHAIN_ORIGIN(id)           \
  GET_CALLER_PC_BP_SP;                 \
  (void)sp;                            \
  GET_STORE_STACK_TRACE_PC_BP(pc, bp); \
  return ChainOrigin(id, &stack);

// Return a new origin chain with the previous ID id and the current stack
// trace.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE dfsan_origin
__dfsan_chain_origin(dfsan_origin id) {
  RET_CHAIN_ORIGIN(id)
}

// Return a new origin chain with the previous ID id and the current stack
// trace if the label is tainted.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE dfsan_origin
__dfsan_chain_origin_if_tainted(dfsan_label label, dfsan_origin id) {
  if (!label)
    return id;
  RET_CHAIN_ORIGIN(id)
}

// Copy or move the origins of the len bytes from src to dst.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __dfsan_mem_origin_transfer(
    const void *dst, const void *src, uptr len) {
  if (src == dst)
    return;
  GET_CALLER_PC_BP;
  GET_STORE_STACK_TRACE_PC_BP(pc, bp);
  MoveOrigin(dst, src, len, &stack);
}

SANITIZER_INTERFACE_ATTRIBUTE void dfsan_mem_origin_transfer(const void *dst,
                                                             const void *src,
                                                             uptr len) {
  __dfsan_mem_origin_transfer(dst, src, len);
}

namespace __dfsan {

bool dfsan_inited = false;
bool dfsan_init_is_running = false;

void dfsan_copy_memory(void *dst, const void *src, uptr size) {
  internal_memcpy(dst, src, size);
  internal_memcpy((void *)shadow_for(dst), (const void *)shadow_for(src),
                  size * sizeof(dfsan_label));
  if (__dfsan_get_track_origins())
    dfsan_mem_origin_transfer(dst, src, size);
}

}  // namespace __dfsan

// If the label s is tainted, set the size bytes from the address p to be a new
// origin chain with the previous ID o and the current stack trace. This is
// used by instrumentation to reduce code size when too much code is inserted.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __dfsan_maybe_store_origin(
    dfsan_label s, void *p, uptr size, dfsan_origin o) {
  if (UNLIKELY(s)) {
    GET_CALLER_PC_BP_SP;
    (void)sp;
    GET_STORE_STACK_TRACE_PC_BP(pc, bp);
    SetOrigin(p, size, ChainOrigin(o, &stack));
  }
}

// Releases the pages within the origin address range.
static void ReleaseOrigins(void *addr, uptr size) {
  const uptr beg_origin_addr = (uptr)__dfsan::origin_for(addr);
  const void *end_addr = (void *)((uptr)addr + size);
  const uptr end_origin_addr = (uptr)__dfsan::origin_for(end_addr);

  if (end_origin_addr - beg_origin_addr <
      common_flags()->clear_shadow_mmap_threshold)
    return;

  const uptr page_size = GetPageSizeCached();
  const uptr beg_aligned = RoundUpTo(beg_origin_addr, page_size);
  const uptr end_aligned = RoundDownTo(end_origin_addr, page_size);

  if (!MmapFixedSuperNoReserve(beg_aligned, end_aligned - beg_aligned))
    Die();
}

// Releases the pages within the shadow address range, and sets
// the shadow addresses not on the pages to be 0.
static void ReleaseOrClearShadows(void *addr, uptr size) {
  const uptr beg_shadow_addr = (uptr)__dfsan::shadow_for(addr);
  const void *end_addr = (void *)((uptr)addr + size);
  const uptr end_shadow_addr = (uptr)__dfsan::shadow_for(end_addr);

  if (end_shadow_addr - beg_shadow_addr <
      common_flags()->clear_shadow_mmap_threshold)
    return WriteShadowWithSize(0, beg_shadow_addr, size);

  const uptr page_size = GetPageSizeCached();
  const uptr beg_aligned = RoundUpTo(beg_shadow_addr, page_size);
  const uptr end_aligned = RoundDownTo(end_shadow_addr, page_size);

  if (beg_aligned >= end_aligned) {
    WriteShadowWithSize(0, beg_shadow_addr, size);
  } else {
    if (beg_aligned != beg_shadow_addr)
      WriteShadowInRange(0, beg_shadow_addr, beg_aligned);
    if (end_aligned != end_shadow_addr)
      WriteShadowInRange(0, end_aligned, end_shadow_addr);
    if (!MmapFixedSuperNoReserve(beg_aligned, end_aligned - beg_aligned))
      Die();
  }
}

void SetShadow(dfsan_label label, void *addr, uptr size, dfsan_origin origin) {
  if (0 != label) {
    const uptr beg_shadow_addr = (uptr)__dfsan::shadow_for(addr);
    WriteShadowWithSize(label, beg_shadow_addr, size);
    if (__dfsan_get_track_origins())
      SetOrigin(addr, size, origin);
    return;
  }

  if (__dfsan_get_track_origins())
    ReleaseOrigins(addr, size);

  ReleaseOrClearShadows(addr, size);
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __dfsan_set_label(
    dfsan_label label, dfsan_origin origin, void *addr, uptr size) {
  SetShadow(label, addr, size, origin);
}

SANITIZER_INTERFACE_ATTRIBUTE
void dfsan_set_label(dfsan_label label, void *addr, uptr size) {
  dfsan_origin init_origin = 0;
  if (label && __dfsan_get_track_origins()) {
    GET_CALLER_PC_BP;
    GET_STORE_STACK_TRACE_PC_BP(pc, bp);
    init_origin = ChainOrigin(0, &stack, true);
  }
  SetShadow(label, addr, size, init_origin);
}

SANITIZER_INTERFACE_ATTRIBUTE
void dfsan_add_label(dfsan_label label, void *addr, uptr size) {
  if (0 == label)
    return;

  if (__dfsan_get_track_origins()) {
    GET_CALLER_PC_BP;
    GET_STORE_STACK_TRACE_PC_BP(pc, bp);
    dfsan_origin init_origin = ChainOrigin(0, &stack, true);
    SetOrigin(addr, size, init_origin);
  }

  for (dfsan_label *labelp = shadow_for(addr); size != 0; --size, ++labelp)
    *labelp |= label;
}

// Unlike the other dfsan interface functions the behavior of this function
// depends on the label of one of its arguments.  Hence it is implemented as a
// custom function.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE dfsan_label
__dfsw_dfsan_get_label(long data, dfsan_label data_label,
                       dfsan_label *ret_label) {
  *ret_label = 0;
  return data_label;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE dfsan_label __dfso_dfsan_get_label(
    long data, dfsan_label data_label, dfsan_label *ret_label,
    dfsan_origin data_origin, dfsan_origin *ret_origin) {
  *ret_label = 0;
  *ret_origin = 0;
  return data_label;
}

// This function is used if dfsan_get_origin is called when origin tracking is
// off.
extern "C" SANITIZER_INTERFACE_ATTRIBUTE dfsan_origin __dfsw_dfsan_get_origin(
    long data, dfsan_label data_label, dfsan_label *ret_label) {
  *ret_label = 0;
  return 0;
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE dfsan_origin __dfso_dfsan_get_origin(
    long data, dfsan_label data_label, dfsan_label *ret_label,
    dfsan_origin data_origin, dfsan_origin *ret_origin) {
  *ret_label = 0;
  *ret_origin = 0;
  return data_origin;
}

SANITIZER_INTERFACE_ATTRIBUTE dfsan_label
dfsan_read_label(const void *addr, uptr size) {
  if (size == 0)
    return 0;
  return __dfsan_union_load(shadow_for(addr), size);
}

SANITIZER_INTERFACE_ATTRIBUTE dfsan_origin
dfsan_read_origin_of_first_taint(const void *addr, uptr size) {
  return GetOriginIfTainted((uptr)addr, size);
}

SANITIZER_INTERFACE_ATTRIBUTE void dfsan_set_label_origin(dfsan_label label,
                                                          dfsan_origin origin,
                                                          void *addr,
                                                          uptr size) {
  __dfsan_set_label(label, origin, addr, size);
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE int
dfsan_has_label(dfsan_label label, dfsan_label elem) {
  return (label & elem) == elem;
}

class Decorator : public __sanitizer::SanitizerCommonDecorator {
 public:
  Decorator() : SanitizerCommonDecorator() {}
  const char *Origin() const { return Magenta(); }
};

namespace {

void PrintNoOriginTrackingWarning() {
  Decorator d;
  Printf(
      "  %sDFSan: origin tracking is not enabled. Did you specify the "
      "-dfsan-track-origins=1 option?%s\n",
      d.Warning(), d.Default());
}

void PrintNoTaintWarning(const void *address) {
  Decorator d;
  Printf("  %sDFSan: no tainted value at %x%s\n", d.Warning(), address,
         d.Default());
}

void PrintInvalidOriginWarning(dfsan_label label, const void *address) {
  Decorator d;
  Printf(
      "  %sTaint value 0x%x (at %p) has invalid origin tracking. This can "
      "be a DFSan bug.%s\n",
      d.Warning(), label, address, d.Default());
}

bool PrintOriginTraceToStr(const void *addr, const char *description,
                           InternalScopedString *out) {
  CHECK(out);
  CHECK(__dfsan_get_track_origins());
  Decorator d;

  const dfsan_label label = *__dfsan::shadow_for(addr);
  CHECK(label);

  const dfsan_origin origin = *__dfsan::origin_for(addr);

  out->append("  %sTaint value 0x%x (at %p) origin tracking (%s)%s\n",
              d.Origin(), label, addr, description ? description : "",
              d.Default());

  Origin o = Origin::FromRawId(origin);
  bool found = false;

  while (o.isChainedOrigin()) {
    StackTrace stack;
    dfsan_origin origin_id = o.raw_id();
    o = o.getNextChainedOrigin(&stack);
    if (o.isChainedOrigin())
      out->append(
          "  %sOrigin value: 0x%x, Taint value was stored to memory at%s\n",
          d.Origin(), origin_id, d.Default());
    else
      out->append("  %sOrigin value: 0x%x, Taint value was created at%s\n",
                  d.Origin(), origin_id, d.Default());

    // Includes a trailing newline, so no need to add it again.
    stack.PrintTo(out);
    found = true;
  }

  return found;
}

}  // namespace

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void dfsan_print_origin_trace(
    const void *addr, const char *description) {
  if (!__dfsan_get_track_origins()) {
    PrintNoOriginTrackingWarning();
    return;
  }

  const dfsan_label label = *__dfsan::shadow_for(addr);
  if (!label) {
    PrintNoTaintWarning(addr);
    return;
  }

  InternalScopedString trace;
  bool success = PrintOriginTraceToStr(addr, description, &trace);

  if (trace.length())
    Printf("%s", trace.data());

  if (!success)
    PrintInvalidOriginWarning(label, addr);
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE size_t
dfsan_sprint_origin_trace(const void *addr, const char *description,
                          char *out_buf, size_t out_buf_size) {
  CHECK(out_buf);

  if (!__dfsan_get_track_origins()) {
    PrintNoOriginTrackingWarning();
    return 0;
  }

  const dfsan_label label = *__dfsan::shadow_for(addr);
  if (!label) {
    PrintNoTaintWarning(addr);
    return 0;
  }

  InternalScopedString trace;
  bool success = PrintOriginTraceToStr(addr, description, &trace);

  if (!success) {
    PrintInvalidOriginWarning(label, addr);
    return 0;
  }

  if (out_buf_size) {
    internal_strncpy(out_buf, trace.data(), out_buf_size - 1);
    out_buf[out_buf_size - 1] = '\0';
  }

  return trace.length();
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE dfsan_origin
dfsan_get_init_origin(const void *addr) {
  if (!__dfsan_get_track_origins())
    return 0;

  const dfsan_label label = *__dfsan::shadow_for(addr);
  if (!label)
    return 0;

  const dfsan_origin origin = *__dfsan::origin_for(addr);

  Origin o = Origin::FromRawId(origin);
  dfsan_origin origin_id = o.raw_id();
  while (o.isChainedOrigin()) {
    StackTrace stack;
    origin_id = o.raw_id();
    o = o.getNextChainedOrigin(&stack);
  }
  return origin_id;
}

void __sanitizer::BufferedStackTrace::UnwindImpl(uptr pc, uptr bp,
                                                 void *context,
                                                 bool request_fast,
                                                 u32 max_depth) {
  using namespace __dfsan;
  DFsanThread *t = GetCurrentThread();
  if (!t || !StackTrace::WillUseFastUnwind(request_fast)) {
    return Unwind(max_depth, pc, bp, context, 0, 0, false);
  }
  Unwind(max_depth, pc, bp, nullptr, t->stack_top(), t->stack_bottom(), true);
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE void __sanitizer_print_stack_trace() {
  GET_CALLER_PC_BP;
  GET_STORE_STACK_TRACE_PC_BP(pc, bp);
  stack.Print();
}

extern "C" SANITIZER_INTERFACE_ATTRIBUTE size_t
dfsan_sprint_stack_trace(char *out_buf, size_t out_buf_size) {
  CHECK(out_buf);
  GET_CALLER_PC_BP;
  GET_STORE_STACK_TRACE_PC_BP(pc, bp);
  return stack.PrintTo(out_buf, out_buf_size);
}

void Flags::SetDefaults() {
#define DFSAN_FLAG(Type, Name, DefaultValue, Description) Name = DefaultValue;
#include "dfsan_flags.inc"
#undef DFSAN_FLAG
}

static void RegisterDfsanFlags(FlagParser *parser, Flags *f) {
#define DFSAN_FLAG(Type, Name, DefaultValue, Description) \
  RegisterFlag(parser, #Name, Description, &f->Name);
#include "dfsan_flags.inc"
#undef DFSAN_FLAG
}

static void InitializeFlags() {
  SetCommonFlagsDefaults();
  {
    CommonFlags cf;
    cf.CopyFrom(*common_flags());
    cf.intercept_tls_get_addr = true;
    OverrideCommonFlags(cf);
  }
  flags().SetDefaults();

  FlagParser parser;
  RegisterCommonFlags(&parser);
  RegisterDfsanFlags(&parser, &flags());
  parser.ParseStringFromEnv("DFSAN_OPTIONS");
  InitializeCommonFlags();
  if (Verbosity()) ReportUnrecognizedFlags();
  if (common_flags()->help) parser.PrintFlagDescriptions();
}

SANITIZER_INTERFACE_ATTRIBUTE
void dfsan_clear_arg_tls(uptr offset, uptr size) {
  internal_memset((void *)((uptr)__dfsan_arg_tls + offset), 0, size);
}

SANITIZER_INTERFACE_ATTRIBUTE
void dfsan_clear_thread_local_state() {
  internal_memset(__dfsan_arg_tls, 0, sizeof(__dfsan_arg_tls));
  internal_memset(__dfsan_retval_tls, 0, sizeof(__dfsan_retval_tls));

  if (__dfsan_get_track_origins()) {
    internal_memset(__dfsan_arg_origin_tls, 0, sizeof(__dfsan_arg_origin_tls));
    internal_memset(&__dfsan_retval_origin_tls, 0,
                    sizeof(__dfsan_retval_origin_tls));
  }
}

extern "C" void dfsan_flush() {
  if (!MmapFixedSuperNoReserve(ShadowAddr(), UnusedAddr() - ShadowAddr()))
    Die();
}

static void DFsanInit(int argc, char **argv, char **envp) {
  CHECK(!dfsan_init_is_running);
  if (dfsan_inited)
    return;
  dfsan_init_is_running = true;
  SanitizerToolName = "DataflowSanitizer";

  AvoidCVE_2016_2143();

  InitializeFlags();

  dfsan_flush();
  if (common_flags()->use_madv_dontdump)
    DontDumpShadowMemory(ShadowAddr(), UnusedAddr() - ShadowAddr());

  // Protect the region of memory we don't use, to preserve the one-to-one
  // mapping from application to shadow memory. But if ASLR is disabled, Linux
  // will load our executable in the middle of our unused region. This mostly
  // works so long as the program doesn't use too much memory. We support this
  // case by disabling memory protection when ASLR is disabled.
  uptr init_addr = (uptr)&DFsanInit;
  if (!(init_addr >= UnusedAddr() && init_addr < AppAddr()))
    MmapFixedNoAccess(UnusedAddr(), AppAddr() - UnusedAddr());

  initialize_interceptors();

  // Set up threads
  DFsanTSDInit(DFsanTSDDtor);

  dfsan_allocator_init();

  DFsanThread *main_thread = DFsanThread::Create(nullptr, nullptr, nullptr);
  SetCurrentThread(main_thread);
  main_thread->ThreadStart();

  dfsan_init_is_running = false;
  dfsan_inited = true;
}

namespace __dfsan {

void dfsan_init() { DFsanInit(0, nullptr, nullptr); }

}  // namespace __dfsan

#if SANITIZER_CAN_USE_PREINIT_ARRAY
__attribute__((section(".preinit_array"),
               used)) static void (*dfsan_init_ptr)(int, char **,
                                                    char **) = DFsanInit;
#endif
