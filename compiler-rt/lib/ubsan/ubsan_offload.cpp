//===-- ubsan_offload.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ubsan_handlers.h"
#include "ubsan_value.h"

#include "ubsan_offload_packet.h"

#include <gpuintrin.h>

#include "shared/rpc.h"

using namespace __ubsan;
using namespace __sanitizer;

[[gnu::visibility("protected"),
  gnu::weak]] rpc::Client __ubsan_rpc_client asm("__llvm_rpc_client");

// CHECK() has no host reporter on the device.
namespace __sanitizer {
void NORETURN CheckFailed(const char *, int, const char *, u64, u64) {
  __builtin_verbose_trap("UndefinedBehaviorSanitizer", "internal check failed");
}
} // namespace __sanitizer

namespace {

// Shallow deduplication pool to avoid flooding the interface.
bool seen(uptr Pc) {
  static constexpr u64 Bits = 6;
  static constexpr u64 Golden = 0x9E3779B97F4A7C15ull; // 2^64 / phi.
  static uptr Table[1u << Bits] = {};
  unsigned Index = (Pc * Golden) >> (sizeof(u64) * 8u - Bits);
  uptr *Last = &Table[Index];
  return __scoped_atomic_exchange_n(Last, Pc, __ATOMIC_RELAXED,
                                    __MEMORY_SCOPE_DEVICE) == Pc;
}

// TODO: Handle __int128 arguments that are passed by-pointer instead.
void report(uptr Pc, __ubsan_report_kind Kind, bool Fatal, const void *Data,
            ValueHandle Val0 = 0, ValueHandle Val1 = 0, ValueHandle Val2 = 0) {
  if (seen(Pc))
    return;

  rpc::Client::Port Port =
      __ubsan_rpc_client.open<UBSAN_OFFLOAD_REPORT_OPCODE>();
  Port.send([&](rpc::Buffer *Buf, uint32_t) {
    auto &Rep = *reinterpret_cast<__ubsan_offload_report *>(Buf);
    Rep.pc = static_cast<uint64_t>(Pc);
    Rep.data = static_cast<uint64_t>(reinterpret_cast<uptr>(Data));
    Rep.val0 = static_cast<uint64_t>(Val0);
    Rep.val1 = static_cast<uint64_t>(Val1);
    Rep.val2 = static_cast<uint64_t>(Val2);
    Rep.kind = static_cast<uint8_t>(Kind);
    Rep.fatal = Fatal;
  });
}

} // namespace

extern "C" {

#define UBSAN_OFFLOAD_HANDLER(kind, name, reason, size, locoff, nloc, ntype,   \
                              flags, params, ...)                              \
  [[gnu::cold, gnu::noinline]] void __ubsan_handle_##name params {             \
    report(GET_CALLER_PC(), UBSAN_OFFLOAD_##kind, false, __VA_ARGS__);         \
  }                                                                            \
  [[gnu::cold, gnu::noinline]] void __ubsan_handle_##name##_abort params {     \
    report(GET_CALLER_PC(), UBSAN_OFFLOAD_##kind, true, __VA_ARGS__);          \
    __builtin_verbose_trap("UndefinedBehaviorSanitizer", reason);              \
  }

#define UBSAN_OFFLOAD_HANDLER_NORETURN(kind, name, reason, size, locoff, nloc, \
                                       ntype, flags, params, ...)              \
  [[gnu::cold, gnu::noinline]] void __ubsan_handle_##name params {             \
    report(GET_CALLER_PC(), UBSAN_OFFLOAD_##kind, true, __VA_ARGS__);          \
    __builtin_verbose_trap("UndefinedBehaviorSanitizer", reason);              \
  }

#include "ubsan_offload_checks.inc"

} // extern "C"
