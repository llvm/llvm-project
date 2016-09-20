#include "xray_interface_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include <atomic>
#include <cstdint>
#include <limits>

namespace __xray {

static constexpr uint8_t CallOpCode = 0xe8;
static constexpr uint16_t MovR10Seq = 0xba41;
static constexpr uint16_t Jmp9Seq = 0x09eb;
static constexpr uint8_t JmpOpCode = 0xe9;
static constexpr uint8_t RetOpCode = 0xc3;

static constexpr int64_t MinOffset{std::numeric_limits<int32_t>::min()};
static constexpr int64_t MaxOffset{std::numeric_limits<int32_t>::max()};

bool patchFunctionEntry(const bool Enable, const uint32_t FuncId, const XRaySledEntry& Sled)
{
  // Here we do the dance of replacing the following sled:
  //
  // xray_sled_n:
  //   jmp +9
  //   <9 byte nop>
  //
  // With the following:
  //
  //   mov r10d, <function id>
  //   call <relative 32bit offset to entry trampoline>
  //
  // We need to do this in the following order:
  //
  // 1. Put the function id first, 2 bytes from the start of the sled (just
  // after the 2-byte jmp instruction).
  // 2. Put the call opcode 6 bytes from the start of the sled.
  // 3. Put the relative offset 7 bytes from the start of the sled.
  // 4. Do an atomic write over the jmp instruction for the "mov r10d"
  // opcode and first operand.
  //
  // Prerequisite is to compute the relative offset to the
  // __xray_FunctionEntry function's address.
  int64_t TrampolineOffset =
      reinterpret_cast<int64_t>(__xray_FunctionEntry) -
      (static_cast<int64_t>(Sled.Address) + 11);
  if (TrampolineOffset < MinOffset || TrampolineOffset > MaxOffset) {
    Report("XRay Entry trampoline (%p) too far from sled (%p); distance = "
           "%ld\n",
           __xray_FunctionEntry, reinterpret_cast<void *>(Sled.Address),
           TrampolineOffset);
    return false;
  }
  if (Enable) {
    *reinterpret_cast<uint32_t *>(Sled.Address + 2) = FuncId;
    *reinterpret_cast<uint8_t *>(Sled.Address + 6) = CallOpCode;
    *reinterpret_cast<uint32_t *>(Sled.Address + 7) = TrampolineOffset;
    std::atomic_store_explicit(
        reinterpret_cast<std::atomic<uint16_t> *>(Sled.Address), MovR10Seq,
        std::memory_order_release);
  } else {
    std::atomic_store_explicit(
        reinterpret_cast<std::atomic<uint16_t> *>(Sled.Address), Jmp9Seq,
        std::memory_order_release);
    // FIXME: Write out the nops still?
  }
  return true;
}

bool patchFunctionExit(const bool Enable, const uint32_t FuncId, const XRaySledEntry& Sled)
{
  // Here we do the dance of replacing the following sled:
  //
  // xray_sled_n:
  //   ret
  //   <10 byte nop>
  //
  // With the following:
  //
  //   mov r10d, <function id>
  //   jmp <relative 32bit offset to exit trampoline>
  //
  // 1. Put the function id first, 2 bytes from the start of the sled (just
  // after the 1-byte ret instruction).
  // 2. Put the jmp opcode 6 bytes from the start of the sled.
  // 3. Put the relative offset 7 bytes from the start of the sled.
  // 4. Do an atomic write over the jmp instruction for the "mov r10d"
  // opcode and first operand.
  //
  // Prerequisite is to compute the relative offset fo the
  // __xray_FunctionExit function's address.
  int64_t TrampolineOffset =
      reinterpret_cast<int64_t>(__xray_FunctionExit) -
      (static_cast<int64_t>(Sled.Address) + 11);
  if (TrampolineOffset < MinOffset || TrampolineOffset > MaxOffset) {
    Report("XRay Exit trampoline (%p) too far from sled (%p); distance = "
           "%ld\n",
           __xray_FunctionExit, reinterpret_cast<void *>(Sled.Address),
           TrampolineOffset);
    return false;
  }
  if (Enable) {
    *reinterpret_cast<uint32_t *>(Sled.Address + 2) = FuncId;
    *reinterpret_cast<uint8_t *>(Sled.Address + 6) = JmpOpCode;
    *reinterpret_cast<uint32_t *>(Sled.Address + 7) = TrampolineOffset;
    std::atomic_store_explicit(
        reinterpret_cast<std::atomic<uint16_t> *>(Sled.Address), MovR10Seq,
        std::memory_order_release);
  } else {
    std::atomic_store_explicit(
        reinterpret_cast<std::atomic<uint8_t> *>(Sled.Address), RetOpCode,
        std::memory_order_release);
    // FIXME: Write out the nops still?
  }
  return true;
}

} // namespace __xray
