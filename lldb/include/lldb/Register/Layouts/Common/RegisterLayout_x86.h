//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_X86_H
#define LLDB_REGISTER_LAYOUTS_COMMON_REGISTERLAYOUT_X86_H

#include "lldb/Register/RegisterLayout.h"
#include "llvm/Support/Compiler.h"
#include <array>
#include <cstdint>
#include <utility>

namespace lldb_private {

/// Map shared x86 saved formats to an OS-selected register description.
class RegisterLayout_x86 : public RegisterLayout {
protected:
  RegisterLayout_x86() = default;
  RegisterLayout_x86(const RegisterLayout_x86 &) = delete;
  RegisterLayout_x86(RegisterLayout_x86 &&) = delete;

public:
  // Generic floating-point registers

  LLVM_PACKED_START
  struct MMSRegComp {
    uint64_t mantissa;
    uint16_t sign_exp;
  };

  struct MMSReg {
    union {
      uint8_t bytes[10];
      MMSRegComp comp;
    };
    uint8_t pad[6];
  };
  LLVM_PACKED_END

  static_assert(sizeof(MMSRegComp) == 10, "MMSRegComp is not 10 bytes of size");
  static_assert(sizeof(MMSReg) == 16, "MMSReg is not 16 bytes of size");

  struct XMMReg {
    uint8_t bytes[16]; // 128-bits for each XMM register
  };

  // i387_fxsave_struct
  struct FXSAVE {
    uint16_t fctrl; // FPU Control Word (fcw)
    uint16_t fstat; // FPU Status Word (fsw)
    uint16_t ftag;  // FPU Tag Word (ftw)
    uint16_t fop;   // Last Instruction Opcode (fop)
    union {
      struct {
        uint64_t fip; // Instruction Pointer
        uint64_t fdp; // Data Pointer
      } x86_64;
      struct {
        uint32_t fioff; // FPU IP Offset (fip)
        uint32_t fiseg; // FPU IP Selector (fcs)
        uint32_t fooff; // FPU Operand Pointer Offset (foo)
        uint32_t foseg; // FPU Operand Pointer Selector (fos)
      } x86_32;
    } ptr;
    uint32_t mxcsr;     // MXCSR Register State
    uint32_t mxcsrmask; // MXCSR Mask
    MMSReg stmm[8];     // 8*16 bytes for each FP-reg = 128 bytes
    XMMReg xmm[16];     // 16*16 bytes for each XMM-reg = 256 bytes
    uint8_t padding1[48];
    uint64_t xcr0;
    uint8_t padding2[40];
  };

  // Extended floating-point registers

  struct YMMHReg {
    uint8_t bytes[16]; // 16 * 8 bits for the high bytes of each YMM register
  };

  struct MPXReg {
    uint8_t bytes[16]; // MPX 128 bit bound registers
  };

  struct MPXCsr {
    uint8_t bytes[8]; // MPX 64 bit bndcfgu and bndstatus registers
                      // (collectively BNDCSR state)
  };

  enum Bank : RegisterBankID { GP, FP, YMMHigh, Bounds, BoundsControl, Debug };

  /// Live contexts expose a full x87 tag word; ELF-core retains its historical
  /// saved, abridged view. Both interpretations use the ordinary layout API.
  enum class TagWordView { Full, Abridged };

  /// Shared conversion between the FXSAVE tag byte and a full logical tag word.
  /// Layouts declare the tag byte, status word, and saved x87 values as inputs.
  /// The returned object has static lifetime and performs no process access.
  static const RegisterConversion &GetX87TagConversion();

  /// Native 32-bit slots or the corresponding fields of a 64-bit GPR bank.
  enum class GPRMode { Native, Compat64 };

protected:
  using GPRLookup =
      llvm::function_ref<std::optional<RegisterPiece>(RegisterID)>;

  /// Return the saved 64-bit field ID for a 32-bit logical register.
  static RegisterID GetX86_64GPR(RegisterID id);

  static std::vector<std::pair<RegisterID, uint32_t>>
  GetGPRWidths(RegisterID register_count, GPRLookup get_field);

  static std::array<RegisterBank, 5> GetHardwareBanks(uint32_t debug_slot_size);

  static std::optional<RegisterLocation>
  GetRegisterLocation(const RegisterDescriptor &reg, GPRLookup get_field,
                      bool is_64_bit, uint32_t debug_slot_size,
                      TagWordView tag_word);

private:
  static uint16_t AbridgedToFullTagWord(uint8_t abridged_tw, uint16_t sw,
                                        llvm::ArrayRef<MMSReg> st_regs);

  static uint8_t FullToAbridgedTagWord(uint16_t tw);

  static llvm::Error ReadX87Tag(llvm::ArrayRef<RegisterData> banks,
                                lldb::ByteOrder byte_order,
                                llvm::MutableArrayRef<uint8_t> value);

  static llvm::Error WriteX87Tag(llvm::ArrayRef<uint8_t> value,
                                 lldb::ByteOrder byte_order,
                                 llvm::ArrayRef<MutableRegisterData> banks);
};

} // namespace lldb_private

#endif
