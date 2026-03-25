//===-- ABI.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_ABI_H
#define LLDB_TARGET_ABI_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/DynamicRegisterInfo.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace llvm {
class Type;
}

namespace lldb_private {

/// \class ABI ABI.h "lldb/Target/ABI.h"
/// An abstract base class for ABI (Application Binary Interface) plugins.
///
/// ABI plugins encapsulate the calling conventions, register usage, data type
/// layouts, and low-level details specific to a particular architecture and
/// operating system combination. These plugins are essential for:
///
/// - Setting up function calls (PrepareTrivialCall)
/// - Extracting function arguments and return values from registers/stack
/// - Understanding register volatility and preserved registers across calls
/// - Creating stack frame unwind plans for exception handling and backtraces
/// - Fixing code/data addresses (e.g., ARM Thumb bit manipulation)
/// - Validating code and stack addresses
///
/// LLDB creates an ABI instance for each process being debugged. The ABI is
/// selected based on the process's architecture (ArchSpec) using the FindPlugin
/// static method, which iterates through all registered ABI plugins until one
/// returns a valid instance for the given architecture.
///
/// ABI plugins are typically instantiated early in the debugging session and
/// remain active for the lifetime of the process. The selection happens via:
/// ABI::FindPlugin(process_sp, arch) which queries all registered ABI
/// implementations through the PluginManager.
///
/// Key methods that subclasses must implement:
/// - PrepareTrivialCall: Set up registers/stack to call a function
/// - GetArgumentValues: Extract function arguments from the current frame
/// - GetReturnValueObjectImpl: Extract the return value after a function call
/// - SetReturnValueObject: Modify the return value in a stack frame
/// - CreateFunctionEntryUnwindPlan: Generate unwind rules at function entry
/// - CreateDefaultUnwindPlan: Generate default unwind rules for assembly
/// - RegisterIsVolatile: Determine if a register is caller-saved
/// - CallFrameAddressIsValid: Validate stack pointer alignment/values
/// - CodeAddressIsValid: Validate instruction pointer values
/// - AugmentRegisterInfo: Add ABI-specific register metadata
///
/// Implementations should be careful about:
/// - Thread safety: ABI methods may be called from multiple threads
/// - Handling both user-space and kernel-space calling conventions
/// - Properly masking address bits (e.g., ARM Thumb, AArch64 PAC/TBI)
/// - Supporting multiple calling conventions on the same architecture
/// - Correctly handling variadic functions and special argument types
class ABI : public PluginInterface {
public:
  struct CallArgument {
    enum eType {
      HostPointer = 0, /* pointer to host data */
      TargetValue,     /* value is on the target or literal */
    };
    eType type;  /* value of eType */
    size_t size; /* size in bytes of this argument */

    lldb::addr_t value;                 /* literal value */
    std::unique_ptr<uint8_t[]> data_up; /* host data pointer */
  };

  ~ABI() override;

  virtual size_t GetRedZoneSize() const = 0;

  virtual bool PrepareTrivialCall(lldb_private::Thread &thread, lldb::addr_t sp,
                                  lldb::addr_t functionAddress,
                                  lldb::addr_t returnAddress,
                                  llvm::ArrayRef<lldb::addr_t> args) const = 0;

  // Prepare trivial call used from ThreadPlanFunctionCallUsingABI
  // AD:
  //  . Because i don't want to change other ABI's this is not declared pure
  //  virtual.
  //    The dummy implementation will simply fail.  Only HexagonABI will
  //    currently
  //    use this method.
  //  . Two PrepareTrivialCall's is not good design so perhaps this should be
  //  combined.
  //
  virtual bool PrepareTrivialCall(lldb_private::Thread &thread, lldb::addr_t sp,
                                  lldb::addr_t functionAddress,
                                  lldb::addr_t returnAddress,
                                  llvm::Type &prototype,
                                  llvm::ArrayRef<CallArgument> args) const;

  virtual bool GetArgumentValues(Thread &thread, ValueList &values) const = 0;

  lldb::ValueObjectSP GetReturnValueObject(Thread &thread, CompilerType &type,
                                           bool persistent = true) const;

  // specialized to work with llvm IR types
  lldb::ValueObjectSP GetReturnValueObject(Thread &thread, llvm::Type &type,
                                           bool persistent = true) const;

  // Set the Return value object in the current frame as though a function with
  virtual Status SetReturnValueObject(lldb::StackFrameSP &frame_sp,
                                      lldb::ValueObjectSP &new_value) = 0;

protected:
  // This is the method the ABI will call to actually calculate the return
  // value. Don't put it in a persistent value object, that will be done by the
  // ABI::GetReturnValueObject.
  virtual lldb::ValueObjectSP
  GetReturnValueObjectImpl(Thread &thread, CompilerType &ast_type) const = 0;

  // specialized to work with llvm IR types
  virtual lldb::ValueObjectSP
  GetReturnValueObjectImpl(Thread &thread, llvm::Type &ir_type) const;

  /// Request to get a Process shared pointer.
  ///
  /// This ABI object may not have been created with a Process object,
  /// or the Process object may no longer be alive.  Be sure to handle
  /// the case where the shared pointer returned does not have an
  /// object inside it.
  lldb::ProcessSP GetProcessSP() const { return m_process_wp.lock(); }

public:
  virtual lldb::UnwindPlanSP CreateFunctionEntryUnwindPlan() = 0;

  virtual lldb::UnwindPlanSP CreateDefaultUnwindPlan() = 0;

  virtual bool RegisterIsVolatile(const RegisterInfo *reg_info) = 0;

  virtual bool GetFallbackRegisterLocation(
      const RegisterInfo *reg_info,
      UnwindPlan::Row::AbstractRegisterLocation &unwind_regloc);

  // Should take a look at a call frame address (CFA) which is just the stack
  // pointer value upon entry to a function. ABIs usually impose alignment
  // restrictions (4, 8 or 16 byte aligned), and zero is usually not allowed.
  // This function should return true if "cfa" is valid call frame address for
  // the ABI, and false otherwise. This is used by the generic stack frame
  // unwinding code to help determine when a stack ends.
  virtual bool CallFrameAddressIsValid(lldb::addr_t cfa) = 0;

  // Validates a possible PC value and returns true if an opcode can be at
  // "pc".
  virtual bool CodeAddressIsValid(lldb::addr_t pc) = 0;

  /// Some targets might use bits in a code address to indicate a mode switch.
  /// ARM uses bit zero to signify a code address is thumb, so any ARM ABI
  /// plug-ins would strip those bits.
  /// @{
  virtual lldb::addr_t FixCodeAddress(lldb::addr_t pc);
  virtual lldb::addr_t FixDataAddress(lldb::addr_t pc);
  /// @}

  /// Use this method when you do not know, or do not care what kind of address
  /// you are fixing. On platforms where there would be a difference between the
  /// two types, it will pick the safest option.
  ///
  /// Its purpose is to signal that no specific choice was made and provide an
  /// alternative to randomly picking FixCode/FixData address. Which could break
  /// platforms where there is a difference (only Arm Thumb at this time).
  virtual lldb::addr_t FixAnyAddress(lldb::addr_t pc) {
    // On Arm Thumb fixing a code address zeroes the bottom bit, so FixData is
    // the safe choice. On any other platform (so far) code and data addresses
    // are fixed in the same way.
    return FixDataAddress(pc);
  }

  llvm::MCRegisterInfo &GetMCRegisterInfo() { return *m_mc_register_info_up; }

  virtual void
  AugmentRegisterInfo(std::vector<DynamicRegisterInfo::Register> &regs) = 0;

  virtual bool GetPointerReturnRegister(const char *&name) { return false; }

  virtual uint64_t GetStackFrameSize() { return 512 * 1024; }

  static lldb::ABISP FindPlugin(lldb::ProcessSP process_sp, const ArchSpec &arch);

protected:
  ABI(lldb::ProcessSP process_sp, std::unique_ptr<llvm::MCRegisterInfo> info_up)
      : m_process_wp(process_sp), m_mc_register_info_up(std::move(info_up)) {
    assert(m_mc_register_info_up && "ABI must have MCRegisterInfo");
  }

  /// Utility function to construct a MCRegisterInfo using the ArchSpec triple.
  /// Plugins wishing to customize the construction can construct the
  /// MCRegisterInfo themselves.
  static std::unique_ptr<llvm::MCRegisterInfo>
  MakeMCRegisterInfo(const ArchSpec &arch);

  lldb::ProcessWP m_process_wp;
  std::unique_ptr<llvm::MCRegisterInfo> m_mc_register_info_up;

private:
  ABI(const ABI &) = delete;
  const ABI &operator=(const ABI &) = delete;
};

/// \class RegInfoBasedABI ABI.h "lldb/Target/ABI.h"
/// A concrete ABI base class that uses RegisterInfo arrays for register metadata.
///
/// RegInfoBasedABI is designed for ABI implementations that define their
/// register information using static RegisterInfo arrays, which is the
/// traditional approach used by most LLDB ABI plugins. This class provides
/// the AugmentRegisterInfo implementation that looks up register details
/// (DWARF/eh_frame register numbers, generic register kinds) from the
/// RegisterInfo array provided by GetRegisterInfoArray.
///
/// Subclasses must implement:
/// - GetRegisterInfoArray: Return a pointer to a static array of RegisterInfo
///   structures that describe all registers for this ABI
///
/// This approach is suitable for architectures where the register set is
/// well-defined and doesn't vary at runtime. For architectures with dynamic
/// register information (e.g., varying vector lengths), consider using
/// MCBasedABI instead.
class RegInfoBasedABI : public ABI {
public:
  void AugmentRegisterInfo(
      std::vector<DynamicRegisterInfo::Register> &regs) override;

protected:
  using ABI::ABI;

  bool GetRegisterInfoByName(llvm::StringRef name, RegisterInfo &info);

  virtual const RegisterInfo *GetRegisterInfoArray(uint32_t &count) = 0;
};

/// \class MCBasedABI ABI.h "lldb/Target/ABI.h"
/// A concrete ABI base class that derives register metadata from LLVM's MCRegisterInfo.
///
/// MCBasedABI is designed for ABI implementations that obtain register
/// information from LLVM's Machine Code (MC) layer rather than maintaining
/// separate static RegisterInfo arrays. This leverages LLVM's existing
/// register definitions and mappings (DWARF/eh_frame register numbers),
/// reducing code duplication and maintenance burden.
///
/// This class provides the AugmentRegisterInfo implementation that queries
/// the MCRegisterInfo object (passed to the ABI constructor) to populate
/// register metadata like DWARF numbers and eh_frame numbers.
///
/// Subclasses must implement:
/// - GetGenericNum: Map register names to LLDB generic register numbers
///   (e.g., LLDB_REGNUM_GENERIC_PC, LLDB_REGNUM_GENERIC_SP)
///
/// Subclasses may optionally override:
/// - GetMCName: Transform LLDB register names to MCRegisterInfo naming
///   conventions (e.g., case transformations, prefix changes)
/// - GetEHAndDWARFNums: Provide custom DWARF/eh_frame number mappings
///
/// This approach is particularly useful for:
/// - New architectures where LLVM support already exists
/// - Reducing code duplication between LLDB and LLVM
/// - Architectures with complex register aliasing schemes
class MCBasedABI : public ABI {
public:
  void AugmentRegisterInfo(
      std::vector<DynamicRegisterInfo::Register> &regs) override;

  /// If the register name is of the form "<from_prefix>[<number>]" then change
  /// the name to "<to_prefix>[<number>]". Otherwise, leave the name unchanged.
  static void MapRegisterName(std::string &reg, llvm::StringRef from_prefix,
                              llvm::StringRef to_prefix);

protected:
  using ABI::ABI;

  /// Return eh_frame and dwarf numbers for the given register.
  virtual std::pair<uint32_t, uint32_t> GetEHAndDWARFNums(llvm::StringRef reg);

  /// Return the generic number of the given register.
  virtual uint32_t GetGenericNum(llvm::StringRef reg) = 0;

  /// For the given (capitalized) lldb register name, return the name of this
  /// register in the MCRegisterInfo struct.
  virtual std::string GetMCName(std::string reg) { return reg; }
};

} // namespace lldb_private

#endif // LLDB_TARGET_ABI_H
