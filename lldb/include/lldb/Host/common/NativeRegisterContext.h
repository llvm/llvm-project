//===-- NativeRegisterContext.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_COMMON_NATIVEREGISTERCONTEXT_H
#define LLDB_HOST_COMMON_NATIVEREGISTERCONTEXT_H

#include "lldb/Host/common/NativeWatchpointList.h"
#include "lldb/Register/RegisterContext.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class NativeThreadProtocol;

enum class ExpeditedRegs { Minimal, Full };

class NativeRegisterContext : public RegisterContext {
public:
  // Constructors and Destructors
  NativeRegisterContext(NativeThreadProtocol &thread);
  NativeRegisterContext(NativeThreadProtocol &thread, const ArchSpec &arch,
                        RegisterLayout layout);

  NativeRegisterContext(NativeThreadProtocol &thread,
                        std::shared_ptr<const RegisterDescription> description);

  ~NativeRegisterContext() override;

  /// Byte position of an alias within its parent value, in target byte order.
  virtual uint32_t GetRegisterValueOffset(RegisterID id) const;

  std::shared_ptr<const RegisterDescription> GetDescription() const override;
  const RegisterLayout &GetLayout() const;
  const ArchSpec &GetTargetArchitecture() const { return m_arch; }
  size_t GetBankSize(RegisterBankID bank) const;
  void SetRegisterLayout(RegisterLayout layout);

  void InvalidateAllRegisters() override {}

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override = 0;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                const lldb_private::RegisterValue &reg_value) override = 0;

  virtual Status ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) = 0;

  virtual Status WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) = 0;

  // Subclasses can override these functions if desired
  virtual uint32_t NumSupportedHardwareBreakpoints();

  virtual uint32_t SetHardwareBreakpoint(lldb::addr_t addr, size_t size);

  virtual bool ClearHardwareBreakpoint(uint32_t hw_idx);

  virtual Status ClearAllHardwareBreakpoints();

  virtual Status GetHardwareBreakHitIndex(uint32_t &bp_index,
                                          lldb::addr_t trap_addr);

  virtual uint32_t NumSupportedHardwareWatchpoints();

  virtual uint32_t SetHardwareWatchpoint(lldb::addr_t addr, size_t size,
                                         uint32_t watch_flags);

  virtual bool ClearHardwareWatchpoint(uint32_t hw_index);

  virtual Status ClearWatchpointHit(uint32_t hw_index);

  virtual Status ClearAllHardwareWatchpoints();

  virtual Status IsWatchpointHit(uint32_t wp_index, bool &is_hit);

  virtual Status GetWatchpointHitIndex(uint32_t &wp_index,
                                       lldb::addr_t trap_addr);

  virtual Status IsWatchpointVacant(uint32_t wp_index, bool &is_vacant);

  virtual lldb::addr_t GetWatchpointAddress(uint32_t wp_index);

  // MIPS Linux kernel returns a masked address (last 3bits are masked)
  // when a HW watchpoint is hit. However user may not have set a watchpoint on
  // this address. This function emulates the instruction at PC and finds the
  // base address used in the load/store instruction. This gives the exact
  // address used to read/write the variable being watched. For example: 'n' is
  // at 0x120010d00 and 'm' is 0x120010d04. When a watchpoint is set at 'm',
  // then watch exception is generated even when 'n' is read/written. This
  // function returns address of 'n' so that client can check whether a
  // watchpoint is set on this address or not.
  virtual lldb::addr_t GetWatchpointHitAddress(uint32_t wp_index);

  virtual bool HardwareSingleStep(bool enable);

  // Subclasses should not override these
  virtual lldb::tid_t GetThreadID() const;

  virtual NativeThreadProtocol &GetThread() { return m_thread; }

  virtual std::vector<RegisterID>
  GetExpeditedRegisters(ExpeditedRegs expType) const;

  virtual bool RegisterOffsetIsDynamic() const { return false; }

  lldb::addr_t GetPC(lldb::addr_t fail_value = LLDB_INVALID_ADDRESS);

  virtual lldb::addr_t
  GetPCfromBreakpointLocation(lldb::addr_t fail_value = LLDB_INVALID_ADDRESS);

  Status SetPC(lldb::addr_t pc);

  lldb::addr_t GetSP(lldb::addr_t fail_value = LLDB_INVALID_ADDRESS);

  Status SetSP(lldb::addr_t sp);

  lldb::addr_t GetFP(lldb::addr_t fail_value = LLDB_INVALID_ADDRESS);

  Status SetFP(lldb::addr_t fp);

  lldb::addr_t GetReturnAddress(lldb::addr_t fail_value = LLDB_INVALID_ADDRESS);

  lldb::addr_t GetFlags(lldb::addr_t fail_value = 0);

protected:
  llvm::Expected<RegisterValue>
  ReadBankRegister(RegisterID id, llvm::ArrayRef<RegisterData> banks) const;
  llvm::Error
  WriteBankRegister(RegisterID id, const RegisterValue &value,
                    llvm::ArrayRef<MutableRegisterData> banks) const;
  // Classes that inherit from ThreadRegisterContext can see and modify these
  NativeThreadProtocol
      &m_thread; // The thread that this register context belongs to.
  // uint32_t m_stop_id;             // The stop ID that any data in this
  // context is valid for

private:
  ArchSpec m_arch;
  std::shared_ptr<const RegisterDescription> m_description;
  std::shared_ptr<const RegisterLayout> m_layout;
  NativeRegisterContext(const NativeRegisterContext &) = delete;
  const NativeRegisterContext &
  operator=(const NativeRegisterContext &) = delete;
};

} // namespace lldb_private

#endif // LLDB_HOST_COMMON_NATIVEREGISTERCONTEXT_H
