//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_THREADREGISTERCONTEXT_H
#define LLDB_TARGET_THREADREGISTERCONTEXT_H

#include "lldb/Register/RegisterContext.h"
#include "lldb/Register/RegisterDescription.h"
#include "lldb/Register/RegisterLayout.h"
#include "lldb/Target/ExecutionContextScope.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

class ThreadRegisterContext
    : public RegisterContext,
      public std::enable_shared_from_this<ThreadRegisterContext>,
                        public ExecutionContextScope {
public:
  // Constructors and Destructors
  ThreadRegisterContext(Thread &thread, uint32_t concrete_frame_idx);
  ThreadRegisterContext(Thread &thread, uint32_t concrete_frame_idx,
                        RegisterLayout layout);
  ThreadRegisterContext(Thread &thread, uint32_t concrete_frame_idx,
                        std::shared_ptr<const RegisterDescription> description);

  ~ThreadRegisterContext() override;

  void InvalidateIfNeeded(bool force);

  // Bank-backed contexts use the layout for metadata and byte decoding.
  void InvalidateAllRegisters() override {}

  std::shared_ptr<const RegisterDescription> GetDescription() const override;
  const RegisterLayout &GetLayout() const;
  const ArchSpec &GetTargetArchitecture() const;
  size_t GetBankSize(RegisterBankID bank) const;

  virtual lldb::ByteOrder GetByteOrder();

  llvm::Expected<lldb_private::RegisterValue>
  ReadRegister(lldb_private::RegisterID id) override = 0;

  llvm::Error
  WriteRegister(lldb_private::RegisterID id,
                const lldb_private::RegisterValue &reg_value) override = 0;

  virtual bool ReadAllRegisterValues(lldb::WritableDataBufferSP &data_sp) {
    return false;
  }

  virtual bool WriteAllRegisterValues(const lldb::DataBufferSP &data_sp) {
    return false;
  }

  virtual bool RegisterWriteCausesReconfigure(const llvm::StringRef name) {
    return false;
  }

  virtual bool ReconfigureRegisterLayout() { return false; }

  llvm::Expected<RegisterCheckpoint> SaveCheckpoint() override;

  bool CopyFromRegisterContext(lldb::ThreadRegisterContextSP context);

  // Subclasses can override these functions if desired
  virtual uint32_t NumSupportedHardwareBreakpoints();

  virtual uint32_t SetHardwareBreakpoint(lldb::addr_t addr, size_t size);

  virtual bool ClearHardwareBreakpoint(uint32_t hw_idx);

  virtual uint32_t NumSupportedHardwareWatchpoints();

  virtual uint32_t SetHardwareWatchpoint(lldb::addr_t addr, size_t size,
                                         bool read, bool write);

  virtual bool ClearHardwareWatchpoint(uint32_t hw_index);

  virtual bool HardwareSingleStep(bool enable);

  virtual Status
  ReadRegisterValueFromMemory(const lldb_private::RegisterDescriptor *reg_info,
                              lldb::addr_t src_addr, uint32_t src_len,
                              RegisterValue &reg_value);

  virtual Status
  WriteRegisterValueToMemory(const lldb_private::RegisterDescriptor *reg_info,
                             lldb::addr_t dst_addr, uint32_t dst_len,
                             const RegisterValue &reg_value);

  // Subclasses should not override these
  virtual lldb::tid_t GetThreadID() const;

  virtual Thread &GetThread() { return m_thread; }

  uint64_t GetPC(uint64_t fail_value = LLDB_INVALID_ADDRESS);

  // Returns the register value containing thread specific data, like TLS data
  // and other thread specific stuff.
  uint64_t GetThreadPointer(uint64_t fail_value = LLDB_INVALID_ADDRESS);

  /// Get an address suitable for symbolication.
  /// When symbolicating -- computing line, block, function --
  /// for a function in the middle of the stack, using the return
  /// address can lead to unexpected results for the user.
  /// A function that ends in a tail-call may have another function
  /// as the "return" address, but it will never actually return.
  /// Or a noreturn call in the middle of a function is the end of
  /// a block of instructions, and a DWARF location list entry for
  /// the return address may be a very different code path with
  /// incorrect results when printing variables for this frame.
  ///
  /// At a source line view, the user expects the current-line indictation
  /// to point to the function call they're under, not the next source line.
  ///
  /// The return address (GetPC()) should always be shown to the user,
  /// but when computing context, keeping within the bounds of the
  /// call instruction is what the user expects to see.
  ///
  /// \param [out] address
  ///     An Address object that will be filled in, if a PC can be retrieved.
  ///
  /// \return
  ///     Returns true if the Address param was filled in.
  bool GetPCForSymbolication(Address &address);

  bool SetPC(uint64_t pc);

  bool SetPC(Address addr);

  uint64_t GetSP(uint64_t fail_value = LLDB_INVALID_ADDRESS);

  bool SetSP(uint64_t sp);

  uint64_t GetFP(uint64_t fail_value = LLDB_INVALID_ADDRESS);

  bool SetFP(uint64_t fp);

  uint64_t GetReturnAddress(uint64_t fail_value = LLDB_INVALID_ADDRESS);

  uint64_t GetFlags(uint64_t fail_value = 0);

  // lldb::ExecutionContextScope pure virtual functions
  lldb::TargetSP CalculateTarget() override;

  lldb::ProcessSP CalculateProcess() override;

  lldb::ThreadSP CalculateThread() override;

  lldb::StackFrameSP CalculateStackFrame() override;

  void CalculateExecutionContext(ExecutionContext &exe_ctx) override;

  uint32_t GetStopID() const { return m_stop_id; }

  void SetStopID(uint32_t stop_id) { m_stop_id = stop_id; }

protected:
  void SetRegisterLayout(RegisterLayout layout);
  void SetRegisterDescription(
      std::shared_ptr<const RegisterDescription> description);
  llvm::Expected<RegisterValue>
  ReadBankRegister(RegisterID id, llvm::ArrayRef<RegisterData> banks) const;

  /// Indicates that this frame is currently executing code,
  /// that the PC value is not a return-pc but an actual executing
  /// instruction.  Some places in lldb will treat a return-pc
  /// value differently than the currently-executing-pc value,
  /// and this method can indicate if that should be done.
  /// The base class implementation only uses the frame index,
  /// but subclasses may have additional information that they
  /// can use to detect frames in this state, for instance a
  /// frame above a trap handler (sigtramp etc)..
  virtual bool BehavesLikeZerothFrame() const {
    return m_concrete_frame_idx == 0;
  }

  // Classes that inherit from ThreadRegisterContext can see and modify these
  Thread &m_thread; // The thread that this register context belongs to.
  uint32_t m_concrete_frame_idx; // The concrete frame index for this register
                                 // context
  uint32_t m_stop_id; // The stop ID that any data in this context is valid for
private:
  std::shared_ptr<const RegisterDescription> m_description;
  std::shared_ptr<const RegisterLayout> m_layout;

  // For ThreadRegisterContext only
  ThreadRegisterContext(const ThreadRegisterContext &) = delete;
  const ThreadRegisterContext &operator=(const ThreadRegisterContext &) = delete;
};

} // namespace lldb_private

#endif // LLDB_TARGET_THREADREGISTERCONTEXT_H
