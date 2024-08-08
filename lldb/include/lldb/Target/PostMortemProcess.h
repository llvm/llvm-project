//===-- PostMortemProcess.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_POSTMORTEMPROCESS_H
#define LLDB_TARGET_POSTMORTEMPROCESS_H

#include "lldb/Target/Process.h"
#include "lldb/Utility/RangeMap.h"

namespace lldb_private {

/// \class PostMortemProcess
/// Base class for all processes that don't represent a live process, such as
/// coredumps or processes traced in the past.
///
/// \a lldb_private::Process virtual functions overrides that are common
/// between these kinds of processes can have default implementations in this
/// class.
class PostMortemProcess : public Process {
  using Process::Process;

public:
  PostMortemProcess(lldb::TargetSP target_sp, lldb::ListenerSP listener_sp,
                    const FileSpec &core_file)
      : Process(target_sp, listener_sp), m_core_file(core_file) {}

  bool IsLiveDebugSession() const override { return false; }

  FileSpec GetCoreFile() const override { return m_core_file; }

protected:
  typedef lldb_private::Range<lldb::addr_t, lldb::addr_t> FileRange;
  typedef lldb_private::RangeDataVector<lldb::addr_t, lldb::addr_t, FileRange>
      VMRangeToFileOffset;
  typedef lldb_private::RangeDataVector<lldb::addr_t, lldb::addr_t, uint32_t>
      VMRangeToPermissions;

  virtual llvm::ArrayRef<uint8_t> PeekMemory(lldb::addr_t low,
                                             lldb::addr_t high);

  lldb::addr_t FindInMemory(lldb::addr_t low, lldb::addr_t high,
                            const uint8_t *buf, size_t size) override;

  llvm::ArrayRef<uint8_t> DoPeekMemory(lldb::ModuleSP &core_module_sp,
                                       VMRangeToFileOffset &core_aranges,
                                       lldb::addr_t low, lldb::addr_t high);

  FileSpec m_core_file;
};

} // namespace lldb_private

#endif // LLDB_TARGET_POSTMORTEMPROCESS_H
