//===-- MainLoopWindows.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_WINDOWS_MAINLOOPWINDOWS_H
#define LLDB_HOST_WINDOWS_MAINLOOPWINDOWS_H

#include "lldb/Host/Config.h"
#include "lldb/Host/MainLoopBase.h"
#include <csignal>
#include <list>
#include <vector>

namespace lldb_private {

// Windows-specific implementation of the MainLoopBase class. It can monitor
// socket descriptors for readability using WSAEventSelect. Non-socket file
// descriptors are not supported.
class MainLoopWindows : public MainLoopBase {
public:
  MainLoopWindows();
  ~MainLoopWindows() override;

  ReadHandleUP RegisterReadObject(const lldb::IOObjectSP &object_sp,
                                  const Callback &callback,
                                  Status &error) override;

  Status Run() override;

  struct FdInfo {
    FdInfo(intptr_t event, Callback callback)
        : event(event), callback(callback) {}
    virtual ~FdInfo() {}
    virtual void WillPoll() {}
    virtual void DidPoll() {}
    virtual void Disarm() {}
    intptr_t event;
    Callback callback;
  };
  using FdInfoUP = std::unique_ptr<FdInfo>;

protected:
  void UnregisterReadObject(IOObject::WaitableHandle handle) override;

  void Interrupt() override;

private:
  llvm::Expected<size_t> Poll();

  llvm::DenseMap<IOObject::WaitableHandle, FdInfoUP> m_read_fds;
  void *m_interrupt_event;
};

} // namespace lldb_private

#endif // LLDB_HOST_WINDOWS_MAINLOOPWINDOWS_H
