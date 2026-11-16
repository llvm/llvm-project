//===-- NativeThreadProtocol.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_COMMON_NATIVETHREADPROTOCOL_H
#define LLDB_HOST_COMMON_NATIVETHREADPROTOCOL_H

#include <memory>

#include "lldb/Host/Debug.h"
#include "lldb/Utility/UnimplementedError.h"
#include "lldb/lldb-private-forward.h"
#include "lldb/lldb-types.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include <optional>

namespace lldb_private {
// NativeThreadProtocol
class NativeThreadProtocol {
public:
  NativeThreadProtocol(NativeProcessProtocol &process, lldb::tid_t tid,
                       std::optional<lldb::tid_t> lane_id = std::nullopt,
                       std::optional<lldb::tid_t> simd_id = std::nullopt);

  virtual ~NativeThreadProtocol() = default;

  virtual std::string GetName() = 0;

  virtual lldb::StateType GetState() = 0;

  virtual NativeRegisterContext &GetRegisterContext() = 0;

  virtual bool GetStopReason(ThreadStopInfo &stop_info,
                             std::string &description) = 0;

  lldb::tid_t GetID() const { return m_tid; }
  std::optional<lldb::tid_t> GetLaneID() const { return m_lane_id; }
  std::optional<lldb::tid_t> GetSIMD() const { return m_simd_id; }
  virtual bool GetIsActive() const { return m_active; }
  void SetIsActive(bool b) { m_active = b; }
  bool GetImplicitlyResumed() const { return m_implicitly_resumed; }
  void SetImplicitlyResumed(bool b) { m_implicitly_resumed = b; }
  NativeProcessProtocol &GetProcess() { return m_process; }

  // Thread-specific watchpoints
  virtual Status SetWatchpoint(lldb::addr_t addr, size_t size,
                               uint32_t watch_flags, bool hardware) = 0;

  virtual Status RemoveWatchpoint(lldb::addr_t addr) = 0;

  // Thread-specific Hardware Breakpoint routines
  virtual Status SetHardwareBreakpoint(lldb::addr_t addr, size_t size) = 0;

  virtual Status RemoveHardwareBreakpoint(lldb::addr_t addr) = 0;

  virtual llvm::Expected<std::unique_ptr<llvm::MemoryBuffer>>
  GetSiginfo() const {
    return llvm::make_error<UnimplementedError>();
  }

  /// Get extended information about this thread as JSON.
  ///
  /// This backs the "jThreadExtendedInfo" GDB remote packet which allows a
  /// thread to supply arbitrary extra information about itself that the
  /// standard packets have no way to express. There are no requirements on
  /// what the returned JSON contains.
  ///
  /// \param[in] args
  ///     The JSON arguments that were decoded from the "jThreadExtendedInfo"
  ///     packet. This is normally a dictionary containing a "thread" key that
  ///     selects this thread, along with any extra hints the client chose to
  ///     supply. It is an empty dictionary when the client is only asking
  ///     whether the packet is supported.
  ///
  /// \return
  ///     A JSON value that describes this thread, or std::nullopt if this
  ///     thread has no extended information to supply. Returning std::nullopt
  ///     causes the "jThreadExtendedInfo" packet to be reported as
  ///     unsupported.
  virtual std::optional<llvm::json::Value>
  GetExtendedInfo(const llvm::json::Value &args) {
    return std::nullopt;
  }

  bool HasValidStopReason();

protected:
  NativeProcessProtocol &m_process;
  lldb::tid_t m_tid;
  // An optional lane number for the thread used to identify the GPU lane for
  // this thread object.
  std::optional<lldb::tid_t> m_lane_id;
  /// An optional SIMD group ID for the thread used to identify the GPU SIMD.
  /// This can be used to identify the ID of the wave or warp for the lane.
  std::optional<lldb::tid_t> m_simd_id;
  /// Set to false if this thread is not active. This can be used to indicate
  /// that a GPU lane is not active. When single stepping this can be used to
  /// step until the thread becomes active again.
  bool m_active = true;
  /// Set if the last resume ran this thread although the client asked for it
  /// to stay suspended, as a GPU lane runs when another lane of its SIMD group
  /// runs. The process sets this when it stops and must clear it at the next
  /// resume. LLGS lists such threads to the client in the
  /// "implicitly_resumed_threads" key of the stop reply.
  bool m_implicitly_resumed = false;
};
}

#endif // LLDB_HOST_COMMON_NATIVETHREADPROTOCOL_H
