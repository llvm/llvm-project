//===-- lldb-private-forward.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_lldb_private_forward_h_
#define LLDB_lldb_private_forward_h_

#if defined(__cplusplus)

#include <memory>

namespace lldb_private
{
    // ---------------------------------------------------------------
    // Class forward decls.
    // ---------------------------------------------------------------
    class NativeBreakpoint;
    class NativeBreakpointList;
    class NativeProcessProtocol;
    class NativeRegisterContext;
    class NativeThreadProtocol;
    class ResumeActionList;
    class UnixSignals;

    // ---------------------------------------------------------------
    // SP/WP decls.
    // ---------------------------------------------------------------
    typedef std::shared_ptr<NativeBreakpoint> NativeBreakpointSP;
    typedef std::shared_ptr<lldb_private::NativeProcessProtocol> NativeProcessProtocolSP;
    typedef std::weak_ptr<lldb_private::NativeProcessProtocol> NativeProcessProtocolWP;
    typedef std::shared_ptr<lldb_private::NativeRegisterContext> NativeRegisterContextSP;
    typedef std::shared_ptr<lldb_private::NativeThreadProtocol> NativeThreadProtocolSP;
}

#endif // #if defined(__cplusplus)
#endif // #ifndef LLDB_lldb_private_forward_h_
