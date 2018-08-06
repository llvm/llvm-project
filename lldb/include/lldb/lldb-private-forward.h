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

namespace lldb_private {
// --------------------------------------------------------------- Class
// forward decls.
// ---------------------------------------------------------------
class NativeBreakpoint;
class NativeBreakpointList;
class NativeProcessProtocol;
class NativeRegisterContext;
class NativeThreadProtocol;
class ResumeActionList;
class UnixSignals;

// --------------------------------------------------------------- SP/WP decls.
// ---------------------------------------------------------------
typedef std::shared_ptr<NativeBreakpoint> NativeBreakpointSP;
}

#endif // #if defined(__cplusplus)
#endif // #ifndef LLDB_lldb_private_forward_h_
