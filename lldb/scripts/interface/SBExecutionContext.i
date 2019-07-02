//===-- SWIG Interface for SBExecutionContext ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

class SBExecutionContext
{
public:
    SBExecutionContext();

    SBExecutionContext (const lldb::SBExecutionContext &rhs);

    SBExecutionContext (const lldb::SBTarget &target);

    SBExecutionContext (const lldb::SBProcess &process);

    SBExecutionContext (lldb::SBThread thread); // can't be a const& because SBThread::get() isn't itself a const function

    SBExecutionContext (const lldb::SBFrame &frame);

    ~SBExecutionContext();

    SBTarget
    GetTarget () const;

    SBProcess
    GetProcess () const;

    SBThread
    GetThread () const;

    SBFrame
    GetFrame () const;
};

%attributeref(lldb::SBExecutionContext, lldb::SBTarget, target, GetTarget);
%attributeref(lldb::SBExecutionContext, lldb::SBProcess, process, GetProcess);
%attributeref(lldb::SBExecutionContext, lldb::SBThread, process, GetThread);
%attributeref(lldb::SBExecutionContext, lldb::SBFrame, process, GetFrame);

} // namespace lldb
