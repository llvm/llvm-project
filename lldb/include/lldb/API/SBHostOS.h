//===-- SBHostOS.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBHostOS_h_
#define LLDB_SBHostOS_h_

#include "lldb/API/SBDefines.h"
#include "lldb/API/SBFileSpec.h"

namespace lldb {

class SBHostOS
{
public:

    static lldb::SBFileSpec
    GetProgramFileSpec ();

    static void
    ThreadCreated (const char *name);

    static lldb::thread_t
    ThreadCreate (const char *name,
                  void *(*thread_function)(void *),
                  void *thread_arg,
                  lldb::SBError *err);

    static bool
    ThreadCancel (lldb::thread_t thread,
                  lldb::SBError *err);

    static bool
    ThreadDetach (lldb::thread_t thread,
                  lldb::SBError *err);
    static bool
    ThreadJoin (lldb::thread_t thread,
                void **result,
                lldb::SBError *err);


private:

};


} // namespace lldb

#endif // LLDB_SBHostOS_h_
