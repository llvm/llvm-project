//===-- SBHostOS.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBHostOS.h"
#include "lldb/API/SBError.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Core/Log.h"
#include "lldb/Host/Host.h"

using namespace lldb;
using namespace lldb_private;



SBFileSpec
SBHostOS::GetProgramFileSpec ()
{
    SBFileSpec sb_filespec;
    sb_filespec.SetFileSpec (Host::GetProgramFileSpec ());
    return sb_filespec;
}

lldb::thread_t
SBHostOS::ThreadCreate
(
    const char *name,
    void *(*thread_function)(void *),
    void *thread_arg,
    SBError *error_ptr
)
{
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (log)
        log->Printf ("SBHostOS::ThreadCreate (name=\"%s\", thread_function=%p, thread_arg=%p, error_ptr=%p)", name, 
                     thread_function, thread_arg, error_ptr);

    // FIXME: You should log the return value?

    return Host::ThreadCreate (name, thread_function, thread_arg, error_ptr ? error_ptr->get() : NULL);
}

void
SBHostOS::ThreadCreated (const char *name)
{
    Host::ThreadCreated (name);
}

bool
SBHostOS::ThreadCancel (lldb::thread_t thread, SBError *error_ptr)
{
    return Host::ThreadCancel (thread, error_ptr ? error_ptr->get() : NULL);
}

bool
SBHostOS::ThreadDetach (lldb::thread_t thread, SBError *error_ptr)
{
    return Host::ThreadDetach (thread, error_ptr ? error_ptr->get() : NULL);
}

bool
SBHostOS::ThreadJoin (lldb::thread_t thread, void **result, SBError *error_ptr)
{
    return Host::ThreadJoin (thread, result, error_ptr ? error_ptr->get() : NULL);
}


