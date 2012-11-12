//===-- lldb-private.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_lldb_private_h_
#define lldb_lldb_private_h_

#if defined(__cplusplus)

#include "lldb/lldb-public.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-private-interfaces.h"
#include "lldb/lldb-private-log.h"
#include "lldb/lldb-private-types.h"

namespace lldb_private {

//------------------------------------------------------------------
/// Initializes lldb.
///
/// This function should be called prior to using any lldb
/// classes to ensure they have a chance to do any static
/// initialization that they need to do.
//------------------------------------------------------------------
void
Initialize();


//------------------------------------------------------------------
/// Notifies any classes that lldb will be terminating soon.
///
/// This function will be called when the Debugger shared instance
/// is being destructed and will give classes the ability to clean
/// up any threads or other resources they have that they might not
/// be able to clean up in their own destructors.
///
/// Internal classes that need this ability will need to add their
/// void T::WillTerminate() method in the body of this function in
/// lldb.cpp to ensure it will get called.
///
/// TODO: when we start having external plug-ins, we will need a way
/// for plug-ins to register a WillTerminate callback.
//------------------------------------------------------------------
void
WillTerminate();

//------------------------------------------------------------------
/// Terminates lldb
///
/// This function optionally can be called when clients are done
/// using lldb functionality to free up any static resources
/// that have been allocated during initialization or during
/// function calls. No lldb functions should be called after
/// calling this function without again calling DCInitialize()
/// again.
//------------------------------------------------------------------
void
Terminate();


const char *
GetVersion ();

const char *
GetVoteAsCString (Vote vote);

const char *
GetSectionTypeAsCString (lldb::SectionType sect_type);
    
bool
NameMatches (const char *name, NameMatchType match_type, const char *match);

} // namespace lldb_private


#endif  // defined(__cplusplus)


#endif  // lldb_lldb_private_h_
