//===-- ConvertEnum.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_UTILITY_CONVERTENUM_H
#define LLDB_UTILITY_CONVERTENUM_H

#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-private-enumerations.h"

namespace lldb_private {

const char *GetVoteAsCString(Vote vote);
const char *GetSectionTypeAsCString(lldb::SectionType sect_type);
}

#endif
