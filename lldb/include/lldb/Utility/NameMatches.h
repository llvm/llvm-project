//===-- NameMatches.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef LLDB_UTILITY_NAMEMATCHES_H
#define LLDB_UTILITY_NAMEMATCHES_H

#include "lldb/lldb-private-enumerations.h"

namespace lldb_private {
bool NameMatches(const char *name, NameMatchType match_type, const char *match);
}

#endif
