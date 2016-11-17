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

#include "llvm/ADT/StringRef.h"

namespace lldb_private {
bool NameMatches(llvm::StringRef name, NameMatchType match_type,
                 llvm::StringRef match);
}

#endif
