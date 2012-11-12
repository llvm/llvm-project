//===-- History.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/History.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Stream.h"

using namespace lldb;
using namespace lldb_private;

void
HistorySourceUInt::DumpHistoryEvent (Stream &strm, HistoryEvent event)
{
    strm.Printf ("%s %llu", m_name.c_str(), (uint64_t)((uintptr_t)event));
}
