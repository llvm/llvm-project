//===-- Stoppoint.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/Stoppoint.h"
#include "lldb/lldb-private.h"


using namespace lldb;
using namespace lldb_private;

// Stoppoint constructor
Stoppoint::Stoppoint() : m_bid(LLDB_INVALID_BREAK_ID) {}

// Destructor
Stoppoint::~Stoppoint() {}

break_id_t Stoppoint::GetID() const { return m_bid; }

void Stoppoint::SetID(break_id_t bid) { m_bid = bid; }
