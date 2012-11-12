//===-- StoppointLocation.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/StoppointLocation.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// StoppointLocation constructor
//----------------------------------------------------------------------
StoppointLocation::StoppointLocation (break_id_t bid, addr_t addr, bool hardware) :
    m_loc_id(bid),
    m_addr(addr),
    m_hw_preferred(hardware),
    m_hw_index(LLDB_INVALID_INDEX32),
    m_byte_size(0),
    m_hit_count(0)
{
}

StoppointLocation::StoppointLocation (break_id_t bid, addr_t addr, uint32_t byte_size, bool hardware) :
    m_loc_id(bid),
    m_addr(addr),
    m_hw_preferred(hardware),
    m_hw_index(LLDB_INVALID_INDEX32),
    m_byte_size(byte_size),
    m_hit_count(0)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
StoppointLocation::~StoppointLocation()
{
}
