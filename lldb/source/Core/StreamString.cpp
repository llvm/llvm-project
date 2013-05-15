//===-- StreamString.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/StreamString.h"
#include <stdio.h>

using namespace lldb;
using namespace lldb_private;

StreamString::StreamString () :
    Stream (0, 4, eByteOrderBig)
{
}

StreamString::StreamString(uint32_t flags, uint32_t addr_size, ByteOrder byte_order) :
    Stream (flags, addr_size, byte_order),
    m_packet ()
{
}

StreamString::~StreamString()
{
}

void
StreamString::Flush ()
{
    // Nothing to do when flushing a buffer based stream...
}

size_t
StreamString::Write (const void *s, size_t length)
{
    m_packet.append ((char *)s, length);
    return length;
}

void
StreamString::Clear()
{
    m_packet.clear();
}

bool
StreamString::Empty() const
{
    return GetSize() == 0;
}

const char *
StreamString::GetData () const
{
    return m_packet.c_str();
}

size_t
StreamString::GetSize () const
{
    return m_packet.size();
}

std::string &
StreamString::GetString()
{
    return m_packet;
}

const std::string &
StreamString::GetString() const
{
    return m_packet;
}

void
StreamString::FillLastLineToColumn (uint32_t column, char fill_char)
{
    const size_t length = m_packet.size();
    size_t last_line_begin_pos = m_packet.find_last_of("\r\n");
    if (last_line_begin_pos == std::string::npos)
    {
        last_line_begin_pos = 0;
    }
    else
    {
        ++last_line_begin_pos;
    }
    
    const size_t line_columns = length - last_line_begin_pos;
    if (column > line_columns)
    {
        m_packet.append(column - line_columns, fill_char);
    }
}

