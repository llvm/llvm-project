//===-- DataBufferHeap.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DataBufferHeap.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// Default constructor
//----------------------------------------------------------------------
DataBufferHeap::DataBufferHeap () :
    m_data()
{
}

//----------------------------------------------------------------------
// Initialize this class with "n" characters and fill the buffer
// with "ch".
//----------------------------------------------------------------------
DataBufferHeap::DataBufferHeap (lldb::offset_t n, uint8_t ch) :
    m_data(n, ch)
{
}

//----------------------------------------------------------------------
// Initialize this class with a copy of the "n" bytes from the "bytes"
// buffer.
//----------------------------------------------------------------------
DataBufferHeap::DataBufferHeap (const void *src, lldb::offset_t src_len) :
    m_data()
{
    CopyData (src, src_len);
}

//----------------------------------------------------------------------
// Virtual destructor since this class inherits from a pure virtual
// base class.
//----------------------------------------------------------------------
DataBufferHeap::~DataBufferHeap ()
{
}

//----------------------------------------------------------------------
// Return a pointer to the bytes owned by this object, or NULL if
// the object contains no bytes.
//----------------------------------------------------------------------
uint8_t *
DataBufferHeap::GetBytes ()
{
    if (m_data.empty())
        return NULL;
    return &m_data[0];
}

//----------------------------------------------------------------------
// Return a const pointer to the bytes owned by this object, or NULL
// if the object contains no bytes.
//----------------------------------------------------------------------
const uint8_t *
DataBufferHeap::GetBytes () const
{
    if (m_data.empty())
        return NULL;
    return &m_data[0];
}

//----------------------------------------------------------------------
// Return the number of bytes this object currently contains.
//----------------------------------------------------------------------
uint64_t
DataBufferHeap::GetByteSize () const
{
    return m_data.size();
}


//----------------------------------------------------------------------
// Sets the number of bytes that this object should be able to
// contain. This can be used prior to copying data into the buffer.
//----------------------------------------------------------------------
uint64_t
DataBufferHeap::SetByteSize (uint64_t new_size)
{
    m_data.resize(new_size);
    return m_data.size();
}

void
DataBufferHeap::CopyData (const void *src, uint64_t src_len)
{
    const uint8_t *src_u8 = (const uint8_t *)src;
    if (src && src_len > 0)
        m_data.assign (src_u8, src_u8 + src_len);
    else
        m_data.clear();
}



