//===-- StreamFile.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/StreamFile.h"

// C Includes
#include <stdio.h>
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Error.h"


using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// StreamFile constructor
//----------------------------------------------------------------------
StreamFile::StreamFile () :
    Stream (),
    m_file ()
{
}

StreamFile::StreamFile (uint32_t flags, uint32_t addr_size, ByteOrder byte_order) :
    Stream (flags, addr_size, byte_order),
    m_file ()
{
}

StreamFile::StreamFile (int fd, bool transfer_ownership) :
    Stream (),
    m_file (fd, transfer_ownership)
{
}

StreamFile::StreamFile (FILE *fh, bool transfer_ownership) :
    Stream (),
    m_file (fh, transfer_ownership)
{
}

StreamFile::StreamFile (const char *path) :
    Stream (),
    m_file (path, File::eOpenOptionWrite | File::eOpenOptionCanCreate, File::ePermissionsDefault)
{
}


StreamFile::~StreamFile()
{
}

void
StreamFile::Flush ()
{
    m_file.Flush();
}

int
StreamFile::Write (const void *s, size_t length)
{
    m_file.Write (s, length);
    return length;
}
