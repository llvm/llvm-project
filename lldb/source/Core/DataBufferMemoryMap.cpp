//===-- DataBufferMemoryMap.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "lldb/Core/DataBufferMemoryMap.h"
#include "lldb/Core/Error.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Host/Host.h"

using namespace lldb_private;

//----------------------------------------------------------------------
// Default Constructor
//----------------------------------------------------------------------
DataBufferMemoryMap::DataBufferMemoryMap() :
    m_mmap_addr(NULL),
    m_mmap_size(0),
    m_data(NULL),
    m_size(0)
{
}

//----------------------------------------------------------------------
// Virtual destructor since this class inherits from a pure virtual
// base class.
//----------------------------------------------------------------------
DataBufferMemoryMap::~DataBufferMemoryMap()
{
    Clear();
}

//----------------------------------------------------------------------
// Return a pointer to the bytes owned by this object, or NULL if
// the object contains no bytes.
//----------------------------------------------------------------------
uint8_t *
DataBufferMemoryMap::GetBytes()
{
    return m_data;
}

//----------------------------------------------------------------------
// Return a const pointer to the bytes owned by this object, or NULL
// if the object contains no bytes.
//----------------------------------------------------------------------
const uint8_t *
DataBufferMemoryMap::GetBytes() const
{
    return m_data;
}

//----------------------------------------------------------------------
// Return the number of bytes this object currently contains.
//----------------------------------------------------------------------
size_t
DataBufferMemoryMap::GetByteSize() const
{
    return m_size;
}

//----------------------------------------------------------------------
// Reverts this object to an empty state by unmapping any memory
// that is currently owned.
//----------------------------------------------------------------------
void
DataBufferMemoryMap::Clear()
{
    if (m_mmap_addr != NULL)
    {
        ::munmap((void *)m_mmap_addr, m_mmap_size);
        m_mmap_addr = NULL;
        m_mmap_size = 0;
        m_data = NULL;
        m_size = 0;
    }
}

//----------------------------------------------------------------------
// Memory map "length" bytes from "file" starting "offset"
// bytes into the file. If "length" is set to SIZE_MAX, then
// map as many bytes as possible.
//
// Returns the number of bytes mapped starting from the requested
// offset.
//----------------------------------------------------------------------
size_t
DataBufferMemoryMap::MemoryMapFromFileSpec (const FileSpec* filespec,
                                            off_t offset, 
                                            size_t length,
                                            bool writeable)
{
    if (filespec != NULL)
    {
        char path[PATH_MAX];
        if (filespec->GetPath(path, sizeof(path)))
        {
            uint32_t options = File::eOpenOptionRead;
            if (writeable)
                options |= File::eOpenOptionWrite;

            File file;
            Error error (file.Open(path, options));
            if (error.Success())
            {
                const bool fd_is_file = true;
                MemoryMapFromFileDescriptor (file.GetDescriptor(), offset, length, writeable, fd_is_file);
                return GetByteSize();
            }
        }
    }
    // We should only get here if there was an error
    Clear();
    return 0;
}


//----------------------------------------------------------------------
// The file descriptor FD is assumed to already be opened as read only
// and the STAT structure is assumed to a valid pointer and already
// containing valid data from a call to stat().
//
// Memory map FILE_LENGTH bytes in FILE starting FILE_OFFSET bytes into
// the file. If FILE_LENGTH is set to SIZE_MAX, then map as many bytes
// as possible.
//
// RETURNS
//  Number of bytes mapped starting from the requested offset.
//----------------------------------------------------------------------
size_t
DataBufferMemoryMap::MemoryMapFromFileDescriptor (int fd, 
                                                  off_t offset, 
                                                  size_t length,
                                                  bool writeable,
                                                  bool fd_is_file)
{
    Clear();
    if (fd >= 0)
    {
        struct stat stat;
        if (::fstat(fd, &stat) == 0)
        {
            if (S_ISREG(stat.st_mode) && (stat.st_size > offset))
            {
                const size_t max_bytes_available = stat.st_size - offset;
                if (length == SIZE_MAX)
                {
                    length = max_bytes_available;
                }
                else if (length > max_bytes_available)
                {
                    // Cap the length if too much data was requested
                    length = max_bytes_available;
                }

                if (length > 0)
                {
                    int prot = PROT_READ;
                    if (writeable)
                        prot |= PROT_WRITE;

                    int flags = MAP_PRIVATE;
                    if (fd_is_file)
                        flags |= MAP_FILE;

                    m_mmap_addr = (uint8_t *)::mmap(NULL, length, prot, flags, fd, offset);

                    if (m_mmap_addr == (void*)-1)
                    {
                        Error error;
                        error.SetErrorToErrno ();
                        if (error.GetError() == EINVAL)
                        {
                            // We may still have a shot at memory mapping if we align things correctly
                            size_t page_offset = offset % Host::GetPageSize();
                            if (page_offset != 0)
                            {
                                m_mmap_addr = (uint8_t *)::mmap(NULL, length + page_offset, prot, flags, fd, offset - page_offset);
                                if (m_mmap_addr == (void*)-1)
                                {
                                    // Failed to map file
                                    m_mmap_addr = NULL;
                                }
                                else if (m_mmap_addr != NULL)
                                {
                                    // We recovered and were able to memory map
                                    // after we aligned things to page boundaries

                                    // Save the actual mmap'ed size
                                    m_mmap_size = length + page_offset;
                                    // Our data is at an offset into the the mapped data
                                    m_data = m_mmap_addr + page_offset;
                                    // Our pretend size is the size that was requestd
                                    m_size = length;
                                }
                            }
                        }
                        if (error.GetError() == ENOMEM)
                        {
                           error.SetErrorStringWithFormat("could not allocate %lld bytes of memory to mmap in file", (uint64_t) length);
                        }
                    }
                    else
                    {
                        // We were able to map the requested data in one chunk
                        // where our mmap and actual data are the same.
                        m_mmap_size = length;
                        m_data = m_mmap_addr;
                        m_size = length;
                    }
                }
            }
        }
    }
    return GetByteSize ();
}
