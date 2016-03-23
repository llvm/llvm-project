//===-- ConnectionSharedMemory.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __ANDROID_NDK__

#include "lldb/Core/ConnectionSharedMemory.h"

// C Includes
#ifdef _WIN32
#include "lldb/Host/windows/windows.h"
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#endif

// C++ Includes
#include <cerrno>
#include <cstdlib>

// Other libraries and framework includes
#include "llvm/Support/MathExtras.h"

// Project includes
#include "lldb/Core/Communication.h"
#include "lldb/Core/Log.h"

#include "llvm/Support/ConvertUTF.h"

using namespace lldb;
using namespace lldb_private;

ConnectionSharedMemory::ConnectionSharedMemory () :
    Connection(),
    m_name(),
    m_fd (-1),
    m_mmap()
{
}

ConnectionSharedMemory::~ConnectionSharedMemory ()
{
    Disconnect(nullptr);
}

bool
ConnectionSharedMemory::IsConnected () const
{
    return m_fd >= 0;
}

ConnectionStatus
ConnectionSharedMemory::Connect (const char *s, Error *error_ptr)
{
//    if (s && s[0])
//    {
//        if (strstr(s, "shm-create://"))
//        {
//        }
//        else if (strstr(s, "shm-connect://"))
//        {
//        }
//        if (error_ptr)
//            error_ptr->SetErrorStringWithFormat ("unsupported connection URL: '%s'", s);
//        return eConnectionStatusError;
//    }
    if (error_ptr)
        error_ptr->SetErrorString("invalid connect arguments");
    return eConnectionStatusError;
}

ConnectionStatus
ConnectionSharedMemory::Disconnect (Error *error_ptr)
{
    m_mmap.Clear();
    if (!m_name.empty())
    {
#ifdef _WIN32
        close(m_fd);
        m_fd = -1;
#else
        shm_unlink (m_name.c_str());
#endif
        m_name.clear();
    }
    return eConnectionStatusSuccess;
}

size_t
ConnectionSharedMemory::Read (void *dst, 
                              size_t dst_len, 
                              uint32_t timeout_usec,
                              ConnectionStatus &status, 
                              Error *error_ptr)
{
    status = eConnectionStatusSuccess;
    return 0;
}

size_t
ConnectionSharedMemory::Write (const void *src, size_t src_len, ConnectionStatus &status, Error *error_ptr)
{
    status = eConnectionStatusSuccess;
    return 0;
}

std::string
ConnectionSharedMemory::GetURI()
{
    // TODO: fix when Connect is fixed?
    return "";
}

ConnectionStatus
ConnectionSharedMemory::BytesAvailable (uint32_t timeout_usec, Error *error_ptr)
{
    return eConnectionStatusLostConnection;
}

ConnectionStatus
ConnectionSharedMemory::Open (bool create, const char *name, size_t size, Error *error_ptr)
{
    if (m_fd != -1)
    {
        if (error_ptr)
            error_ptr->SetErrorString("already open");
        return eConnectionStatusError;
    }
    
    m_name.assign (name);

#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
    std::wstring wname;
    if (llvm::ConvertUTF8toWide(name, wname))
    {
        if (create)
        {
            handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, llvm::Hi_32(size),
                                        llvm::Lo_32(size), wname.c_str());
        }
        else
            handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wname.c_str());
    }

    m_fd = _open_osfhandle((intptr_t)handle, 0);
#else
    int oflag = O_RDWR;
    if (create)
        oflag |= O_CREAT;
    m_fd = ::shm_open (m_name.c_str(), oflag, S_IRUSR|S_IWUSR);

    if (create)
        ::ftruncate (m_fd, size);
#endif

    if (m_mmap.MemoryMapFromFileDescriptor(m_fd, 0, size, true, false) == size)
        return eConnectionStatusSuccess;

    Disconnect(nullptr);
    return eConnectionStatusError;
}

#endif // __ANDROID_NDK__
