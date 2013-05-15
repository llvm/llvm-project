//===-- StreamTee.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StreamTee_h_
#define liblldb_StreamTee_h_

#include <limits.h>

#include "lldb/Core/Stream.h"
#include "lldb/Host/Mutex.h"

namespace lldb_private {

class StreamTee : public Stream
{
public:
    StreamTee () :
        Stream (),
        m_streams_mutex (Mutex::eMutexTypeRecursive),
        m_streams ()
    {
    }

    StreamTee (lldb::StreamSP &stream_sp):
        Stream (),
        m_streams_mutex (Mutex::eMutexTypeRecursive),
        m_streams ()
    {
        // No need to lock mutex during construction
        if (stream_sp)
            m_streams.push_back (stream_sp);
    }
    

    StreamTee (lldb::StreamSP &stream_sp, lldb::StreamSP &stream_2_sp) :
        Stream (),
        m_streams_mutex (Mutex::eMutexTypeRecursive),
        m_streams ()
    {
        // No need to lock mutex during construction
        if (stream_sp)
            m_streams.push_back (stream_sp);
        if (stream_2_sp)
            m_streams.push_back (stream_2_sp);
    }
    
    StreamTee (const StreamTee &rhs) :
        Stream (rhs),
        m_streams_mutex (Mutex::eMutexTypeRecursive),
        m_streams() // Don't copy until we lock down "rhs"
    {
        Mutex::Locker locker (rhs.m_streams_mutex);
        m_streams = rhs.m_streams;
    }

    virtual
    ~StreamTee ()
    {
    }

    StreamTee &
    operator = (const StreamTee &rhs)
    {
        if (this != &rhs) {
            Stream::operator=(rhs);
            Mutex::Locker lhs_locker (m_streams_mutex);
            Mutex::Locker rhs_locker (rhs.m_streams_mutex);
            m_streams = rhs.m_streams;            
        }
        return *this;
    }

    virtual void
    Flush ()
    {
        Mutex::Locker locker (m_streams_mutex);
        collection::iterator pos, end;
        for (pos = m_streams.begin(), end = m_streams.end(); pos != end; ++pos)
        {
            // Allow for our collection to contain NULL streams. This allows
            // the StreamTee to be used with hard coded indexes for clients
            // that might want N total streams with only a few that are set
            // to valid values.
            Stream *strm = pos->get();
            if (strm)
                strm->Flush ();
        }
    }

    virtual size_t
    Write (const void *s, size_t length)
    {
        Mutex::Locker locker (m_streams_mutex);
        if (m_streams.empty())
            return 0;
    
        size_t min_bytes_written = SIZE_MAX;
        collection::iterator pos, end;
        for (pos = m_streams.begin(), end = m_streams.end(); pos != end; ++pos)
        {
            // Allow for our collection to contain NULL streams. This allows
            // the StreamTee to be used with hard coded indexes for clients
            // that might want N total streams with only a few that are set
            // to valid values.
            Stream *strm = pos->get();
            if (strm)
            {
                const size_t bytes_written = strm->Write (s, length);
                if (min_bytes_written > bytes_written)
                    min_bytes_written = bytes_written;
            }
        }
        if (min_bytes_written == SIZE_MAX)
            return 0;
        return min_bytes_written;
    }

    size_t
    AppendStream (const lldb::StreamSP &stream_sp)
    {
        size_t new_idx = m_streams.size();
        Mutex::Locker locker (m_streams_mutex);
        m_streams.push_back (stream_sp);
        return new_idx;
    }

    size_t
    GetNumStreams () const
    {
        size_t result = 0;
        {
            Mutex::Locker locker (m_streams_mutex);
            result = m_streams.size();
        }
        return result;
    }

    lldb::StreamSP
    GetStreamAtIndex (uint32_t idx)
    {
        lldb::StreamSP stream_sp;
        Mutex::Locker locker (m_streams_mutex);
        if (idx < m_streams.size())
            stream_sp = m_streams[idx];
        return stream_sp;
    }

    void
    SetStreamAtIndex (uint32_t idx, const lldb::StreamSP& stream_sp)
    {
        Mutex::Locker locker (m_streams_mutex);
        // Resize our stream vector as necessary to fit as many streams
        // as needed. This also allows this class to be used with hard
        // coded indexes that can be used contain many streams, not all
        // of which are valid.
        if (idx >= m_streams.size())
            m_streams.resize(idx + 1);
        m_streams[idx] = stream_sp;
    }
    

protected:
    typedef std::vector<lldb::StreamSP> collection;
    mutable Mutex m_streams_mutex;
    collection m_streams;
};

} // namespace lldb_private
#endif  // #ifndef liblldb_StreamTee_h_
