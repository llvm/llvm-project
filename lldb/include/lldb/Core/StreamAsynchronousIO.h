//===-- StreamAsynchronousIO.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StreamAsynchronousIO_h_
#define liblldb_StreamAsynchronousIO_h_

#include <string>

#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"

namespace lldb_private {

class StreamAsynchronousIO : 
    public Stream
{
public:
    StreamAsynchronousIO (Broadcaster &broadcaster, uint32_t broadcast_event_type);
    
    virtual ~StreamAsynchronousIO ();
    
    virtual void
    Flush ();
    
    virtual size_t
    Write (const void *src, size_t src_len);
    
    
private:
    Broadcaster &m_broadcaster;
    uint32_t m_broadcast_event_type;
    StreamString m_accumulated_data;
};

} // namespace lldb_private
#endif // #ifndef liblldb_StreamAsynchronousIO_h
