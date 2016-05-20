//===-- StreamCallback.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StreamCallback_h_
#define liblldb_StreamCallback_h_

#include <mutex>
#include <string>

#include "lldb/Core/Stream.h"
#include "lldb/Core/StreamString.h"

namespace lldb_private {

class StreamCallback : 
    public Stream
{
public:
    StreamCallback (lldb::LogOutputCallback callback, void *baton);
    
    ~StreamCallback () override;
    
    void
    Flush () override;
    
    size_t
    Write (const void *src, size_t src_len) override;
    
private:
    typedef std::map<lldb::tid_t, StreamString> collection;
    lldb::LogOutputCallback m_callback;
    void *m_baton;
    collection m_accumulated_data;
    std::mutex m_collection_mutex;

    StreamString &FindStreamForThread(lldb::tid_t cur_tid);
};

} // namespace lldb_private

#endif // liblldb_StreamCallback_h
