//===-- StreamCallback.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#include "lldb/Core/Broadcaster.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/StreamCallback.h"
#include "lldb/Host/Host.h"
#include "lldb/lldb-private.h"

using namespace lldb;
using namespace lldb_private;

StreamCallback::StreamCallback(lldb::LogOutputCallback callback, void *baton)
    : Stream(0, 4, eByteOrderBig), m_callback(callback), m_baton(baton),
      m_accumulated_data(), m_collection_mutex() {}

StreamCallback::~StreamCallback() {}

StreamString &StreamCallback::FindStreamForThread(lldb::tid_t cur_tid) {
  Mutex::Locker locker(m_collection_mutex);
  collection::iterator iter = m_accumulated_data.find(cur_tid);
  if (iter == m_accumulated_data.end()) {
    std::pair<collection::iterator, bool> ret;
    ret = m_accumulated_data.insert(
        std::pair<lldb::tid_t, StreamString>(cur_tid, StreamString()));
    iter = ret.first;
  }
  return (*iter).second;
}

void StreamCallback::Flush() {
  lldb::tid_t cur_tid = Host::GetCurrentThreadID();
  StreamString &out_stream = FindStreamForThread(cur_tid);
  m_callback(out_stream.GetData(), m_baton);
  out_stream.Clear();
}

size_t StreamCallback::Write(const void *s, size_t length) {
  lldb::tid_t cur_tid = Host::GetCurrentThreadID();
  FindStreamForThread(cur_tid).Write(s, length);
  return length;
}
