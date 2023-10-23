//===-- WatchpointCollection.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Breakpoint/WatchpointCollection.h"
#include "lldb/Breakpoint/Watchpoint.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadSpec.h"

using namespace lldb;
using namespace lldb_private;

// WatchpointCollection constructor
WatchpointCollection::WatchpointCollection() = default;

// Destructor
WatchpointCollection::~WatchpointCollection() = default;

void WatchpointCollection::Add(const WatchpointSP &wp) {
  std::lock_guard<std::mutex> guard(m_collection_mutex);
  m_collection.push_back(wp);
}

bool WatchpointCollection::Remove(WatchpointSP &wp) {
  std::lock_guard<std::mutex> guard(m_collection_mutex);
  for (collection::iterator pos = m_collection.begin();
       pos != m_collection.end(); ++pos) {
    if (*pos == wp) {
      m_collection.erase(pos);
      return true;
    }
  }
  return false;
}

WatchpointSP WatchpointCollection::GetByIndex(size_t i) {
  std::lock_guard<std::mutex> guard(m_collection_mutex);
  if (i < m_collection.size())
    return m_collection[i];
  return {};
}

const WatchpointSP WatchpointCollection::GetByIndex(size_t i) const {
  std::lock_guard<std::mutex> guard(m_collection_mutex);
  if (i < m_collection.size())
    return m_collection[i];
  return {};
}

WatchpointCollection &
WatchpointCollection::operator=(const WatchpointCollection &rhs) {
  if (this != &rhs) {
    std::lock(m_collection_mutex, rhs.m_collection_mutex);
    std::lock_guard<std::mutex> lhs_guard(m_collection_mutex, std::adopt_lock);
    std::lock_guard<std::mutex> rhs_guard(rhs.m_collection_mutex,
                                          std::adopt_lock);
    m_collection = rhs.m_collection;
  }
  return *this;
}

void WatchpointCollection::Clear() {
  std::lock_guard<std::mutex> guard(m_collection_mutex);
  m_collection.clear();
}

bool WatchpointCollection::Contains(const WatchpointSP &wp_sp) {
  std::lock_guard<std::mutex> guard(m_collection_mutex);
  const size_t size = m_collection.size();
  for (size_t i = 0; i < size; ++i) {
    if (m_collection[i] == wp_sp)
      return true;
  }
  return false;
}

bool WatchpointCollection::Contains(const Watchpoint *wp) {
  std::lock_guard<std::mutex> guard(m_collection_mutex);
  const size_t size = m_collection.size();
  for (size_t i = 0; i < size; ++i) {
    if (m_collection[i].get() == wp)
      return true;
  }
  return false;
}
