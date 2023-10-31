//===-- WatchpointResource.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <assert.h>

#include "lldb/Breakpoint/WatchpointResource.h"

using namespace lldb;
using namespace lldb_private;

WatchpointResource::WatchpointResource(lldb::addr_t addr, size_t size,
                                       bool read, bool write)
    : m_id(LLDB_INVALID_WATCHPOINT_RESOURCE_ID), m_addr(addr), m_size(size),
      m_watch_read(read), m_watch_write(write) {}

WatchpointResource::~WatchpointResource() {
  std::lock_guard<std::mutex> guard(m_owners_mutex);
  m_owners.clear();
}

addr_t WatchpointResource::GetAddress() const { return m_addr; }

size_t WatchpointResource::GetByteSize() const { return m_size; }

bool WatchpointResource::WatchpointResourceRead() const { return m_watch_read; }

bool WatchpointResource::WatchpointResourceWrite() const {
  return m_watch_write;
}

void WatchpointResource::SetType(bool read, bool write) {
  m_watch_read = read;
  m_watch_write = write;
}

wp_resource_id_t WatchpointResource::GetID() const { return m_id; }

void WatchpointResource::SetID(wp_resource_id_t id) { m_id = id; }

bool WatchpointResource::Contains(addr_t addr) {
  if (addr >= m_addr && addr < m_addr + m_size)
    return true;
  return false;
}

void WatchpointResource::AddOwner(const WatchpointSP &wp_sp) {
  std::lock_guard<std::mutex> guard(m_owners_mutex);
  m_owners.push_back(wp_sp);
}

void WatchpointResource::RemoveOwner(WatchpointSP &wp_sp) {
  std::lock_guard<std::mutex> guard(m_owners_mutex);
  const auto &it = std::find(m_owners.begin(), m_owners.end(), wp_sp);
  if (it != m_owners.end())
    m_owners.erase(it);
}

size_t WatchpointResource::GetNumberOfOwners() {
  std::lock_guard<std::mutex> guard(m_owners_mutex);
  return m_owners.size();
}

bool WatchpointResource::OwnersContains(const WatchpointSP &wp_sp) {
  return OwnersContains(wp_sp.get());
}

bool WatchpointResource::OwnersContains(const Watchpoint *wp) {
  std::lock_guard<std::mutex> guard(m_owners_mutex);
  WatchpointCollection::const_iterator match =
      std::find_if(m_owners.begin(), m_owners.end(),
                   [&wp](const WatchpointSP &x) { return x.get() == wp; });
  return match != m_owners.end();
}

WatchpointSP WatchpointResource::GetOwnerAtIndex(size_t idx) {
  std::lock_guard<std::mutex> guard(m_owners_mutex);
  assert(idx < m_owners.size());
  if (idx >= m_owners.size())
    return {};

  return m_owners[idx];
}

WatchpointResource::WatchpointCollection WatchpointResource::CopyOwnersList() {
  std::lock_guard<std::mutex> guard(m_owners_mutex);
  return m_owners;
}
