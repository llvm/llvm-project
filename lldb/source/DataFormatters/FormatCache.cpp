//===-- FormatCache.cpp ------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//




#include "lldb/DataFormatters/FormatCache.h"

using namespace lldb;
using namespace lldb_private;

FormatCache::Entry::Entry()
    : m_format_cached(false), m_summary_cached(false),
      m_synthetic_cached(false) {}

bool FormatCache::Entry::IsFormatCached() { return m_format_cached; }

bool FormatCache::Entry::IsSummaryCached() { return m_summary_cached; }

bool FormatCache::Entry::IsSyntheticCached() { return m_synthetic_cached; }

void FormatCache::Entry::Get(lldb::TypeFormatImplSP &retval) {
  retval = m_format_sp;
}

void FormatCache::Entry::Get(lldb::TypeSummaryImplSP &retval) {
  retval = m_summary_sp;
}

void FormatCache::Entry::Get(lldb::SyntheticChildrenSP &retval) {
  retval = m_synthetic_sp;
}

void FormatCache::Entry::Set(lldb::TypeFormatImplSP format_sp) {
  m_format_cached = true;
  m_format_sp = format_sp;
}

void FormatCache::Entry::Set(lldb::TypeSummaryImplSP summary_sp) {
  m_summary_cached = true;
  m_summary_sp = summary_sp;
}

void FormatCache::Entry::Set(lldb::SyntheticChildrenSP synthetic_sp) {
  m_synthetic_cached = true;
  m_synthetic_sp = synthetic_sp;
}

FormatCache::FormatCache()
    : m_map(), m_mutex()
#ifdef LLDB_CONFIGURATION_DEBUG
      ,
      m_cache_hits(0), m_cache_misses(0)
#endif
{
}

FormatCache::Entry &FormatCache::GetEntry(ConstString type) {
  auto i = m_map.find(type), e = m_map.end();
  if (i != e)
    return i->second;
  m_map[type] = FormatCache::Entry();
  return m_map[type];
}

template<> bool FormatCache::Entry::IsCached<lldb::TypeFormatImplSP>() {
  return IsFormatCached();
}
template<> bool FormatCache::Entry::IsCached<lldb::TypeSummaryImplSP> () {
  return IsSummaryCached();
}
template<> bool FormatCache::Entry::IsCached<lldb::SyntheticChildrenSP>() {
  return IsSyntheticCached();
}

template <typename ImplSP>
bool FormatCache::Get(ConstString type, ImplSP &format_impl_sp) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  auto entry = GetEntry(type);
  if (entry.IsCached<ImplSP>()) {
#ifdef LLDB_CONFIGURATION_DEBUG
    m_cache_hits++;
#endif
    entry.Get(format_impl_sp);
    return true;
  }
#ifdef LLDB_CONFIGURATION_DEBUG
  m_cache_misses++;
#endif
  format_impl_sp.reset();
  return false;
}

/// Explicit instantiations for the three types.
/// \{
template bool
FormatCache::Get<lldb::TypeFormatImplSP>(ConstString, lldb::TypeFormatImplSP &);
template bool
FormatCache::Get<lldb::TypeSummaryImplSP>(ConstString,
                                          lldb::TypeSummaryImplSP &);
template bool
FormatCache::Get<lldb::SyntheticChildrenSP>(ConstString,
                                            lldb::SyntheticChildrenSP &);
/// \}

void FormatCache::Set(ConstString type, lldb::TypeFormatImplSP &format_sp) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  GetEntry(type).Set(format_sp);
}

void FormatCache::Set(ConstString type, lldb::TypeSummaryImplSP &summary_sp) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  GetEntry(type).Set(summary_sp);
}

void FormatCache::Set(ConstString type,
                      lldb::SyntheticChildrenSP &synthetic_sp) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  GetEntry(type).Set(synthetic_sp);
}

void FormatCache::Clear() {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  m_map.clear();
}
