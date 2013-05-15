//===-- FormatCache.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

// C Includes

// C++ Includes

// Other libraries and framework includes

// Project includes
#include "lldb/DataFormatters/FormatCache.h"

using namespace lldb;
using namespace lldb_private;

FormatCache::Entry::Entry () :
m_summary_cached(false),
m_synthetic_cached(false),
m_summary_sp(),
m_synthetic_sp()
{}

FormatCache::Entry::Entry (lldb::TypeSummaryImplSP summary_sp) :
m_synthetic_cached(false),
m_synthetic_sp()
{
    SetSummary (summary_sp);
}

FormatCache::Entry::Entry (lldb::SyntheticChildrenSP synthetic_sp) :
m_summary_cached(false),
m_summary_sp()
{
    SetSynthetic (synthetic_sp);
}

FormatCache::Entry::Entry (lldb::TypeSummaryImplSP summary_sp,lldb::SyntheticChildrenSP synthetic_sp)
{
    SetSummary (summary_sp);
    SetSynthetic (synthetic_sp);
}

bool
FormatCache::Entry::IsSummaryCached ()
{
    return m_summary_cached;
}

bool
FormatCache::Entry::IsSyntheticCached ()
{
    return m_synthetic_cached;
}

lldb::TypeSummaryImplSP
FormatCache::Entry::GetSummary ()
{
    return m_summary_sp;
}

lldb::SyntheticChildrenSP
FormatCache::Entry::GetSynthetic ()
{
    return m_synthetic_sp;
}

void
FormatCache::Entry::SetSummary (lldb::TypeSummaryImplSP summary_sp)
{
    m_summary_cached = true;
    m_summary_sp = summary_sp;
}

void
FormatCache::Entry::SetSynthetic (lldb::SyntheticChildrenSP synthetic_sp)
{
    m_synthetic_cached = true;
    m_synthetic_sp = synthetic_sp;
}

FormatCache::FormatCache () :
m_map(),
m_mutex (Mutex::eMutexTypeRecursive)
#ifdef LLDB_CONFIGURATION_DEBUG
,m_cache_hits(0),m_cache_misses(0)
#endif
{
}

FormatCache::Entry&
FormatCache::GetEntry (const ConstString& type)
{
    auto i = m_map.find(type),
    e = m_map.end();
    if (i != e)
        return i->second;
    m_map[type] = FormatCache::Entry();
    return m_map[type];
}

bool
FormatCache::GetSummary (const ConstString& type,lldb::TypeSummaryImplSP& summary_sp)
{
    Mutex::Locker lock(m_mutex);
    auto entry = GetEntry(type);
    if (entry.IsSummaryCached())
    {
#ifdef LLDB_CONFIGURATION_DEBUG
        m_cache_hits++;
#endif
        summary_sp = entry.GetSummary();
        return true;
    }
#ifdef LLDB_CONFIGURATION_DEBUG
    m_cache_misses++;
#endif
    summary_sp.reset();
    return false;
}

bool
FormatCache::GetSynthetic (const ConstString& type,lldb::SyntheticChildrenSP& synthetic_sp)
{
    Mutex::Locker lock(m_mutex);
    auto entry = GetEntry(type);
    if (entry.IsSyntheticCached())
    {
#ifdef LLDB_CONFIGURATION_DEBUG
        m_cache_hits++;
#endif
        synthetic_sp = entry.GetSynthetic();
        return true;
    }
#ifdef LLDB_CONFIGURATION_DEBUG
    m_cache_misses++;
#endif
    synthetic_sp.reset();
    return false;
}

void
FormatCache::SetSummary (const ConstString& type,lldb::TypeSummaryImplSP& summary_sp)
{
    Mutex::Locker lock(m_mutex);
    GetEntry(type).SetSummary(summary_sp);
}

void
FormatCache::SetSynthetic (const ConstString& type,lldb::SyntheticChildrenSP& synthetic_sp)
{
    Mutex::Locker lock(m_mutex);
    GetEntry(type).SetSynthetic(synthetic_sp);
}

void
FormatCache::Clear ()
{
    Mutex::Locker lock(m_mutex);
    m_map.clear();
}

