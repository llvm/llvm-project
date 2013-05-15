//===-- FormatCache.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_FormatCache_h_
#define lldb_FormatCache_h_

// C Includes
// C++ Includes
#include <map>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/Core/ConstString.h"
#include "lldb/DataFormatters/FormatClasses.h"

namespace lldb_private {
class FormatCache
{
private:
    struct Entry
    {
    private:
        bool m_summary_cached : 1;
        bool m_synthetic_cached : 1;
        
        lldb::TypeSummaryImplSP m_summary_sp;
        lldb::SyntheticChildrenSP m_synthetic_sp;
    public:
        Entry ();
        Entry (lldb::TypeSummaryImplSP);
        Entry (lldb::SyntheticChildrenSP);
        Entry (lldb::TypeSummaryImplSP,lldb::SyntheticChildrenSP);

        bool
        IsSummaryCached ();
        
        bool
        IsSyntheticCached ();
        
        lldb::TypeSummaryImplSP
        GetSummary ();
        
        lldb::SyntheticChildrenSP
        GetSynthetic ();
        
        void
        SetSummary (lldb::TypeSummaryImplSP);
        
        void
        SetSynthetic (lldb::SyntheticChildrenSP);
    };
    typedef std::map<ConstString,Entry> CacheMap;
    CacheMap m_map;
    Mutex m_mutex;
    
#ifdef LLDB_CONFIGURATION_DEBUG
    uint64_t m_cache_hits;
    uint64_t m_cache_misses;
#endif
    
    Entry&
    GetEntry (const ConstString& type);
    
public:
    FormatCache ();
    
    bool
    GetSummary (const ConstString& type,lldb::TypeSummaryImplSP& summary_sp);

    bool
    GetSynthetic (const ConstString& type,lldb::SyntheticChildrenSP& synthetic_sp);
    
    void
    SetSummary (const ConstString& type,lldb::TypeSummaryImplSP& summary_sp);
    
    void
    SetSynthetic (const ConstString& type,lldb::SyntheticChildrenSP& synthetic_sp);
    
    void
    Clear ();
    
#ifdef LLDB_CONFIGURATION_DEBUG
    uint64_t
    GetCacheHits ()
    {
        return m_cache_hits;
    }
    
    uint64_t
    GetCacheMisses ()
    {
        return m_cache_misses;
    }
#endif
};
} // namespace lldb_private

#endif	// lldb_FormatCache_h_
