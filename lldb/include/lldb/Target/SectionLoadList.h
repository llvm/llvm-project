//===-- SectionLoadList.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SectionLoadList_h_
#define liblldb_SectionLoadList_h_

// C Includes
// C++ Includes
#include <map>

// Other libraries and framework includes
#include "llvm/ADT/DenseMap.h"
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/Host/Mutex.h"

namespace lldb_private {

class SectionLoadList
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    SectionLoadList () :
        m_addr_to_sect (),
        m_sect_to_addr (),
        m_mutex (Mutex::eMutexTypeRecursive)

    {
    }

    ~SectionLoadList()
    {
        // Call clear since this takes a lock and clears the section load list
        // in case another thread is currently using this section load list
        Clear();
    }

    bool
    IsEmpty() const;

    void
    Clear ();

    lldb::addr_t
    GetSectionLoadAddress (const lldb::SectionSP &section_sp) const;

    bool
    ResolveLoadAddress (lldb::addr_t load_addr, Address &so_addr) const;

    bool
    SetSectionLoadAddress (const lldb::SectionSP &section_sp, lldb::addr_t load_addr, bool warn_multiple = false);

    // The old load address should be specified when unloading to ensure we get
    // the correct instance of the section as a shared library could be loaded
    // at more than one location.
    bool
    SetSectionUnloaded (const lldb::SectionSP &section_sp, lldb::addr_t load_addr);

    // Unload all instances of a section. This function can be used on systems
    // that don't support multiple copies of the same shared library to be
    // loaded at the same time.
    size_t
    SetSectionUnloaded (const lldb::SectionSP &section_sp);

    void
    Dump (Stream &s, Target *target);

protected:
    typedef std::map<lldb::addr_t, lldb::SectionSP> addr_to_sect_collection;
    typedef llvm::DenseMap<const Section *, lldb::addr_t> sect_to_addr_collection;
    addr_to_sect_collection m_addr_to_sect;
    sect_to_addr_collection m_sect_to_addr;
    mutable Mutex m_mutex;

private:
    DISALLOW_COPY_AND_ASSIGN (SectionLoadList);
};

} // namespace lldb_private

#endif  // liblldb_SectionLoadList_h_
