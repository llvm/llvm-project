//===-- Section.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Section_h_
#define liblldb_Section_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/AddressRange.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/ModuleChild.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/RangeMap.h"
#include "lldb/Core/UserID.h"
#include "lldb/Core/VMRange.h"
#include <limits.h>

namespace lldb_private {

class SectionList
{
public:
    typedef std::vector<lldb::SectionSP>  collection;
    typedef collection::iterator        iterator;
    typedef collection::const_iterator  const_iterator;

    SectionList();

    virtual
    ~SectionList();

    uint32_t
    AddSection (const lldb::SectionSP& section_sp);

    uint32_t
    AddUniqueSection (const lldb::SectionSP& section_sp);

    uint32_t
    FindSectionIndex (const Section* sect);

    bool
    ContainsSection(lldb::user_id_t sect_id) const;

    void
    Dump (Stream *s, Target *target, bool show_header, uint32_t depth) const;

    lldb::SectionSP
    FindSectionByName (const ConstString &section_dstr) const;

    lldb::SectionSP
    FindSectionByID (lldb::user_id_t sect_id) const;

    lldb::SectionSP
    FindSectionByType (lldb::SectionType sect_type, bool check_children, uint32_t start_idx = 0) const;

    lldb::SectionSP
    FindSectionContainingFileAddress (lldb::addr_t addr, uint32_t depth = UINT32_MAX) const;

    lldb::SectionSP
    FindSectionContainingLinkedFileAddress (lldb::addr_t vm_addr, uint32_t depth) const;

    bool
    GetSectionData (const DataExtractor& module_data, DataExtractor& section_data) const;

    // Get the number of sections in this list only
    size_t
    GetSize () const
    {
        return m_sections.size();
    }

    // Get the number of sections in this list, and any contained child sections
    size_t
    GetNumSections (uint32_t depth) const;

    bool
    ReplaceSection (lldb::user_id_t sect_id, const lldb::SectionSP& section_sp, uint32_t depth = UINT32_MAX);

    lldb::SectionSP
    GetSectionAtIndex (uint32_t idx) const;

    size_t
    Slide (lldb::addr_t slide_amount, bool slide_children);
    
    // Update all section lookup caches
    void
    Finalize ();

protected:
    collection  m_sections;

    typedef RangeDataArray<uint64_t, uint64_t, collection::size_type, 1> SectionRangeCache;
    mutable SectionRangeCache   m_range_cache;
#ifdef LLDB_CONFIGURATION_DEBUG
    mutable bool                m_finalized;
#endif
    
    void BuildRangeCache() const;
    
    void InvalidateRangeCache() const;
};


class Section :
    public STD_ENABLE_SHARED_FROM_THIS(Section),
    public ModuleChild,
    public UserID,
    public Flags
{
public:
    // Create a root section (one that has no parent)
    Section (const lldb::ModuleSP &module_sp,
             lldb::user_id_t sect_id,
             const ConstString &name,
             lldb::SectionType sect_type,
             lldb::addr_t file_vm_addr,
             lldb::addr_t vm_size,
             uint64_t file_offset,
             uint64_t file_size,
             uint32_t flags);

    // Create a section that is a child of parent_section_sp
    Section (const lldb::SectionSP &parent_section_sp,    // NULL for top level sections, non-NULL for child sections
             const lldb::ModuleSP &module_sp,
             lldb::user_id_t sect_id,
             const ConstString &name,
             lldb::SectionType sect_type,
             lldb::addr_t file_vm_addr,
             lldb::addr_t vm_size,
             uint64_t file_offset,
             uint64_t file_size,
             uint32_t flags);

    ~Section ();

    static int
    Compare (const Section& a, const Section& b);

    bool
    ContainsFileAddress (lldb::addr_t vm_addr) const;

    SectionList&
    GetChildren ()
    {
        return m_children;
    }

    const SectionList&
    GetChildren () const
    {
        return m_children;
    }

    void
    Dump (Stream *s, Target *target, uint32_t depth) const;

    void
    DumpName (Stream *s) const;

    lldb::addr_t
    GetLoadBaseAddress (Target *target) const;

    bool
    ResolveContainedAddress (lldb::addr_t offset, Address &so_addr) const;

    uint64_t
    GetFileOffset () const
    {
        return m_file_offset;
    }

    void
    SetFileOffset (uint64_t file_offset) 
    {
        m_file_offset = file_offset;
    }

    uint64_t
    GetFileSize () const
    {
        return m_file_size;
    }

    void
    SetFileSize (uint64_t file_size)
    {
        m_file_size = file_size;
    }

    lldb::addr_t
    GetFileAddress () const;

    lldb::addr_t
    GetOffset () const;


    lldb::addr_t
    GetByteSize () const
    {
        return m_byte_size;
    }
    
    void
    SetByteSize (lldb::addr_t byte_size)
    {
        m_byte_size = byte_size;
    }
    
    bool
    IsFake() const
    {
        return m_fake;
    }

    void
    SetIsFake(bool fake)
    {
        m_fake = fake;
    }
    
    bool
    IsEncrypted () const
    {
        return m_encrypted;
    }
    
    void
    SetIsEncrypted (bool b)
    {
        m_encrypted = b;
    }

    bool
    IsDescendant (const Section *section);

    const ConstString&
    GetName () const;

    bool
    Slide (lldb::addr_t slide_amount, bool slide_children);

    void
    SetLinkedLocation (const lldb::SectionSP &linked_section_sp, uint64_t linked_offset);

    bool
    ContainsLinkedFileAddress (lldb::addr_t vm_addr) const;

    lldb::SectionSP
    GetLinkedSection () const
    {
        return m_linked_section_wp.lock();
    }

    uint64_t
    GetLinkedOffset () const
    {
        return m_linked_offset;
    }

    lldb::addr_t
    GetLinkedFileAddress () const;

    lldb::SectionType
    GetType () const
    {
        return m_type;
    }

    lldb::SectionSP
    GetParent () const
    {
        return m_parent_wp.lock();
    }
    
    bool
    IsThreadSpecific () const
    {
        return m_thread_specific;
    }

    void
    SetIsThreadSpecific (bool b)
    {
        m_thread_specific = b;
    }
    
    // Update all section lookup caches
    void
    Finalize ()
    {
        m_children.Finalize();
    }

protected:

    lldb::SectionWP m_parent_wp;        // Weak pointer to parent section
    ConstString     m_name;             // Name of this section
    lldb::SectionType m_type;           // The type of this section
    lldb::addr_t    m_file_addr;        // The absolute file virtual address range of this section if m_parent == NULL,
                                        // offset from parent file virtual address if m_parent != NULL
    lldb::addr_t    m_byte_size;        // Size in bytes that this section will occupy in memory at runtime
    uint64_t        m_file_offset;      // Object file offset (if any)
    uint64_t        m_file_size;        // Object file size (can be smaller than m_byte_size for zero filled sections...)
    SectionList     m_children;         // Child sections
    bool            m_fake:1,           // If true, then this section only can contain the address if one of its
                                        // children contains an address. This allows for gaps between the children
                                        // that are contained in the address range for this section, but do not produce
                                        // hits unless the children contain the address.
                    m_encrypted:1,      // Set to true if the contents are encrypted
                    m_thread_specific:1;// This section is thread specific
    lldb::SectionWP m_linked_section_wp;
    uint64_t        m_linked_offset;
private:
    DISALLOW_COPY_AND_ASSIGN (Section);
};


} // namespace lldb_private

#endif  // liblldb_Section_h_
