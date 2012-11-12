//===-- Section.cpp ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Section.h"
#include "lldb/Core/Module.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;

Section::Section (const ModuleSP &module_sp,
                  user_id_t sect_id,
                  const ConstString &name,
                  SectionType sect_type,
                  addr_t file_addr,
                  addr_t byte_size,
                  uint64_t file_offset,
                  uint64_t file_size,
                  uint32_t flags) :
    ModuleChild     (module_sp),
    UserID          (sect_id),
    Flags           (flags),
    m_parent_wp     (),
    m_name          (name),
    m_type          (sect_type),
    m_file_addr     (file_addr),
    m_byte_size     (byte_size),
    m_file_offset   (file_offset),
    m_file_size     (file_size),
    m_children      (),
    m_fake          (false),
    m_encrypted     (false),
    m_thread_specific (false),
    m_linked_section_wp(),
    m_linked_offset (0)
{
//    printf ("Section::Section(%p): module=%p, sect_id = 0x%16.16llx, addr=[0x%16.16llx - 0x%16.16llx), file [0x%16.16llx - 0x%16.16llx), flags = 0x%8.8x, name = %s\n",
//            this, module_sp.get(), sect_id, file_addr, file_addr + byte_size, file_offset, file_offset + file_size, flags, name.GetCString());
}

Section::Section (const lldb::SectionSP &parent_section_sp,
                  const ModuleSP &module_sp,
                  user_id_t sect_id,
                  const ConstString &name,
                  SectionType sect_type,
                  addr_t file_addr,
                  addr_t byte_size,
                  uint64_t file_offset,
                  uint64_t file_size,
                  uint32_t flags) :
    ModuleChild     (module_sp),
    UserID          (sect_id),
    Flags           (flags),
    m_parent_wp     (),
    m_name          (name),
    m_type          (sect_type),
    m_file_addr     (file_addr),
    m_byte_size     (byte_size),
    m_file_offset   (file_offset),
    m_file_size     (file_size),
    m_children      (),
    m_fake          (false),
    m_encrypted     (false),
    m_thread_specific (false),
    m_linked_section_wp(),
    m_linked_offset (0)
{
//    printf ("Section::Section(%p): module=%p, sect_id = 0x%16.16llx, addr=[0x%16.16llx - 0x%16.16llx), file [0x%16.16llx - 0x%16.16llx), flags = 0x%8.8x, name = %s.%s\n",
//            this, module_sp.get(), sect_id, file_addr, file_addr + byte_size, file_offset, file_offset + file_size, flags, parent_section_sp->GetName().GetCString(), name.GetCString());
    if (parent_section_sp)
        m_parent_wp = parent_section_sp;
}

Section::~Section()
{
//    printf ("Section::~Section(%p)\n", this);
}

const ConstString&
Section::GetName() const
{
    SectionSP linked_section_sp (m_linked_section_wp.lock());
    if (linked_section_sp)
        return linked_section_sp->GetName();
    return m_name;
}

addr_t
Section::GetFileAddress () const
{
    SectionSP parent_sp (GetParent ());
    if (parent_sp)
    {
        // This section has a parent which means m_file_addr is an offset into
        // the parent section, so the file address for this section is the file
        // address of the parent plus the offset
        return parent_sp->GetFileAddress() + m_file_addr;
    }
    // This section has no parent, so m_file_addr is the file base address
    return m_file_addr;
}

lldb::addr_t
Section::GetOffset () const
{
    // This section has a parent which means m_file_addr is an offset.
    SectionSP parent_sp (GetParent ());
    if (parent_sp)
        return m_file_addr;
    
    // This section has no parent, so there is no offset to be had
    return 0;
}


addr_t
Section::GetLinkedFileAddress () const
{
    SectionSP linked_section_sp (m_linked_section_wp.lock());
    if (linked_section_sp)
        return linked_section_sp->GetFileAddress() + m_linked_offset;
    return LLDB_INVALID_ADDRESS;
}


addr_t
Section::GetLoadBaseAddress (Target *target) const
{
    addr_t load_base_addr = LLDB_INVALID_ADDRESS;
    SectionSP linked_section_sp (m_linked_section_wp.lock());
    if (linked_section_sp)
    {
        load_base_addr = linked_section_sp->GetLoadBaseAddress(target);
        if (load_base_addr != LLDB_INVALID_ADDRESS)
            load_base_addr += m_linked_offset;
    }
    else
    {
        SectionSP parent_sp (GetParent ());
        if (parent_sp)
        {
            load_base_addr = parent_sp->GetLoadBaseAddress (target);
            if (load_base_addr != LLDB_INVALID_ADDRESS)
                load_base_addr += GetOffset();
        }
        else
        {
            load_base_addr = target->GetSectionLoadList().GetSectionLoadAddress (const_cast<Section *>(this)->shared_from_this());
        }
    }

    return load_base_addr;
}

bool
Section::ResolveContainedAddress (addr_t offset, Address &so_addr) const
{
    const uint32_t num_children = m_children.GetSize();
    if (num_children > 0)
    {
        for (uint32_t i=0; i<num_children; i++)
        {
            Section* child_section = m_children.GetSectionAtIndex (i).get();

            addr_t child_offset = child_section->GetOffset();
            if (child_offset <= offset && offset - child_offset < child_section->GetByteSize())
                return child_section->ResolveContainedAddress (offset - child_offset, so_addr);
        }
    }
    SectionSP linked_section_sp (m_linked_section_wp.lock());
    if (linked_section_sp)
    {
        so_addr.SetOffset(m_linked_offset + offset);
        so_addr.SetSection(linked_section_sp);
    }
    else
    {
        so_addr.SetOffset(offset);
        so_addr.SetSection(const_cast<Section *>(this)->shared_from_this());
        
#ifdef LLDB_CONFIGURATION_DEBUG
        // For debug builds, ensure that there are no orphaned (i.e., moduleless) sections.
        assert(GetModule().get());
#endif
    }
    return true;
}

bool
Section::ContainsFileAddress (addr_t vm_addr) const
{
    const addr_t file_addr = GetFileAddress();
    if (file_addr != LLDB_INVALID_ADDRESS)
    {
        if (file_addr <= vm_addr)
        {
            const addr_t offset = vm_addr - file_addr;
            return offset < GetByteSize();
        }
    }
    return false;
}

bool
Section::ContainsLinkedFileAddress (addr_t vm_addr) const
{
    const addr_t linked_file_addr = GetLinkedFileAddress();
    if (linked_file_addr != LLDB_INVALID_ADDRESS)
    {
        if (linked_file_addr <= vm_addr)
        {
            const addr_t offset = vm_addr - linked_file_addr;
            return offset < GetByteSize();
        }
    }
    return false;
}

int
Section::Compare (const Section& a, const Section& b)
{
    if (&a == &b)
        return 0;

    const ModuleSP a_module_sp = a.GetModule();
    const ModuleSP b_module_sp = b.GetModule();
    if (a_module_sp == b_module_sp)
    {
        user_id_t a_sect_uid = a.GetID();
        user_id_t b_sect_uid = b.GetID();
        if (a_sect_uid < b_sect_uid)
            return -1;
        if (a_sect_uid > b_sect_uid)
            return 1;
        return 0;
    }
    else
    {
        // The modules are different, just compare the module pointers
        if (a_module_sp.get() < b_module_sp.get())
            return -1;
        else
            return 1;   // We already know the modules aren't equal
    }
}


void
Section::Dump (Stream *s, Target *target, uint32_t depth) const
{
//    s->Printf("%.*p: ", (int)sizeof(void*) * 2, this);
    s->Indent();
    s->Printf("0x%8.8llx %-16s ", GetID(), GetSectionTypeAsCString (m_type));
    bool resolved = true;
    addr_t addr = LLDB_INVALID_ADDRESS;

    SectionSP linked_section_sp (m_linked_section_wp.lock());
    if (GetByteSize() == 0)
        s->Printf("%39s", "");
    else
    {
        if (target && linked_section_sp.get() == NULL)
            addr = GetLoadBaseAddress (target);

        if (addr == LLDB_INVALID_ADDRESS)
        {
            if (target)
                resolved = false;
            addr = GetFileAddress();
        }

        VMRange range(addr, addr + m_byte_size);
        range.Dump (s, 0);
    }

    s->Printf("%c 0x%8.8llx 0x%8.8llx 0x%8.8x ", resolved ? ' ' : '*', m_file_offset, m_file_size, Get());

    DumpName (s);

    s->EOL();

    if (linked_section_sp)
    {
        addr = LLDB_INVALID_ADDRESS;
        resolved = true;
        if (target)
        {
            addr = linked_section_sp->GetLoadBaseAddress(target);
            if (addr != LLDB_INVALID_ADDRESS)
                addr += m_linked_offset;
        }

        if (addr == LLDB_INVALID_ADDRESS)
        {
            if (target)
                resolved = false;
            addr = linked_section_sp->GetFileAddress() + m_linked_offset;
        }

        int indent = 28 + s->GetIndentLevel();
        s->Printf("%*.*s", indent, indent, "");
        VMRange linked_range(addr, addr + m_byte_size);
        linked_range.Dump (s, 0);
        indent = 3 * (sizeof(uint32_t) * 2 + 2 + 1) + 1;
        s->Printf("%c%*.*s", resolved ? ' ' : '*', indent, indent, "");

        linked_section_sp->DumpName(s);
        s->Printf(" + 0x%llx\n", m_linked_offset);
    }

    if (depth > 0)
        m_children.Dump(s, target, false, depth - 1);
}

void
Section::DumpName (Stream *s) const
{
    SectionSP parent_sp (GetParent ());
    if (parent_sp)
    {
        parent_sp->DumpName (s);
        s->PutChar('.');
    }
    else
    {
        // The top most section prints the module basename
        ModuleSP module_sp (GetModule());
        if (module_sp)
        {
            const char *module_basename = module_sp->GetFileSpec().GetFilename().AsCString();
            if (module_basename && module_basename[0])
                s->Printf("%s.", module_basename);
        }
    }
    m_name.Dump(s);
}

bool
Section::IsDescendant (const Section *section)
{
    if (this == section)
        return true;
    SectionSP parent_sp (GetParent ());
    if (parent_sp)
        return parent_sp->IsDescendant (section);
    return false;
}

bool
Section::Slide (addr_t slide_amount, bool slide_children)
{
    if (m_file_addr != LLDB_INVALID_ADDRESS)
    {
        if (slide_amount == 0)
            return true;

        m_file_addr += slide_amount;

        if (slide_children)
            m_children.Slide (slide_amount, slide_children);

        return true;
    }
    return false;
}

void
Section::SetLinkedLocation (const lldb::SectionSP &linked_section_sp, uint64_t linked_offset)
{
    if (linked_section_sp)
        m_module_wp = linked_section_sp->GetModule();
    m_linked_section_wp = linked_section_sp;
    m_linked_offset  = linked_offset;
}

#pragma mark SectionList

SectionList::SectionList () :
    m_sections()
#ifdef LLDB_CONFIGURATION_DEBUG
    , m_finalized(false)
#endif
{
}


SectionList::~SectionList ()
{
}

uint32_t
SectionList::AddSection (const lldb::SectionSP& section_sp)
{
    assert (section_sp.get());
    uint32_t section_index = m_sections.size();
    m_sections.push_back(section_sp);
    InvalidateRangeCache();
    return section_index;
}

uint32_t
SectionList::FindSectionIndex (const Section* sect)
{
    iterator sect_iter;
    iterator begin = m_sections.begin();
    iterator end = m_sections.end();
    for (sect_iter = begin; sect_iter != end; ++sect_iter)
    {
        if (sect_iter->get() == sect)
        {
            // The secton was already in this section list
            return std::distance (begin, sect_iter);
        }
    }
    return UINT32_MAX;
}

uint32_t
SectionList::AddUniqueSection (const lldb::SectionSP& sect_sp)
{
    uint32_t sect_idx = FindSectionIndex (sect_sp.get());
    if (sect_idx == UINT32_MAX)
        sect_idx = AddSection (sect_sp);
    return sect_idx;
}


bool
SectionList::ReplaceSection (user_id_t sect_id, const lldb::SectionSP& sect_sp, uint32_t depth)
{
    iterator sect_iter, end = m_sections.end();
    for (sect_iter = m_sections.begin(); sect_iter != end; ++sect_iter)
    {
        if ((*sect_iter)->GetID() == sect_id)
        {
            *sect_iter = sect_sp;
            InvalidateRangeCache();
            return true;
        }
        else if (depth > 0)
        {
            if ((*sect_iter)->GetChildren().ReplaceSection(sect_id, sect_sp, depth - 1))
                return true;
        }
    }
    return false;
}


size_t
SectionList::GetNumSections (uint32_t depth) const
{
    size_t count = m_sections.size();
    if (depth > 0)
    {
        const_iterator sect_iter, end = m_sections.end();
        for (sect_iter = m_sections.begin(); sect_iter != end; ++sect_iter)
        {
            count += (*sect_iter)->GetChildren().GetNumSections(depth - 1);
        }
    }
    return count;
}

SectionSP
SectionList::GetSectionAtIndex (uint32_t idx) const
{
    SectionSP sect_sp;
    if (idx < m_sections.size())
        sect_sp = m_sections[idx];
    return sect_sp;
}

SectionSP
SectionList::FindSectionByName (const ConstString &section_dstr) const
{
    SectionSP sect_sp;
    // Check if we have a valid section string
    if (section_dstr && !m_sections.empty())
    {
        const_iterator sect_iter;
        const_iterator end = m_sections.end();
        for (sect_iter = m_sections.begin(); sect_iter != end && sect_sp.get() == NULL; ++sect_iter)
        {
            Section *child_section = sect_iter->get();
            assert (child_section);
            if (child_section->GetName() == section_dstr)
            {
                sect_sp = *sect_iter;
            }
            else
            {
                sect_sp = child_section->GetChildren().FindSectionByName(section_dstr);
            }
        }
    }
    return sect_sp;
}

SectionSP
SectionList::FindSectionByID (user_id_t sect_id) const
{
    SectionSP sect_sp;
    if (sect_id)
    {
        const_iterator sect_iter;
        const_iterator end = m_sections.end();
        for (sect_iter = m_sections.begin(); sect_iter != end && sect_sp.get() == NULL; ++sect_iter)
        {
            if ((*sect_iter)->GetID() == sect_id)
            {
                sect_sp = *sect_iter;
                break;
            }
            else
            {
                sect_sp = (*sect_iter)->GetChildren().FindSectionByID (sect_id);
            }
        }
    }
    return sect_sp;
}


SectionSP
SectionList::FindSectionByType (SectionType sect_type, bool check_children, uint32_t start_idx) const
{
    SectionSP sect_sp;
    uint32_t num_sections = m_sections.size();
    for (uint32_t idx = start_idx; idx < num_sections; ++idx)
    {
        if (m_sections[idx]->GetType() == sect_type)
        {
            sect_sp = m_sections[idx];
            break;
        }
        else if (check_children)
        {
            sect_sp = m_sections[idx]->GetChildren().FindSectionByType (sect_type, check_children, 0);
            if (sect_sp)
                break;
        }
    }
    return sect_sp;
}

SectionSP
SectionList::FindSectionContainingFileAddress (addr_t vm_addr, uint32_t depth) const
{
    SectionSP sect_sp;
    const_iterator sect_iter;
    const_iterator end = m_sections.end();
    for (sect_iter = m_sections.begin(); sect_iter != end && sect_sp.get() == NULL; ++sect_iter)
    {
        Section *sect = sect_iter->get();
        if (sect->ContainsFileAddress (vm_addr))
        {
            // The file address is in this section. We need to make sure one of our child
            // sections doesn't contain this address as well as obeying the depth limit
            // that was passed in.
            if (depth > 0)
                sect_sp = sect->GetChildren().FindSectionContainingFileAddress(vm_addr, depth - 1);

            if (sect_sp.get() == NULL && !sect->IsFake())
                sect_sp = *sect_iter;
        }
    }
    return sect_sp;
}

void
SectionList::BuildRangeCache() const
{
    m_range_cache.Clear();
    
    for (collection::size_type idx = 0, last_idx = m_sections.size();
         idx < last_idx;
         ++idx)
    {
        Section *sect = m_sections[idx].get();
        
        addr_t linked_file_address = sect->GetLinkedFileAddress();
        
        if (linked_file_address != LLDB_INVALID_ADDRESS)
            m_range_cache.Append(SectionRangeCache::Entry(linked_file_address, sect->GetByteSize(), idx));
    }
    
    m_range_cache.Sort();
    
#ifdef LLDB_CONFIGURATION_DEBUG
    m_finalized = true;
#endif
}

void
SectionList::InvalidateRangeCache() const
{
#ifdef LLDB_CONFIGURATION_DEBUG
    assert(!m_finalized);
#endif
    m_range_cache.Clear();
}

SectionSP
SectionList::FindSectionContainingLinkedFileAddress (addr_t vm_addr, uint32_t depth) const
{
    //if (m_range_cache.IsEmpty())
    //    BuildRangeCache();
#ifdef LLDB_CONFIGURATION_DEBUG
    assert(m_finalized);
#endif
    
    SectionRangeCache::Entry *entry = m_range_cache.FindEntryThatContains(vm_addr);
    
    if (entry)
        return m_sections[entry->data];
        
    if (depth == 0)
        return SectionSP();
    
    for (const_iterator si = m_sections.begin(), se = m_sections.end();
         si != se;
         ++si)
    {
        Section *sect = si->get();
        
        SectionSP sect_sp = sect->GetChildren().FindSectionContainingLinkedFileAddress(vm_addr, depth - 1);
            
        if (sect_sp)
            return sect_sp;
    }
    
    return SectionSP();
}

bool
SectionList::ContainsSection(user_id_t sect_id) const
{
    return FindSectionByID (sect_id).get() != NULL;
}

void
SectionList::Dump (Stream *s, Target *target, bool show_header, uint32_t depth) const
{
    bool target_has_loaded_sections = target && !target->GetSectionLoadList().IsEmpty();
    if (show_header && !m_sections.empty())
    {
        s->Indent();
        s->Printf(    "SectID     Type             %s Address                             File Off.  File Size  Flags      Section Name\n", target_has_loaded_sections ? "Load" : "File");
        s->Indent();
        s->PutCString("---------- ---------------- ---------------------------------------  ---------- ---------- ---------- ----------------------------\n");
    }


    const_iterator sect_iter;
    const_iterator end = m_sections.end();
    for (sect_iter = m_sections.begin(); sect_iter != end; ++sect_iter)
    {
        (*sect_iter)->Dump(s, target_has_loaded_sections ? target : NULL, depth);
    }

    if (show_header && !m_sections.empty())
        s->IndentLess();

}

size_t
SectionList::Slide (addr_t slide_amount, bool slide_children)
{
    size_t count = 0;
    const_iterator pos, end = m_sections.end();
    for (pos = m_sections.begin(); pos != end; ++pos)
    {
        if ((*pos)->Slide(slide_amount, slide_children))
            ++count;
    }
    InvalidateRangeCache();
    return count;
}

void
SectionList::Finalize ()
{
    BuildRangeCache();
    
    for (const_iterator si = m_sections.begin(), se = m_sections.end();
         si != se;
         ++si)
    {
        Section *sect = si->get();
        
        sect->GetChildren().Finalize();
    }
}

