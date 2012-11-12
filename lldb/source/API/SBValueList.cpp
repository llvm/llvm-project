//===-- SBValueList.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lldb/API/SBValueList.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBStream.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/ValueObjectList.h"

using namespace lldb;
using namespace lldb_private;

SBValueList::SBValueList () :
    m_opaque_ap ()
{
}

SBValueList::SBValueList (const SBValueList &rhs) :
    m_opaque_ap ()
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (rhs.IsValid())
        m_opaque_ap.reset (new ValueObjectList (*rhs));

    if (log)
    {
        log->Printf ("SBValueList::SBValueList (rhs.ap=%p) => this.ap = %p",
                     (rhs.IsValid() ? rhs.m_opaque_ap.get() : NULL), 
                     m_opaque_ap.get());
    }
}

SBValueList::SBValueList (const ValueObjectList *lldb_object_ptr) :
    m_opaque_ap ()
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    if (lldb_object_ptr)
        m_opaque_ap.reset (new ValueObjectList (*lldb_object_ptr));

    if (log)
    {
        log->Printf ("SBValueList::SBValueList (lldb_object_ptr=%p) => this.ap = %p", 
                     lldb_object_ptr, 
                     m_opaque_ap.get());
    }
}

SBValueList::~SBValueList ()
{
}

bool
SBValueList::IsValid () const
{
    return (m_opaque_ap.get() != NULL);
}

void
SBValueList::Clear()
{
    m_opaque_ap.reset();
}

const SBValueList &
SBValueList::operator = (const SBValueList &rhs)
{
    if (this != &rhs)
    {
        if (rhs.IsValid())
            m_opaque_ap.reset (new ValueObjectList (*rhs));
        else
            m_opaque_ap.reset ();
    }
    return *this;
}

ValueObjectList *
SBValueList::operator->()
{
    return m_opaque_ap.get();
}

ValueObjectList &
SBValueList::operator*()
{
    return *m_opaque_ap;
}

const ValueObjectList *
SBValueList::operator->() const
{
    return m_opaque_ap.get();
}

const ValueObjectList &
SBValueList::operator*() const
{
    return *m_opaque_ap;
}

void
SBValueList::Append (const SBValue &val_obj)
{
    ValueObjectSP value_sp (val_obj.GetSP());
    if (value_sp)
    {
        CreateIfNeeded ();
        m_opaque_ap->Append (value_sp);
    }
}

void
SBValueList::Append (lldb::ValueObjectSP& val_obj_sp)
{
    if (val_obj_sp)
    {
        CreateIfNeeded ();
        m_opaque_ap->Append (val_obj_sp);
    }
}

void
SBValueList::Append (const lldb::SBValueList& value_list)
{
    if (value_list.IsValid())
    {
        CreateIfNeeded ();
        m_opaque_ap->Append (*value_list);
    }
}


SBValue
SBValueList::GetValueAtIndex (uint32_t idx) const
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    //if (log)
    //    log->Printf ("SBValueList::GetValueAtIndex (uint32_t idx) idx = %d", idx);

    SBValue sb_value;
    ValueObjectSP value_sp;
    if (m_opaque_ap.get())
    {
        value_sp = m_opaque_ap->GetValueObjectAtIndex (idx);
        sb_value.SetSP (value_sp);
    }

    if (log)
    {
        SBStream sstr;
        sb_value.GetDescription (sstr);
        log->Printf ("SBValueList::GetValueAtIndex (this.ap=%p, idx=%d) => SBValue (this.sp = %p, '%s')", 
                     m_opaque_ap.get(), idx, value_sp.get(), sstr.GetData());
    }

    return sb_value;
}

uint32_t
SBValueList::GetSize () const
{
    LogSP log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_API));

    //if (log)
    //    log->Printf ("SBValueList::GetSize ()");

    uint32_t size = 0;
    if (m_opaque_ap.get())
        size = m_opaque_ap->GetSize();

    if (log)
        log->Printf ("SBValueList::GetSize (this.ap=%p) => %d", m_opaque_ap.get(), size);

    return size;
}

void
SBValueList::CreateIfNeeded ()
{
    if (m_opaque_ap.get() == NULL)
        m_opaque_ap.reset (new ValueObjectList());
}


SBValue
SBValueList::FindValueObjectByUID (lldb::user_id_t uid)
{
    SBValue sb_value;
    if (m_opaque_ap.get())
        sb_value.SetSP (m_opaque_ap->FindValueObjectByUID (uid));
    return sb_value;
}

ValueObjectList *
SBValueList::get ()
{
    return m_opaque_ap.get();
}

ValueObjectList &
SBValueList::ref ()
{
    CreateIfNeeded();
    return *m_opaque_ap.get();
}


