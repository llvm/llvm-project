//===-- SBTypeSummary.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/API/SBTypeSummary.h"

#include "lldb/API/SBStream.h"

#include "lldb/DataFormatters/DataVisualization.h"

using namespace lldb;
using namespace lldb_private;

#ifndef LLDB_DISABLE_PYTHON

SBTypeSummary::SBTypeSummary() :
m_opaque_sp()
{
}

SBTypeSummary
SBTypeSummary::CreateWithSummaryString (const char* data, uint32_t options)
{
    if (!data || data[0] == 0)
        return SBTypeSummary();
        
    return SBTypeSummary(TypeSummaryImplSP(new StringSummaryFormat(options, data)));
}

SBTypeSummary
SBTypeSummary::CreateWithFunctionName (const char* data, uint32_t options)
{
    if (!data || data[0] == 0)
        return SBTypeSummary();
    
    return SBTypeSummary(TypeSummaryImplSP(new ScriptSummaryFormat(options, data)));
}

SBTypeSummary
SBTypeSummary::CreateWithScriptCode (const char* data, uint32_t options)
{
    if (!data || data[0] == 0)
        return SBTypeSummary();
    
    return SBTypeSummary(TypeSummaryImplSP(new ScriptSummaryFormat(options, "", data)));
}

SBTypeSummary::SBTypeSummary (const lldb::SBTypeSummary &rhs) :
m_opaque_sp(rhs.m_opaque_sp)
{
}

SBTypeSummary::~SBTypeSummary ()
{
}

bool
SBTypeSummary::IsValid() const
{
    return m_opaque_sp.get() != NULL;
}

bool
SBTypeSummary::IsFunctionCode()
{
    if (!IsValid())
        return false;
    if (m_opaque_sp->IsScripted())
    {
        ScriptSummaryFormat* script_summary_ptr = (ScriptSummaryFormat*)m_opaque_sp.get();
        const char* ftext = script_summary_ptr->GetPythonScript();
        return (ftext && *ftext != 0);
    }
    return false;
}

bool
SBTypeSummary::IsFunctionName()
{
    if (!IsValid())
        return false;
    if (m_opaque_sp->IsScripted())
    {
        ScriptSummaryFormat* script_summary_ptr = (ScriptSummaryFormat*)m_opaque_sp.get();
        const char* ftext = script_summary_ptr->GetPythonScript();
        return (!ftext || *ftext == 0);
    }
    return false;
}

bool
SBTypeSummary::IsSummaryString()
{
    if (!IsValid())
        return false;
    
    if (m_opaque_sp->GetType() == lldb_private::TypeSummaryImpl::eTypeCallback)
        return false;
    
    return !m_opaque_sp->IsScripted();
}

const char*
SBTypeSummary::GetData ()
{
    if (!IsValid())
        return NULL;
    if (m_opaque_sp->GetType() == lldb_private::TypeSummaryImpl::eTypeCallback)
        return NULL;
    if (m_opaque_sp->IsScripted())
    {
        ScriptSummaryFormat* script_summary_ptr = (ScriptSummaryFormat*)m_opaque_sp.get();
        const char* fname = script_summary_ptr->GetFunctionName();
        const char* ftext = script_summary_ptr->GetPythonScript();
        if (ftext && *ftext)
            return ftext;
        return fname;
    }
    else
    {
        StringSummaryFormat* string_summary_ptr = (StringSummaryFormat*)m_opaque_sp.get();
        return string_summary_ptr->GetSummaryString();
    }
}

uint32_t
SBTypeSummary::GetOptions ()
{
    if (!IsValid())
        return lldb::eTypeOptionNone;
    return m_opaque_sp->GetOptions();
}

void
SBTypeSummary::SetOptions (uint32_t value)
{
    if (!CopyOnWrite_Impl())
        return;
    m_opaque_sp->SetOptions(value);
}

void
SBTypeSummary::SetSummaryString (const char* data)
{
    if (!IsValid())
        return;
    if (m_opaque_sp->IsScripted() || (m_opaque_sp->GetType() == lldb_private::TypeSummaryImpl::eTypeCallback))
        ChangeSummaryType(false);
    ((StringSummaryFormat*)m_opaque_sp.get())->SetSummaryString(data);
}

void
SBTypeSummary::SetFunctionName (const char* data)
{
    if (!IsValid())
        return;
    if (!m_opaque_sp->IsScripted())
        ChangeSummaryType(true);
    ((ScriptSummaryFormat*)m_opaque_sp.get())->SetFunctionName(data);
}

void
SBTypeSummary::SetFunctionCode (const char* data)
{
    if (!IsValid())
        return;
    if (!m_opaque_sp->IsScripted())
        ChangeSummaryType(true);
    ((ScriptSummaryFormat*)m_opaque_sp.get())->SetPythonScript(data);
}

bool
SBTypeSummary::GetDescription (lldb::SBStream &description, 
                              lldb::DescriptionLevel description_level)
{
    if (!CopyOnWrite_Impl())
        return false;
    else {
        description.Printf("%s\n",
                           m_opaque_sp->GetDescription().c_str());
        return true;
    }
}

lldb::SBTypeSummary &
SBTypeSummary::operator = (const lldb::SBTypeSummary &rhs)
{
    if (this != &rhs)
    {
        m_opaque_sp = rhs.m_opaque_sp;
    }
    return *this;
}

bool
SBTypeSummary::operator == (lldb::SBTypeSummary &rhs)
{
    if (IsValid() == false)
        return !rhs.IsValid();
    return m_opaque_sp == rhs.m_opaque_sp;
}

bool
SBTypeSummary::IsEqualTo (lldb::SBTypeSummary &rhs)
{
    if (IsValid() == false)
        return !rhs.IsValid();

    if (m_opaque_sp->GetType() != rhs.m_opaque_sp->GetType())
        return false;
    
    if (m_opaque_sp->GetType() == lldb_private::TypeSummaryImpl::eTypeCallback)
    {
        lldb_private::CXXFunctionSummaryFormat *self_cxx = (lldb_private::CXXFunctionSummaryFormat*)m_opaque_sp.get();
        lldb_private::CXXFunctionSummaryFormat *other_cxx = (lldb_private::CXXFunctionSummaryFormat*)rhs.m_opaque_sp.get();
        return (self_cxx->m_impl == other_cxx->m_impl);
    }
    
    if (m_opaque_sp->IsScripted() != rhs.m_opaque_sp->IsScripted())
        return false;
    
    if (IsFunctionCode() != rhs.IsFunctionCode())
        return false;

    if (IsSummaryString() != rhs.IsSummaryString())
        return false;

    if (IsFunctionName() != rhs.IsFunctionName())
        return false;
    
    if ( GetData() == NULL || rhs.GetData() == NULL || strcmp(GetData(), rhs.GetData()) )
        return false;
    
    return GetOptions() == rhs.GetOptions();
    
}

bool
SBTypeSummary::operator != (lldb::SBTypeSummary &rhs)
{
    if (IsValid() == false)
        return !rhs.IsValid();
    return m_opaque_sp != rhs.m_opaque_sp;
}

lldb::TypeSummaryImplSP
SBTypeSummary::GetSP ()
{
    return m_opaque_sp;
}

void
SBTypeSummary::SetSP (const lldb::TypeSummaryImplSP &typesummary_impl_sp)
{
    m_opaque_sp = typesummary_impl_sp;
}

SBTypeSummary::SBTypeSummary (const lldb::TypeSummaryImplSP &typesummary_impl_sp) :
m_opaque_sp(typesummary_impl_sp)
{
}

bool
SBTypeSummary::CopyOnWrite_Impl()
{
    if (!IsValid())
        return false;
    
    if (m_opaque_sp.unique())
        return true;
    
    TypeSummaryImplSP new_sp;
    
    if (m_opaque_sp->GetType() == lldb_private::TypeSummaryImpl::eTypeCallback)
    {
        CXXFunctionSummaryFormat* current_summary_ptr = (CXXFunctionSummaryFormat*)m_opaque_sp.get();
        new_sp = TypeSummaryImplSP(new CXXFunctionSummaryFormat(GetOptions(),
                                                                current_summary_ptr->m_impl,
                                                                current_summary_ptr->m_description.c_str()));
    }
    else if (m_opaque_sp->IsScripted())
    {
        ScriptSummaryFormat* current_summary_ptr = (ScriptSummaryFormat*)m_opaque_sp.get();
        new_sp = TypeSummaryImplSP(new ScriptSummaryFormat(GetOptions(),
                                                           current_summary_ptr->GetFunctionName(),
                                                           current_summary_ptr->GetPythonScript()));
    }
    else {
        StringSummaryFormat* current_summary_ptr = (StringSummaryFormat*)m_opaque_sp.get();
        new_sp = TypeSummaryImplSP(new StringSummaryFormat(GetOptions(),
                                                           current_summary_ptr->GetSummaryString()));
    }
    
    SetSP(new_sp);
    
    return true;
}

bool
SBTypeSummary::ChangeSummaryType (bool want_script)
{
    if (!IsValid())
        return false;
    
    TypeSummaryImplSP new_sp;
    
    if (want_script == m_opaque_sp->IsScripted())
    {
        if (m_opaque_sp->GetType() == lldb_private::TypeSummaryImpl::eTypeCallback && !want_script)
            new_sp = TypeSummaryImplSP(new StringSummaryFormat(GetOptions(), ""));
        else
            return CopyOnWrite_Impl();
    }
    
    if (!new_sp)
    {
        if (want_script)
            new_sp = TypeSummaryImplSP(new ScriptSummaryFormat(GetOptions(), "", ""));
        else
            new_sp = TypeSummaryImplSP(new StringSummaryFormat(GetOptions(), ""));
    }
    
    SetSP(new_sp);
    
    return true;
}

#endif // LLDB_DISABLE_PYTHON
