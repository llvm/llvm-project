//===-- SBTypeFormat.cpp ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/API/SBTypeFormat.h"

#include "lldb/API/SBStream.h"

#include "lldb/DataFormatters/DataVisualization.h"

using namespace lldb;
using namespace lldb_private;

SBTypeFormat::SBTypeFormat() :
m_opaque_sp()
{
}

SBTypeFormat::SBTypeFormat (lldb::Format format,
                            uint32_t options)
: m_opaque_sp(TypeFormatImplSP(new TypeFormatImpl(format,options)))
{
}

SBTypeFormat::SBTypeFormat (const lldb::SBTypeFormat &rhs) :
m_opaque_sp(rhs.m_opaque_sp)
{
}

SBTypeFormat::~SBTypeFormat ()
{
}

bool
SBTypeFormat::IsValid() const
{
    return m_opaque_sp.get() != NULL;
}

lldb::Format
SBTypeFormat::GetFormat ()
{
    if (IsValid())
        return m_opaque_sp->GetFormat();
    return lldb::eFormatInvalid;
}

uint32_t
SBTypeFormat::GetOptions()
{
    if (IsValid())
        return m_opaque_sp->GetOptions();
    return 0;
}

void
SBTypeFormat::SetFormat (lldb::Format fmt)
{
    if (CopyOnWrite_Impl())
        m_opaque_sp->SetFormat(fmt);
}

void
SBTypeFormat::SetOptions (uint32_t value)
{
    if (CopyOnWrite_Impl())
        m_opaque_sp->SetOptions(value);
}

bool
SBTypeFormat::GetDescription (lldb::SBStream &description, 
                              lldb::DescriptionLevel description_level)
{
    if (!IsValid())
        return false;
    else {
        description.Printf("%s\n",
                           m_opaque_sp->GetDescription().c_str());
        return true;
    }
}

lldb::SBTypeFormat &
SBTypeFormat::operator = (const lldb::SBTypeFormat &rhs)
{
    if (this != &rhs)
    {
        m_opaque_sp = rhs.m_opaque_sp;
    }
    return *this;
}

bool
SBTypeFormat::operator == (lldb::SBTypeFormat &rhs)
{
    if (IsValid() == false)
        return !rhs.IsValid();
    return m_opaque_sp == rhs.m_opaque_sp;
}

bool
SBTypeFormat::IsEqualTo (lldb::SBTypeFormat &rhs)
{
    if (IsValid() == false)
        return !rhs.IsValid();
    
    if (GetFormat() == rhs.GetFormat())
        return GetOptions() == rhs.GetOptions();
    else
        return false;
}

bool
SBTypeFormat::operator != (lldb::SBTypeFormat &rhs)
{
    if (IsValid() == false)
        return !rhs.IsValid();
    return m_opaque_sp != rhs.m_opaque_sp;
}

lldb::TypeFormatImplSP
SBTypeFormat::GetSP ()
{
    return m_opaque_sp;
}

void
SBTypeFormat::SetSP (const lldb::TypeFormatImplSP &typeformat_impl_sp)
{
    m_opaque_sp = typeformat_impl_sp;
}

SBTypeFormat::SBTypeFormat (const lldb::TypeFormatImplSP &typeformat_impl_sp) :
    m_opaque_sp(typeformat_impl_sp)
{
}

bool
SBTypeFormat::CopyOnWrite_Impl()
{
    if (!IsValid())
        return false;
    if (m_opaque_sp.unique())
        return true;

    SetSP(TypeFormatImplSP(new TypeFormatImpl(GetFormat(),GetOptions())));
    return true;
}
