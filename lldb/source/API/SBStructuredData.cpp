//===-- SBStructuredData.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBStructuredData.h"

#include "lldb/API/SBStream.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Event.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/StructuredData.h"
#include "lldb/Target/StructuredDataPlugin.h"

using namespace lldb;
using namespace lldb_private;

#pragma mark --
#pragma mark Impl

class SBStructuredData::Impl
{
public:

    Impl() :
        m_plugin_wp(),
        m_data_sp()
    {
    }

    Impl(const Impl &rhs) = default;

    Impl(const EventSP &event_sp) :
       m_plugin_wp(EventDataStructuredData::GetPluginFromEvent(event_sp.get())),
        m_data_sp(EventDataStructuredData::GetObjectFromEvent(event_sp.get()))
    {
    }

    ~Impl() = default;

    Impl&
    operator =(const Impl &rhs) = default;

    bool
    IsValid() const
    {
        return m_data_sp.get() != nullptr;
    }

    void
    Clear()
    {
        m_plugin_wp.reset();
        m_data_sp.reset();
    }

    SBError
    GetAsJSON(lldb::SBStream &stream) const
    {
        SBError sb_error;

        if (!m_data_sp)
        {
            sb_error.SetErrorString("No structured data.");
            return sb_error;
        }

        m_data_sp->Dump(stream.ref());
        return sb_error;
    }

    lldb::SBError
    GetDescription(lldb::SBStream &stream) const
    {
        SBError sb_error;

        if (!m_data_sp)
        {
            sb_error.SetErrorString("Cannot pretty print structured data: "
                                    "no data to print.");
            return sb_error;
        }

        // Grab the plugin.
        auto plugin_sp = StructuredDataPluginSP(m_plugin_wp);
        if (!plugin_sp)
        {
            sb_error.SetErrorString("Cannot pretty print structured data: "
                                    "plugin doesn't exist.");
            return sb_error;
        }

        // Get the data's description.
        auto error = plugin_sp->GetDescription(m_data_sp, stream.ref());
        if (!error.Success())
            sb_error.SetError(error);

        return sb_error;
    }

private:

    StructuredDataPluginWP    m_plugin_wp;
    StructuredData::ObjectSP  m_data_sp;

};

#pragma mark --
#pragma mark SBStructuredData


SBStructuredData::SBStructuredData() :
    m_impl_up(new Impl())
{
}

SBStructuredData::SBStructuredData(const lldb::SBStructuredData &rhs) :
    m_impl_up(new Impl(*rhs.m_impl_up.get()))
{
}

SBStructuredData::SBStructuredData(const lldb::EventSP &event_sp) :
    m_impl_up(new Impl(event_sp))
{
}

SBStructuredData::~SBStructuredData()
{
}

SBStructuredData &
SBStructuredData::operator =(const lldb::SBStructuredData &rhs)
{
    *m_impl_up = *rhs.m_impl_up;
    return *this;
}

bool
SBStructuredData::IsValid() const
{
    return m_impl_up->IsValid();
}

void
SBStructuredData::Clear()
{
    m_impl_up->Clear();
}

SBError
SBStructuredData::GetAsJSON(lldb::SBStream &stream) const
{
    return m_impl_up->GetAsJSON(stream);
}

lldb::SBError
SBStructuredData::GetDescription(lldb::SBStream &stream) const
{
    return m_impl_up->GetDescription(stream);
}

