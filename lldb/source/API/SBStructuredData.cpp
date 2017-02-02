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
#include "lldb/Core/Event.h"
#include "lldb/Core/StructuredData.h"
#include "lldb/Target/StructuredDataPlugin.h"
#include "lldb/Utility/Error.h"
#include "lldb/Utility/Stream.h"

using namespace lldb;
using namespace lldb_private;

#pragma mark--
#pragma mark StructuredDataImpl

class StructuredDataImpl {
public:
  StructuredDataImpl() : m_plugin_wp(), m_data_sp() {}

  StructuredDataImpl(const StructuredDataImpl &rhs) = default;

  StructuredDataImpl(const EventSP &event_sp)
      : m_plugin_wp(
            EventDataStructuredData::GetPluginFromEvent(event_sp.get())),
        m_data_sp(EventDataStructuredData::GetObjectFromEvent(event_sp.get())) {
  }

  ~StructuredDataImpl() = default;

  StructuredDataImpl &operator=(const StructuredDataImpl &rhs) = default;

  bool IsValid() const { return m_data_sp.get() != nullptr; }

  void Clear() {
    m_plugin_wp.reset();
    m_data_sp.reset();
  }

  SBError GetAsJSON(lldb_private::Stream &stream) const {
    SBError sb_error;

    if (!m_data_sp) {
      sb_error.SetErrorString("No structured data.");
      return sb_error;
    }

    m_data_sp->Dump(stream);
    return sb_error;
  }

  Error GetDescription(lldb_private::Stream &stream) const {
    Error error;

    if (!m_data_sp) {
      error.SetErrorString("Cannot pretty print structured data: "
                           "no data to print.");
      return error;
    }

    // Grab the plugin.
    auto plugin_sp = StructuredDataPluginSP(m_plugin_wp);
    if (!plugin_sp) {
      error.SetErrorString("Cannot pretty print structured data: "
                           "plugin doesn't exist.");
      return error;
    }

    // Get the data's description.
    return plugin_sp->GetDescription(m_data_sp, stream);
  }

private:
  StructuredDataPluginWP m_plugin_wp;
  StructuredData::ObjectSP m_data_sp;
};

#pragma mark--
#pragma mark SBStructuredData

SBStructuredData::SBStructuredData() : m_impl_up(new StructuredDataImpl()) {}

SBStructuredData::SBStructuredData(const lldb::SBStructuredData &rhs)
    : m_impl_up(new StructuredDataImpl(*rhs.m_impl_up.get())) {}

SBStructuredData::SBStructuredData(const lldb::EventSP &event_sp)
    : m_impl_up(new StructuredDataImpl(event_sp)) {}

SBStructuredData::~SBStructuredData() {}

SBStructuredData &SBStructuredData::
operator=(const lldb::SBStructuredData &rhs) {
  *m_impl_up = *rhs.m_impl_up;
  return *this;
}

bool SBStructuredData::IsValid() const { return m_impl_up->IsValid(); }

void SBStructuredData::Clear() { m_impl_up->Clear(); }

SBError SBStructuredData::GetAsJSON(lldb::SBStream &stream) const {
  return m_impl_up->GetAsJSON(stream.ref());
}

lldb::SBError SBStructuredData::GetDescription(lldb::SBStream &stream) const {
  Error error = m_impl_up->GetDescription(stream.ref());
  SBError sb_error;
  sb_error.SetError(error);
  return sb_error;
}
