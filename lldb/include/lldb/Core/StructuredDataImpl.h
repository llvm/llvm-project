//===-- StructuredDataImpl.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StructuredDataImpl_h_
#define liblldb_StructuredDataImpl_h_

#include "lldb/Core/Event.h"
#include "lldb/Core/StructuredData.h"
#include "lldb/Utility/Error.h"
#include "lldb/Utility/Stream.h"
#include "lldb/Target/StructuredDataPlugin.h"
#include "lldb/lldb-forward.h"

#pragma mark--
#pragma mark StructuredDataImpl

namespace lldb_private {

class StructuredDataImpl {
public:
  StructuredDataImpl() : m_plugin_wp(), m_data_sp() {}

  StructuredDataImpl(const StructuredDataImpl &rhs) = default;

  StructuredDataImpl(const lldb::EventSP &event_sp)
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

  Error GetAsJSON(Stream &stream) const {
    Error error;

    if (!m_data_sp) {
      error.SetErrorString("No structured data.");
      return error;
    }

    m_data_sp->Dump(stream);
    return error;
  }

  Error GetDescription(Stream &stream) const {
    Error error;

    if (!m_data_sp) {
      error.SetErrorString("Cannot pretty print structured data: "
                           "no data to print.");
      return error;
    }

    // Grab the plugin.
    auto plugin_sp = lldb::StructuredDataPluginSP(m_plugin_wp);
    if (!plugin_sp) {
      error.SetErrorString("Cannot pretty print structured data: "
                           "plugin doesn't exist.");
      return error;
    }

    // Get the data's description.
    return plugin_sp->GetDescription(m_data_sp, stream);
  }

  StructuredData::ObjectSP GetObjectSP() {
    return m_data_sp;
  }

  void SetObjectSP(const StructuredData::ObjectSP &obj) {
    m_data_sp = obj;
  }

private:

  lldb::StructuredDataPluginWP m_plugin_wp;
  StructuredData::ObjectSP m_data_sp;
};
}
#endif
