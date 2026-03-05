//===-- UserSettingsController.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/UserSettingsController.h"

#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"

#include <memory>

namespace lldb_private {
class CommandInterpreter;
}
namespace lldb_private {
class ConstString;
}
namespace lldb_private {
class ExecutionContext;
}
namespace lldb_private {
class Property;
}

using namespace lldb;
using namespace lldb_private;

Properties::Properties() = default;

Properties::Properties(const lldb::OptionValuePropertiesSP &collection_sp)
    : m_collection_sp(collection_sp) {}

Properties::~Properties() = default;

lldb::OptionValueSP
Properties::GetPropertyValue(const ExecutionContext *exe_ctx,
                             llvm::StringRef path, Status &error) const {
  return m_collection_sp->GetSubValue(exe_ctx, path, error);
}

Status Properties::SetPropertyValue(const ExecutionContext *exe_ctx,
                                    VarSetOperationType op,
                                    llvm::StringRef path,
                                    llvm::StringRef value) {
  return m_collection_sp->SetSubValue(exe_ctx, op, path, value);
}

void Properties::DumpAllPropertyValues(const ExecutionContext *exe_ctx,
                                       Stream &strm, uint32_t dump_mask,
                                       bool is_json) {
  if (is_json) {
    llvm::json::Value json = m_collection_sp->ToJSON(exe_ctx);
    strm.Printf("%s", llvm::formatv("{0:2}", json).str().c_str());
  } else
    m_collection_sp->DumpValue(exe_ctx, strm, dump_mask);
}

void Properties::DumpAllDescriptions(CommandInterpreter &interpreter,
                                     Stream &strm) const {
  strm.PutCString("Top level variables:\n\n");

  return m_collection_sp->DumpAllDescriptions(interpreter, strm);
}

Status Properties::DumpPropertyValue(const ExecutionContext *exe_ctx,
                                     Stream &strm,
                                     llvm::StringRef property_path,
                                     uint32_t dump_mask, bool is_json) {
  return m_collection_sp->DumpPropertyValue(exe_ctx, strm, property_path,
                                            dump_mask, is_json);
}

size_t
Properties::Apropos(llvm::StringRef keyword,
                    std::vector<const Property *> &matching_properties) const {
  m_collection_sp->Apropos(keyword, matching_properties);
  return matching_properties.size();
}

llvm::StringRef Properties::GetExperimentalSettingsName() {
  static constexpr llvm::StringLiteral g_experimental("experimental");
  return g_experimental;
}

bool Properties::IsSettingExperimental(llvm::StringRef setting) {
  if (setting.empty())
    return false;

  llvm::StringRef experimental = GetExperimentalSettingsName();
  size_t dot_pos = setting.find_first_of('.');
  return setting.take_front(dot_pos) == experimental;
}

void Properties::SetPropertiesAtPathIfNotExists(
    llvm::StringRef path, const lldb::OptionValuePropertiesSP &properties_sp,
    llvm::StringRef description, bool is_global_property) {
  assert(!path.empty());
  assert(m_collection_sp != nullptr);

  llvm::SmallVector<llvm::StringRef, 3> segments;
  path.split(segments, '.');
  llvm::StringRef last_segment = segments.pop_back_val();

  OptionValuePropertiesSP collection_sp = m_collection_sp;
  for (llvm::StringRef segment : segments) {
    const Property *inner = collection_sp->GetProperty(segment);
    if (!inner) {
      auto inner_sp = std::make_shared<OptionValueProperties>(segment);
      // `segment` is a substring of `path`, so `segment.end()` includes
      // everything up until and including the current segment.
      inner_sp->SetExpectedPath(std::string(path.begin(), segment.end()));

      collection_sp->AppendProperty(segment, ("Settings for " + segment).str(),
                                    /*is_global=*/true, inner_sp);
      collection_sp = inner_sp;
      continue;
    }

    OptionValueProperties *inner_properties =
        inner->GetValue()->GetAsProperties();
    if (!inner_properties) {
      assert(false && "Intermediate properties must be OptionValueProperties");
      return;
    }
    collection_sp = inner_properties->shared_from_this();
  }

  const Property *last_property = collection_sp->GetProperty(last_segment);
  if (last_property)
    return; // already exists

  collection_sp->AppendProperty(last_segment, description, is_global_property,
                                properties_sp);
}
