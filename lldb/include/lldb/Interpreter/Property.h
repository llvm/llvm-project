//===-- Property.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_PROPERTY_H
#define LLDB_INTERPRETER_PROPERTY_H

#include "lldb/Interpreter/OptionValue.h"
#include "lldb/Utility/Flags.h"
#include "lldb/Utility/Stream.h"
#include "lldb/lldb-defines.h"
#include "lldb/lldb-private-types.h"

#include <optional>
#include <string>

namespace lldb_private {

// A structure that can be used to create a global table for all properties.
// Property class instances can be constructed using one of these.
struct PropertyDefinition {
  const char *name;
  OptionValue::Type type;
  bool global; // false == this setting is a global setting by default
  uintptr_t default_uint_value;
  const char *default_cstr_value;
  OptionEnumValues enum_values;
  const char *description;
  // Deprecated settings can still be set and shown by name, but are mostly
  // hidden otherwise. See the Deprecated class in PropertiesBase.td.
  bool deprecated;
  // What users of a deprecated setting should do instead, or nullptr.
  const char *deprecation_message;
};

struct PropertyCollectionDefinition {
  llvm::ArrayRef<PropertyDefinition> definitions;
  llvm::StringRef expected_path;
};

class Property {
public:
  Property(const PropertyDefinition &definition);

  Property(llvm::StringRef name, llvm::StringRef desc, bool is_global,
           const lldb::OptionValueSP &value_sp);

  llvm::StringRef GetName() const { return m_name; }
  llvm::StringRef GetDescription() const { return m_description; }

  const lldb::OptionValueSP &GetValue() const { return m_value_sp; }

  void SetOptionValue(const lldb::OptionValueSP &value_sp) {
    m_value_sp = value_sp;
  }

  bool IsValid() const { return (bool)m_value_sp; }

  bool IsGlobal() const { return m_is_global; }

  bool IsDeprecated() const { return m_is_deprecated; }

  /// What users of this deprecated property should do instead.
  llvm::StringRef GetDeprecationMessage() const {
    return m_deprecation_message;
  }

  /// Whether listings should leave this property out. Deprecated properties
  /// are only listed when \a include_deprecated is set.
  bool IsHidden(bool include_deprecated) const {
    return m_is_deprecated && !include_deprecated;
  }

  /// If this property is deprecated, and output produced with \a dump_mask can
  /// be tagged, write the deprecation tag that follows the property's name.
  void DumpDeprecatedTag(Stream &strm, uint32_t dump_mask) const;

  /// The description, starting with a deprecation tag if this property is
  /// deprecated.
  std::string GetTaggedDescription() const;

  void Dump(const ExecutionContext *exe_ctx, Stream &strm,
            uint32_t dump_mask) const;

  bool DumpQualifiedName(
      Stream &strm,
      std::optional<Stream::HighlightSettings> highlight = std::nullopt) const;

  void DumpDescription(
      CommandInterpreter &interpreter, Stream &strm, uint32_t output_width,
      bool display_qualified_name, bool include_deprecated,
      std::optional<Stream::HighlightSettings> highlight = std::nullopt) const;

  void SetValueChangedCallback(std::function<void()> callback);

protected:
  std::string m_name;
  std::string m_description;
  lldb::OptionValueSP m_value_sp;
  bool m_is_global;
  bool m_is_deprecated = false;
  /// Points into the static PropertyDefinition table, or is nullptr.
  const char *m_deprecation_message = nullptr;
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_PROPERTY_H
