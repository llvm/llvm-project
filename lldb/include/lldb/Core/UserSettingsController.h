//====-- UserSettingsController.h --------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_USERSETTINGSCONTROLLER_H
#define LLDB_CORE_USERSETTINGSCONTROLLER_H

#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private-enumerations.h"

#include "llvm/ADT/StringRef.h"

#include <vector>

#include <cstddef>
#include <cstdint>

namespace lldb_private {
class CommandInterpreter;
class ConstString;
class ExecutionContext;
class Property;
class Stream;
}

namespace lldb_private {

class Properties {
public:
  Properties() = default;

  Properties(const lldb::OptionValuePropertiesSP &collection_sp)
      : m_collection_sp(collection_sp) {}

  virtual ~Properties() = default;

  virtual lldb::OptionValuePropertiesSP GetValueProperties() const {
    // This function is virtual in case subclasses want to lazily implement
    // creating the properties.
    return m_collection_sp;
  }

  virtual lldb::OptionValueSP GetPropertyValue(const ExecutionContext *exe_ctx,
                                               llvm::StringRef property_path,
                                               bool will_modify,
                                               Status &error) const;

  virtual Status SetPropertyValue(const ExecutionContext *exe_ctx,
                                  VarSetOperationType op,
                                  llvm::StringRef property_path,
                                  llvm::StringRef value);

  virtual Status DumpPropertyValue(const ExecutionContext *exe_ctx,
                                   Stream &strm, llvm::StringRef property_path,
                                   uint32_t dump_mask);

  virtual void DumpAllPropertyValues(const ExecutionContext *exe_ctx,
                                     Stream &strm, uint32_t dump_mask);

  virtual void DumpAllDescriptions(CommandInterpreter &interpreter,
                                   Stream &strm) const;

  size_t Apropos(llvm::StringRef keyword,
                 std::vector<const Property *> &matching_properties) const;

  lldb::OptionValuePropertiesSP GetSubProperty(const ExecutionContext *exe_ctx,
                                               ConstString name);

  // We sometimes need to introduce a setting to enable experimental features,
  // but then we don't want the setting for these to cause errors when the
  // setting goes away.  Add a sub-topic of the settings using this
  // experimental name, and two things will happen.  One is that settings that
  // don't find the name will not be treated as errors.  Also, if you decide to
  // keep the settings just move them into the containing properties, and we
  // will auto-forward the experimental settings to the real one.
  static const char *GetExperimentalSettingsName();

  static bool IsSettingExperimental(llvm::StringRef setting);

protected:
  lldb::OptionValuePropertiesSP m_collection_sp;
};

} // namespace lldb_private

#endif // LLDB_CORE_USERSETTINGSCONTROLLER_H
