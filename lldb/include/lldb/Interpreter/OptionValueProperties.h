//===-- OptionValueProperties.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_INTERPRETER_OPTIONVALUEPROPERTIES_H
#define LLDB_INTERPRETER_OPTIONVALUEPROPERTIES_H

#include <vector>

#include "lldb/Core/FormatEntity.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Interpreter/OptionValue.h"
#include "lldb/Interpreter/Property.h"
#include "lldb/Utility/ConstString.h"

namespace lldb_private {
class Properties;

class OptionValueProperties
    : public Cloneable<OptionValueProperties, OptionValue>,
      public std::enable_shared_from_this<OptionValueProperties> {
public:
  OptionValueProperties() = default;

  OptionValueProperties(ConstString name);

  ~OptionValueProperties() override = default;

  Type GetType() const override { return eTypeProperties; }

  void Clear() override;

  static lldb::OptionValuePropertiesSP
  CreateLocalCopy(const Properties &global_properties);

  lldb::OptionValueSP
  DeepCopy(const lldb::OptionValueSP &new_parent) const override;

  Status
  SetValueFromString(llvm::StringRef value,
                     VarSetOperationType op = eVarSetOperationAssign) override;

  void DumpValue(const ExecutionContext *exe_ctx, Stream &strm,
                 uint32_t dump_mask) override;

  llvm::json::Value ToJSON(const ExecutionContext *exe_ctx) override;

  ConstString GetName() const override { return m_name; }

  virtual Status DumpPropertyValue(const ExecutionContext *exe_ctx,
                                   Stream &strm, llvm::StringRef property_path,
                                   uint32_t dump_mask, bool is_json = false);

  virtual void DumpAllDescriptions(CommandInterpreter &interpreter,
                                   Stream &strm) const;

  void Apropos(llvm::StringRef keyword,
               std::vector<const Property *> &matching_properties) const;

  void Initialize(const PropertyDefinitions &setting_definitions);

  //    bool
  //    GetQualifiedName (Stream &strm);

  // Subclass specific functions

  // Get the index of a property given its exact name in this property
  // collection, "name" can't be a path to a property path that refers to a
  // property within a property
  virtual uint32_t GetPropertyIndex(ConstString name) const;

  // Get a property by exact name exists in this property collection, name can
  // not be a path to a property path that refers to a property within a
  // property
  virtual const Property *
  GetProperty(ConstString name,
              const ExecutionContext *exe_ctx = nullptr) const;

  virtual const Property *
  GetPropertyAtIndex(uint32_t idx,
                     const ExecutionContext *exe_ctx = nullptr) const {
    return ProtectedGetPropertyAtIndex(idx);
  }

  // Property can be be a property path like
  // "target.process.extra-startup-command"
  virtual const Property *
  GetPropertyAtPath(const ExecutionContext *exe_ctx,

                    llvm::StringRef property_path) const;

  virtual lldb::OptionValueSP
  GetPropertyValueAtIndex(uint32_t idx, const ExecutionContext *exe_ctx) const;

  virtual lldb::OptionValueSP GetValueForKey(const ExecutionContext *exe_ctx,
                                             ConstString key) const;

  lldb::OptionValueSP GetSubValue(const ExecutionContext *exe_ctx,
                                  llvm::StringRef name,
                                  Status &error) const override;

  Status SetSubValue(const ExecutionContext *exe_ctx, VarSetOperationType op,
                     llvm::StringRef path, llvm::StringRef value) override;

  bool
  GetPropertyAtIndexAsArgs(uint32_t idx, Args &args,
                           const ExecutionContext *exe_ctx = nullptr) const;

  bool SetPropertyAtIndexFromArgs(uint32_t idx, const Args &args,
                                  const ExecutionContext *exe_ctx = nullptr);

  OptionValueDictionary *GetPropertyAtIndexAsOptionValueDictionary(
      uint32_t idx, const ExecutionContext *exe_ctx = nullptr) const;

  OptionValueSInt64 *GetPropertyAtIndexAsOptionValueSInt64(
      uint32_t idx, const ExecutionContext *exe_ctx = nullptr) const;

  OptionValueUInt64 *GetPropertyAtIndexAsOptionValueUInt64(
      uint32_t idx, const ExecutionContext *exe_ctx = nullptr) const;

  OptionValueString *GetPropertyAtIndexAsOptionValueString(
      uint32_t idx, const ExecutionContext *exe_ctx = nullptr) const;

  OptionValueFileSpec *GetPropertyAtIndexAsOptionValueFileSpec(
      uint32_t idx, const ExecutionContext *exe_ctx = nullptr) const;

  OptionValuePathMappings *GetPropertyAtIndexAsOptionValuePathMappings(
      uint32_t idx, const ExecutionContext *exe_ctx = nullptr) const;

  OptionValueFileSpecList *GetPropertyAtIndexAsOptionValueFileSpecList(
      uint32_t idx, const ExecutionContext *exe_ctx = nullptr) const;

  void AppendProperty(ConstString name, llvm::StringRef desc, bool is_global,
                      const lldb::OptionValueSP &value_sp);

  lldb::OptionValuePropertiesSP GetSubProperty(const ExecutionContext *exe_ctx,
                                               ConstString name);

  void SetValueChangedCallback(uint32_t property_idx,
                               std::function<void()> callback);

  template <typename T>
  auto GetPropertyAtIndexAs(uint32_t idx,
                            const ExecutionContext *exe_ctx = nullptr) const {
    if (const Property *property = GetPropertyAtIndex(idx, exe_ctx)) {
      if (OptionValue *value = property->GetValue().get())
        return value->GetValueAs<T>();
    }
    if constexpr (std::is_pointer_v<T>)
      return T{nullptr};
    else
      return std::optional<T>{std::nullopt};
  }

  template <typename T>
  bool SetPropertyAtIndex(uint32_t idx, T t,
                          const ExecutionContext *exe_ctx = nullptr) const {
    if (const Property *property = GetPropertyAtIndex(idx, exe_ctx)) {
      if (OptionValue *value = property->GetValue().get()) {
        value->SetValueAs(t);
        return true;
      }
    }
    return false;
  }

protected:
  Property *ProtectedGetPropertyAtIndex(uint32_t idx) {
    assert(idx < m_properties.size() && "invalid property index");
    return ((idx < m_properties.size()) ? &m_properties[idx] : nullptr);
  }

  const Property *ProtectedGetPropertyAtIndex(uint32_t idx) const {
    assert(idx < m_properties.size() && "invalid property index");
    return ((idx < m_properties.size()) ? &m_properties[idx] : nullptr);
  }

  typedef UniqueCStringMap<size_t> NameToIndex;

  ConstString m_name;
  std::vector<Property> m_properties;
  NameToIndex m_name_to_index;
};

} // namespace lldb_private

#endif // LLDB_INTERPRETER_OPTIONVALUEPROPERTIES_H
