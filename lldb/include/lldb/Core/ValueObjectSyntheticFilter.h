//===-- ValueObjectSyntheticFilter.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectSyntheticFilter_h_
#define liblldb_ValueObjectSyntheticFilter_h_

// C Includes
// C++ Includes
#include <memory>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/ThreadSafeSTLMap.h"
#include "lldb/Core/ThreadSafeSTLVector.h"
#include "lldb/Core/ValueObject.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A ValueObject that obtains its children from some source other than
// real information
// This is currently used to implement Python-based children and filters
// but you can bind it to any source of synthetic information and have
// it behave accordingly
//----------------------------------------------------------------------
class ValueObjectSynthetic : public ValueObject {
public:
  ~ValueObjectSynthetic() override;

  uint64_t GetByteSize() override;

  ConstString GetTypeName() override;

  ConstString GetQualifiedTypeName() override;

  ConstString GetDisplayTypeName() override;

  bool MightHaveChildren() override;

  size_t CalculateNumChildren(uint32_t max) override;

  lldb::ValueType GetValueType() const override;

  lldb::ValueObjectSP GetChildAtIndex(size_t idx, bool can_create) override;

  lldb::ValueObjectSP GetChildMemberWithName(const ConstString &name,
                                             bool can_create) override;

  size_t GetIndexOfChildWithName(const ConstString &name) override;

  lldb::ValueObjectSP
  GetDynamicValue(lldb::DynamicValueType valueType) override;

  bool IsInScope() override;

  bool HasSyntheticValue() override { return false; }

  bool IsSynthetic() override { return true; }

  bool IsBaseClass() override {
    if (m_parent)
      return m_parent->IsBaseClass();
    return false;
  }

  void CalculateSyntheticValue(bool use_synthetic) override {}

  bool IsDynamic() override {
    return ((m_parent != nullptr) ? m_parent->IsDynamic() : false);
  }

  lldb::ValueObjectSP GetStaticValue() override {
    return ((m_parent != nullptr) ? m_parent->GetStaticValue() : GetSP());
  }

  virtual lldb::DynamicValueType GetDynamicValueType() {
    return ((m_parent != nullptr) ? m_parent->GetDynamicValueType()
                                  : lldb::eNoDynamicValues);
  }

  ValueObject *GetParent() override {
    return ((m_parent != nullptr) ? m_parent->GetParent() : nullptr);
  }

  const ValueObject *GetParent() const override {
    return ((m_parent != nullptr) ? m_parent->GetParent() : nullptr);
  }

  lldb::ValueObjectSP GetNonSyntheticValue() override;

  bool CanProvideValue() override;

  bool DoesProvideSyntheticValue() override {
    return (UpdateValueIfNeeded(), m_provides_value == eLazyBoolYes);
  }

  lldb::ValueObjectSP
  GetSyntheticChildAtOffset(uint32_t offset, const CompilerType &type,
                            bool can_create,
                            ConstString name = ConstString()) override {
    if (m_parent)
      return m_parent->GetSyntheticChildAtOffset(offset, type, can_create,
                                                 name);
    return nullptr;
  }

  bool GetIsConstant() const override { return false; }

  bool SetValueFromCString(const char *value_str, Error &error) override;

  void SetFormat(lldb::Format format) override;

  lldb::LanguageType GetPreferredDisplayLanguage() override;

  void SetPreferredDisplayLanguage(lldb::LanguageType);

  bool IsSyntheticChildrenGenerated() override;

  void SetSyntheticChildrenGenerated(bool b) override;

  bool GetDeclaration(Declaration &decl) override;

  uint64_t GetLanguageFlags() override;

  void SetLanguageFlags(uint64_t flags) override;

protected:
  bool UpdateValue() override;

  LazyBool CanUpdateWithInvalidExecutionContext() override {
    return eLazyBoolYes;
  }

  CompilerType GetCompilerTypeImpl() override;

  virtual void CreateSynthFilter();

  // we need to hold on to the SyntheticChildren because someone might delete
  // the type binding while we are alive
  lldb::SyntheticChildrenSP m_synth_sp;
  std::unique_ptr<SyntheticChildrenFrontEnd> m_synth_filter_ap;

  typedef ThreadSafeSTLMap<uint32_t, ValueObject *> ByIndexMap;
  typedef ThreadSafeSTLMap<const char *, uint32_t> NameToIndexMap;
  typedef ThreadSafeSTLVector<lldb::ValueObjectSP> SyntheticChildrenCache;

  typedef ByIndexMap::iterator ByIndexIterator;
  typedef NameToIndexMap::iterator NameToIndexIterator;

  ByIndexMap m_children_byindex;
  NameToIndexMap m_name_toindex;
  uint32_t m_synthetic_children_count; // FIXME use the ValueObject's
                                       // ChildrenManager instead of a special
                                       // purpose solution
  SyntheticChildrenCache m_synthetic_children_cache;

  ConstString m_parent_type_name;

  LazyBool m_might_have_children;

  LazyBool m_provides_value;

private:
  friend class ValueObject;
  ValueObjectSynthetic(ValueObject &parent, lldb::SyntheticChildrenSP filter);

  void CopyValueData(ValueObject *source);

  DISALLOW_COPY_AND_ASSIGN(ValueObjectSynthetic);
};

} // namespace lldb_private

#endif // liblldb_ValueObjectSyntheticFilter_h_
