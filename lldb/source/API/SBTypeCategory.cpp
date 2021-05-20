//===-- SBTypeCategory.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBTypeCategory.h"
#include "SBReproducerPrivate.h"

#include "lldb/API/SBStream.h"
#include "lldb/API/SBTypeFilter.h"
#include "lldb/API/SBTypeFormat.h"
#include "lldb/API/SBTypeNameSpecifier.h"
#include "lldb/API/SBTypeSummary.h"
#include "lldb/API/SBTypeSynthetic.h"

#include "lldb/Core/Debugger.h"
#include "lldb/DataFormatters/DataVisualization.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/ScriptInterpreter.h"

using namespace lldb;
using namespace lldb_private;

typedef std::pair<lldb::TypeCategoryImplSP, user_id_t> ImplType;

SBTypeCategory::SBTypeCategory() : m_opaque_sp() {
  LLDB_RECORD_CONSTRUCTOR_NO_ARGS(SBTypeCategory);
}

SBTypeCategory::SBTypeCategory(const char *name) : m_opaque_sp() {
  DataVisualization::Categories::GetCategory(ConstString(name), m_opaque_sp);
}

SBTypeCategory::SBTypeCategory(const lldb::SBTypeCategory &rhs)
    : m_opaque_sp(rhs.m_opaque_sp) {
  LLDB_RECORD_CONSTRUCTOR(SBTypeCategory, (const lldb::SBTypeCategory &), rhs);
}

SBTypeCategory::~SBTypeCategory() = default;

bool SBTypeCategory::IsValid() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeCategory, IsValid);
  return this->operator bool();
}
SBTypeCategory::operator bool() const {
  LLDB_RECORD_METHOD_CONST_NO_ARGS(bool, SBTypeCategory, operator bool);

  return (m_opaque_sp.get() != nullptr);
}

bool SBTypeCategory::GetEnabled() {
  LLDB_RECORD_METHOD_NO_ARGS(bool, SBTypeCategory, GetEnabled);

  if (!IsValid())
    return false;
  return m_opaque_sp->IsEnabled();
}

void SBTypeCategory::SetEnabled(bool enabled) {
  LLDB_RECORD_METHOD(void, SBTypeCategory, SetEnabled, (bool), enabled);

  if (!IsValid())
    return;
  if (enabled)
    DataVisualization::Categories::Enable(m_opaque_sp);
  else
    DataVisualization::Categories::Disable(m_opaque_sp);
}

const char *SBTypeCategory::GetName() {
  LLDB_RECORD_METHOD_NO_ARGS(const char *, SBTypeCategory, GetName);

  if (!IsValid())
    return nullptr;
  return m_opaque_sp->GetName();
}

lldb::LanguageType SBTypeCategory::GetLanguageAtIndex(uint32_t idx) {
  LLDB_RECORD_METHOD(lldb::LanguageType, SBTypeCategory, GetLanguageAtIndex,
                     (uint32_t), idx);

  if (IsValid())
    return m_opaque_sp->GetLanguageAtIndex(idx);
  return lldb::eLanguageTypeUnknown;
}

uint32_t SBTypeCategory::GetNumLanguages() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeCategory, GetNumLanguages);

  if (IsValid())
    return m_opaque_sp->GetNumLanguages();
  return 0;
}

void SBTypeCategory::AddLanguage(lldb::LanguageType language) {
  LLDB_RECORD_METHOD(void, SBTypeCategory, AddLanguage, (lldb::LanguageType),
                     language);

  if (IsValid())
    m_opaque_sp->AddLanguage(language);
}

uint32_t SBTypeCategory::GetNumFormats() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeCategory, GetNumFormats);

  if (!IsValid())
    return 0;

  return m_opaque_sp->GetTypeFormatsContainer()->GetCount() +
         m_opaque_sp->GetRegexTypeFormatsContainer()->GetCount();
}

uint32_t SBTypeCategory::GetNumSummaries() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeCategory, GetNumSummaries);

  if (!IsValid())
    return 0;
  return m_opaque_sp->GetTypeSummariesContainer()->GetCount() +
         m_opaque_sp->GetRegexTypeSummariesContainer()->GetCount();
}

uint32_t SBTypeCategory::GetNumFilters() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeCategory, GetNumFilters);

  if (!IsValid())
    return 0;
  return m_opaque_sp->GetTypeFiltersContainer()->GetCount() +
         m_opaque_sp->GetRegexTypeFiltersContainer()->GetCount();
}

uint32_t SBTypeCategory::GetNumSynthetics() {
  LLDB_RECORD_METHOD_NO_ARGS(uint32_t, SBTypeCategory, GetNumSynthetics);

  if (!IsValid())
    return 0;
  return m_opaque_sp->GetTypeSyntheticsContainer()->GetCount() +
         m_opaque_sp->GetRegexTypeSyntheticsContainer()->GetCount();
}

lldb::SBTypeNameSpecifier
SBTypeCategory::GetTypeNameSpecifierForFilterAtIndex(uint32_t index) {
  LLDB_RECORD_METHOD(lldb::SBTypeNameSpecifier, SBTypeCategory,
                     GetTypeNameSpecifierForFilterAtIndex, (uint32_t), index);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBTypeNameSpecifier());
  return LLDB_RECORD_RESULT(SBTypeNameSpecifier(
      m_opaque_sp->GetTypeNameSpecifierForFilterAtIndex(index)));
}

lldb::SBTypeNameSpecifier
SBTypeCategory::GetTypeNameSpecifierForFormatAtIndex(uint32_t index) {
  LLDB_RECORD_METHOD(lldb::SBTypeNameSpecifier, SBTypeCategory,
                     GetTypeNameSpecifierForFormatAtIndex, (uint32_t), index);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBTypeNameSpecifier());
  return LLDB_RECORD_RESULT(SBTypeNameSpecifier(
      m_opaque_sp->GetTypeNameSpecifierForFormatAtIndex(index)));
}

lldb::SBTypeNameSpecifier
SBTypeCategory::GetTypeNameSpecifierForSummaryAtIndex(uint32_t index) {
  LLDB_RECORD_METHOD(lldb::SBTypeNameSpecifier, SBTypeCategory,
                     GetTypeNameSpecifierForSummaryAtIndex, (uint32_t), index);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBTypeNameSpecifier());
  return LLDB_RECORD_RESULT(SBTypeNameSpecifier(
      m_opaque_sp->GetTypeNameSpecifierForSummaryAtIndex(index)));
}

lldb::SBTypeNameSpecifier
SBTypeCategory::GetTypeNameSpecifierForSyntheticAtIndex(uint32_t index) {
  LLDB_RECORD_METHOD(lldb::SBTypeNameSpecifier, SBTypeCategory,
                     GetTypeNameSpecifierForSyntheticAtIndex, (uint32_t),
                     index);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBTypeNameSpecifier());
  return LLDB_RECORD_RESULT(SBTypeNameSpecifier(
      m_opaque_sp->GetTypeNameSpecifierForSyntheticAtIndex(index)));
}

SBTypeFilter SBTypeCategory::GetFilterForType(SBTypeNameSpecifier spec) {
  LLDB_RECORD_METHOD(lldb::SBTypeFilter, SBTypeCategory, GetFilterForType,
                     (lldb::SBTypeNameSpecifier), spec);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBTypeFilter());

  if (!spec.IsValid())
    return LLDB_RECORD_RESULT(SBTypeFilter());

  lldb::TypeFilterImplSP children_sp;

  if (spec.IsRegex())
    m_opaque_sp->GetRegexTypeFiltersContainer()->GetExact(
        ConstString(spec.GetName()), children_sp);
  else
    m_opaque_sp->GetTypeFiltersContainer()->GetExact(
        ConstString(spec.GetName()), children_sp);

  if (!children_sp)
    return LLDB_RECORD_RESULT(lldb::SBTypeFilter());

  TypeFilterImplSP filter_sp =
      std::static_pointer_cast<TypeFilterImpl>(children_sp);

  return LLDB_RECORD_RESULT(lldb::SBTypeFilter(filter_sp));
}
SBTypeFormat SBTypeCategory::GetFormatForType(SBTypeNameSpecifier spec) {
  LLDB_RECORD_METHOD(lldb::SBTypeFormat, SBTypeCategory, GetFormatForType,
                     (lldb::SBTypeNameSpecifier), spec);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBTypeFormat());

  if (!spec.IsValid())
    return LLDB_RECORD_RESULT(SBTypeFormat());

  lldb::TypeFormatImplSP format_sp;

  if (spec.IsRegex())
    m_opaque_sp->GetRegexTypeFormatsContainer()->GetExact(
        ConstString(spec.GetName()), format_sp);
  else
    m_opaque_sp->GetTypeFormatsContainer()->GetExact(
        ConstString(spec.GetName()), format_sp);

  if (!format_sp)
    return LLDB_RECORD_RESULT(lldb::SBTypeFormat());

  return LLDB_RECORD_RESULT(lldb::SBTypeFormat(format_sp));
}

SBTypeSummary SBTypeCategory::GetSummaryForType(SBTypeNameSpecifier spec) {
  LLDB_RECORD_METHOD(lldb::SBTypeSummary, SBTypeCategory, GetSummaryForType,
                     (lldb::SBTypeNameSpecifier), spec);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBTypeSummary());

  if (!spec.IsValid())
    return LLDB_RECORD_RESULT(SBTypeSummary());

  lldb::TypeSummaryImplSP summary_sp;

  if (spec.IsRegex())
    m_opaque_sp->GetRegexTypeSummariesContainer()->GetExact(
        ConstString(spec.GetName()), summary_sp);
  else
    m_opaque_sp->GetTypeSummariesContainer()->GetExact(
        ConstString(spec.GetName()), summary_sp);

  if (!summary_sp)
    return LLDB_RECORD_RESULT(lldb::SBTypeSummary());

  return LLDB_RECORD_RESULT(lldb::SBTypeSummary(summary_sp));
}

SBTypeSynthetic SBTypeCategory::GetSyntheticForType(SBTypeNameSpecifier spec) {
  LLDB_RECORD_METHOD(lldb::SBTypeSynthetic, SBTypeCategory, GetSyntheticForType,
                     (lldb::SBTypeNameSpecifier), spec);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBTypeSynthetic());

  if (!spec.IsValid())
    return LLDB_RECORD_RESULT(SBTypeSynthetic());

  lldb::SyntheticChildrenSP children_sp;

  if (spec.IsRegex())
    m_opaque_sp->GetRegexTypeSyntheticsContainer()->GetExact(
        ConstString(spec.GetName()), children_sp);
  else
    m_opaque_sp->GetTypeSyntheticsContainer()->GetExact(
        ConstString(spec.GetName()), children_sp);

  if (!children_sp)
    return LLDB_RECORD_RESULT(lldb::SBTypeSynthetic());

  ScriptedSyntheticChildrenSP synth_sp =
      std::static_pointer_cast<ScriptedSyntheticChildren>(children_sp);

  return LLDB_RECORD_RESULT(lldb::SBTypeSynthetic(synth_sp));
}

SBTypeFilter SBTypeCategory::GetFilterAtIndex(uint32_t index) {
  LLDB_RECORD_METHOD(lldb::SBTypeFilter, SBTypeCategory, GetFilterAtIndex,
                     (uint32_t), index);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBTypeFilter());
  lldb::SyntheticChildrenSP children_sp =
      m_opaque_sp->GetSyntheticAtIndex((index));

  if (!children_sp.get())
    return LLDB_RECORD_RESULT(lldb::SBTypeFilter());

  TypeFilterImplSP filter_sp =
      std::static_pointer_cast<TypeFilterImpl>(children_sp);

  return LLDB_RECORD_RESULT(lldb::SBTypeFilter(filter_sp));
}

SBTypeFormat SBTypeCategory::GetFormatAtIndex(uint32_t index) {
  LLDB_RECORD_METHOD(lldb::SBTypeFormat, SBTypeCategory, GetFormatAtIndex,
                     (uint32_t), index);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBTypeFormat());
  return LLDB_RECORD_RESULT(
      SBTypeFormat(m_opaque_sp->GetFormatAtIndex((index))));
}

SBTypeSummary SBTypeCategory::GetSummaryAtIndex(uint32_t index) {
  LLDB_RECORD_METHOD(lldb::SBTypeSummary, SBTypeCategory, GetSummaryAtIndex,
                     (uint32_t), index);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBTypeSummary());
  return LLDB_RECORD_RESULT(
      SBTypeSummary(m_opaque_sp->GetSummaryAtIndex((index))));
}

SBTypeSynthetic SBTypeCategory::GetSyntheticAtIndex(uint32_t index) {
  LLDB_RECORD_METHOD(lldb::SBTypeSynthetic, SBTypeCategory, GetSyntheticAtIndex,
                     (uint32_t), index);

  if (!IsValid())
    return LLDB_RECORD_RESULT(SBTypeSynthetic());
  lldb::SyntheticChildrenSP children_sp =
      m_opaque_sp->GetSyntheticAtIndex((index));

  if (!children_sp.get())
    return LLDB_RECORD_RESULT(lldb::SBTypeSynthetic());

  ScriptedSyntheticChildrenSP synth_sp =
      std::static_pointer_cast<ScriptedSyntheticChildren>(children_sp);

  return LLDB_RECORD_RESULT(lldb::SBTypeSynthetic(synth_sp));
}

bool SBTypeCategory::AddTypeFormat(SBTypeNameSpecifier type_name,
                                   SBTypeFormat format) {
  LLDB_RECORD_METHOD(bool, SBTypeCategory, AddTypeFormat,
                     (lldb::SBTypeNameSpecifier, lldb::SBTypeFormat), type_name,
                     format);

  if (!IsValid())
    return false;

  if (!type_name.IsValid())
    return false;

  if (!format.IsValid())
    return false;

  if (type_name.IsRegex())
    m_opaque_sp->GetRegexTypeFormatsContainer()->Add(
        RegularExpression(type_name.GetName()), format.GetSP());
  else
    m_opaque_sp->GetTypeFormatsContainer()->Add(
        ConstString(type_name.GetName()), format.GetSP());

  return true;
}

bool SBTypeCategory::DeleteTypeFormat(SBTypeNameSpecifier type_name) {
  LLDB_RECORD_METHOD(bool, SBTypeCategory, DeleteTypeFormat,
                     (lldb::SBTypeNameSpecifier), type_name);

  if (!IsValid())
    return false;

  if (!type_name.IsValid())
    return false;

  if (type_name.IsRegex())
    return m_opaque_sp->GetRegexTypeFormatsContainer()->Delete(
        ConstString(type_name.GetName()));
  else
    return m_opaque_sp->GetTypeFormatsContainer()->Delete(
        ConstString(type_name.GetName()));
}

bool SBTypeCategory::AddTypeSummary(SBTypeNameSpecifier type_name,
                                    SBTypeSummary summary) {
  LLDB_RECORD_METHOD(bool, SBTypeCategory, AddTypeSummary,
                     (lldb::SBTypeNameSpecifier, lldb::SBTypeSummary),
                     type_name, summary);

  if (!IsValid())
    return false;

  if (!type_name.IsValid())
    return false;

  if (!summary.IsValid())
    return false;

  // FIXME: we need to iterate over all the Debugger objects and have each of
  // them contain a copy of the function
  // since we currently have formatters live in a global space, while Python
  // code lives in a specific Debugger-related environment this should
  // eventually be fixed by deciding a final location in the LLDB object space
  // for formatters
  if (summary.IsFunctionCode()) {
    const void *name_token =
        (const void *)ConstString(type_name.GetName()).GetCString();
    const char *script = summary.GetData();
    StringList input;
    input.SplitIntoLines(script, strlen(script));
    uint32_t num_debuggers = lldb_private::Debugger::GetNumDebuggers();
    bool need_set = true;
    for (uint32_t j = 0; j < num_debuggers; j++) {
      DebuggerSP debugger_sp = lldb_private::Debugger::GetDebuggerAtIndex(j);
      if (debugger_sp) {
        ScriptInterpreter *interpreter_ptr =
            debugger_sp->GetScriptInterpreter();
        if (interpreter_ptr) {
          std::string output;
          if (interpreter_ptr->GenerateTypeScriptFunction(input, output,
                                                          name_token) &&
              !output.empty()) {
            if (need_set) {
              need_set = false;
              summary.SetFunctionName(output.c_str());
            }
          }
        }
      }
    }
  }

  if (type_name.IsRegex())
    m_opaque_sp->GetRegexTypeSummariesContainer()->Add(
        RegularExpression(type_name.GetName()), summary.GetSP());
  else
    m_opaque_sp->GetTypeSummariesContainer()->Add(
        ConstString(type_name.GetName()), summary.GetSP());

  return true;
}

bool SBTypeCategory::DeleteTypeSummary(SBTypeNameSpecifier type_name) {
  LLDB_RECORD_METHOD(bool, SBTypeCategory, DeleteTypeSummary,
                     (lldb::SBTypeNameSpecifier), type_name);

  if (!IsValid())
    return false;

  if (!type_name.IsValid())
    return false;

  if (type_name.IsRegex())
    return m_opaque_sp->GetRegexTypeSummariesContainer()->Delete(
        ConstString(type_name.GetName()));
  else
    return m_opaque_sp->GetTypeSummariesContainer()->Delete(
        ConstString(type_name.GetName()));
}

bool SBTypeCategory::AddTypeFilter(SBTypeNameSpecifier type_name,
                                   SBTypeFilter filter) {
  LLDB_RECORD_METHOD(bool, SBTypeCategory, AddTypeFilter,
                     (lldb::SBTypeNameSpecifier, lldb::SBTypeFilter), type_name,
                     filter);

  if (!IsValid())
    return false;

  if (!type_name.IsValid())
    return false;

  if (!filter.IsValid())
    return false;

  if (type_name.IsRegex())
    m_opaque_sp->GetRegexTypeFiltersContainer()->Add(
        RegularExpression(type_name.GetName()), filter.GetSP());
  else
    m_opaque_sp->GetTypeFiltersContainer()->Add(
        ConstString(type_name.GetName()), filter.GetSP());

  return true;
}

bool SBTypeCategory::DeleteTypeFilter(SBTypeNameSpecifier type_name) {
  LLDB_RECORD_METHOD(bool, SBTypeCategory, DeleteTypeFilter,
                     (lldb::SBTypeNameSpecifier), type_name);

  if (!IsValid())
    return false;

  if (!type_name.IsValid())
    return false;

  if (type_name.IsRegex())
    return m_opaque_sp->GetRegexTypeFiltersContainer()->Delete(
        ConstString(type_name.GetName()));
  else
    return m_opaque_sp->GetTypeFiltersContainer()->Delete(
        ConstString(type_name.GetName()));
}

bool SBTypeCategory::AddTypeSynthetic(SBTypeNameSpecifier type_name,
                                      SBTypeSynthetic synth) {
  LLDB_RECORD_METHOD(bool, SBTypeCategory, AddTypeSynthetic,
                     (lldb::SBTypeNameSpecifier, lldb::SBTypeSynthetic),
                     type_name, synth);

  if (!IsValid())
    return false;

  if (!type_name.IsValid())
    return false;

  if (!synth.IsValid())
    return false;

  // FIXME: we need to iterate over all the Debugger objects and have each of
  // them contain a copy of the function
  // since we currently have formatters live in a global space, while Python
  // code lives in a specific Debugger-related environment this should
  // eventually be fixed by deciding a final location in the LLDB object space
  // for formatters
  if (synth.IsClassCode()) {
    const void *name_token =
        (const void *)ConstString(type_name.GetName()).GetCString();
    const char *script = synth.GetData();
    StringList input;
    input.SplitIntoLines(script, strlen(script));
    uint32_t num_debuggers = lldb_private::Debugger::GetNumDebuggers();
    bool need_set = true;
    for (uint32_t j = 0; j < num_debuggers; j++) {
      DebuggerSP debugger_sp = lldb_private::Debugger::GetDebuggerAtIndex(j);
      if (debugger_sp) {
        ScriptInterpreter *interpreter_ptr =
            debugger_sp->GetScriptInterpreter();
        if (interpreter_ptr) {
          std::string output;
          if (interpreter_ptr->GenerateTypeSynthClass(input, output,
                                                      name_token) &&
              !output.empty()) {
            if (need_set) {
              need_set = false;
              synth.SetClassName(output.c_str());
            }
          }
        }
      }
    }
  }

  if (type_name.IsRegex())
    m_opaque_sp->GetRegexTypeSyntheticsContainer()->Add(
        RegularExpression(type_name.GetName()), synth.GetSP());
  else
    m_opaque_sp->GetTypeSyntheticsContainer()->Add(
        ConstString(type_name.GetName()), synth.GetSP());

  return true;
}

bool SBTypeCategory::DeleteTypeSynthetic(SBTypeNameSpecifier type_name) {
  LLDB_RECORD_METHOD(bool, SBTypeCategory, DeleteTypeSynthetic,
                     (lldb::SBTypeNameSpecifier), type_name);

  if (!IsValid())
    return false;

  if (!type_name.IsValid())
    return false;

  if (type_name.IsRegex())
    return m_opaque_sp->GetRegexTypeSyntheticsContainer()->Delete(
        ConstString(type_name.GetName()));
  else
    return m_opaque_sp->GetTypeSyntheticsContainer()->Delete(
        ConstString(type_name.GetName()));
}

bool SBTypeCategory::GetDescription(lldb::SBStream &description,
                                    lldb::DescriptionLevel description_level) {
  LLDB_RECORD_METHOD(bool, SBTypeCategory, GetDescription,
                     (lldb::SBStream &, lldb::DescriptionLevel), description,
                     description_level);

  if (!IsValid())
    return false;
  description.Printf("Category name: %s\n", GetName());
  return true;
}

lldb::SBTypeCategory &SBTypeCategory::
operator=(const lldb::SBTypeCategory &rhs) {
  LLDB_RECORD_METHOD(lldb::SBTypeCategory &,
                     SBTypeCategory, operator=,(const lldb::SBTypeCategory &),
                     rhs);

  if (this != &rhs) {
    m_opaque_sp = rhs.m_opaque_sp;
  }
  return LLDB_RECORD_RESULT(*this);
}

bool SBTypeCategory::operator==(lldb::SBTypeCategory &rhs) {
  LLDB_RECORD_METHOD(bool, SBTypeCategory, operator==,(lldb::SBTypeCategory &),
                     rhs);

  if (!IsValid())
    return !rhs.IsValid();

  return m_opaque_sp.get() == rhs.m_opaque_sp.get();
}

bool SBTypeCategory::operator!=(lldb::SBTypeCategory &rhs) {
  LLDB_RECORD_METHOD(bool, SBTypeCategory, operator!=,(lldb::SBTypeCategory &),
                     rhs);

  if (!IsValid())
    return rhs.IsValid();

  return m_opaque_sp.get() != rhs.m_opaque_sp.get();
}

lldb::TypeCategoryImplSP SBTypeCategory::GetSP() {
  if (!IsValid())
    return lldb::TypeCategoryImplSP();
  return m_opaque_sp;
}

void SBTypeCategory::SetSP(
    const lldb::TypeCategoryImplSP &typecategory_impl_sp) {
  m_opaque_sp = typecategory_impl_sp;
}

SBTypeCategory::SBTypeCategory(
    const lldb::TypeCategoryImplSP &typecategory_impl_sp)
    : m_opaque_sp(typecategory_impl_sp) {}

bool SBTypeCategory::IsDefaultCategory() {
  if (!IsValid())
    return false;

  return (strcmp(m_opaque_sp->GetName(), "default") == 0);
}

namespace lldb_private {
namespace repro {

template <>
void RegisterMethods<SBTypeCategory>(Registry &R) {
  LLDB_REGISTER_CONSTRUCTOR(SBTypeCategory, ());
  LLDB_REGISTER_CONSTRUCTOR(SBTypeCategory, (const lldb::SBTypeCategory &));
  LLDB_REGISTER_METHOD_CONST(bool, SBTypeCategory, IsValid, ());
  LLDB_REGISTER_METHOD_CONST(bool, SBTypeCategory, operator bool, ());
  LLDB_REGISTER_METHOD(bool, SBTypeCategory, GetEnabled, ());
  LLDB_REGISTER_METHOD(void, SBTypeCategory, SetEnabled, (bool));
  LLDB_REGISTER_METHOD(const char *, SBTypeCategory, GetName, ());
  LLDB_REGISTER_METHOD(lldb::LanguageType, SBTypeCategory, GetLanguageAtIndex,
                       (uint32_t));
  LLDB_REGISTER_METHOD(uint32_t, SBTypeCategory, GetNumLanguages, ());
  LLDB_REGISTER_METHOD(void, SBTypeCategory, AddLanguage,
                       (lldb::LanguageType));
  LLDB_REGISTER_METHOD(uint32_t, SBTypeCategory, GetNumFormats, ());
  LLDB_REGISTER_METHOD(uint32_t, SBTypeCategory, GetNumSummaries, ());
  LLDB_REGISTER_METHOD(uint32_t, SBTypeCategory, GetNumFilters, ());
  LLDB_REGISTER_METHOD(uint32_t, SBTypeCategory, GetNumSynthetics, ());
  LLDB_REGISTER_METHOD(lldb::SBTypeNameSpecifier, SBTypeCategory,
                       GetTypeNameSpecifierForSyntheticAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBTypeSummary, SBTypeCategory, GetSummaryForType,
                       (lldb::SBTypeNameSpecifier));
  LLDB_REGISTER_METHOD(lldb::SBTypeSynthetic, SBTypeCategory,
                       GetSyntheticForType, (lldb::SBTypeNameSpecifier));
  LLDB_REGISTER_METHOD(lldb::SBTypeFilter, SBTypeCategory, GetFilterAtIndex,
                       (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBTypeSummary, SBTypeCategory, GetSummaryAtIndex,
                       (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBTypeSynthetic, SBTypeCategory,
                       GetSyntheticAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(bool, SBTypeCategory, AddTypeSummary,
                       (lldb::SBTypeNameSpecifier, lldb::SBTypeSummary));
  LLDB_REGISTER_METHOD(bool, SBTypeCategory, AddTypeSynthetic,
                       (lldb::SBTypeNameSpecifier, lldb::SBTypeSynthetic));
  LLDB_REGISTER_METHOD(bool, SBTypeCategory, DeleteTypeSynthetic,
                       (lldb::SBTypeNameSpecifier));
  LLDB_REGISTER_METHOD(lldb::SBTypeNameSpecifier, SBTypeCategory,
                       GetTypeNameSpecifierForFilterAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBTypeNameSpecifier, SBTypeCategory,
                       GetTypeNameSpecifierForFormatAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBTypeNameSpecifier, SBTypeCategory,
                       GetTypeNameSpecifierForSummaryAtIndex, (uint32_t));
  LLDB_REGISTER_METHOD(lldb::SBTypeFilter, SBTypeCategory, GetFilterForType,
                       (lldb::SBTypeNameSpecifier));
  LLDB_REGISTER_METHOD(lldb::SBTypeFormat, SBTypeCategory, GetFormatForType,
                       (lldb::SBTypeNameSpecifier));
  LLDB_REGISTER_METHOD(lldb::SBTypeFormat, SBTypeCategory, GetFormatAtIndex,
                       (uint32_t));
  LLDB_REGISTER_METHOD(bool, SBTypeCategory, AddTypeFormat,
                       (lldb::SBTypeNameSpecifier, lldb::SBTypeFormat));
  LLDB_REGISTER_METHOD(bool, SBTypeCategory, DeleteTypeFormat,
                       (lldb::SBTypeNameSpecifier));
  LLDB_REGISTER_METHOD(bool, SBTypeCategory, DeleteTypeSummary,
                       (lldb::SBTypeNameSpecifier));
  LLDB_REGISTER_METHOD(bool, SBTypeCategory, AddTypeFilter,
                       (lldb::SBTypeNameSpecifier, lldb::SBTypeFilter));
  LLDB_REGISTER_METHOD(bool, SBTypeCategory, DeleteTypeFilter,
                       (lldb::SBTypeNameSpecifier));
  LLDB_REGISTER_METHOD(bool, SBTypeCategory, GetDescription,
                       (lldb::SBStream &, lldb::DescriptionLevel));
  LLDB_REGISTER_METHOD(
      lldb::SBTypeCategory &,
      SBTypeCategory, operator=,(const lldb::SBTypeCategory &));
  LLDB_REGISTER_METHOD(bool,
                       SBTypeCategory, operator==,(lldb::SBTypeCategory &));
  LLDB_REGISTER_METHOD(bool,
                       SBTypeCategory, operator!=,(lldb::SBTypeCategory &));
}

}
}
