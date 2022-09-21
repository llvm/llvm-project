//===-- TypeCategory.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/TypeCategory.h"
#include "lldb/Target/Language.h"


using namespace lldb;
using namespace lldb_private;

TypeCategoryImpl::TypeCategoryImpl(IFormatChangeListener *clist,
                                   ConstString name)
    : m_format_cont(clist), m_summary_cont(clist), m_filter_cont(clist),
      m_synth_cont(clist), m_enabled(false), m_change_listener(clist),
      m_mutex(), m_name(name), m_languages() {}

static bool IsApplicable(lldb::LanguageType category_lang,
                         lldb::LanguageType valobj_lang) {
  switch (category_lang) {
  // Unless we know better, allow only exact equality.
  default:
    return category_lang == valobj_lang;

  // the C family, we consider it as one
  case eLanguageTypeC89:
  case eLanguageTypeC:
  case eLanguageTypeC99:
    return valobj_lang == eLanguageTypeC89 || valobj_lang == eLanguageTypeC ||
           valobj_lang == eLanguageTypeC99;

  // ObjC knows about C and itself
  case eLanguageTypeObjC:
    return valobj_lang == eLanguageTypeC89 || valobj_lang == eLanguageTypeC ||
           valobj_lang == eLanguageTypeC99 || valobj_lang == eLanguageTypeObjC;

  // C++ knows about C and C++
  case eLanguageTypeC_plus_plus:
    return valobj_lang == eLanguageTypeC89 || valobj_lang == eLanguageTypeC ||
           valobj_lang == eLanguageTypeC99 ||
           valobj_lang == eLanguageTypeC_plus_plus;

  // ObjC++ knows about C,C++,ObjC and ObjC++
  case eLanguageTypeObjC_plus_plus:
    return valobj_lang == eLanguageTypeC89 || valobj_lang == eLanguageTypeC ||
           valobj_lang == eLanguageTypeC99 ||
           valobj_lang == eLanguageTypeC_plus_plus ||
           valobj_lang == eLanguageTypeObjC;

  // Categories with unspecified language match everything.
  case eLanguageTypeUnknown:
    return true;
  }
}

bool TypeCategoryImpl::IsApplicable(lldb::LanguageType lang) {
  for (size_t idx = 0; idx < GetNumLanguages(); idx++) {
    const lldb::LanguageType category_lang = GetLanguageAtIndex(idx);
    if (::IsApplicable(category_lang, lang))
      return true;
  }
  return false;
}

size_t TypeCategoryImpl::GetNumLanguages() {
  if (m_languages.empty())
    return 1;
  return m_languages.size();
}

lldb::LanguageType TypeCategoryImpl::GetLanguageAtIndex(size_t idx) {
  if (m_languages.empty())
    return lldb::eLanguageTypeUnknown;
  return m_languages[idx];
}

void TypeCategoryImpl::AddLanguage(lldb::LanguageType lang) {
  m_languages.push_back(lang);
}

bool TypeCategoryImpl::Get(lldb::LanguageType lang,
                           const FormattersMatchVector &candidates,
                           lldb::TypeFormatImplSP &entry) {
  if (!IsEnabled() || !IsApplicable(lang))
    return false;
  if (GetTypeFormatsContainer()->Get(candidates, entry))
    return true;
  bool regex = GetRegexTypeFormatsContainer()->Get(candidates, entry);
  return regex;
}

bool TypeCategoryImpl::Get(lldb::LanguageType lang,
                           const FormattersMatchVector &candidates,
                           lldb::TypeSummaryImplSP &entry) {
  if (!IsEnabled() || !IsApplicable(lang))
    return false;
  if (GetTypeSummariesContainer()->Get(candidates, entry))
    return true;
  bool regex = GetRegexTypeSummariesContainer()->Get(candidates, entry);
  return regex;
}

bool TypeCategoryImpl::Get(lldb::LanguageType lang,
                           const FormattersMatchVector &candidates,
                           lldb::SyntheticChildrenSP &entry) {
  if (!IsEnabled() || !IsApplicable(lang))
    return false;
  TypeFilterImpl::SharedPointer filter_sp;
  // first find both Filter and Synth, and then check which is most recent

  if (!GetTypeFiltersContainer()->Get(candidates, filter_sp))
    GetRegexTypeFiltersContainer()->Get(candidates, filter_sp);

  bool pick_synth = false;
  ScriptedSyntheticChildren::SharedPointer synth;
  if (!GetTypeSyntheticsContainer()->Get(candidates, synth))
    GetRegexTypeSyntheticsContainer()->Get(candidates, synth);
  if (!filter_sp.get() && !synth.get())
    return false;
  else if (!filter_sp.get() && synth.get())
    pick_synth = true;

  else if (filter_sp.get() && !synth.get())
    pick_synth = false;

  else /*if (filter_sp.get() && synth.get())*/
  {
    pick_synth = filter_sp->GetRevision() <= synth->GetRevision();
  }
  if (pick_synth) {
    entry = synth;
    return true;
  } else {
    entry = filter_sp;
    return true;
  }
  return false;
}

void TypeCategoryImpl::Clear(FormatCategoryItems items) {
  if (items & eFormatCategoryItemFormat) {
    GetTypeFormatsContainer()->Clear();
    GetRegexTypeFormatsContainer()->Clear();
  }

  if (items & eFormatCategoryItemSummary) {
    GetTypeSummariesContainer()->Clear();
    GetRegexTypeSummariesContainer()->Clear();
  }

  if (items & eFormatCategoryItemFilter) {
    GetTypeFiltersContainer()->Clear();
    GetRegexTypeFiltersContainer()->Clear();
  }

  if (items & eFormatCategoryItemSynth) {
    GetTypeSyntheticsContainer()->Clear();
    GetRegexTypeSyntheticsContainer()->Clear();
  }
}

bool TypeCategoryImpl::Delete(ConstString name, FormatCategoryItems items) {
  bool success = false;

  if (items & eFormatCategoryItemFormat) {
    success = GetTypeFormatsContainer()->Delete(name) || success;
    success = GetRegexTypeFormatsContainer()->Delete(name) || success;
  }

  if (items & eFormatCategoryItemSummary) {
    success = GetTypeSummariesContainer()->Delete(name) || success;
    success = GetRegexTypeSummariesContainer()->Delete(name) || success;
  }

  if (items & eFormatCategoryItemFilter) {
    success = GetTypeFiltersContainer()->Delete(name) || success;
    success = GetRegexTypeFiltersContainer()->Delete(name) || success;
  }

  if (items & eFormatCategoryItemSynth) {
    success = GetTypeSyntheticsContainer()->Delete(name) || success;
    success = GetRegexTypeSyntheticsContainer()->Delete(name) || success;
  }

  return success;
}

uint32_t TypeCategoryImpl::GetCount(FormatCategoryItems items) {
  uint32_t count = 0;

  if (items & eFormatCategoryItemFormat) {
    count += GetTypeFormatsContainer()->GetCount();
    count += GetRegexTypeFormatsContainer()->GetCount();
  }

  if (items & eFormatCategoryItemSummary) {
    count += GetTypeSummariesContainer()->GetCount();
    count += GetRegexTypeSummariesContainer()->GetCount();
  }

  if (items & eFormatCategoryItemFilter) {
    count += GetTypeFiltersContainer()->GetCount();
    count += GetRegexTypeFiltersContainer()->GetCount();
  }

  if (items & eFormatCategoryItemSynth) {
    count += GetTypeSyntheticsContainer()->GetCount();
    count += GetRegexTypeSyntheticsContainer()->GetCount();
  }

  return count;
}

bool TypeCategoryImpl::AnyMatches(ConstString type_name,
                                  FormatCategoryItems items, bool only_enabled,
                                  const char **matching_category,
                                  FormatCategoryItems *matching_type) {
  if (!IsEnabled() && only_enabled)
    return false;

  lldb::TypeFormatImplSP format_sp;
  lldb::TypeSummaryImplSP summary_sp;
  TypeFilterImpl::SharedPointer filter_sp;
  ScriptedSyntheticChildren::SharedPointer synth_sp;

  if (items & eFormatCategoryItemFormat) {
    if (GetTypeFormatsContainer()->Get(type_name, format_sp) ||
        GetRegexTypeFormatsContainer()->Get(type_name, format_sp)) {
      if (matching_category)
        *matching_category = m_name.GetCString();
      if (matching_type)
        *matching_type = eFormatCategoryItemFormat;
      return true;
    }
  }

  if (items & eFormatCategoryItemSummary) {
    if (GetTypeSummariesContainer()->Get(type_name, summary_sp) ||
        GetRegexTypeSummariesContainer()->Get(type_name, summary_sp)) {
      if (matching_category)
        *matching_category = m_name.GetCString();
      if (matching_type)
        *matching_type = eFormatCategoryItemSummary;
      return true;
    }
  }

  if (items & eFormatCategoryItemFilter) {
    if (GetTypeFiltersContainer()->Get(type_name, filter_sp) ||
        GetRegexTypeFiltersContainer()->Get(type_name, filter_sp)) {
      if (matching_category)
        *matching_category = m_name.GetCString();
      if (matching_type)
        *matching_type = eFormatCategoryItemFilter;
      return true;
    }
  }

  if (items & eFormatCategoryItemSynth) {
    if (GetTypeSyntheticsContainer()->Get(type_name, synth_sp) ||
        GetRegexTypeSyntheticsContainer()->Get(type_name, synth_sp)) {
      if (matching_category)
        *matching_category = m_name.GetCString();
      if (matching_type)
        *matching_type = eFormatCategoryItemSynth;
      return true;
    }
  }

  return false;
}

TypeCategoryImpl::FormatContainer::MapValueType
TypeCategoryImpl::GetFormatForType(lldb::TypeNameSpecifierImplSP type_sp) {
  FormatContainer::MapValueType retval;

  if (type_sp) {
    if (type_sp->IsRegex())
      GetRegexTypeFormatsContainer()->GetExact(ConstString(type_sp->GetName()),
                                               retval);
    else
      GetTypeFormatsContainer()->GetExact(ConstString(type_sp->GetName()),
                                          retval);
  }

  return retval;
}

TypeCategoryImpl::SummaryContainer::MapValueType
TypeCategoryImpl::GetSummaryForType(lldb::TypeNameSpecifierImplSP type_sp) {
  SummaryContainer::MapValueType retval;

  if (type_sp) {
    if (type_sp->IsRegex())
      GetRegexTypeSummariesContainer()->GetExact(
          ConstString(type_sp->GetName()), retval);
    else
      GetTypeSummariesContainer()->GetExact(ConstString(type_sp->GetName()),
                                            retval);
  }

  return retval;
}

TypeCategoryImpl::FilterContainer::MapValueType
TypeCategoryImpl::GetFilterForType(lldb::TypeNameSpecifierImplSP type_sp) {
  FilterContainer::MapValueType retval;

  if (type_sp) {
    if (type_sp->IsRegex())
      GetRegexTypeFiltersContainer()->GetExact(ConstString(type_sp->GetName()),
                                               retval);
    else
      GetTypeFiltersContainer()->GetExact(ConstString(type_sp->GetName()),
                                          retval);
  }

  return retval;
}

TypeCategoryImpl::SynthContainer::MapValueType
TypeCategoryImpl::GetSyntheticForType(lldb::TypeNameSpecifierImplSP type_sp) {
  SynthContainer::MapValueType retval;

  if (type_sp) {
    if (type_sp->IsRegex())
      GetRegexTypeSyntheticsContainer()->GetExact(
          ConstString(type_sp->GetName()), retval);
    else
      GetTypeSyntheticsContainer()->GetExact(ConstString(type_sp->GetName()),
                                             retval);
  }

  return retval;
}

lldb::TypeNameSpecifierImplSP
TypeCategoryImpl::GetTypeNameSpecifierForSummaryAtIndex(size_t index) {
  if (index < GetTypeSummariesContainer()->GetCount())
    return GetTypeSummariesContainer()->GetTypeNameSpecifierAtIndex(index);
  else
    return GetRegexTypeSummariesContainer()->GetTypeNameSpecifierAtIndex(
        index - GetTypeSummariesContainer()->GetCount());
}

TypeCategoryImpl::FormatContainer::MapValueType
TypeCategoryImpl::GetFormatAtIndex(size_t index) {
  if (index < GetTypeFormatsContainer()->GetCount())
    return GetTypeFormatsContainer()->GetAtIndex(index);
  else
    return GetRegexTypeFormatsContainer()->GetAtIndex(
        index - GetTypeFormatsContainer()->GetCount());
}

TypeCategoryImpl::SummaryContainer::MapValueType
TypeCategoryImpl::GetSummaryAtIndex(size_t index) {
  if (index < GetTypeSummariesContainer()->GetCount())
    return GetTypeSummariesContainer()->GetAtIndex(index);
  else
    return GetRegexTypeSummariesContainer()->GetAtIndex(
        index - GetTypeSummariesContainer()->GetCount());
}

TypeCategoryImpl::FilterContainer::MapValueType
TypeCategoryImpl::GetFilterAtIndex(size_t index) {
  if (index < GetTypeFiltersContainer()->GetCount())
    return GetTypeFiltersContainer()->GetAtIndex(index);
  else
    return GetRegexTypeFiltersContainer()->GetAtIndex(
        index - GetTypeFiltersContainer()->GetCount());
}

lldb::TypeNameSpecifierImplSP
TypeCategoryImpl::GetTypeNameSpecifierForFormatAtIndex(size_t index) {
  if (index < GetTypeFormatsContainer()->GetCount())
    return GetTypeFormatsContainer()->GetTypeNameSpecifierAtIndex(index);
  else
    return GetRegexTypeFormatsContainer()->GetTypeNameSpecifierAtIndex(
        index - GetTypeFormatsContainer()->GetCount());
}

lldb::TypeNameSpecifierImplSP
TypeCategoryImpl::GetTypeNameSpecifierForFilterAtIndex(size_t index) {
  if (index < GetTypeFiltersContainer()->GetCount())
    return GetTypeFiltersContainer()->GetTypeNameSpecifierAtIndex(index);
  else
    return GetRegexTypeFiltersContainer()->GetTypeNameSpecifierAtIndex(
        index - GetTypeFiltersContainer()->GetCount());
}

TypeCategoryImpl::SynthContainer::MapValueType
TypeCategoryImpl::GetSyntheticAtIndex(size_t index) {
  if (index < GetTypeSyntheticsContainer()->GetCount())
    return GetTypeSyntheticsContainer()->GetAtIndex(index);
  else
    return GetRegexTypeSyntheticsContainer()->GetAtIndex(
        index - GetTypeSyntheticsContainer()->GetCount());
}

lldb::TypeNameSpecifierImplSP
TypeCategoryImpl::GetTypeNameSpecifierForSyntheticAtIndex(size_t index) {
  if (index < GetTypeSyntheticsContainer()->GetCount())
    return GetTypeSyntheticsContainer()->GetTypeNameSpecifierAtIndex(index);
  else
    return GetRegexTypeSyntheticsContainer()->GetTypeNameSpecifierAtIndex(
        index - GetTypeSyntheticsContainer()->GetCount());
}

void TypeCategoryImpl::Enable(bool value, uint32_t position) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  if ((m_enabled = value))
    m_enabled_position = position;
  if (m_change_listener)
    m_change_listener->Changed();
}

std::string TypeCategoryImpl::GetDescription() {
  StreamString stream;
  stream.Printf("%s (%s", GetName(), (IsEnabled() ? "enabled" : "disabled"));
  StreamString lang_stream;
  lang_stream.Printf(", applicable for language(s): ");
  bool print_lang = false;
  for (size_t idx = 0; idx < GetNumLanguages(); idx++) {
    const lldb::LanguageType lang = GetLanguageAtIndex(idx);
    if (lang != lldb::eLanguageTypeUnknown)
      print_lang = true;
    lang_stream.Printf("%s%s", Language::GetNameForLanguageType(lang),
                       idx + 1 < GetNumLanguages() ? ", " : "");
  }
  if (print_lang)
    stream.PutCString(lang_stream.GetString());
  stream.PutChar(')');
  return std::string(stream.GetString());
}
