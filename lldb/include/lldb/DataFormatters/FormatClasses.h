//===-- FormatClasses.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_DATAFORMATTERS_FORMATCLASSES_H
#define LLDB_DATAFORMATTERS_FORMATCLASSES_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "lldb/DataFormatters/TypeFormat.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/Type.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-public.h"

namespace lldb_private {

class HardcodedFormatters {
public:
  template <typename FormatterType>
  using HardcodedFormatterFinder =
      std::function<typename FormatterType::SharedPointer(
          lldb_private::ValueObject &, lldb::DynamicValueType,
          FormatManager &)>;

  template <typename FormatterType>
  using HardcodedFormatterFinders =
      std::vector<HardcodedFormatterFinder<FormatterType>>;

  typedef HardcodedFormatterFinders<TypeFormatImpl> HardcodedFormatFinder;
  typedef HardcodedFormatterFinders<TypeSummaryImpl> HardcodedSummaryFinder;
  typedef HardcodedFormatterFinders<SyntheticChildren> HardcodedSyntheticFinder;
};

class FormattersMatchCandidate {
public:
  FormattersMatchCandidate(ConstString name, bool strip_ptr,
                           bool strip_ref, bool strip_tydef)
      : m_type_name(name), m_stripped_pointer(strip_ptr),
        m_stripped_reference(strip_ref), m_stripped_typedef(strip_tydef) {}

  ~FormattersMatchCandidate() = default;

  ConstString GetTypeName() const { return m_type_name; }

  bool DidStripPointer() const { return m_stripped_pointer; }

  bool DidStripReference() const { return m_stripped_reference; }

  bool DidStripTypedef() const { return m_stripped_typedef; }

  template <class Formatter>
  bool IsMatch(const std::shared_ptr<Formatter> &formatter_sp) const {
    if (!formatter_sp)
      return false;
    if (formatter_sp->Cascades() == false && DidStripTypedef())
      return false;
    if (formatter_sp->SkipsPointers() && DidStripPointer())
      return false;
    if (formatter_sp->SkipsReferences() && DidStripReference())
      return false;
    return true;
  }

private:
  ConstString m_type_name;
  bool m_stripped_pointer;
  bool m_stripped_reference;
  bool m_stripped_typedef;
};

typedef std::vector<FormattersMatchCandidate> FormattersMatchVector;
typedef std::vector<lldb::LanguageType> CandidateLanguagesVector;

class FormattersMatchData {
public:
  FormattersMatchData(ValueObject &, lldb::DynamicValueType);

  FormattersMatchVector GetMatchesVector();

  ConstString GetTypeForCache();

  CandidateLanguagesVector GetCandidateLanguages();

  ValueObject &GetValueObject();

  lldb::DynamicValueType GetDynamicValueType();

private:
  ValueObject &m_valobj;
  lldb::DynamicValueType m_dynamic_value_type;
  std::pair<FormattersMatchVector, bool> m_formatters_match_vector;
  ConstString m_type_for_cache;
  CandidateLanguagesVector m_candidate_languages;
};

class TypeNameSpecifierImpl {
public:
  TypeNameSpecifierImpl() : m_type() {}

  TypeNameSpecifierImpl(llvm::StringRef name, bool is_regex)
      : m_is_regex(is_regex), m_type() {
    m_type.m_type_name = std::string(name);
  }

  // if constructing with a given type, is_regex cannot be true since we are
  // giving an exact type to match
  TypeNameSpecifierImpl(lldb::TypeSP type) : m_is_regex(false), m_type() {
    if (type) {
      m_type.m_type_name = std::string(type->GetName().GetStringRef());
      m_type.m_compiler_type = type->GetForwardCompilerType();
    }
  }

  TypeNameSpecifierImpl(CompilerType type) : m_is_regex(false), m_type() {
    if (type.IsValid()) {
      m_type.m_type_name.assign(type.GetTypeName().GetCString());
      m_type.m_compiler_type = type;
    }
  }

  const char *GetName() {
    if (m_type.m_type_name.size())
      return m_type.m_type_name.c_str();
    return nullptr;
  }

  CompilerType GetCompilerType() {
    if (m_type.m_compiler_type.IsValid())
      return m_type.m_compiler_type;
    return CompilerType();
  }

  bool IsRegex() { return m_is_regex; }

private:
  bool m_is_regex = false;
  // TODO: Replace this with TypeAndOrName.
  struct TypeOrName {
    std::string m_type_name;
    CompilerType m_compiler_type;
  };
  TypeOrName m_type;

  TypeNameSpecifierImpl(const TypeNameSpecifierImpl &) = delete;
  const TypeNameSpecifierImpl &
  operator=(const TypeNameSpecifierImpl &) = delete;
};

} // namespace lldb_private

#endif // LLDB_DATAFORMATTERS_FORMATCLASSES_H
