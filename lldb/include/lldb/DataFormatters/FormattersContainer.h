//===-- FormattersContainer.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_DATAFORMATTERS_FORMATTERSCONTAINER_H
#define LLDB_DATAFORMATTERS_FORMATTERSCONTAINER_H

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "lldb/lldb-public.h"

#include "lldb/Core/ValueObject.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/TypeFormat.h"
#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Utility/RegularExpression.h"
#include "lldb/Utility/StringLexer.h"

namespace lldb_private {

class IFormatChangeListener {
public:
  virtual ~IFormatChangeListener() = default;

  virtual void Changed() = 0;

  virtual uint32_t GetCurrentRevision() = 0;
};

/// Class for matching type names.
class TypeMatcher {
  RegularExpression m_type_name_regex;
  ConstString m_type_name;
  /// False if m_type_name_regex should be used for matching. False if this is
  /// just matching by comparing with m_type_name string.
  bool m_is_regex;

  // if the user tries to add formatters for, say, "struct Foo" those will not
  // match any type because of the way we strip qualifiers from typenames this
  // method looks for the case where the user is adding a
  // "class","struct","enum" or "union" Foo and strips the unnecessary qualifier
  static ConstString StripTypeName(ConstString type) {
    if (type.IsEmpty())
      return type;

    std::string type_cstr(type.AsCString());
    StringLexer type_lexer(type_cstr);

    type_lexer.AdvanceIf("class ");
    type_lexer.AdvanceIf("enum ");
    type_lexer.AdvanceIf("struct ");
    type_lexer.AdvanceIf("union ");

    while (type_lexer.NextIf({' ', '\t', '\v', '\f'}).first)
      ;

    return ConstString(type_lexer.GetUnlexed());
  }

public:
  TypeMatcher() = delete;
  /// Creates a matcher that accepts any type with exactly the given type name.
  TypeMatcher(ConstString type_name)
      : m_type_name(type_name), m_is_regex(false) {}
  /// Creates a matcher that accepts any type matching the given regex.
  TypeMatcher(RegularExpression regex)
      : m_type_name_regex(regex), m_is_regex(true) {}

  /// True iff this matches the given type name.
  bool Matches(ConstString type_name) const {
    if (m_is_regex)
      return m_type_name_regex.Execute(type_name.GetStringRef());
    return m_type_name == type_name ||
           StripTypeName(m_type_name) == StripTypeName(type_name);
  }

  /// Returns the underlying match string for this TypeMatcher.
  ConstString GetMatchString() const {
    if (m_is_regex)
      return ConstString(m_type_name_regex.GetText());
    return StripTypeName(m_type_name);
  }

  /// Returns true if this TypeMatcher and the given one were most created by
  /// the same match string.
  /// The main purpose of this function is to find existing TypeMatcher
  /// instances by the user input that created them. This is necessary as LLDB
  /// allows referencing existing TypeMatchers in commands by the user input
  /// that originally created them:
  /// (lldb) type summary add --summary-string \"A\" -x TypeName
  /// (lldb) type summary delete TypeName
  bool CreatedBySameMatchString(TypeMatcher other) const {
    return GetMatchString() == other.GetMatchString();
  }
};

template <typename ValueType> class FormattersContainer {
public:
  typedef typename std::shared_ptr<ValueType> ValueSP;
  typedef std::vector<std::pair<TypeMatcher, ValueSP>> MapType;
  typedef std::function<bool(const TypeMatcher &, const ValueSP &)>
      ForEachCallback;
  typedef typename std::shared_ptr<FormattersContainer<ValueType>>
      SharedPointer;

  friend class TypeCategoryImpl;

  FormattersContainer(IFormatChangeListener *lst) : listener(lst) {}

  void Add(TypeMatcher matcher, const ValueSP &entry) {
    if (listener)
      entry->GetRevision() = listener->GetCurrentRevision();
    else
      entry->GetRevision() = 0;

    std::lock_guard<std::recursive_mutex> guard(m_map_mutex);
    Delete(matcher);
    m_map.emplace_back(std::move(matcher), std::move(entry));
    if (listener)
      listener->Changed();
  }

  bool Delete(TypeMatcher matcher) {
    std::lock_guard<std::recursive_mutex> guard(m_map_mutex);
    for (auto iter = m_map.begin(); iter != m_map.end(); ++iter)
      if (iter->first.CreatedBySameMatchString(matcher)) {
        m_map.erase(iter);
        if (listener)
          listener->Changed();
        return true;
      }
    return false;
  }

  bool Get(ConstString type, ValueSP &entry) {
    std::lock_guard<std::recursive_mutex> guard(m_map_mutex);
    for (auto &formatter : llvm::reverse(m_map)) {
      if (formatter.first.Matches(type)) {
        entry = formatter.second;
        return true;
      }
    }
    return false;
  }

  bool GetExact(TypeMatcher matcher, ValueSP &entry) {
    std::lock_guard<std::recursive_mutex> guard(m_map_mutex);
    for (const auto &pos : m_map)
      if (pos.first.CreatedBySameMatchString(matcher)) {
        entry = pos.second;
        return true;
      }
    return false;
  }

  ValueSP GetAtIndex(size_t index) {
    std::lock_guard<std::recursive_mutex> guard(m_map_mutex);
    if (index >= m_map.size())
      return ValueSP();
    return m_map[index].second;
  }

  lldb::TypeNameSpecifierImplSP GetTypeNameSpecifierAtIndex(size_t index) {
    std::lock_guard<std::recursive_mutex> guard(m_map_mutex);
    if (index >= m_map.size())
      return lldb::TypeNameSpecifierImplSP();
    TypeMatcher type_matcher = m_map[index].first;
    return std::make_shared<TypeNameSpecifierImpl>(
        type_matcher.GetMatchString().GetStringRef(), true);
  }

  void Clear() {
    std::lock_guard<std::recursive_mutex> guard(m_map_mutex);
    m_map.clear();
    if (listener)
      listener->Changed();
  }

  void ForEach(ForEachCallback callback) {
    if (callback) {
      std::lock_guard<std::recursive_mutex> guard(m_map_mutex);
      for (const auto &pos : m_map) {
        const TypeMatcher &type = pos.first;
        if (!callback(type, pos.second))
          break;
      }
    }
  }

  uint32_t GetCount() {
    std::lock_guard<std::recursive_mutex> guard(m_map_mutex);
    return m_map.size();
  }

protected:
  FormattersContainer(const FormattersContainer &) = delete;
  const FormattersContainer &operator=(const FormattersContainer &) = delete;

  bool Get(const FormattersMatchVector &candidates, ValueSP &entry) {
    for (const FormattersMatchCandidate &candidate : candidates) {
      if (Get(candidate.GetTypeName(), entry)) {
        if (candidate.IsMatch(entry) == false) {
          entry.reset();
          continue;
        } else {
          return true;
        }
      }
    }
    return false;
  }

  MapType m_map;
  std::recursive_mutex m_map_mutex;
  IFormatChangeListener *listener;
};

} // namespace lldb_private

#endif // LLDB_DATAFORMATTERS_FORMATTERSCONTAINER_H
