//===-- TypeCategoryMap.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_DATAFORMATTERS_TYPECATEGORYMAP_H
#define LLDB_DATAFORMATTERS_TYPECATEGORYMAP_H

#include <functional>
#include <list>
#include <map>
#include <mutex>

#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-public.h"

#include "lldb/DataFormatters/FormattersContainer.h"
#include "lldb/DataFormatters/TypeCategory.h"

namespace lldb_private {
class TypeCategoryMap {
private:
  typedef ConstString KeyType;
  typedef TypeCategoryImpl ValueType;
  typedef ValueType::SharedPointer ValueSP;
  typedef std::list<lldb::TypeCategoryImplSP> ActiveCategoriesList;
  typedef ActiveCategoriesList::iterator ActiveCategoriesIterator;

public:
  typedef std::map<KeyType, ValueSP> MapType;
  typedef MapType::iterator MapIterator;
  typedef std::function<bool(const ValueSP &)> ForEachCallback;

  typedef uint32_t Position;

  static const Position First = 0;
  static const Position Default = 1;
  static const Position Last = UINT32_MAX;

  TypeCategoryMap(IFormatChangeListener *lst);

  void Add(KeyType name, const ValueSP &entry);

  bool Delete(KeyType name);

  bool Enable(KeyType category_name, Position pos = Default);

  bool Disable(KeyType category_name);

  bool Enable(ValueSP category, Position pos = Default);

  bool Disable(ValueSP category);

  void EnableAllCategories();

  void DisableAllCategories();

  void Clear();

  bool Get(KeyType name, ValueSP &entry);

  bool Get(uint32_t pos, ValueSP &entry);

  void ForEach(ForEachCallback callback);

  lldb::TypeCategoryImplSP GetAtIndex(uint32_t);

  bool
  AnyMatches(ConstString type_name,
             TypeCategoryImpl::FormatCategoryItems items =
                 TypeCategoryImpl::ALL_ITEM_TYPES,
             bool only_enabled = true, const char **matching_category = nullptr,
             TypeCategoryImpl::FormatCategoryItems *matching_type = nullptr);

  uint32_t GetCount() { return m_map.size(); }

  template <typename ImplSP> void Get(FormattersMatchData &, ImplSP &);

private:
  class delete_matching_categories {
    lldb::TypeCategoryImplSP ptr;

  public:
    delete_matching_categories(lldb::TypeCategoryImplSP p) : ptr(p) {}

    bool operator()(const lldb::TypeCategoryImplSP &other) {
      return ptr.get() == other.get();
    }
  };

  std::recursive_mutex m_map_mutex;
  IFormatChangeListener *listener;

  MapType m_map;
  ActiveCategoriesList m_active_categories;

  MapType &map() { return m_map; }

  ActiveCategoriesList &active_list() { return m_active_categories; }

  std::recursive_mutex &mutex() { return m_map_mutex; }

  friend class FormattersContainer<KeyType, ValueType>;
  friend class FormatManager;
};
} // namespace lldb_private

#endif // LLDB_DATAFORMATTERS_TYPECATEGORYMAP_H
