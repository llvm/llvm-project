//===-- FormatManager.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_FormatManager_h_
#define lldb_FormatManager_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/DataFormatters/FormatCache.h"
#include "lldb/DataFormatters/FormatNavigator.h"
#include "lldb/DataFormatters/TypeCategory.h"
#include "lldb/DataFormatters/TypeCategoryMap.h"

namespace lldb_private {
    
// this file (and its. cpp) contain the low-level implementation of LLDB Data Visualization
// class DataVisualization is the high-level front-end of this feature
// clients should refer to that class as the entry-point into the data formatters
// unless they have a good reason to bypass it and prefer to use this file's objects directly

class FormatManager : public IFormatChangeListener
{
    typedef FormatNavigator<ConstString, TypeFormatImpl> ValueNavigator;
    typedef ValueNavigator::MapType ValueMap;
    typedef FormatMap<ConstString, TypeSummaryImpl> NamedSummariesMap;
    typedef TypeCategoryMap::MapType::iterator CategoryMapIterator;
public:
    
    typedef TypeCategoryMap::CallbackType CategoryCallback;
    
    FormatManager ();
    
    ValueNavigator&
    GetValueNavigator ()
    {
        return m_value_nav;
    }
    
    NamedSummariesMap&
    GetNamedSummaryNavigator ()
    {
        return m_named_summaries_map;
    }
    
    void
    EnableCategory (const ConstString& category_name,
                    TypeCategoryMap::Position pos = TypeCategoryMap::Default)
    {
        m_categories_map.Enable(category_name,
                                pos);
    }
    
    void
    DisableCategory (const ConstString& category_name)
    {
        m_categories_map.Disable(category_name);
    }
    
    void
    EnableCategory (const lldb::TypeCategoryImplSP& category,
                    TypeCategoryMap::Position pos = TypeCategoryMap::Default)
    {
        m_categories_map.Enable(category,
                                pos);
    }
    
    void
    DisableCategory (const lldb::TypeCategoryImplSP& category)
    {
        m_categories_map.Disable(category);
    }
    
    bool
    DeleteCategory (const ConstString& category_name)
    {
        return m_categories_map.Delete(category_name);
    }
    
    void
    ClearCategories ()
    {
        return m_categories_map.Clear();
    }
    
    uint32_t
    GetCategoriesCount ()
    {
        return m_categories_map.GetCount();
    }
    
    lldb::TypeCategoryImplSP
    GetCategoryAtIndex (size_t index)
    {
        return m_categories_map.GetAtIndex(index);
    }
    
    void
    LoopThroughCategories (CategoryCallback callback, void* param)
    {
        m_categories_map.LoopThrough(callback, param);
    }
    
    lldb::TypeCategoryImplSP
    GetCategory (const char* category_name = NULL,
                 bool can_create = true)
    {
        if (!category_name)
            return GetCategory(m_default_category_name);
        return GetCategory(ConstString(category_name));
    }
    
    lldb::TypeCategoryImplSP
    GetCategory (const ConstString& category_name,
                 bool can_create = true);
    
    lldb::TypeSummaryImplSP
    GetSummaryForType (lldb::TypeNameSpecifierImplSP type_sp);

    lldb::TypeFilterImplSP
    GetFilterForType (lldb::TypeNameSpecifierImplSP type_sp);

#ifndef LLDB_DISABLE_PYTHON
    lldb::ScriptedSyntheticChildrenSP
    GetSyntheticForType (lldb::TypeNameSpecifierImplSP type_sp);
#endif
    
#ifndef LLDB_DISABLE_PYTHON
    lldb::SyntheticChildrenSP
    GetSyntheticChildrenForType (lldb::TypeNameSpecifierImplSP type_sp);
#endif
    
    lldb::TypeSummaryImplSP
    GetSummaryFormat (ValueObject& valobj,
                      lldb::DynamicValueType use_dynamic);

#ifndef LLDB_DISABLE_PYTHON
    lldb::SyntheticChildrenSP
    GetSyntheticChildren (ValueObject& valobj,
                          lldb::DynamicValueType use_dynamic);
#endif
    
    bool
    AnyMatches (ConstString type_name,
                TypeCategoryImpl::FormatCategoryItems items = TypeCategoryImpl::ALL_ITEM_TYPES,
                bool only_enabled = true,
                const char** matching_category = NULL,
                TypeCategoryImpl::FormatCategoryItems* matching_type = NULL)
    {
        return m_categories_map.AnyMatches(type_name,
                                           items,
                                           only_enabled,
                                           matching_category,
                                           matching_type);
    }

    static bool
    GetFormatFromCString (const char *format_cstr,
                          bool partial_match_ok,
                          lldb::Format &format);

    static char
    GetFormatAsFormatChar (lldb::Format format);

    static const char *
    GetFormatAsCString (lldb::Format format);
    
    // if the user tries to add formatters for, say, "struct Foo"
    // those will not match any type because of the way we strip qualifiers from typenames
    // this method looks for the case where the user is adding a "class","struct","enum" or "union" Foo
    // and strips the unnecessary qualifier
    static ConstString
    GetValidTypeName (const ConstString& type);
    
    // when DataExtractor dumps a vectorOfT, it uses a predefined format for each item
    // this method returns it, or eFormatInvalid if vector_format is not a vectorOf
    static lldb::Format
    GetSingleItemFormat (lldb::Format vector_format);
    
    void
    Changed ()
    {
        __sync_add_and_fetch(&m_last_revision, +1);
        m_format_cache.Clear ();
    }
    
    uint32_t
    GetCurrentRevision ()
    {
        return m_last_revision;
    }
    
    ~FormatManager ()
    {
    }
    
private:
    FormatCache m_format_cache;
    ValueNavigator m_value_nav;
    NamedSummariesMap m_named_summaries_map;
    uint32_t m_last_revision;
    TypeCategoryMap m_categories_map;
    
    ConstString m_default_category_name;
    ConstString m_system_category_name;
    ConstString m_gnu_cpp_category_name;
    ConstString m_libcxx_category_name;
    ConstString m_objc_category_name;
    ConstString m_corefoundation_category_name;
    ConstString m_coregraphics_category_name;
    ConstString m_coreservices_category_name;
    ConstString m_vectortypes_category_name;
    ConstString m_appkit_category_name;
    
    TypeCategoryMap&
    GetCategories ()
    {
        return m_categories_map;
    }
    
    // WARNING: these are temporary functions that setup formatters
    // while a few of these actually should be globally available and setup by LLDB itself
    // most would actually belong to the users' lldbinit file or to some other form of configurable
    // storage
    void
    LoadLibStdcppFormatters ();
    
    void
    LoadLibcxxFormatters ();
    
    void
    LoadSystemFormatters ();
    
    void
    LoadObjCFormatters ();
};
    
} // namespace lldb_private

#endif	// lldb_FormatManager_h_
