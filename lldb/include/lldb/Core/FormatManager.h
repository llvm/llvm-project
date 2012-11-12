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

#include "lldb/Core/FormatNavigator.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"

using lldb::LogSP;

namespace lldb_private {
    
// this file (and its. cpp) contain the low-level implementation of LLDB Data Visualization
// class DataVisualization is the high-level front-end of this feature
// clients should refer to that class as the entry-point into the data formatters
// unless they have a good reason to bypass it and prefer to use this file's objects directly
    
class CategoryMap;
    
class TypeCategoryImpl
{
private:
    
    typedef FormatNavigator<ConstString, TypeSummaryImpl> SummaryNavigator;
    typedef FormatNavigator<lldb::RegularExpressionSP, TypeSummaryImpl> RegexSummaryNavigator;
    
    typedef FormatNavigator<ConstString, TypeFilterImpl> FilterNavigator;
    typedef FormatNavigator<lldb::RegularExpressionSP, TypeFilterImpl> RegexFilterNavigator;
    
#ifndef LLDB_DISABLE_PYTHON
    typedef FormatNavigator<ConstString, TypeSyntheticImpl> SynthNavigator;
    typedef FormatNavigator<lldb::RegularExpressionSP, TypeSyntheticImpl> RegexSynthNavigator;
#endif // #ifndef LLDB_DISABLE_PYTHON

    typedef SummaryNavigator::MapType SummaryMap;
    typedef RegexSummaryNavigator::MapType RegexSummaryMap;
    typedef FilterNavigator::MapType FilterMap;
    typedef RegexFilterNavigator::MapType RegexFilterMap;
#ifndef LLDB_DISABLE_PYTHON
    typedef SynthNavigator::MapType SynthMap;
    typedef RegexSynthNavigator::MapType RegexSynthMap;
#endif // #ifndef LLDB_DISABLE_PYTHON

public:
        
    typedef uint16_t FormatCategoryItems;
    static const uint16_t ALL_ITEM_TYPES = UINT16_MAX;
    
    typedef SummaryNavigator::SharedPointer SummaryNavigatorSP;
    typedef RegexSummaryNavigator::SharedPointer RegexSummaryNavigatorSP;
    typedef FilterNavigator::SharedPointer FilterNavigatorSP;
    typedef RegexFilterNavigator::SharedPointer RegexFilterNavigatorSP;
#ifndef LLDB_DISABLE_PYTHON
    typedef SynthNavigator::SharedPointer SynthNavigatorSP;
    typedef RegexSynthNavigator::SharedPointer RegexSynthNavigatorSP;
#endif // #ifndef LLDB_DISABLE_PYTHON

    TypeCategoryImpl (IFormatChangeListener* clist,
                      ConstString name);
    
    SummaryNavigatorSP
    GetSummaryNavigator ()
    {
        return SummaryNavigatorSP(m_summary_nav);
    }
    
    RegexSummaryNavigatorSP
    GetRegexSummaryNavigator ()
    {
        return RegexSummaryNavigatorSP(m_regex_summary_nav);
    }
    
    FilterNavigatorSP
    GetFilterNavigator ()
    {
        return FilterNavigatorSP(m_filter_nav);
    }
    
    RegexFilterNavigatorSP
    GetRegexFilterNavigator ()
    {
        return RegexFilterNavigatorSP(m_regex_filter_nav);
    }
    
    SummaryNavigator::MapValueType
    GetSummaryForType (lldb::TypeNameSpecifierImplSP type_sp)
    {
        SummaryNavigator::MapValueType retval;
        
        if (type_sp)
        {
            if (type_sp->IsRegex())
                m_regex_summary_nav->GetExact(ConstString(type_sp->GetName()),retval);
            else
                m_summary_nav->GetExact(ConstString(type_sp->GetName()),retval);
        }

        return retval;
    }
    
    FilterNavigator::MapValueType
    GetFilterForType (lldb::TypeNameSpecifierImplSP type_sp)
    {
        FilterNavigator::MapValueType retval;
        
        if (type_sp)
        {
            if (type_sp->IsRegex())
                m_regex_filter_nav->GetExact(ConstString(type_sp->GetName()),retval);
            else
                m_filter_nav->GetExact(ConstString(type_sp->GetName()),retval);
        }
        
        return retval;
    }
    
#ifndef LLDB_DISABLE_PYTHON
    SynthNavigator::MapValueType
    GetSyntheticForType (lldb::TypeNameSpecifierImplSP type_sp)
    {
        SynthNavigator::MapValueType retval;
        
        if (type_sp)
        {
            if (type_sp->IsRegex())
                m_regex_synth_nav->GetExact(ConstString(type_sp->GetName()),retval);
            else
                m_synth_nav->GetExact(ConstString(type_sp->GetName()),retval);
        }
        
        return retval;
    }
#endif
    
    lldb::TypeNameSpecifierImplSP
    GetTypeNameSpecifierForSummaryAtIndex (uint32_t index)
    {
        if (index < m_summary_nav->GetCount())
            return m_summary_nav->GetTypeNameSpecifierAtIndex(index);
        else
            return m_regex_summary_nav->GetTypeNameSpecifierAtIndex(index-m_summary_nav->GetCount());
    }
    
    SummaryNavigator::MapValueType
    GetSummaryAtIndex (uint32_t index)
    {
        if (index < m_summary_nav->GetCount())
            return m_summary_nav->GetAtIndex(index);
        else
            return m_regex_summary_nav->GetAtIndex(index-m_summary_nav->GetCount());
    }

    FilterNavigator::MapValueType
    GetFilterAtIndex (uint32_t index)
    {
        if (index < m_filter_nav->GetCount())
            return m_filter_nav->GetAtIndex(index);
        else
            return m_regex_filter_nav->GetAtIndex(index-m_filter_nav->GetCount());
    }
    
    lldb::TypeNameSpecifierImplSP
    GetTypeNameSpecifierForFilterAtIndex (uint32_t index)
    {
        if (index < m_filter_nav->GetCount())
            return m_filter_nav->GetTypeNameSpecifierAtIndex(index);
        else
            return m_regex_filter_nav->GetTypeNameSpecifierAtIndex(index-m_filter_nav->GetCount());
    }

#ifndef LLDB_DISABLE_PYTHON
    SynthNavigatorSP
    GetSyntheticNavigator ()
    {
        return SynthNavigatorSP(m_synth_nav);
    }
    
    RegexSynthNavigatorSP
    GetRegexSyntheticNavigator ()
    {
        return RegexSynthNavigatorSP(m_regex_synth_nav);
    }
    
    SynthNavigator::MapValueType
    GetSyntheticAtIndex (uint32_t index)
    {
        if (index < m_synth_nav->GetCount())
            return m_synth_nav->GetAtIndex(index);
        else
            return m_regex_synth_nav->GetAtIndex(index-m_synth_nav->GetCount());
    }
    
    lldb::TypeNameSpecifierImplSP
    GetTypeNameSpecifierForSyntheticAtIndex (uint32_t index)
    {
        if (index < m_synth_nav->GetCount())
            return m_synth_nav->GetTypeNameSpecifierAtIndex(index);
        else
            return m_regex_synth_nav->GetTypeNameSpecifierAtIndex(index - m_synth_nav->GetCount());
    }
    
#endif // #ifndef LLDB_DISABLE_PYTHON

    bool
    IsEnabled () const
    {
        return m_enabled;
    }
    
    uint32_t
    GetEnabledPosition()
    {
        if (m_enabled == false)
            return UINT32_MAX;
        else
            return m_enabled_position;
    }
    
    bool
    Get (ValueObject& valobj,
         lldb::TypeSummaryImplSP& entry,
         lldb::DynamicValueType use_dynamic,
         uint32_t* reason = NULL);
    
    bool
    Get (ValueObject& valobj,
         lldb::SyntheticChildrenSP& entry,
         lldb::DynamicValueType use_dynamic,
         uint32_t* reason = NULL);
        
    void
    Clear (FormatCategoryItems items = ALL_ITEM_TYPES);
    
    bool
    Delete (ConstString name,
            FormatCategoryItems items = ALL_ITEM_TYPES);
    
    uint32_t
    GetCount (FormatCategoryItems items = ALL_ITEM_TYPES);
    
    const char*
    GetName ()
    {
        return m_name.GetCString();
    }
    
    bool
    AnyMatches (ConstString type_name,
                FormatCategoryItems items = ALL_ITEM_TYPES,
                bool only_enabled = true,
                const char** matching_category = NULL,
                FormatCategoryItems* matching_type = NULL);
    
    typedef STD_SHARED_PTR(TypeCategoryImpl) SharedPointer;
    
private:
    SummaryNavigator::SharedPointer m_summary_nav;
    RegexSummaryNavigator::SharedPointer m_regex_summary_nav;
    FilterNavigator::SharedPointer m_filter_nav;
    RegexFilterNavigator::SharedPointer m_regex_filter_nav;
#ifndef LLDB_DISABLE_PYTHON
    SynthNavigator::SharedPointer m_synth_nav;
    RegexSynthNavigator::SharedPointer m_regex_synth_nav;
#endif // #ifndef LLDB_DISABLE_PYTHON
    
    bool m_enabled;
    
    IFormatChangeListener* m_change_listener;
    
    Mutex m_mutex;
    
    ConstString m_name;
    
    uint32_t m_enabled_position;
    
    void
    Enable (bool value,
            uint32_t position)
    {
        Mutex::Locker locker(m_mutex);
        m_enabled = value;
        m_enabled_position = position;
        if (m_change_listener)
            m_change_listener->Changed();
    }
    
    void
    Disable ()
    {
        Enable(false, UINT32_MAX);
    }
    
    friend class CategoryMap;
    
    friend class FormatNavigator<ConstString, TypeSummaryImpl>;
    friend class FormatNavigator<lldb::RegularExpressionSP, TypeSummaryImpl>;
    
    friend class FormatNavigator<ConstString, TypeFilterImpl>;
    friend class FormatNavigator<lldb::RegularExpressionSP, TypeFilterImpl>;
    
#ifndef LLDB_DISABLE_PYTHON
    friend class FormatNavigator<ConstString, TypeSyntheticImpl>;
    friend class FormatNavigator<lldb::RegularExpressionSP, TypeSyntheticImpl>;
#endif // #ifndef LLDB_DISABLE_PYTHON
    

};

class CategoryMap
{
private:
    typedef ConstString KeyType;
    typedef TypeCategoryImpl ValueType;
    typedef ValueType::SharedPointer ValueSP;
    typedef std::list<lldb::TypeCategoryImplSP> ActiveCategoriesList;
    typedef ActiveCategoriesList::iterator ActiveCategoriesIterator;
        
public:
    typedef std::map<KeyType, ValueSP> MapType;
    typedef MapType::iterator MapIterator;
    typedef bool(*CallbackType)(void*, const ValueSP&);
    typedef uint32_t Position;
    
    static const Position First = 0;
    static const Position Default = 1;
    static const Position Last = UINT32_MAX;
    
    CategoryMap (IFormatChangeListener* lst) :
        m_map_mutex(Mutex::eMutexTypeRecursive),
        listener(lst),
        m_map(),
        m_active_categories()
    {
        ConstString default_cs("default");
        lldb::TypeCategoryImplSP default_sp = lldb::TypeCategoryImplSP(new TypeCategoryImpl(listener, default_cs));
        Add(default_cs,default_sp);
        Enable(default_cs,First);
    }
    
    void
    Add (KeyType name,
         const ValueSP& entry)
    {
        Mutex::Locker locker(m_map_mutex);
        m_map[name] = entry;
        if (listener)
            listener->Changed();
    }
    
    bool
    Delete (KeyType name)
    {
        Mutex::Locker locker(m_map_mutex);
        MapIterator iter = m_map.find(name);
        if (iter == m_map.end())
            return false;
        m_map.erase(name);
        Disable(name);
        if (listener)
            listener->Changed();
        return true;
    }
    
    bool
    Enable (KeyType category_name,
            Position pos = Default)
    {
        Mutex::Locker locker(m_map_mutex);
        ValueSP category;
        if (!Get(category_name,category))
            return false;
        return Enable(category, pos);
    }
    
    bool
    Disable (KeyType category_name)
    {
        Mutex::Locker locker(m_map_mutex);
        ValueSP category;
        if (!Get(category_name,category))
            return false;
        return Disable(category);
    }
    
    bool
    Enable (ValueSP category,
            Position pos = Default)
    {
        Mutex::Locker locker(m_map_mutex);
        if (category.get())
        {
            Position pos_w = pos;
            if (pos == First || m_active_categories.size() == 0)
                m_active_categories.push_front(category);
            else if (pos == Last || pos == m_active_categories.size())
                m_active_categories.push_back(category);
            else if (pos < m_active_categories.size())
            {
                ActiveCategoriesList::iterator iter = m_active_categories.begin();
                while (pos_w)
                {
                    pos_w--,iter++;
                }
                m_active_categories.insert(iter,category);
            }
            else
                return false;
            category->Enable(true,
                             pos);
            return true;
        }
        return false;
    }
    
    bool
    Disable (ValueSP category)
    {
        Mutex::Locker locker(m_map_mutex);
        if (category.get())
        {
            m_active_categories.remove_if(delete_matching_categories(category));
            category->Disable();
            return true;
        }
        return false;
    }
    
    void
    Clear ()
    {
        Mutex::Locker locker(m_map_mutex);
        m_map.clear();
        m_active_categories.clear();
        if (listener)
            listener->Changed();
    }
    
    bool
    Get (KeyType name,
         ValueSP& entry)
    {
        Mutex::Locker locker(m_map_mutex);
        MapIterator iter = m_map.find(name);
        if (iter == m_map.end())
            return false;
        entry = iter->second;
        return true;
    }
    
    bool
    Get (uint32_t pos,
         ValueSP& entry)
    {
        Mutex::Locker locker(m_map_mutex);
        MapIterator iter = m_map.begin();
        MapIterator end = m_map.end();
        while (pos > 0)
        {
            iter++;
            pos--;
            if (iter == end)
                return false;
        }
        entry = iter->second;
        return false;
    }
    
    void
    LoopThrough (CallbackType callback, void* param);
    
    lldb::TypeCategoryImplSP
    GetAtIndex (uint32_t);
    
    bool
    AnyMatches (ConstString type_name,
                TypeCategoryImpl::FormatCategoryItems items = TypeCategoryImpl::ALL_ITEM_TYPES,
                bool only_enabled = true,
                const char** matching_category = NULL,
                TypeCategoryImpl::FormatCategoryItems* matching_type = NULL);
    
    uint32_t
    GetCount ()
    {
        return m_map.size();
    }
    
    lldb::TypeSummaryImplSP
    GetSummaryFormat (ValueObject& valobj,
         lldb::DynamicValueType use_dynamic);
    
#ifndef LLDB_DISABLE_PYTHON
    lldb::SyntheticChildrenSP
    GetSyntheticChildren (ValueObject& valobj,
                          lldb::DynamicValueType use_dynamic);
#endif
    
private:
    
    class delete_matching_categories
    {
        lldb::TypeCategoryImplSP ptr;
    public:
        delete_matching_categories(lldb::TypeCategoryImplSP p) : ptr(p)
        {}
        
        bool operator()(const lldb::TypeCategoryImplSP& other)
        {
            return ptr.get() == other.get();
        }
    };
    
    Mutex m_map_mutex;
    IFormatChangeListener* listener;
    
    MapType m_map;
    ActiveCategoriesList m_active_categories;
    
    MapType& map ()
    {
        return m_map;
    }
    
    ActiveCategoriesList& active_list ()
    {
        return m_active_categories;
    }
    
    Mutex& mutex ()
    {
        return m_map_mutex;
    }
    
    friend class FormatNavigator<KeyType, ValueType>;
    friend class FormatManager;
};

class FormatManager : public IFormatChangeListener
{
    typedef FormatNavigator<ConstString, TypeFormatImpl> ValueNavigator;
    typedef ValueNavigator::MapType ValueMap;
    typedef FormatMap<ConstString, TypeSummaryImpl> NamedSummariesMap;
    typedef CategoryMap::MapType::iterator CategoryMapIterator;
public:
    
    typedef CategoryMap::CallbackType CategoryCallback;
    
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
                    CategoryMap::Position pos = CategoryMap::Default)
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
                    CategoryMap::Position pos = CategoryMap::Default)
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
    GetCategoryAtIndex (uint32_t index)
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
    GetSummaryFormat (ValueObject& valobj,
                      lldb::DynamicValueType use_dynamic)
    {
        return m_categories_map.GetSummaryFormat(valobj, use_dynamic);
    }
    
    lldb::TypeSummaryImplSP
    GetSummaryForType (lldb::TypeNameSpecifierImplSP type_sp);

    lldb::TypeFilterImplSP
    GetFilterForType (lldb::TypeNameSpecifierImplSP type_sp);

#ifndef LLDB_DISABLE_PYTHON
    lldb::TypeSyntheticImplSP
    GetSyntheticForType (lldb::TypeNameSpecifierImplSP type_sp);
#endif
    
#ifndef LLDB_DISABLE_PYTHON
    lldb::SyntheticChildrenSP
    GetSyntheticChildrenForType (lldb::TypeNameSpecifierImplSP type_sp);
#endif
    
#ifndef LLDB_DISABLE_PYTHON
    lldb::SyntheticChildrenSP
    GetSyntheticChildren (ValueObject& valobj,
                          lldb::DynamicValueType use_dynamic)
    {
        return m_categories_map.GetSyntheticChildren(valobj, use_dynamic);
    }
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
    ValueNavigator m_value_nav;
    NamedSummariesMap m_named_summaries_map;
    uint32_t m_last_revision;
    CategoryMap m_categories_map;
    
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
    
    CategoryMap&
    GetCategories ()
    {
        return m_categories_map;
    }
    
    // WARNING: these are temporary functions that setup formatters
    // while a few of these actually should be globally available and setup by LLDB itself
    // most would actually belong to the users' lldbinit file or to some other form of configurable
    // storage
    void
    LoadSTLFormatters();
    
    void
    LoadLibcxxFormatters();
    
    void
    LoadSystemFormatters();
    
    void
    LoadObjCFormatters();
};
    
} // namespace lldb_private

#endif	// lldb_FormatManager_h_
