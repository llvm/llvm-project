//===-- FormatManager.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/FormatManager.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/Core/CXXFormatterFunctions.h"
#include "lldb/Core/Debugger.h"

using namespace lldb;
using namespace lldb_private;


struct FormatInfo
{
    Format format;
    const char format_char; // One or more format characters that can be used for this format.
    const char *format_name;    // Long format name that can be used to specify the current format
};

static FormatInfo 
g_format_infos[] = 
{
    { eFormatDefault        , '\0'  , "default"             },
    { eFormatBoolean        , 'B'   , "boolean"             },
    { eFormatBinary         , 'b'   , "binary"              },
    { eFormatBytes          , 'y'   , "bytes"               },
    { eFormatBytesWithASCII , 'Y'   , "bytes with ASCII"    },
    { eFormatChar           , 'c'   , "character"           },
    { eFormatCharPrintable  , 'C'   , "printable character" },
    { eFormatComplexFloat   , 'F'   , "complex float"       },
    { eFormatCString        , 's'   , "c-string"            },
    { eFormatDecimal        , 'd'   , "decimal"             },
    { eFormatEnum           , 'E'   , "enumeration"         },
    { eFormatHex            , 'x'   , "hex"                 },
    { eFormatHexUppercase   , 'X'   , "uppercase hex"       },
    { eFormatFloat          , 'f'   , "float"               },
    { eFormatOctal          , 'o'   , "octal"               },
    { eFormatOSType         , 'O'   , "OSType"              },
    { eFormatUnicode16      , 'U'   , "unicode16"           },
    { eFormatUnicode32      , '\0'  , "unicode32"           },
    { eFormatUnsigned       , 'u'   , "unsigned decimal"    },
    { eFormatPointer        , 'p'   , "pointer"             },
    { eFormatVectorOfChar   , '\0'  , "char[]"              },
    { eFormatVectorOfSInt8  , '\0'  , "int8_t[]"            },
    { eFormatVectorOfUInt8  , '\0'  , "uint8_t[]"           },
    { eFormatVectorOfSInt16 , '\0'  , "int16_t[]"           },
    { eFormatVectorOfUInt16 , '\0'  , "uint16_t[]"          },
    { eFormatVectorOfSInt32 , '\0'  , "int32_t[]"           },
    { eFormatVectorOfUInt32 , '\0'  , "uint32_t[]"          },
    { eFormatVectorOfSInt64 , '\0'  , "int64_t[]"           },
    { eFormatVectorOfUInt64 , '\0'  , "uint64_t[]"          },
    { eFormatVectorOfFloat32, '\0'  , "float32[]"           },
    { eFormatVectorOfFloat64, '\0'  , "float64[]"           },
    { eFormatVectorOfUInt128, '\0'  , "uint128_t[]"         },
    { eFormatComplexInteger , 'I'   , "complex integer"     },
    { eFormatCharArray      , 'a'   , "character array"     },
    { eFormatAddressInfo    , 'A'   , "address"             },
    { eFormatHexFloat       , '\0'  , "hex float"           },
    { eFormatInstruction    , 'i'   , "instruction"         },
    { eFormatVoid           , 'v'   , "void"                }
};

static uint32_t 
g_num_format_infos = sizeof(g_format_infos)/sizeof(FormatInfo);

static bool
GetFormatFromFormatChar (char format_char, Format &format)
{
    for (uint32_t i=0; i<g_num_format_infos; ++i)
    {
        if (g_format_infos[i].format_char == format_char)
        {
            format = g_format_infos[i].format;
            return true;
        }
    }
    format = eFormatInvalid;
    return false;
}

static bool
GetFormatFromFormatName (const char *format_name, bool partial_match_ok, Format &format)
{
    uint32_t i;
    for (i=0; i<g_num_format_infos; ++i)
    {
        if (strcasecmp (g_format_infos[i].format_name, format_name) == 0)
        {
            format = g_format_infos[i].format;
            return true;
        }
    }
    
    if (partial_match_ok)
    {
        for (i=0; i<g_num_format_infos; ++i)
        {
            if (strcasestr (g_format_infos[i].format_name, format_name) == g_format_infos[i].format_name)
            {
                format = g_format_infos[i].format;
                return true;
            }
        }
    }
    format = eFormatInvalid;
    return false;
}

bool
FormatManager::GetFormatFromCString (const char *format_cstr,
                                     bool partial_match_ok,
                                     lldb::Format &format)
{
    bool success = false;
    if (format_cstr && format_cstr[0])
    {
        if (format_cstr[1] == '\0')
        {
            success = GetFormatFromFormatChar (format_cstr[0], format);
            if (success)
                return true;
        }
        
        success = GetFormatFromFormatName (format_cstr, partial_match_ok, format);
    }
    if (!success)
        format = eFormatInvalid;
    return success;
}

char
FormatManager::GetFormatAsFormatChar (lldb::Format format)
{
    for (uint32_t i=0; i<g_num_format_infos; ++i)
    {
        if (g_format_infos[i].format == format)
            return g_format_infos[i].format_char;
    }
    return '\0';
}
    


const char *
FormatManager::GetFormatAsCString (Format format)
{
    if (format >= eFormatDefault && format < kNumFormats)
        return g_format_infos[format].format_name;
    return NULL;
}

TypeCategoryImpl::TypeCategoryImpl(IFormatChangeListener* clist,
                                   ConstString name) :
    m_summary_nav(new SummaryNavigator("summary",clist)),
    m_regex_summary_nav(new RegexSummaryNavigator("regex-summary",clist)),
    m_filter_nav(new FilterNavigator("filter",clist)),
    m_regex_filter_nav(new RegexFilterNavigator("regex-filter",clist)),
#ifndef LLDB_DISABLE_PYTHON
    m_synth_nav(new SynthNavigator("synth",clist)),
    m_regex_synth_nav(new RegexSynthNavigator("regex-synth",clist)),
#endif
    m_enabled(false),
    m_change_listener(clist),
    m_mutex(Mutex::eMutexTypeRecursive),
    m_name(name)
{}

bool
TypeCategoryImpl::Get (ValueObject& valobj,
                     lldb::TypeSummaryImplSP& entry,
                     lldb::DynamicValueType use_dynamic,
                     uint32_t* reason)
{
    if (!IsEnabled())
        return false;
    if (GetSummaryNavigator()->Get(valobj, entry, use_dynamic, reason))
        return true;
    bool regex = GetRegexSummaryNavigator()->Get(valobj, entry, use_dynamic, reason);
    if (regex && reason)
        *reason |= lldb_private::eFormatterChoiceCriterionRegularExpressionSummary;
    return regex;
}

bool
TypeCategoryImpl::Get(ValueObject& valobj,
                    lldb::SyntheticChildrenSP& entry_sp,
                    lldb::DynamicValueType use_dynamic,
                    uint32_t* reason)
{
    if (!IsEnabled())
        return false;
    TypeFilterImpl::SharedPointer filter_sp;
    uint32_t reason_filter = 0;
    bool regex_filter = false;
    // first find both Filter and Synth, and then check which is most recent
    
    if (!GetFilterNavigator()->Get(valobj, filter_sp, use_dynamic, &reason_filter))
        regex_filter = GetRegexFilterNavigator()->Get (valobj, filter_sp, use_dynamic, &reason_filter);

#ifndef LLDB_DISABLE_PYTHON
    bool regex_synth = false;
    uint32_t reason_synth = 0;    
    bool pick_synth = false;
    TypeSyntheticImpl::SharedPointer synth;
    if (!GetSyntheticNavigator()->Get(valobj, synth, use_dynamic, &reason_synth))
        regex_synth = GetRegexSyntheticNavigator()->Get (valobj, synth, use_dynamic, &reason_synth);
    if (!filter_sp.get() && !synth.get())
        return false;
    else if (!filter_sp.get() && synth.get())
        pick_synth = true;
    
    else if (filter_sp.get() && !synth.get())
        pick_synth = false;
    
    else /*if (filter_sp.get() && synth.get())*/
    {
        if (filter_sp->GetRevision() > synth->GetRevision())
            pick_synth = false;
        else
            pick_synth = true;
    }
    if (pick_synth)
    {
        if (regex_synth && reason)
            *reason |= lldb_private::eFormatterChoiceCriterionRegularExpressionFilter;
        entry_sp = synth;
        return true;
    }
    else
    {
        if (regex_filter && reason)
            *reason |= lldb_private::eFormatterChoiceCriterionRegularExpressionFilter;
        entry_sp = filter_sp;
        return true;
    }

#else
    if (filter_sp)
    {
        entry_sp = filter_sp;
        return true;
    }
#endif

    return false;    
    
}

void
TypeCategoryImpl::Clear (FormatCategoryItems items)
{
    if ( (items & eFormatCategoryItemSummary) == eFormatCategoryItemSummary )
        m_summary_nav->Clear();
    if ( (items & eFormatCategoryItemRegexSummary) == eFormatCategoryItemRegexSummary )
        m_regex_summary_nav->Clear();
    if ( (items & eFormatCategoryItemFilter)  == eFormatCategoryItemFilter )
        m_filter_nav->Clear();
    if ( (items & eFormatCategoryItemRegexFilter) == eFormatCategoryItemRegexFilter )
        m_regex_filter_nav->Clear();
#ifndef LLDB_DISABLE_PYTHON
    if ( (items & eFormatCategoryItemSynth)  == eFormatCategoryItemSynth )
        m_synth_nav->Clear();
    if ( (items & eFormatCategoryItemRegexSynth) == eFormatCategoryItemRegexSynth )
        m_regex_synth_nav->Clear();
#endif
}

bool
TypeCategoryImpl::Delete (ConstString name,
                        FormatCategoryItems items)
{
    bool success = false;
    if ( (items & eFormatCategoryItemSummary) == eFormatCategoryItemSummary )
        success = m_summary_nav->Delete(name) || success;
    if ( (items & eFormatCategoryItemRegexSummary) == eFormatCategoryItemRegexSummary )
        success = m_regex_summary_nav->Delete(name) || success;
    if ( (items & eFormatCategoryItemFilter)  == eFormatCategoryItemFilter )
        success = m_filter_nav->Delete(name) || success;
    if ( (items & eFormatCategoryItemRegexFilter) == eFormatCategoryItemRegexFilter )
        success = m_regex_filter_nav->Delete(name) || success;
#ifndef LLDB_DISABLE_PYTHON
    if ( (items & eFormatCategoryItemSynth)  == eFormatCategoryItemSynth )
        success = m_synth_nav->Delete(name) || success;
    if ( (items & eFormatCategoryItemRegexSynth) == eFormatCategoryItemRegexSynth )
        success = m_regex_synth_nav->Delete(name) || success;
#endif
    return success;
}

uint32_t
TypeCategoryImpl::GetCount (FormatCategoryItems items)
{
    uint32_t count = 0;
    if ( (items & eFormatCategoryItemSummary) == eFormatCategoryItemSummary )
        count += m_summary_nav->GetCount();
    if ( (items & eFormatCategoryItemRegexSummary) == eFormatCategoryItemRegexSummary )
        count += m_regex_summary_nav->GetCount();
    if ( (items & eFormatCategoryItemFilter)  == eFormatCategoryItemFilter )
        count += m_filter_nav->GetCount();
    if ( (items & eFormatCategoryItemRegexFilter) == eFormatCategoryItemRegexFilter )
        count += m_regex_filter_nav->GetCount();
#ifndef LLDB_DISABLE_PYTHON
    if ( (items & eFormatCategoryItemSynth)  == eFormatCategoryItemSynth )
        count += m_synth_nav->GetCount();
    if ( (items & eFormatCategoryItemRegexSynth) == eFormatCategoryItemRegexSynth )
        count += m_regex_synth_nav->GetCount();
#endif
    return count;
}

bool
TypeCategoryImpl::AnyMatches(ConstString type_name,
                           FormatCategoryItems items,
                           bool only_enabled,
                           const char** matching_category,
                           FormatCategoryItems* matching_type)
{
    if (!IsEnabled() && only_enabled)
        return false;
    
    lldb::TypeSummaryImplSP summary;
    TypeFilterImpl::SharedPointer filter;
#ifndef LLDB_DISABLE_PYTHON
    TypeSyntheticImpl::SharedPointer synth;
#endif
    
    if ( (items & eFormatCategoryItemSummary) == eFormatCategoryItemSummary )
    {
        if (m_summary_nav->Get(type_name, summary))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemSummary;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemRegexSummary) == eFormatCategoryItemRegexSummary )
    {
        if (m_regex_summary_nav->Get(type_name, summary))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemRegexSummary;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemFilter)  == eFormatCategoryItemFilter )
    {
        if (m_filter_nav->Get(type_name, filter))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemFilter;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemRegexFilter) == eFormatCategoryItemRegexFilter )
    {
        if (m_regex_filter_nav->Get(type_name, filter))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemRegexFilter;
            return true;
        }
    }
#ifndef LLDB_DISABLE_PYTHON
    if ( (items & eFormatCategoryItemSynth)  == eFormatCategoryItemSynth )
    {
        if (m_synth_nav->Get(type_name, synth))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemSynth;
            return true;
        }
    }
    if ( (items & eFormatCategoryItemRegexSynth) == eFormatCategoryItemRegexSynth )
    {
        if (m_regex_synth_nav->Get(type_name, synth))
        {
            if (matching_category)
                *matching_category = m_name.GetCString();
            if (matching_type)
                *matching_type = eFormatCategoryItemRegexSynth;
            return true;
        }
    }
#endif
    return false;
}

bool
CategoryMap::AnyMatches (ConstString type_name,
                         TypeCategoryImpl::FormatCategoryItems items,
                         bool only_enabled,
                         const char** matching_category,
                         TypeCategoryImpl::FormatCategoryItems* matching_type)
{
    Mutex::Locker locker(m_map_mutex);
    
    MapIterator pos, end = m_map.end();
    for (pos = m_map.begin(); pos != end; pos++)
    {
        if (pos->second->AnyMatches(type_name,
                                    items,
                                    only_enabled,
                                    matching_category,
                                    matching_type))
            return true;
    }
    return false;
}

lldb::TypeSummaryImplSP
CategoryMap::GetSummaryFormat (ValueObject& valobj,
                               lldb::DynamicValueType use_dynamic)
{
    Mutex::Locker locker(m_map_mutex);
    
    uint32_t reason_why;        
    ActiveCategoriesIterator begin, end = m_active_categories.end();
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    
    for (begin = m_active_categories.begin(); begin != end; begin++)
    {
        lldb::TypeCategoryImplSP category_sp = *begin;
        lldb::TypeSummaryImplSP current_format;
        if (log)
            log->Printf("[CategoryMap::GetSummaryFormat] Trying to use category %s\n", category_sp->GetName());
        if (!category_sp->Get(valobj, current_format, use_dynamic, &reason_why))
            continue;
        return current_format;
    }
    if (log)
        log->Printf("[CategoryMap::GetSummaryFormat] nothing found - returning empty SP\n");
    return lldb::TypeSummaryImplSP();
}

lldb::TypeSummaryImplSP
FormatManager::GetSummaryForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    if (!type_sp)
        return lldb::TypeSummaryImplSP();
    lldb::TypeSummaryImplSP summary_chosen_sp;
    uint32_t num_categories = m_categories_map.GetCount();
    lldb::TypeCategoryImplSP category_sp;
    uint32_t prio_category = UINT32_MAX;
    for (uint32_t category_id = 0;
         category_id < num_categories;
         category_id++)
    {
        category_sp = GetCategoryAtIndex(category_id);
        if (category_sp->IsEnabled() == false)
            continue;
        lldb::TypeSummaryImplSP summary_current_sp = category_sp->GetSummaryForType(type_sp);
        if (summary_current_sp && (summary_chosen_sp.get() == NULL || (prio_category > category_sp->GetEnabledPosition())))
        {
            prio_category = category_sp->GetEnabledPosition();
            summary_chosen_sp = summary_current_sp;
        }
    }
    return summary_chosen_sp;
}

lldb::TypeFilterImplSP
FormatManager::GetFilterForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    if (!type_sp)
        return lldb::TypeFilterImplSP();
    lldb::TypeFilterImplSP filter_chosen_sp;
    uint32_t num_categories = m_categories_map.GetCount();
    lldb::TypeCategoryImplSP category_sp;
    uint32_t prio_category = UINT32_MAX;
    for (uint32_t category_id = 0;
         category_id < num_categories;
         category_id++)
    {
        category_sp = GetCategoryAtIndex(category_id);
        if (category_sp->IsEnabled() == false)
            continue;
        lldb::TypeFilterImplSP filter_current_sp((TypeFilterImpl*)category_sp->GetFilterForType(type_sp).get());
        if (filter_current_sp && (filter_chosen_sp.get() == NULL || (prio_category > category_sp->GetEnabledPosition())))
        {
            prio_category = category_sp->GetEnabledPosition();
            filter_chosen_sp = filter_current_sp;
        }
    }
    return filter_chosen_sp;
}

#ifndef LLDB_DISABLE_PYTHON
lldb::TypeSyntheticImplSP
FormatManager::GetSyntheticForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    if (!type_sp)
        return lldb::TypeSyntheticImplSP();
    lldb::TypeSyntheticImplSP synth_chosen_sp;
    uint32_t num_categories = m_categories_map.GetCount();
    lldb::TypeCategoryImplSP category_sp;
    uint32_t prio_category = UINT32_MAX;
    for (uint32_t category_id = 0;
         category_id < num_categories;
         category_id++)
    {
        category_sp = GetCategoryAtIndex(category_id);
        if (category_sp->IsEnabled() == false)
            continue;
        lldb::TypeSyntheticImplSP synth_current_sp((TypeSyntheticImpl*)category_sp->GetSyntheticForType(type_sp).get());
        if (synth_current_sp && (synth_chosen_sp.get() == NULL || (prio_category > category_sp->GetEnabledPosition())))
        {
            prio_category = category_sp->GetEnabledPosition();
            synth_chosen_sp = synth_current_sp;
        }
    }
    return synth_chosen_sp;
}
#endif

#ifndef LLDB_DISABLE_PYTHON
lldb::SyntheticChildrenSP
FormatManager::GetSyntheticChildrenForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    if (!type_sp)
        return lldb::SyntheticChildrenSP();
    lldb::TypeFilterImplSP filter_sp = GetFilterForType(type_sp);
    lldb::TypeSyntheticImplSP synth_sp = GetSyntheticForType(type_sp);
    if (filter_sp->GetRevision() > synth_sp->GetRevision())
        return lldb::SyntheticChildrenSP(filter_sp.get());
    else
        return lldb::SyntheticChildrenSP(synth_sp.get());
}
#endif

#ifndef LLDB_DISABLE_PYTHON
lldb::SyntheticChildrenSP
CategoryMap::GetSyntheticChildren (ValueObject& valobj,
                                   lldb::DynamicValueType use_dynamic)
{
    Mutex::Locker locker(m_map_mutex);
    
    uint32_t reason_why;
    
    ActiveCategoriesIterator begin, end = m_active_categories.end();
    
    LogSP log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    
    for (begin = m_active_categories.begin(); begin != end; begin++)
    {
        lldb::TypeCategoryImplSP category_sp = *begin;
        lldb::SyntheticChildrenSP current_format;
        if (log)
            log->Printf("[CategoryMap::GetSyntheticChildren] Trying to use category %s\n", category_sp->GetName());
        if (!category_sp->Get(valobj, current_format, use_dynamic, &reason_why))
            continue;
        return current_format;
    }
    if (log)
        log->Printf("[CategoryMap::GetSyntheticChildren] nothing found - returning empty SP\n");
    return lldb::SyntheticChildrenSP();
}
#endif

void
CategoryMap::LoopThrough(CallbackType callback, void* param)
{
    if (callback)
    {
        Mutex::Locker locker(m_map_mutex);
        
        // loop through enabled categories in respective order
        {
            ActiveCategoriesIterator begin, end = m_active_categories.end();
            for (begin = m_active_categories.begin(); begin != end; begin++)
            {
                lldb::TypeCategoryImplSP category = *begin;
                ConstString type = ConstString(category->GetName());
                if (!callback(param, category))
                    break;
            }
        }
        
        // loop through disabled categories in just any order
        {
            MapIterator pos, end = m_map.end();
            for (pos = m_map.begin(); pos != end; pos++)
            {
                if (pos->second->IsEnabled())
                    continue;
                KeyType type = pos->first;
                if (!callback(param, pos->second))
                    break;
            }
        }
    }
}

TypeCategoryImplSP
CategoryMap::GetAtIndex (uint32_t index)
{
    Mutex::Locker locker(m_map_mutex);
    
    if (index < m_map.size())
    {
        MapIterator pos, end = m_map.end();
        for (pos = m_map.begin(); pos != end; pos++)
        {
            if (index == 0)
                return pos->second;
            index--;
        }
    }
    
    return TypeCategoryImplSP();
}

lldb::TypeCategoryImplSP
FormatManager::GetCategory (const ConstString& category_name,
                         bool can_create)
{
    if (!category_name)
        return GetCategory(m_default_category_name);
    lldb::TypeCategoryImplSP category;
    if (m_categories_map.Get(category_name, category))
        return category;
    
    if (!can_create)
        return lldb::TypeCategoryImplSP();
    
    m_categories_map.Add(category_name,lldb::TypeCategoryImplSP(new TypeCategoryImpl(this, category_name)));
    return GetCategory(category_name);
}

lldb::Format
FormatManager::GetSingleItemFormat(lldb::Format vector_format)
{
    switch(vector_format)
    {
        case eFormatVectorOfChar:
            return eFormatCharArray;
            
        case eFormatVectorOfSInt8:
        case eFormatVectorOfSInt16:
        case eFormatVectorOfSInt32:
        case eFormatVectorOfSInt64:
            return eFormatDecimal;
            
        case eFormatVectorOfUInt8:
        case eFormatVectorOfUInt16:
        case eFormatVectorOfUInt32:
        case eFormatVectorOfUInt64:
        case eFormatVectorOfUInt128:
            return eFormatHex;
            
        case eFormatVectorOfFloat32:
        case eFormatVectorOfFloat64:
            return eFormatFloat;
            
        default:
            return lldb::eFormatInvalid;
    }
}

ConstString
FormatManager::GetValidTypeName (const ConstString& type)
{
    return ::GetValidTypeName_Impl(type);
}

FormatManager::FormatManager() : 
    m_value_nav("format",this),
    m_named_summaries_map(this),
    m_last_revision(0),
    m_categories_map(this),
    m_default_category_name(ConstString("default")),
    m_system_category_name(ConstString("system")), 
    m_gnu_cpp_category_name(ConstString("gnu-libstdc++")),
    m_libcxx_category_name(ConstString("libcxx")),
    m_objc_category_name(ConstString("objc")),
    m_corefoundation_category_name(ConstString("CoreFoundation")),
    m_coregraphics_category_name(ConstString("CoreGraphics")),
    m_coreservices_category_name(ConstString("CoreServices")),
    m_vectortypes_category_name(ConstString("VectorTypes")),
    m_appkit_category_name(ConstString("AppKit"))
{
    
    LoadSystemFormatters();
    LoadSTLFormatters();
    LoadLibcxxFormatters();
    LoadObjCFormatters();
    
    EnableCategory(m_objc_category_name,CategoryMap::Last);
    EnableCategory(m_corefoundation_category_name,CategoryMap::Last);
    EnableCategory(m_appkit_category_name,CategoryMap::Last);
    EnableCategory(m_coreservices_category_name,CategoryMap::Last);
    EnableCategory(m_coregraphics_category_name,CategoryMap::Last);
    EnableCategory(m_gnu_cpp_category_name,CategoryMap::Last);
    EnableCategory(m_libcxx_category_name,CategoryMap::Last);
    EnableCategory(m_vectortypes_category_name,CategoryMap::Last);
    EnableCategory(m_system_category_name,CategoryMap::Last);
}

void
FormatManager::LoadSTLFormatters()
{
    TypeSummaryImpl::Flags stl_summary_flags;
    stl_summary_flags.SetCascades(true)
    .SetSkipPointers(false)
    .SetSkipReferences(false)
    .SetDontShowChildren(true)
    .SetDontShowValue(true)
    .SetShowMembersOneLiner(false)
    .SetHideItemNames(false);
    
    lldb::TypeSummaryImplSP std_string_summary_sp(new StringSummaryFormat(stl_summary_flags,
                                                                          "${var._M_dataplus._M_p}"));
    
    TypeCategoryImpl::SharedPointer gnu_category_sp = GetCategory(m_gnu_cpp_category_name);
    
    gnu_category_sp->GetSummaryNavigator()->Add(ConstString("std::string"),
                                                std_string_summary_sp);
    gnu_category_sp->GetSummaryNavigator()->Add(ConstString("std::basic_string<char>"),
                                                std_string_summary_sp);
    gnu_category_sp->GetSummaryNavigator()->Add(ConstString("std::basic_string<char,std::char_traits<char>,std::allocator<char> >"),
                                                std_string_summary_sp);
    gnu_category_sp->GetSummaryNavigator()->Add(ConstString("std::basic_string<char, std::char_traits<char>, std::allocator<char> >"),
                                                std_string_summary_sp);
    
    
#ifndef LLDB_DISABLE_PYTHON
    
    SyntheticChildren::Flags stl_synth_flags;
    stl_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(false);
    
    gnu_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::vector<.+>(( )?&)?$")),
                                                       SyntheticChildrenSP(new TypeSyntheticImpl(stl_synth_flags,
                                                                                                 "lldb.formatters.cpp.gnu_libstdcpp.StdVectorSynthProvider")));
    gnu_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::map<.+> >(( )?&)?$")),
                                                       SyntheticChildrenSP(new TypeSyntheticImpl(stl_synth_flags,
                                                                                                 "lldb.formatters.cpp.gnu_libstdcpp.StdMapSynthProvider")));
    gnu_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::list<.+>(( )?&)?$")),
                                                       SyntheticChildrenSP(new TypeSyntheticImpl(stl_synth_flags,
                                                                                                 "lldb.formatters.cpp.gnu_libstdcpp.StdListSynthProvider")));
    
    stl_summary_flags.SetDontShowChildren(false);
    gnu_category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::vector<.+>(( )?&)?$")),
                                                     TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags,
                                                                                               "size=${svar%#}")));
    gnu_category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::map<.+> >(( )?&)?$")),
                                                     TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags,
                                                                                               "size=${svar%#}")));
    gnu_category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::list<.+>(( )?&)?$")),
                                                     TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags,
                                                                                               "size=${svar%#}")));
#endif
}

void
FormatManager::LoadLibcxxFormatters()
{
    TypeSummaryImpl::Flags stl_summary_flags;
    stl_summary_flags.SetCascades(true)
    .SetSkipPointers(false)
    .SetSkipReferences(false)
    .SetDontShowChildren(true)
    .SetDontShowValue(true)
    .SetShowMembersOneLiner(false)
    .SetHideItemNames(false);
    
#ifndef LLDB_DISABLE_PYTHON
    std::string code("     lldb.formatters.cpp.libcxx.stdstring_SummaryProvider(valobj,internal_dict)");
    lldb::TypeSummaryImplSP std_string_summary_sp(new ScriptSummaryFormat(stl_summary_flags, "lldb.formatters.cpp.libcxx.stdstring_SummaryProvider",code.c_str()));
    
    TypeCategoryImpl::SharedPointer libcxx_category_sp = GetCategory(m_libcxx_category_name);
    
    libcxx_category_sp->GetSummaryNavigator()->Add(ConstString("std::__1::string"),
                                                   std_string_summary_sp);
    libcxx_category_sp->GetSummaryNavigator()->Add(ConstString("std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >"),
                                                   std_string_summary_sp);

    SyntheticChildren::Flags stl_synth_flags;
    stl_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(false);
    
    libcxx_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::__1::vector<.+>(( )?&)?$")),
                                                       SyntheticChildrenSP(new TypeSyntheticImpl(stl_synth_flags,
                                                                                                 "lldb.formatters.cpp.libcxx.stdvector_SynthProvider")));
    libcxx_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::__1::list<.+>(( )?&)?$")),
                                                       SyntheticChildrenSP(new TypeSyntheticImpl(stl_synth_flags,
                                                                                                 "lldb.formatters.cpp.libcxx.stdlist_SynthProvider")));
    libcxx_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::__1::map<.+> >(( )?&)?$")),
                                                       SyntheticChildrenSP(new TypeSyntheticImpl(stl_synth_flags,
                                                                                                 "lldb.formatters.cpp.libcxx.stdmap_SynthProvider")));
    libcxx_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^(std::__1::)deque<.+>(( )?&)?$")),
                                                          SyntheticChildrenSP(new TypeSyntheticImpl(stl_synth_flags,
                                                                                                    "lldb.formatters.cpp.libcxx.stddeque_SynthProvider")));
    libcxx_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^(std::__1::)shared_ptr<.+>(( )?&)?$")),
                                                          SyntheticChildrenSP(new TypeSyntheticImpl(stl_synth_flags,
                                                                                                    "lldb.formatters.cpp.libcxx.stdsharedptr_SynthProvider")));
    libcxx_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^(std::__1::)weak_ptr<.+>(( )?&)?$")),
                                                          SyntheticChildrenSP(new TypeSyntheticImpl(stl_synth_flags,
                                                                                                    "lldb.formatters.cpp.libcxx.stdsharedptr_SynthProvider")));
    
    stl_summary_flags.SetDontShowChildren(false);stl_summary_flags.SetSkipPointers(true);
    libcxx_category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::__1::vector<.+>(( )?&)?$")),
                                                        TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags, "size=${svar%#}")));
    libcxx_category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::__1::list<.+>(( )?&)?$")),
                                                        TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags, "size=${svar%#}")));
    libcxx_category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::__1::map<.+> >(( )?&)?$")),
                                                        TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags, "size=${svar%#}")));
    libcxx_category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::__1::deque<.+>(( )?&)?$")),
                                                        TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags, "size=${svar%#}")));
    libcxx_category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::__1::shared_ptr<.+>(( )?&)?$")),
                                                        TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags, "{${var.__ptr_%S}} (strong=${var.count} weak=${var.weak_count})")));
    libcxx_category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::__1::weak_ptr<.+>(( )?&)?$")),
                                                        TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags, "{${var.__ptr_%S}} (strong=${var.count} weak=${var.weak_count})")));

#endif
}

void
FormatManager::LoadSystemFormatters()
{
    lldb::TypeSummaryImplSP string_format(new StringSummaryFormat(TypeSummaryImpl::Flags().SetCascades(false)
                                                                  .SetSkipPointers(true)
                                                                  .SetSkipReferences(false)
                                                                  .SetDontShowChildren(true)
                                                                  .SetDontShowValue(false)
                                                                  .SetShowMembersOneLiner(false)
                                                                  .SetHideItemNames(false),
                                                                  "${var%s}"));
    
    
    lldb::TypeSummaryImplSP string_array_format(new StringSummaryFormat(TypeSummaryImpl::Flags().SetCascades(false)
                                                                        .SetSkipPointers(true)
                                                                        .SetSkipReferences(false)
                                                                        .SetDontShowChildren(false)
                                                                        .SetDontShowValue(true)
                                                                        .SetShowMembersOneLiner(false)
                                                                        .SetHideItemNames(false),
                                                                        "${var%s}"));
    
    lldb::RegularExpressionSP any_size_char_arr(new RegularExpression("char \\[[0-9]+\\]"));
    
    TypeCategoryImpl::SharedPointer sys_category_sp = GetCategory(m_system_category_name);
    
    sys_category_sp->GetSummaryNavigator()->Add(ConstString("char *"), string_format);
    sys_category_sp->GetSummaryNavigator()->Add(ConstString("const char *"), string_format);
    sys_category_sp->GetRegexSummaryNavigator()->Add(any_size_char_arr, string_array_format);
    
    lldb::TypeSummaryImplSP ostype_summary(new StringSummaryFormat(TypeSummaryImpl::Flags().SetCascades(false)
                                                                   .SetSkipPointers(true)
                                                                   .SetSkipReferences(true)
                                                                   .SetDontShowChildren(true)
                                                                   .SetDontShowValue(false)
                                                                   .SetShowMembersOneLiner(false)
                                                                   .SetHideItemNames(false),
                                                                   "${var%O}"));
    
    sys_category_sp->GetSummaryNavigator()->Add(ConstString("OSType"), ostype_summary);
}

static void
AddSummary(TypeCategoryImpl::SharedPointer category_sp,
           const char* string,
           ConstString type_name,
           TypeSummaryImpl::Flags flags)
{
    lldb::TypeSummaryImplSP summary_sp(new StringSummaryFormat(flags,
                                                               string));
    category_sp->GetSummaryNavigator()->Add(type_name,
                                            summary_sp);
}

#ifndef LLDB_DISABLE_PYTHON
static void
AddScriptSummary(TypeCategoryImpl::SharedPointer category_sp,
                 const char* funct_name,
                 ConstString type_name,
                 TypeSummaryImpl::Flags flags)
{
    
    std::string code("     ");
    code.append(funct_name).append("(valobj,internal_dict)");
    
    lldb::TypeSummaryImplSP summary_sp(new ScriptSummaryFormat(flags,
                                                               funct_name,
                                                               code.c_str()));
    category_sp->GetSummaryNavigator()->Add(type_name,
                                            summary_sp);
}
#endif

#ifndef LLDB_DISABLE_PYTHON
static void
AddCXXSummary (TypeCategoryImpl::SharedPointer category_sp,
               CXXFunctionSummaryFormat::Callback funct,
               const char* description,
               ConstString type_name,
               TypeSummaryImpl::Flags flags)
{
    lldb::TypeSummaryImplSP summary_sp(new CXXFunctionSummaryFormat(flags,funct,description));
    category_sp->GetSummaryNavigator()->Add(type_name,
                                            summary_sp);
}
#endif

#ifndef LLDB_DISABLE_PYTHON
static void AddCXXSynthetic  (TypeCategoryImpl::SharedPointer category_sp,
                              CXXSyntheticChildren::CreateFrontEndCallback generator,
                              const char* description,
                              ConstString type_name,
                              TypeSyntheticImpl::Flags flags)
{
    lldb::SyntheticChildrenSP synth_sp(new CXXSyntheticChildren(flags,description,generator));
    category_sp->GetSyntheticNavigator()->Add(type_name,synth_sp);
}
#endif

void
FormatManager::LoadObjCFormatters()
{
    TypeSummaryImpl::Flags objc_flags;
    objc_flags.SetCascades(false)
    .SetSkipPointers(true)
    .SetSkipReferences(true)
    .SetDontShowChildren(true)
    .SetDontShowValue(true)
    .SetShowMembersOneLiner(false)
    .SetHideItemNames(false);

    TypeCategoryImpl::SharedPointer objc_category_sp = GetCategory(m_objc_category_name);
    
    lldb::TypeSummaryImplSP ObjC_BOOL_summary(new CXXFunctionSummaryFormat(objc_flags, lldb_private::formatters::ObjCBOOLSummaryProvider,""));
    objc_category_sp->GetSummaryNavigator()->Add(ConstString("BOOL"),
                                                 ObjC_BOOL_summary);
    objc_category_sp->GetSummaryNavigator()->Add(ConstString("BOOL &"),
                                                 ObjC_BOOL_summary);
    objc_category_sp->GetSummaryNavigator()->Add(ConstString("BOOL *"),
                                                 ObjC_BOOL_summary);

#ifndef LLDB_DISABLE_PYTHON
    // we need to skip pointers here since we are special casing a SEL* when retrieving its value
    objc_flags.SetSkipPointers(true);
    AddCXXSummary(objc_category_sp, lldb_private::formatters::ObjCSELSummaryProvider<false>, "SEL summary", ConstString("SEL"), objc_flags);
    AddCXXSummary(objc_category_sp, lldb_private::formatters::ObjCSELSummaryProvider<false>, "SEL summary", ConstString("struct objc_selector"), objc_flags);
    AddCXXSummary(objc_category_sp, lldb_private::formatters::ObjCSELSummaryProvider<false>, "SEL summary", ConstString("objc_selector"), objc_flags);
    AddCXXSummary(objc_category_sp, lldb_private::formatters::ObjCSELSummaryProvider<true>, "SEL summary", ConstString("objc_selector *"), objc_flags);
    
    AddScriptSummary(objc_category_sp, "lldb.formatters.objc.Class.Class_Summary", ConstString("Class"), objc_flags);
#endif // LLDB_DISABLE_PYTHON

    objc_flags.SetSkipPointers(false);

    TypeCategoryImpl::SharedPointer corefoundation_category_sp = GetCategory(m_corefoundation_category_name);

    AddSummary(corefoundation_category_sp,
               "${var.years} years, ${var.months} months, ${var.days} days, ${var.hours} hours, ${var.minutes} minutes ${var.seconds} seconds",
               ConstString("CFGregorianUnits"),
               objc_flags);
    AddSummary(corefoundation_category_sp,
               "location=${var.location} length=${var.length}",
               ConstString("CFRange"),
               objc_flags);
    AddSummary(corefoundation_category_sp,
               "(x=${var.x}, y=${var.y})",
               ConstString("NSPoint"),
               objc_flags);
    AddSummary(corefoundation_category_sp,
               "location=${var.location}, length=${var.length}",
               ConstString("NSRange"),
               objc_flags);
    AddSummary(corefoundation_category_sp,
               "${var.origin}, ${var.size}",
               ConstString("NSRect"),
               objc_flags);
    AddSummary(corefoundation_category_sp,
               "(${var.origin}, ${var.size}), ...",
               ConstString("NSRectArray"),
               objc_flags);
    AddSummary(objc_category_sp,
               "(width=${var.width}, height=${var.height})",
               ConstString("NSSize"),
               objc_flags);
    
    TypeCategoryImpl::SharedPointer coregraphics_category_sp = GetCategory(m_coregraphics_category_name);
    
    AddSummary(coregraphics_category_sp,
               "(width=${var.width}, height=${var.height})",
               ConstString("CGSize"),
               objc_flags);
    AddSummary(coregraphics_category_sp,
               "(x=${var.x}, y=${var.y})",
               ConstString("CGPoint"),
               objc_flags);
    AddSummary(coregraphics_category_sp,
               "origin=${var.origin} size=${var.size}",
               ConstString("CGRect"),
               objc_flags);
    
    TypeCategoryImpl::SharedPointer coreservices_category_sp = GetCategory(m_coreservices_category_name);
    
    AddSummary(coreservices_category_sp,
               "red=${var.red} green=${var.green} blue=${var.blue}",
               ConstString("RGBColor"),
               objc_flags);
    AddSummary(coreservices_category_sp,
               "(t=${var.top}, l=${var.left}, b=${var.bottom}, r=${var.right})",
               ConstString("Rect"),
               objc_flags);
    AddSummary(coreservices_category_sp,
               "(v=${var.v}, h=${var.h})",
               ConstString("Point"),
               objc_flags);
    AddSummary(coreservices_category_sp,
               "${var.month}/${var.day}/${var.year}  ${var.hour} :${var.minute} :${var.second} dayOfWeek:${var.dayOfWeek}",
               ConstString("DateTimeRect *"),
               objc_flags);
    AddSummary(coreservices_category_sp,
               "${var.ld.month}/${var.ld.day}/${var.ld.year} ${var.ld.hour} :${var.ld.minute} :${var.ld.second} dayOfWeek:${var.ld.dayOfWeek}",
               ConstString("LongDateRect"),
               objc_flags);
    AddSummary(coreservices_category_sp,
               "(x=${var.x}, y=${var.y})",
               ConstString("HIPoint"),
               objc_flags);
    AddSummary(coreservices_category_sp,
               "origin=${var.origin} size=${var.size}",
               ConstString("HIRect"),
               objc_flags);
    
    TypeCategoryImpl::SharedPointer appkit_category_sp = GetCategory(m_appkit_category_name);
    
    TypeSummaryImpl::Flags appkit_flags;
    appkit_flags.SetCascades(true)
    .SetSkipPointers(false)
    .SetSkipReferences(false)
    .SetDontShowChildren(true)
    .SetDontShowValue(false)
    .SetShowMembersOneLiner(false)
    .SetHideItemNames(false);

    appkit_flags.SetDontShowChildren(false);
    

#ifndef LLDB_DISABLE_PYTHON
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSArraySummaryProvider, "NSArray summary provider", ConstString("NSArray"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSArraySummaryProvider, "NSArray summary provider", ConstString("NSMutableArray"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSArraySummaryProvider, "NSArray summary provider", ConstString("__NSArrayI"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSArraySummaryProvider, "NSArray summary provider", ConstString("__NSArrayM"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSArraySummaryProvider, "NSArray summary provider", ConstString("__NSCFArray"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSArraySummaryProvider, "NSArray summary provider", ConstString("CFArrayRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSArraySummaryProvider, "NSArray summary provider", ConstString("CFMutableArrayRef"), appkit_flags);

    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDictionarySummaryProvider<false>, "NSDictionary summary provider", ConstString("NSDictionary"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDictionarySummaryProvider<false>, "NSDictionary summary provider", ConstString("NSMutableDictionary"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDictionarySummaryProvider<false>, "NSDictionary summary provider", ConstString("__NSCFDictionary"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDictionarySummaryProvider<false>, "NSDictionary summary provider", ConstString("__NSDictionaryI"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDictionarySummaryProvider<false>, "NSDictionary summary provider", ConstString("__NSDictionaryM"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDictionarySummaryProvider<true>, "NSDictionary summary provider", ConstString("CFDictionaryRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDictionarySummaryProvider<true>, "NSDictionary summary provider", ConstString("CFMutableDictionaryRef"), appkit_flags);
    
    // AddSummary(appkit_category_sp, "${var.key%@} -> ${var.value%@}", ConstString("$_lldb_typegen_nspair"), appkit_flags);
    
    appkit_flags.SetDontShowChildren(true);
    
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSArraySyntheticFrontEndCreator, "NSArray synthetic children", ConstString("__NSArrayM"), TypeSyntheticImpl::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSArraySyntheticFrontEndCreator, "NSArray synthetic children", ConstString("__NSArrayI"), TypeSyntheticImpl::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSArraySyntheticFrontEndCreator, "NSArray synthetic children", ConstString("NSArray"), TypeSyntheticImpl::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSArraySyntheticFrontEndCreator, "NSArray synthetic children", ConstString("NSMutableArray"), TypeSyntheticImpl::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSArraySyntheticFrontEndCreator, "NSArray synthetic children", ConstString("__NSCFArray"), TypeSyntheticImpl::Flags());

    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSDictionarySyntheticFrontEndCreator, "NSDictionary synthetic children", ConstString("__NSDictionaryM"), TypeSyntheticImpl::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSDictionarySyntheticFrontEndCreator, "NSDictionary synthetic children", ConstString("__NSDictionaryI"), TypeSyntheticImpl::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSDictionarySyntheticFrontEndCreator, "NSDictionary synthetic children", ConstString("NSDictionary"), TypeSyntheticImpl::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSDictionarySyntheticFrontEndCreator, "NSDictionary synthetic children", ConstString("NSMutableDictionary"), TypeSyntheticImpl::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSDictionarySyntheticFrontEndCreator, "NSDictionary synthetic children", ConstString("CFDictionaryRef"), TypeSyntheticImpl::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSDictionarySyntheticFrontEndCreator, "NSDictionary synthetic children", ConstString("CFMutableDictionaryRef"), TypeSyntheticImpl::Flags());

    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.CFBag.CFBag_SummaryProvider", ConstString("CFBagRef"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.CFBag.CFBag_SummaryProvider", ConstString("__CFBag"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.CFBag.CFBag_SummaryProvider", ConstString("const struct __CFBag"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.CFBag.CFBag_SummaryProvider", ConstString("CFMutableBagRef"), appkit_flags);
    
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.CFBinaryHeap.CFBinaryHeap_SummaryProvider", ConstString("CFBinaryHeapRef"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.CFBinaryHeap.CFBinaryHeap_SummaryProvider", ConstString("__CFBinaryHeap"), appkit_flags);

    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("NSString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("CFStringRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("CFMutableStringRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("__NSCFConstantString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("__NSCFString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("NSCFConstantString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("NSCFString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("NSPathStore2"), appkit_flags);
    
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.CFString.CFAttributedString_SummaryProvider", ConstString("NSAttributedString"), appkit_flags);
    
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSBundle.NSBundle_SummaryProvider", ConstString("NSBundle"), appkit_flags);

    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>, "NSData summary provider", ConstString("NSData"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>, "NSData summary provider", ConstString("NSConcreteData"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>, "NSData summary provider", ConstString("NSConcreteMutableData"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>, "NSData summary provider", ConstString("__NSCFData"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDataSummaryProvider<true>, "NSData summary provider", ConstString("CFDataRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDataSummaryProvider<true>, "NSData summary provider", ConstString("CFMutableDataRef"), appkit_flags);

    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSException.NSException_SummaryProvider", ConstString("NSException"), appkit_flags);

    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSMachPort.NSMachPort_SummaryProvider", ConstString("NSMachPort"), appkit_flags);
    
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSNotification.NSNotification_SummaryProvider", ConstString("NSNotification"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSNotification.NSNotification_SummaryProvider", ConstString("NSConcreteNotification"), appkit_flags);

    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSNumberSummaryProvider, "NSNumber summary provider", ConstString("NSNumber"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSNumberSummaryProvider, "NSNumber summary provider", ConstString("__NSCFBoolean"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSNumberSummaryProvider, "NSNumber summary provider", ConstString("__NSCFNumber"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSNumberSummaryProvider, "NSNumber summary provider", ConstString("NSCFBoolean"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSNumberSummaryProvider, "NSNumber summary provider", ConstString("NSCFNumber"), appkit_flags);
    
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider, "NSDecimalNumber summary provider", ConstString("NSDecimalNumber"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider, "NSHost summary provider", ConstString("NSHost"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider, "NSTask summary provider", ConstString("NSTask"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider, "NSValue summary provider", ConstString("NSValue"), appkit_flags);

    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSSet.NSSet_SummaryProvider", ConstString("NSSet"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSSet.NSSet_SummaryProvider2", ConstString("CFSetRef"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSSet.NSSet_SummaryProvider2", ConstString("CFMutableSetRef"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSSet.NSSet_SummaryProvider", ConstString("__NSCFSet"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSSet.NSSet_SummaryProvider", ConstString("__NSSetI"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSSet.NSSet_SummaryProvider", ConstString("__NSSetM"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSSet.NSSet_SummaryProvider", ConstString("NSCountedSet"), appkit_flags);

    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSURL.NSURL_SummaryProvider", ConstString("NSURL"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSURL.NSURL_SummaryProvider", ConstString("CFURLRef"), appkit_flags);
    
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSDate.NSDate_SummaryProvider", ConstString("NSDate"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSDate.NSDate_SummaryProvider", ConstString("__NSDate"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSDate.NSDate_SummaryProvider", ConstString("__NSTaggedDate"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSDate.NSDate_SummaryProvider", ConstString("NSCalendarDate"), appkit_flags);

    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSDate.NSTimeZone_SummaryProvider", ConstString("NSTimeZone"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSDate.NSTimeZone_SummaryProvider", ConstString("CFTimeZoneRef"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSDate.NSTimeZone_SummaryProvider", ConstString("__NSTimeZone"), appkit_flags);

    // CFAbsoluteTime is actually a double rather than a pointer to an object
    // we do not care about the numeric value, since it is probably meaningless to users
    appkit_flags.SetDontShowValue(true);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSDate.CFAbsoluteTime_SummaryProvider", ConstString("CFAbsoluteTime"), appkit_flags);
    appkit_flags.SetDontShowValue(false);
    
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSIndexSet.NSIndexSet_SummaryProvider", ConstString("NSIndexSet"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.NSIndexSet.NSIndexSet_SummaryProvider", ConstString("NSMutableIndexSet"), appkit_flags);

    AddSummary(appkit_category_sp, "@\"${var.month%d}/${var.day%d}/${var.year%d} ${var.hour%d}:${var.minute%d}:${var.second}\"", ConstString("CFGregorianDate"), appkit_flags);
    
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.CFBitVector.CFBitVector_SummaryProvider", ConstString("CFBitVectorRef"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.CFBitVector.CFBitVector_SummaryProvider", ConstString("CFMutableBitVectorRef"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.CFBitVector.CFBitVector_SummaryProvider", ConstString("__CFBitVector"), appkit_flags);
    AddScriptSummary(appkit_category_sp, "lldb.formatters.objc.CFBitVector.CFBitVector_SummaryProvider", ConstString("__CFMutableBitVector"), appkit_flags);
#endif // LLDB_DISABLE_PYTHON
    
    TypeCategoryImpl::SharedPointer vectors_category_sp = GetCategory(m_vectortypes_category_name);

    TypeSummaryImpl::Flags vector_flags;
    vector_flags.SetCascades(true)
    .SetSkipPointers(true)
    .SetSkipReferences(false)
    .SetDontShowChildren(true)
    .SetDontShowValue(false)
    .SetShowMembersOneLiner(true)
    .SetHideItemNames(true);
    
    AddSummary(vectors_category_sp,
               "${var.uint128}",
               ConstString("builtin_type_vec128"),
               objc_flags);

    AddSummary(vectors_category_sp,
               "",
               ConstString("float [4]"),
               vector_flags);
    AddSummary(vectors_category_sp,
               "",
               ConstString("int32_t [4]"),
               vector_flags);
    AddSummary(vectors_category_sp,
               "",
               ConstString("int16_t [8]"),
               vector_flags);
    AddSummary(vectors_category_sp,
               "",
               ConstString("vDouble"),
               vector_flags);
    AddSummary(vectors_category_sp,
               "",
               ConstString("vFloat"),
               vector_flags);
    AddSummary(vectors_category_sp,
               "",
               ConstString("vSInt8"),
               vector_flags);
    AddSummary(vectors_category_sp,
               "",
               ConstString("vSInt16"),
               vector_flags);
    AddSummary(vectors_category_sp,
               "",
               ConstString("vSInt32"),
               vector_flags);
    AddSummary(vectors_category_sp,
               "",
               ConstString("vUInt16"),
               vector_flags);
    AddSummary(vectors_category_sp,
               "",
               ConstString("vUInt8"),
               vector_flags);
    AddSummary(vectors_category_sp,
               "",
               ConstString("vUInt16"),
               vector_flags);
    AddSummary(vectors_category_sp,
               "",
               ConstString("vUInt32"),
               vector_flags);
    AddSummary(vectors_category_sp,
               "",
               ConstString("vBool32"),
               vector_flags);
}
