//===-- FormatManager.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/DataFormatters/FormatManager.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/Core/Debugger.h"
#include "lldb/DataFormatters/CXXFormatterFunctions.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"

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
lldb::ScriptedSyntheticChildrenSP
FormatManager::GetSyntheticForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    if (!type_sp)
        return lldb::ScriptedSyntheticChildrenSP();
    lldb::ScriptedSyntheticChildrenSP synth_chosen_sp;
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
        lldb::ScriptedSyntheticChildrenSP synth_current_sp((ScriptedSyntheticChildren*)category_sp->GetSyntheticForType(type_sp).get());
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
    lldb::ScriptedSyntheticChildrenSP synth_sp = GetSyntheticForType(type_sp);
    if (filter_sp->GetRevision() > synth_sp->GetRevision())
        return lldb::SyntheticChildrenSP(filter_sp.get());
    else
        return lldb::SyntheticChildrenSP(synth_sp.get());
}
#endif

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

ConstString
GetTypeForCache (ValueObject& valobj,
                 lldb::DynamicValueType use_dynamic)
{
    if (use_dynamic == lldb::eNoDynamicValues)
    {
        if (valobj.IsDynamic())
        {
            if (valobj.GetStaticValue())
                return valobj.GetStaticValue()->GetQualifiedTypeName();
            else
                return ConstString();
        }
        else
            return valobj.GetQualifiedTypeName();
    }
    if (valobj.IsDynamic())
        return valobj.GetQualifiedTypeName();
    if (valobj.GetDynamicValue(use_dynamic))
        return valobj.GetDynamicValue(use_dynamic)->GetQualifiedTypeName();
    return ConstString();
}

#define USE_CACHE 1
lldb::TypeSummaryImplSP
FormatManager::GetSummaryFormat (ValueObject& valobj,
                                 lldb::DynamicValueType use_dynamic)
{
    TypeSummaryImplSP retval;
#if USE_CACHE
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    ConstString valobj_type(GetTypeForCache(valobj, use_dynamic));
    if (valobj_type)
    {
        if (log)
            log->Printf("[FormatManager::GetSummaryFormat] Looking into cache for type %s", valobj_type.AsCString("<invalid>"));
        if (m_format_cache.GetSummary(valobj_type,retval))
            return retval;
        if (log)
            log->Printf("[FormatManager::GetSummaryFormat] Cache search failed. Going normal route");
    }
#endif
    retval = m_categories_map.GetSummaryFormat(valobj, use_dynamic);
#if USE_CACHE
    if (valobj_type)
    {
        if (log)
            log->Printf("[FormatManager::GetSummaryFormat] Caching %p for type %s",retval.get(),valobj_type.AsCString("<invalid>"));
        m_format_cache.SetSummary(valobj_type,retval);
    }
#ifdef LLDB_CONFIGURATION_DEBUG
    if (log)
        log->Printf("[FormatManager::GetSummaryFormat] Cache hits: %llu - Cache Misses: %llu", m_format_cache.GetCacheHits(), m_format_cache.GetCacheMisses());
#endif
#endif
    return retval;
}

#ifndef LLDB_DISABLE_PYTHON
lldb::SyntheticChildrenSP
FormatManager::GetSyntheticChildren (ValueObject& valobj,
                      lldb::DynamicValueType use_dynamic)
{
    SyntheticChildrenSP retval;
#if USE_CACHE
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_TYPES));
    ConstString valobj_type(GetTypeForCache(valobj, use_dynamic));
    if (valobj_type)
    {
        if (log)
            log->Printf("[FormatManager::GetSyntheticChildren] Looking into cache for type %s\n", valobj_type.AsCString("<invalid>"));
        if (m_format_cache.GetSynthetic(valobj_type,retval))
            return retval;
        if (log)
            log->Printf("[FormatManager::GetSyntheticChildren] Cache search failed. Going normal route\n");
    }
#endif
    retval = m_categories_map.GetSyntheticChildren(valobj, use_dynamic);
#if USE_CACHE
    if (valobj_type)
    {
        if (log)
            log->Printf("[FormatManager::GetSyntheticChildren] Caching %p for type %s\n",retval.get(),valobj_type.AsCString("<invalid>"));
        m_format_cache.SetSynthetic(valobj_type,retval);
    }
#ifdef LLDB_CONFIGURATION_DEBUG
    if (log)
        log->Printf("[FormatManager::GetSyntheticChildren] Cache hits: %llu - Cache Misses: %llu", m_format_cache.GetCacheHits(), m_format_cache.GetCacheMisses());
#endif
#endif
    return retval;
}
#endif
#undef USE_CACHE

FormatManager::FormatManager() :
    m_format_cache(),
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
    LoadLibStdcppFormatters();
    LoadLibcxxFormatters();
    LoadObjCFormatters();
    
    EnableCategory(m_objc_category_name,TypeCategoryMap::Last);
    EnableCategory(m_corefoundation_category_name,TypeCategoryMap::Last);
    EnableCategory(m_appkit_category_name,TypeCategoryMap::Last);
    EnableCategory(m_coreservices_category_name,TypeCategoryMap::Last);
    EnableCategory(m_coregraphics_category_name,TypeCategoryMap::Last);
    EnableCategory(m_gnu_cpp_category_name,TypeCategoryMap::Last);
    EnableCategory(m_libcxx_category_name,TypeCategoryMap::Last);
    EnableCategory(m_vectortypes_category_name,TypeCategoryMap::Last);
    EnableCategory(m_system_category_name,TypeCategoryMap::Last);
}

static void
AddStringSummary(TypeCategoryImpl::SharedPointer category_sp,
                 const char* string,
                 ConstString type_name,
                 TypeSummaryImpl::Flags flags,
                 bool regex = false)
{
    lldb::TypeSummaryImplSP summary_sp(new StringSummaryFormat(flags,
                                                               string));
    
    if (regex)
        category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression(type_name.AsCString())),summary_sp);
    else
        category_sp->GetSummaryNavigator()->Add(type_name, summary_sp);
}

#ifndef LLDB_DISABLE_PYTHON
static void
AddScriptSummary(TypeCategoryImpl::SharedPointer category_sp,
                 const char* funct_name,
                 ConstString type_name,
                 TypeSummaryImpl::Flags flags,
                 bool regex = false)
{
    
    std::string code("     ");
    code.append(funct_name).append("(valobj,internal_dict)");
    
    lldb::TypeSummaryImplSP summary_sp(new ScriptSummaryFormat(flags,
                                                               funct_name,
                                                               code.c_str()));
    if (regex)
        category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression(type_name.AsCString())),summary_sp);
    else
        category_sp->GetSummaryNavigator()->Add(type_name, summary_sp);
}
#endif

#ifndef LLDB_DISABLE_PYTHON
static void
AddCXXSummary (TypeCategoryImpl::SharedPointer category_sp,
               CXXFunctionSummaryFormat::Callback funct,
               const char* description,
               ConstString type_name,
               TypeSummaryImpl::Flags flags,
               bool regex = false)
{
    lldb::TypeSummaryImplSP summary_sp(new CXXFunctionSummaryFormat(flags,funct,description));
    if (regex)
        category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression(type_name.AsCString())),summary_sp);
    else
        category_sp->GetSummaryNavigator()->Add(type_name, summary_sp);
}
#endif

#ifndef LLDB_DISABLE_PYTHON
static void AddCXXSynthetic  (TypeCategoryImpl::SharedPointer category_sp,
                              CXXSyntheticChildren::CreateFrontEndCallback generator,
                              const char* description,
                              ConstString type_name,
                              ScriptedSyntheticChildren::Flags flags,
                              bool regex = false)
{
    lldb::SyntheticChildrenSP synth_sp(new CXXSyntheticChildren(flags,description,generator));
    if (regex)
        category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression(type_name.AsCString())), synth_sp);
    else
        category_sp->GetSyntheticNavigator()->Add(type_name,synth_sp);
}
#endif

void
FormatManager::LoadLibStdcppFormatters()
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
    
    // making sure we force-pick the summary for printing wstring (_M_p is a wchar_t*)
    lldb::TypeSummaryImplSP std_wstring_summary_sp(new StringSummaryFormat(stl_summary_flags,
                                                                           "${var._M_dataplus._M_p%S}"));
    
    gnu_category_sp->GetSummaryNavigator()->Add(ConstString("std::wstring"),
                                                std_wstring_summary_sp);
    gnu_category_sp->GetSummaryNavigator()->Add(ConstString("std::basic_string<wchar_t>"),
                                                std_wstring_summary_sp);
    gnu_category_sp->GetSummaryNavigator()->Add(ConstString("std::basic_string<wchar_t,std::char_traits<wchar_t>,std::allocator<wchar_t> >"),
                                                std_wstring_summary_sp);
    gnu_category_sp->GetSummaryNavigator()->Add(ConstString("std::basic_string<wchar_t, std::char_traits<wchar_t>, std::allocator<wchar_t> >"),
                                                std_wstring_summary_sp);
    
    
#ifndef LLDB_DISABLE_PYTHON
    
    SyntheticChildren::Flags stl_synth_flags;
    stl_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(false);
    
    gnu_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::vector<.+>(( )?&)?$")),
                                                       SyntheticChildrenSP(new ScriptedSyntheticChildren(stl_synth_flags,
                                                                                                 "lldb.formatters.cpp.gnu_libstdcpp.StdVectorSynthProvider")));
    gnu_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::map<.+> >(( )?&)?$")),
                                                       SyntheticChildrenSP(new ScriptedSyntheticChildren(stl_synth_flags,
                                                                                                 "lldb.formatters.cpp.gnu_libstdcpp.StdMapSynthProvider")));
    gnu_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::list<.+>(( )?&)?$")),
                                                       SyntheticChildrenSP(new ScriptedSyntheticChildren(stl_synth_flags,
                                                                                                 "lldb.formatters.cpp.gnu_libstdcpp.StdListSynthProvider")));
    
    stl_summary_flags.SetDontShowChildren(false);stl_summary_flags.SetSkipPointers(true);
    gnu_category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::vector<.+>(( )?&)?$")),
                                                     TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags,
                                                                                               "size=${svar%#}")));
    gnu_category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::map<.+> >(( )?&)?$")),
                                                     TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags,
                                                                                               "size=${svar%#}")));
    gnu_category_sp->GetRegexSummaryNavigator()->Add(RegularExpressionSP(new RegularExpression("^std::list<.+>(( )?&)?$")),
                                                     TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags,
                                                                                               "size=${svar%#}")));

    AddCXXSynthetic(gnu_category_sp, lldb_private::formatters::LibStdcppVectorIteratorSyntheticFrontEndCreator, "std::vector iterator synthetic children", ConstString("^__gnu_cxx::__normal_iterator<.+>$"), stl_synth_flags, true);
    
    AddCXXSynthetic(gnu_category_sp, lldb_private::formatters::LibstdcppMapIteratorSyntheticFrontEndCreator, "std::map iterator synthetic children", ConstString("^std::_Rb_tree_iterator<.+>$"), stl_synth_flags, true);
    
    gnu_category_sp->GetSummaryNavigator()->Add(ConstString("std::vector<std::allocator<bool> >"),
                                                   TypeSummaryImplSP(new StringSummaryFormat(stl_summary_flags, "size=${svar%#}")));
    
    gnu_category_sp->GetSyntheticNavigator()->Add(ConstString("std::vector<std::allocator<bool> >"),
                                                     SyntheticChildrenSP(new CXXSyntheticChildren(stl_synth_flags,"libc++ std::vector<bool> synthetic children",lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEndCreator)));

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
    //std::string code("     lldb.formatters.cpp.libcxx.stdstring_SummaryProvider(valobj,internal_dict)");
    //lldb::TypeSummaryImplSP std_string_summary_sp(new ScriptSummaryFormat(stl_summary_flags, "lldb.formatters.cpp.libcxx.stdstring_SummaryProvider",code.c_str()));
    
    lldb::TypeSummaryImplSP std_string_summary_sp(new CXXFunctionSummaryFormat(stl_summary_flags, lldb_private::formatters::LibcxxStringSummaryProvider, "std::string summary provider"));
    lldb::TypeSummaryImplSP std_wstring_summary_sp(new CXXFunctionSummaryFormat(stl_summary_flags, lldb_private::formatters::LibcxxWStringSummaryProvider, "std::wstring summary provider"));

    TypeCategoryImpl::SharedPointer libcxx_category_sp = GetCategory(m_libcxx_category_name);
    
    libcxx_category_sp->GetSummaryNavigator()->Add(ConstString("std::__1::string"),
                                                   std_string_summary_sp);
    libcxx_category_sp->GetSummaryNavigator()->Add(ConstString("std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >"),
                                                   std_string_summary_sp);

    libcxx_category_sp->GetSummaryNavigator()->Add(ConstString("std::__1::wstring"),
                                                   std_wstring_summary_sp);
    libcxx_category_sp->GetSummaryNavigator()->Add(ConstString("std::__1::basic_string<wchar_t, std::__1::char_traits<wchar_t>, std::__1::allocator<wchar_t> >"),
                                                   std_wstring_summary_sp);
    
    SyntheticChildren::Flags stl_synth_flags;
    stl_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(false);
    
    AddCXXSynthetic(libcxx_category_sp, lldb_private::formatters::LibcxxStdVectorSyntheticFrontEndCreator, "libc++ std::vector synthetic children", ConstString("^std::__1::vector<.+>(( )?&)?$"), stl_synth_flags, true);
    AddCXXSynthetic(libcxx_category_sp, lldb_private::formatters::LibcxxStdListSyntheticFrontEndCreator, "libc++ std::list synthetic children", ConstString("^std::__1::list<.+>(( )?&)?$"), stl_synth_flags, true);
    AddCXXSynthetic(libcxx_category_sp, lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator, "libc++ std::map synthetic children", ConstString("^std::__1::map<.+> >(( )?&)?$"), stl_synth_flags, true);
    AddCXXSynthetic(libcxx_category_sp, lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEndCreator, "libc++ std::vector<bool> synthetic children", ConstString("std::__1::vector<std::__1::allocator<bool> >"), stl_synth_flags);
    AddCXXSynthetic(libcxx_category_sp, lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator, "libc++ std::set synthetic children", ConstString("^std::__1::set<.+> >(( )?&)?$"), stl_synth_flags, true);
    AddCXXSynthetic(libcxx_category_sp, lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator, "libc++ std::multiset synthetic children", ConstString("^std::__1::multiset<.+> >(( )?&)?$"), stl_synth_flags, true);
    AddCXXSynthetic(libcxx_category_sp, lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator, "libc++ std::multimap synthetic children", ConstString("^std::__1::multimap<.+> >(( )?&)?$"), stl_synth_flags, true);

    libcxx_category_sp->GetRegexSyntheticNavigator()->Add(RegularExpressionSP(new RegularExpression("^(std::__1::)deque<.+>(( )?&)?$")),
                                                          SyntheticChildrenSP(new ScriptedSyntheticChildren(stl_synth_flags,
                                                                                                    "lldb.formatters.cpp.libcxx.stddeque_SynthProvider")));
    
    AddCXXSynthetic(libcxx_category_sp, lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEndCreator, "shared_ptr synthetic children", ConstString("^(std::__1::)shared_ptr<.+>(( )?&)?$"), stl_synth_flags, true);
    AddCXXSynthetic(libcxx_category_sp, lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEndCreator, "weak_ptr synthetic children", ConstString("^(std::__1::)weak_ptr<.+>(( )?&)?$"), stl_synth_flags, true);
    
    stl_summary_flags.SetDontShowChildren(false);stl_summary_flags.SetSkipPointers(false);
    
    AddCXXSummary(libcxx_category_sp, lldb_private::formatters::LibcxxContainerSummaryProvider, "libc++ std::vector summary provider", ConstString("^std::__1::vector<.+>(( )?&)?$"), stl_summary_flags, true);
    AddCXXSummary(libcxx_category_sp, lldb_private::formatters::LibcxxContainerSummaryProvider, "libc++ std::list summary provider", ConstString("^std::__1::list<.+>(( )?&)?$"), stl_summary_flags, true);
    AddCXXSummary(libcxx_category_sp, lldb_private::formatters::LibcxxContainerSummaryProvider, "libc++ std::map summary provider", ConstString("^std::__1::map<.+>(( )?&)?$"), stl_summary_flags, true);
    AddCXXSummary(libcxx_category_sp, lldb_private::formatters::LibcxxContainerSummaryProvider, "libc++ std::deque summary provider", ConstString("^std::__1::deque<.+>(( )?&)?$"), stl_summary_flags, true);
    AddCXXSummary(libcxx_category_sp, lldb_private::formatters::LibcxxContainerSummaryProvider, "libc++ std::vector<bool> summary provider", ConstString("std::__1::vector<std::__1::allocator<bool> >"), stl_summary_flags);
    AddCXXSummary(libcxx_category_sp, lldb_private::formatters::LibcxxContainerSummaryProvider, "libc++ std::set summary provider", ConstString("^std::__1::set<.+>(( )?&)?$"), stl_summary_flags, true);
    AddCXXSummary(libcxx_category_sp, lldb_private::formatters::LibcxxContainerSummaryProvider, "libc++ std::multiset summary provider", ConstString("^std::__1::multiset<.+>(( )?&)?$"), stl_summary_flags, true);
    AddCXXSummary(libcxx_category_sp, lldb_private::formatters::LibcxxContainerSummaryProvider, "libc++ std::multimap summary provider", ConstString("^std::__1::multimap<.+>(( )?&)?$"), stl_summary_flags, true);

    stl_summary_flags.SetSkipPointers(true);
    AddStringSummary(libcxx_category_sp, "{${var.__ptr_%S}} (strong=${var.count} weak=${var.weak_count})}", ConstString("^std::__1::shared_ptr<.+>(( )?&)?$"), stl_summary_flags, true);
    AddStringSummary(libcxx_category_sp, "{${var.__ptr_%S}} (strong=${var.count} weak=${var.weak_count})}", ConstString("^std::__1::weak_ptr<.+>(( )?&)?$"), stl_summary_flags, true);
    
    AddCXXSynthetic(libcxx_category_sp, lldb_private::formatters::LibCxxVectorIteratorSyntheticFrontEndCreator, "std::vector iterator synthetic children", ConstString("^std::__1::__wrap_iter<.+>$"), stl_synth_flags, true);
    
    AddCXXSynthetic(libcxx_category_sp, lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEndCreator, "std::map iterator synthetic children", ConstString("^std::__1::__map_iterator<.+>$"), stl_synth_flags, true);
    
#endif
}

void
FormatManager::LoadSystemFormatters()
{
    
    TypeSummaryImpl::Flags string_flags;
    string_flags.SetCascades(true)
    .SetSkipPointers(true)
    .SetSkipReferences(false)
    .SetDontShowChildren(true)
    .SetDontShowValue(false)
    .SetShowMembersOneLiner(false)
    .SetHideItemNames(false);
    
    lldb::TypeSummaryImplSP string_format(new StringSummaryFormat(string_flags, "${var%s}"));
    
    
    lldb::TypeSummaryImplSP string_array_format(new StringSummaryFormat(TypeSummaryImpl::Flags().SetCascades(false)
                                                                        .SetSkipPointers(true)
                                                                        .SetSkipReferences(false)
                                                                        .SetDontShowChildren(true)
                                                                        .SetDontShowValue(true)
                                                                        .SetShowMembersOneLiner(false)
                                                                        .SetHideItemNames(false),
                                                                        "${var%s}"));
    
    lldb::RegularExpressionSP any_size_char_arr(new RegularExpression("char \\[[0-9]+\\]"));
    
    TypeCategoryImpl::SharedPointer sys_category_sp = GetCategory(m_system_category_name);
    
    sys_category_sp->GetSummaryNavigator()->Add(ConstString("char *"), string_format);
    sys_category_sp->GetSummaryNavigator()->Add(ConstString("const char *"), string_format);
    sys_category_sp->GetSummaryNavigator()->Add(ConstString("unsigned char *"), string_format);
    sys_category_sp->GetSummaryNavigator()->Add(ConstString("const unsigned char *"), string_format);
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
    
#ifndef LLDB_DISABLE_PYTHON
    // FIXME because of a bug in the FormatNavigator we need to add a summary for both X* and const X* (<rdar://problem/12717717>)
    AddCXXSummary(sys_category_sp, lldb_private::formatters::Char16StringSummaryProvider, "char16_t * summary provider", ConstString("char16_t *"), string_flags);
    AddCXXSummary(sys_category_sp, lldb_private::formatters::Char16StringSummaryProvider, "char16_t * summary provider", ConstString("const char16_t *"), string_flags);
    
    AddCXXSummary(sys_category_sp, lldb_private::formatters::Char32StringSummaryProvider, "char32_t * summary provider", ConstString("char32_t *"), string_flags);
    AddCXXSummary(sys_category_sp, lldb_private::formatters::Char32StringSummaryProvider, "char32_t * summary provider", ConstString("const char32_t *"), string_flags);
    
    AddCXXSummary(sys_category_sp, lldb_private::formatters::WCharStringSummaryProvider, "wchar_t * summary provider", ConstString("wchar_t *"), string_flags);
    AddCXXSummary(sys_category_sp, lldb_private::formatters::WCharStringSummaryProvider, "wchar_t * summary provider", ConstString("const wchar_t *"), string_flags);
    
    AddCXXSummary(sys_category_sp, lldb_private::formatters::Char16StringSummaryProvider, "unichar * summary provider", ConstString("unichar *"), string_flags);
    AddCXXSummary(sys_category_sp, lldb_private::formatters::Char16StringSummaryProvider, "unichar * summary provider", ConstString("const unichar *"), string_flags);
    
    TypeSummaryImpl::Flags widechar_flags;
    widechar_flags.SetDontShowValue(true)
    .SetSkipPointers(true)
    .SetSkipReferences(false)
    .SetCascades(true)
    .SetDontShowChildren(true)
    .SetHideItemNames(true)
    .SetShowMembersOneLiner(false);
    
    AddCXXSummary(sys_category_sp, lldb_private::formatters::Char16SummaryProvider, "char16_t summary provider", ConstString("char16_t"), widechar_flags);
    AddCXXSummary(sys_category_sp, lldb_private::formatters::Char32SummaryProvider, "char32_t summary provider", ConstString("char32_t"), widechar_flags);
    AddCXXSummary(sys_category_sp, lldb_private::formatters::WCharSummaryProvider, "wchar_t summary provider", ConstString("wchar_t"), widechar_flags);

    AddCXXSummary(sys_category_sp, lldb_private::formatters::Char16SummaryProvider, "unichar summary provider", ConstString("unichar"), widechar_flags);
    
#endif
}

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
    AddCXXSummary(objc_category_sp, lldb_private::formatters::ObjCSELSummaryProvider<false>, "SEL summary provider", ConstString("SEL"), objc_flags);
    AddCXXSummary(objc_category_sp, lldb_private::formatters::ObjCSELSummaryProvider<false>, "SEL summary provider", ConstString("struct objc_selector"), objc_flags);
    AddCXXSummary(objc_category_sp, lldb_private::formatters::ObjCSELSummaryProvider<false>, "SEL summary provider", ConstString("objc_selector"), objc_flags);
    AddCXXSummary(objc_category_sp, lldb_private::formatters::ObjCSELSummaryProvider<true>, "SEL summary provider", ConstString("objc_selector *"), objc_flags);
    AddCXXSummary(objc_category_sp, lldb_private::formatters::ObjCSELSummaryProvider<true>, "SEL summary provider", ConstString("SEL *"), objc_flags);
    
    AddCXXSummary(objc_category_sp, lldb_private::formatters::ObjCClassSummaryProvider, "Class summary provider", ConstString("Class"), objc_flags);
    
    SyntheticChildren::Flags class_synth_flags;
    class_synth_flags.SetCascades(true).SetSkipPointers(false).SetSkipReferences(false);
    
    AddCXXSynthetic(objc_category_sp, lldb_private::formatters::ObjCClassSyntheticFrontEndCreator, "Class synthetic children", ConstString("Class"), class_synth_flags);
#endif // LLDB_DISABLE_PYTHON

    objc_flags.SetSkipPointers(false);

    TypeCategoryImpl::SharedPointer corefoundation_category_sp = GetCategory(m_corefoundation_category_name);

    AddStringSummary(corefoundation_category_sp,
                     "${var.years} years, ${var.months} months, ${var.days} days, ${var.hours} hours, ${var.minutes} minutes ${var.seconds} seconds",
                     ConstString("CFGregorianUnits"),
                     objc_flags);
    AddStringSummary(corefoundation_category_sp,
                     "location=${var.location} length=${var.length}",
                     ConstString("CFRange"),
                     objc_flags);
    AddStringSummary(corefoundation_category_sp,
                     "(x=${var.x}, y=${var.y})",
                     ConstString("NSPoint"),
                     objc_flags);
    AddStringSummary(corefoundation_category_sp,
                     "location=${var.location}, length=${var.length}",
                     ConstString("NSRange"),
                     objc_flags);
    AddStringSummary(corefoundation_category_sp,
                     "${var.origin}, ${var.size}",
                     ConstString("NSRect"),
                     objc_flags);
    AddStringSummary(corefoundation_category_sp,
                     "(${var.origin}, ${var.size}), ...",
                     ConstString("NSRectArray"),
                     objc_flags);
    AddStringSummary(objc_category_sp,
                     "(width=${var.width}, height=${var.height})",
                     ConstString("NSSize"),
                     objc_flags);
    
    TypeCategoryImpl::SharedPointer coregraphics_category_sp = GetCategory(m_coregraphics_category_name);
    
    AddStringSummary(coregraphics_category_sp,
                     "(width=${var.width}, height=${var.height})",
                     ConstString("CGSize"),
                     objc_flags);
    AddStringSummary(coregraphics_category_sp,
                     "(x=${var.x}, y=${var.y})",
                     ConstString("CGPoint"),
                     objc_flags);
    AddStringSummary(coregraphics_category_sp,
                     "origin=${var.origin} size=${var.size}",
                     ConstString("CGRect"),
                     objc_flags);
    
    TypeCategoryImpl::SharedPointer coreservices_category_sp = GetCategory(m_coreservices_category_name);
    
    AddStringSummary(coreservices_category_sp,
                     "red=${var.red} green=${var.green} blue=${var.blue}",
                     ConstString("RGBColor"),
                     objc_flags);
    AddStringSummary(coreservices_category_sp,
                     "(t=${var.top}, l=${var.left}, b=${var.bottom}, r=${var.right})",
                     ConstString("Rect"),
                     objc_flags);
    AddStringSummary(coreservices_category_sp,
                     "(v=${var.v}, h=${var.h})",
                     ConstString("Point"),
                     objc_flags);
    AddStringSummary(coreservices_category_sp,
                     "${var.month}/${var.day}/${var.year}  ${var.hour} :${var.minute} :${var.second} dayOfWeek:${var.dayOfWeek}",
                     ConstString("DateTimeRect *"),
                     objc_flags);
    AddStringSummary(coreservices_category_sp,
                     "${var.ld.month}/${var.ld.day}/${var.ld.year} ${var.ld.hour} :${var.ld.minute} :${var.ld.second} dayOfWeek:${var.ld.dayOfWeek}",
                     ConstString("LongDateRect"),
                     objc_flags);
    AddStringSummary(coreservices_category_sp,
                     "(x=${var.x}, y=${var.y})",
                     ConstString("HIPoint"),
                     objc_flags);
    AddStringSummary(coreservices_category_sp,
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
    
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>, "NSSet summary", ConstString("NSSet"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>, "NSMutableSet summary", ConstString("NSMutableSet"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSSetSummaryProvider<true>, "CFSetRef summary", ConstString("CFSetRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSSetSummaryProvider<true>, "CFMutableSetRef summary", ConstString("CFMutableSetRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>, "__NSCFSet summary", ConstString("__NSCFSet"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>, "__NSSetI summary", ConstString("__NSSetI"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>, "__NSSetM summary", ConstString("__NSSetM"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>, "NSCountedSet summary", ConstString("NSCountedSet"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>, "NSMutableSet summary", ConstString("NSMutableSet"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>, "NSOrderedSet summary", ConstString("NSOrderedSet"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>, "__NSOrderedSetI summary", ConstString("__NSOrderedSetI"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSSetSummaryProvider<false>, "__NSOrderedSetM summary", ConstString("__NSOrderedSetM"), appkit_flags);

    // AddSummary(appkit_category_sp, "${var.key%@} -> ${var.value%@}", ConstString("$_lldb_typegen_nspair"), appkit_flags);
    
    appkit_flags.SetDontShowChildren(true);
    
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSArraySyntheticFrontEndCreator, "NSArray synthetic children", ConstString("__NSArrayM"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSArraySyntheticFrontEndCreator, "NSArray synthetic children", ConstString("__NSArrayI"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSArraySyntheticFrontEndCreator, "NSArray synthetic children", ConstString("NSArray"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSArraySyntheticFrontEndCreator, "NSArray synthetic children", ConstString("NSMutableArray"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSArraySyntheticFrontEndCreator, "NSArray synthetic children", ConstString("__NSCFArray"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSArraySyntheticFrontEndCreator, "NSArray synthetic children", ConstString("CFMutableArrayRef"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSArraySyntheticFrontEndCreator, "NSArray synthetic children", ConstString("CFArrayRef"), ScriptedSyntheticChildren::Flags());

    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSDictionarySyntheticFrontEndCreator, "NSDictionary synthetic children", ConstString("__NSDictionaryM"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSDictionarySyntheticFrontEndCreator, "NSDictionary synthetic children", ConstString("__NSDictionaryI"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSDictionarySyntheticFrontEndCreator, "NSDictionary synthetic children", ConstString("NSDictionary"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSDictionarySyntheticFrontEndCreator, "NSDictionary synthetic children", ConstString("NSMutableDictionary"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSDictionarySyntheticFrontEndCreator, "NSDictionary synthetic children", ConstString("CFDictionaryRef"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSDictionarySyntheticFrontEndCreator, "NSDictionary synthetic children", ConstString("CFMutableDictionaryRef"), ScriptedSyntheticChildren::Flags());

    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSSetSyntheticFrontEndCreator, "NSSet synthetic children", ConstString("NSSet"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSSetSyntheticFrontEndCreator, "__NSSetI synthetic children", ConstString("__NSSetI"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSSetSyntheticFrontEndCreator, "__NSSetM synthetic children", ConstString("__NSSetM"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSSetSyntheticFrontEndCreator, "NSMutableSet synthetic children", ConstString("NSMutableSet"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSSetSyntheticFrontEndCreator, "NSOrderedSet synthetic children", ConstString("NSOrderedSet"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSSetSyntheticFrontEndCreator, "__NSOrderedSetI synthetic children", ConstString("__NSOrderedSetI"), ScriptedSyntheticChildren::Flags());
    AddCXXSynthetic(appkit_category_sp, lldb_private::formatters::NSSetSyntheticFrontEndCreator, "__NSOrderedSetM synthetic children", ConstString("__NSOrderedSetM"), ScriptedSyntheticChildren::Flags());
    
    AddCXXSummary(appkit_category_sp,lldb_private::formatters::CFBagSummaryProvider, "CFBag summary provider", ConstString("CFBagRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp,lldb_private::formatters::CFBagSummaryProvider, "CFBag summary provider", ConstString("__CFBag"), appkit_flags);
    AddCXXSummary(appkit_category_sp,lldb_private::formatters::CFBagSummaryProvider, "CFBag summary provider", ConstString("const struct __CFBag"), appkit_flags);
    AddCXXSummary(appkit_category_sp,lldb_private::formatters::CFBagSummaryProvider, "CFBag summary provider", ConstString("CFMutableBagRef"), appkit_flags);
    
    AddCXXSummary(appkit_category_sp,lldb_private::formatters::CFBinaryHeapSummaryProvider, "CFBinaryHeap summary provider", ConstString("CFBinaryHeapRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp,lldb_private::formatters::CFBinaryHeapSummaryProvider, "CFBinaryHeap summary provider", ConstString("__CFBinaryHeap"), appkit_flags);

    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("NSString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("CFStringRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("CFMutableStringRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("NSMutableString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("__NSCFConstantString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("__NSCFString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("NSCFConstantString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("NSCFString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSStringSummaryProvider, "NSString summary provider", ConstString("NSPathStore2"), appkit_flags);
    
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSAttributedStringSummaryProvider, "NSAttributedString summary provider", ConstString("NSAttributedString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSMutableAttributedStringSummaryProvider, "NSMutableAttributedString summary provider", ConstString("NSMutableAttributedString"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSMutableAttributedStringSummaryProvider, "NSMutableAttributedString summary provider", ConstString("NSConcreteMutableAttributedString"), appkit_flags);

    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSBundleSummaryProvider, "NSBundle summary provider", ConstString("NSBundle"), appkit_flags);

    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>, "NSData summary provider", ConstString("NSData"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>, "NSData summary provider", ConstString("NSConcreteData"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>, "NSData summary provider", ConstString("NSConcreteMutableData"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDataSummaryProvider<false>, "NSData summary provider", ConstString("__NSCFData"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDataSummaryProvider<true>, "NSData summary provider", ConstString("CFDataRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDataSummaryProvider<true>, "NSData summary provider", ConstString("CFMutableDataRef"), appkit_flags);

    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSMachPortSummaryProvider, "NSMachPort summary provider", ConstString("NSMachPort"), appkit_flags);

    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSNotificationSummaryProvider, "NSNotification summary provider", ConstString("NSNotification"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSNotificationSummaryProvider, "NSNotification summary provider", ConstString("NSConcreteNotification"), appkit_flags);

    AddStringSummary(appkit_category_sp, "domain: ${var._domain} - code: ${var._code}", ConstString("NSError"), appkit_flags);
    AddStringSummary(appkit_category_sp,"name:${var.name%S} reason:${var.reason%S}",ConstString("NSException"),appkit_flags);

    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSNumberSummaryProvider, "NSNumber summary provider", ConstString("NSNumber"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSNumberSummaryProvider, "NSNumber summary provider", ConstString("__NSCFBoolean"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSNumberSummaryProvider, "NSNumber summary provider", ConstString("__NSCFNumber"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSNumberSummaryProvider, "NSNumber summary provider", ConstString("NSCFBoolean"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSNumberSummaryProvider, "NSNumber summary provider", ConstString("NSCFNumber"), appkit_flags);
    
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider, "NSDecimalNumber summary provider", ConstString("NSDecimalNumber"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider, "NSHost summary provider", ConstString("NSHost"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider, "NSTask summary provider", ConstString("NSTask"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::RuntimeSpecificDescriptionSummaryProvider, "NSValue summary provider", ConstString("NSValue"), appkit_flags);
    
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSURLSummaryProvider, "NSURL summary provider", ConstString("NSURL"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSURLSummaryProvider, "NSURL summary provider", ConstString("CFURLRef"), appkit_flags);
    
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDateSummaryProvider, "NSDate summary provider", ConstString("NSDate"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDateSummaryProvider, "NSDate summary provider", ConstString("__NSDate"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDateSummaryProvider, "NSDate summary provider", ConstString("__NSTaggedDate"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSDateSummaryProvider, "NSDate summary provider", ConstString("NSCalendarDate"), appkit_flags);

    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSTimeZoneSummaryProvider, "NSTimeZone summary provider", ConstString("NSTimeZone"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSTimeZoneSummaryProvider, "NSTimeZone summary provider", ConstString("CFTimeZoneRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSTimeZoneSummaryProvider, "NSTimeZone summary provider", ConstString("__NSTimeZone"), appkit_flags);

    // CFAbsoluteTime is actually a double rather than a pointer to an object
    // we do not care about the numeric value, since it is probably meaningless to users
    appkit_flags.SetDontShowValue(true);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::CFAbsoluteTimeSummaryProvider, "CFAbsoluteTime summary provider", ConstString("CFAbsoluteTime"), appkit_flags);
    appkit_flags.SetDontShowValue(false);
    
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSIndexSetSummaryProvider, "NSIndexSet summary provider", ConstString("NSIndexSet"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::NSIndexSetSummaryProvider, "NSIndexSet summary provider", ConstString("NSMutableIndexSet"), appkit_flags);

    AddStringSummary(appkit_category_sp,
                     "@\"${var.month%d}/${var.day%d}/${var.year%d} ${var.hour%d}:${var.minute%d}:${var.second}\"",
                     ConstString("CFGregorianDate"),
                     appkit_flags);
    
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::CFBitVectorSummaryProvider, "CFBitVector summary provider", ConstString("CFBitVectorRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::CFBitVectorSummaryProvider, "CFBitVector summary provider", ConstString("CFMutableBitVectorRef"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::CFBitVectorSummaryProvider, "CFBitVector summary provider", ConstString("__CFBitVector"), appkit_flags);
    AddCXXSummary(appkit_category_sp, lldb_private::formatters::CFBitVectorSummaryProvider, "CFBitVector summary provider", ConstString("__CFMutableBitVector"), appkit_flags);
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
    
    AddStringSummary(vectors_category_sp,
                     "${var.uint128}",
                     ConstString("builtin_type_vec128"),
                     objc_flags);

    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("float [4]"),
                     vector_flags);
    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("int32_t [4]"),
                     vector_flags);
    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("int16_t [8]"),
                     vector_flags);
    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("vDouble"),
                     vector_flags);
    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("vFloat"),
                     vector_flags);
    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("vSInt8"),
                     vector_flags);
    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("vSInt16"),
                     vector_flags);
    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("vSInt32"),
                     vector_flags);
    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("vUInt16"),
                     vector_flags);
    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("vUInt8"),
                     vector_flags);
    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("vUInt16"),
                     vector_flags);
    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("vUInt32"),
                     vector_flags);
    AddStringSummary(vectors_category_sp,
                     "",
                     ConstString("vBool32"),
                     vector_flags);
}
