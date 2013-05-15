//===-- DataVisualization.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/DataFormatters/DataVisualization.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#include "lldb/Core/Debugger.h"

using namespace lldb;
using namespace lldb_private;

static FormatManager&
GetFormatManager()
{
    static FormatManager g_format_manager;
    return g_format_manager;
}

void
DataVisualization::ForceUpdate ()
{
    GetFormatManager().Changed();
}

uint32_t
DataVisualization::GetCurrentRevision ()
{
    return GetFormatManager().GetCurrentRevision();
}

lldb::TypeFormatImplSP
DataVisualization::ValueFormats::GetFormat (ValueObject& valobj, lldb::DynamicValueType use_dynamic)
{
    lldb::TypeFormatImplSP entry;
    GetFormatManager().GetValueNavigator().Get(valobj, entry, use_dynamic);
    return entry;
}

lldb::TypeFormatImplSP
DataVisualization::ValueFormats::GetFormat (const ConstString &type)
{
    lldb::TypeFormatImplSP entry;
    GetFormatManager().GetValueNavigator().Get(type, entry);
    return entry;
}

void
DataVisualization::ValueFormats::Add (const ConstString &type, const lldb::TypeFormatImplSP &entry)
{
    GetFormatManager().GetValueNavigator().Add(FormatManager::GetValidTypeName(type),entry);
}

bool
DataVisualization::ValueFormats::Delete (const ConstString &type)
{
    return GetFormatManager().GetValueNavigator().Delete(type);
}

void
DataVisualization::ValueFormats::Clear ()
{
    GetFormatManager().GetValueNavigator().Clear();
}

void
DataVisualization::ValueFormats::LoopThrough (TypeFormatImpl::ValueCallback callback, void* callback_baton)
{
    GetFormatManager().GetValueNavigator().LoopThrough(callback, callback_baton);
}

size_t
DataVisualization::ValueFormats::GetCount ()
{
    return GetFormatManager().GetValueNavigator().GetCount();
}

lldb::TypeNameSpecifierImplSP
DataVisualization::ValueFormats::GetTypeNameSpecifierForFormatAtIndex (size_t index)
{
    return GetFormatManager().GetValueNavigator().GetTypeNameSpecifierAtIndex(index);
}

lldb::TypeFormatImplSP
DataVisualization::ValueFormats::GetFormatAtIndex (size_t index)
{
    return GetFormatManager().GetValueNavigator().GetAtIndex(index);
}

lldb::TypeSummaryImplSP
DataVisualization::GetSummaryFormat (ValueObject& valobj,
                                     lldb::DynamicValueType use_dynamic)
{
    return GetFormatManager().GetSummaryFormat(valobj, use_dynamic);
}

lldb::TypeSummaryImplSP
DataVisualization::GetSummaryForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    return GetFormatManager().GetSummaryForType(type_sp);
}

#ifndef LLDB_DISABLE_PYTHON
lldb::SyntheticChildrenSP
DataVisualization::GetSyntheticChildren (ValueObject& valobj,
                                         lldb::DynamicValueType use_dynamic)
{
    return GetFormatManager().GetSyntheticChildren(valobj, use_dynamic);
}
#endif

#ifndef LLDB_DISABLE_PYTHON
lldb::SyntheticChildrenSP
DataVisualization::GetSyntheticChildrenForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    return GetFormatManager().GetSyntheticChildrenForType(type_sp);
}
#endif

lldb::TypeFilterImplSP
DataVisualization::GetFilterForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    return GetFormatManager().GetFilterForType(type_sp);
}

#ifndef LLDB_DISABLE_PYTHON
lldb::ScriptedSyntheticChildrenSP
DataVisualization::GetSyntheticForType (lldb::TypeNameSpecifierImplSP type_sp)
{
    return GetFormatManager().GetSyntheticForType(type_sp);
}
#endif

bool
DataVisualization::AnyMatches (ConstString type_name,
                               TypeCategoryImpl::FormatCategoryItems items,
                               bool only_enabled,
                               const char** matching_category,
                               TypeCategoryImpl::FormatCategoryItems* matching_type)
{
    return GetFormatManager().AnyMatches(type_name,
                                         items,
                                         only_enabled,
                                         matching_category,
                                         matching_type);
}

bool
DataVisualization::Categories::GetCategory (const ConstString &category, lldb::TypeCategoryImplSP &entry,
                                            bool allow_create)
{
    entry = GetFormatManager().GetCategory(category, allow_create);
    return (entry.get() != NULL);
}

void
DataVisualization::Categories::Add (const ConstString &category)
{
    GetFormatManager().GetCategory(category);
}

bool
DataVisualization::Categories::Delete (const ConstString &category)
{
    GetFormatManager().DisableCategory(category);
    return GetFormatManager().DeleteCategory(category);
}

void
DataVisualization::Categories::Clear ()
{
    GetFormatManager().ClearCategories();
}

void
DataVisualization::Categories::Clear (const ConstString &category)
{
    GetFormatManager().GetCategory(category)->Clear(eFormatCategoryItemSummary | eFormatCategoryItemRegexSummary);
}

void
DataVisualization::Categories::Enable (const ConstString& category,
                                       TypeCategoryMap::Position pos)
{
    if (GetFormatManager().GetCategory(category)->IsEnabled())
        GetFormatManager().DisableCategory(category);
    GetFormatManager().EnableCategory(category, pos);
}

void
DataVisualization::Categories::Disable (const ConstString& category)
{
    if (GetFormatManager().GetCategory(category)->IsEnabled() == true)
        GetFormatManager().DisableCategory(category);
}

void
DataVisualization::Categories::Enable (const lldb::TypeCategoryImplSP& category,
                                       TypeCategoryMap::Position pos)
{
    if (category.get())
    {
        if (category->IsEnabled())
            GetFormatManager().DisableCategory(category);
        GetFormatManager().EnableCategory(category, pos);
    }
}

void
DataVisualization::Categories::Disable (const lldb::TypeCategoryImplSP& category)
{
    if (category.get() && category->IsEnabled() == true)
        GetFormatManager().DisableCategory(category);
}

void
DataVisualization::Categories::LoopThrough (FormatManager::CategoryCallback callback, void* callback_baton)
{
    GetFormatManager().LoopThroughCategories(callback, callback_baton);
}

uint32_t
DataVisualization::Categories::GetCount ()
{
    return GetFormatManager().GetCategoriesCount();
}

lldb::TypeCategoryImplSP
DataVisualization::Categories::GetCategoryAtIndex (size_t index)
{
    return GetFormatManager().GetCategoryAtIndex(index);
}

bool
DataVisualization::NamedSummaryFormats::GetSummaryFormat (const ConstString &type, lldb::TypeSummaryImplSP &entry)
{
    return GetFormatManager().GetNamedSummaryNavigator().Get(type,entry);
}

void
DataVisualization::NamedSummaryFormats::Add (const ConstString &type, const lldb::TypeSummaryImplSP &entry)
{
    GetFormatManager().GetNamedSummaryNavigator().Add(FormatManager::GetValidTypeName(type),entry);
}

bool
DataVisualization::NamedSummaryFormats::Delete (const ConstString &type)
{
    return GetFormatManager().GetNamedSummaryNavigator().Delete(type);
}

void
DataVisualization::NamedSummaryFormats::Clear ()
{
    GetFormatManager().GetNamedSummaryNavigator().Clear();
}

void
DataVisualization::NamedSummaryFormats::LoopThrough (TypeSummaryImpl::SummaryCallback callback, void* callback_baton)
{
    GetFormatManager().GetNamedSummaryNavigator().LoopThrough(callback, callback_baton);
}

uint32_t
DataVisualization::NamedSummaryFormats::GetCount ()
{
    return GetFormatManager().GetNamedSummaryNavigator().GetCount();
}
