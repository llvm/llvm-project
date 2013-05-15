//===-- DataVisualization.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_DataVisualization_h_
#define lldb_DataVisualization_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "lldb/Core/ConstString.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/FormatManager.h"

namespace lldb_private {

// this class is the high-level front-end of LLDB Data Visualization
// code in FormatManager.h/cpp is the low-level implementation of this feature
// clients should refer to this class as the entry-point into the data formatters
// unless they have a good reason to bypass this and go to the backend
class DataVisualization
{
public:
    
    // use this call to force the FM to consider itself updated even when there is no apparent reason for that
    static void
    ForceUpdate();
    
    static uint32_t
    GetCurrentRevision ();
    
    class ValueFormats
    {
    public:
        static lldb::TypeFormatImplSP
        GetFormat (ValueObject& valobj, lldb::DynamicValueType use_dynamic);
        
        static lldb::TypeFormatImplSP
        GetFormat (const ConstString &type);
        
        static void
        Add (const ConstString &type, const lldb::TypeFormatImplSP &entry);
        
        static bool
        Delete (const ConstString &type);
        
        static void
        Clear ();
        
        static void
        LoopThrough (TypeFormatImpl::ValueCallback callback, void* callback_baton);
        
        static size_t
        GetCount ();
        
        static lldb::TypeNameSpecifierImplSP
        GetTypeNameSpecifierForFormatAtIndex (size_t);
        
        static lldb::TypeFormatImplSP
        GetFormatAtIndex (size_t);
    };
    
    static lldb::TypeSummaryImplSP
    GetSummaryFormat(ValueObject& valobj,
                     lldb::DynamicValueType use_dynamic);

    static lldb::TypeSummaryImplSP
    GetSummaryForType (lldb::TypeNameSpecifierImplSP type_sp);
    
#ifndef LLDB_DISABLE_PYTHON
    static lldb::SyntheticChildrenSP
    GetSyntheticChildrenForType (lldb::TypeNameSpecifierImplSP type_sp);
#endif
    
    static lldb::TypeFilterImplSP
    GetFilterForType (lldb::TypeNameSpecifierImplSP type_sp);

#ifndef LLDB_DISABLE_PYTHON
    static lldb::ScriptedSyntheticChildrenSP
    GetSyntheticForType (lldb::TypeNameSpecifierImplSP type_sp);
#endif
    
#ifndef LLDB_DISABLE_PYTHON
    static lldb::SyntheticChildrenSP
    GetSyntheticChildren(ValueObject& valobj,
                         lldb::DynamicValueType use_dynamic);
#endif
    
    static bool
    AnyMatches(ConstString type_name,
               TypeCategoryImpl::FormatCategoryItems items = TypeCategoryImpl::ALL_ITEM_TYPES,
               bool only_enabled = true,
               const char** matching_category = NULL,
               TypeCategoryImpl::FormatCategoryItems* matching_type = NULL);
    
    class NamedSummaryFormats
    {
    public:
        static bool
        GetSummaryFormat (const ConstString &type, lldb::TypeSummaryImplSP &entry);
        
        static void
        Add (const ConstString &type, const lldb::TypeSummaryImplSP &entry);
        
        static bool
        Delete (const ConstString &type);
        
        static void
        Clear ();
        
        static void
        LoopThrough (TypeSummaryImpl::SummaryCallback callback, void* callback_baton);
        
        static uint32_t
        GetCount ();
    };
    
    class Categories
    {
    public:
        
        static bool
        GetCategory (const ConstString &category,
                     lldb::TypeCategoryImplSP &entry,
                     bool allow_create = true);

        static void
        Add (const ConstString &category);
        
        static bool
        Delete (const ConstString &category);
        
        static void
        Clear ();
        
        static void
        Clear (const ConstString &category);
        
        static void
        Enable (const ConstString& category,
                TypeCategoryMap::Position = TypeCategoryMap::Default);
        
        static void
        Disable (const ConstString& category);

        static void
        Enable (const lldb::TypeCategoryImplSP& category,
                TypeCategoryMap::Position = TypeCategoryMap::Default);
        
        static void
        Disable (const lldb::TypeCategoryImplSP& category);
        
        static void
        LoopThrough (FormatManager::CategoryCallback callback, void* callback_baton);
        
        static uint32_t
        GetCount ();
        
        static lldb::TypeCategoryImplSP
        GetCategoryAtIndex (size_t);
    };
};

    
} // namespace lldb_private

#endif	// lldb_DataVisualization_h_
