//===-- TypeCategory.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_TypeCategory_h_
#define lldb_TypeCategory_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/DataFormatters/FormatNavigator.h"

namespace lldb_private {    
    class TypeCategoryImpl
    {
    private:
        
        typedef FormatNavigator<ConstString, TypeSummaryImpl> SummaryNavigator;
        typedef FormatNavigator<lldb::RegularExpressionSP, TypeSummaryImpl> RegexSummaryNavigator;
        
        typedef FormatNavigator<ConstString, TypeFilterImpl> FilterNavigator;
        typedef FormatNavigator<lldb::RegularExpressionSP, TypeFilterImpl> RegexFilterNavigator;
        
#ifndef LLDB_DISABLE_PYTHON
        typedef FormatNavigator<ConstString, ScriptedSyntheticChildren> SynthNavigator;
        typedef FormatNavigator<lldb::RegularExpressionSP, ScriptedSyntheticChildren> RegexSynthNavigator;
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
        GetSummaryForType (lldb::TypeNameSpecifierImplSP type_sp);
        
        FilterNavigator::MapValueType
        GetFilterForType (lldb::TypeNameSpecifierImplSP type_sp);
        
#ifndef LLDB_DISABLE_PYTHON
        SynthNavigator::MapValueType
        GetSyntheticForType (lldb::TypeNameSpecifierImplSP type_sp);
#endif
        
        lldb::TypeNameSpecifierImplSP
        GetTypeNameSpecifierForSummaryAtIndex (size_t index);
        
        SummaryNavigator::MapValueType
        GetSummaryAtIndex (size_t index);
        
        FilterNavigator::MapValueType
        GetFilterAtIndex (size_t index);
        
        lldb::TypeNameSpecifierImplSP
        GetTypeNameSpecifierForFilterAtIndex (size_t index);
        
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
        GetSyntheticAtIndex (size_t index);
        
        lldb::TypeNameSpecifierImplSP
        GetTypeNameSpecifierForSyntheticAtIndex (size_t index);
        
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
        
        typedef std::shared_ptr<TypeCategoryImpl> SharedPointer;
        
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
        Enable (bool value, uint32_t position);
        
        void
        Disable ()
        {
            Enable(false, UINT32_MAX);
        }
        
        friend class TypeCategoryMap;
        
        friend class FormatNavigator<ConstString, TypeSummaryImpl>;
        friend class FormatNavigator<lldb::RegularExpressionSP, TypeSummaryImpl>;
        
        friend class FormatNavigator<ConstString, TypeFilterImpl>;
        friend class FormatNavigator<lldb::RegularExpressionSP, TypeFilterImpl>;
        
#ifndef LLDB_DISABLE_PYTHON
        friend class FormatNavigator<ConstString, ScriptedSyntheticChildren>;
        friend class FormatNavigator<lldb::RegularExpressionSP, ScriptedSyntheticChildren>;
#endif // #ifndef LLDB_DISABLE_PYTHON
    };
    
} // namespace lldb_private

#endif	// lldb_TypeCategory_h_
