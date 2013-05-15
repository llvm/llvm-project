//===-- TypeCategory.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/DataFormatters/TypeCategory.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

using namespace lldb;
using namespace lldb_private;

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
    ScriptedSyntheticChildren::SharedPointer synth;
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
    ScriptedSyntheticChildren::SharedPointer synth;
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

TypeCategoryImpl::SummaryNavigator::MapValueType
TypeCategoryImpl::GetSummaryForType (lldb::TypeNameSpecifierImplSP type_sp)
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

TypeCategoryImpl::FilterNavigator::MapValueType
TypeCategoryImpl::GetFilterForType (lldb::TypeNameSpecifierImplSP type_sp)
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
TypeCategoryImpl::SynthNavigator::MapValueType
TypeCategoryImpl::GetSyntheticForType (lldb::TypeNameSpecifierImplSP type_sp)
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
TypeCategoryImpl::GetTypeNameSpecifierForSummaryAtIndex (size_t index)
{
    if (index < m_summary_nav->GetCount())
        return m_summary_nav->GetTypeNameSpecifierAtIndex(index);
    else
        return m_regex_summary_nav->GetTypeNameSpecifierAtIndex(index-m_summary_nav->GetCount());
}

TypeCategoryImpl::SummaryNavigator::MapValueType
TypeCategoryImpl::GetSummaryAtIndex (size_t index)
{
    if (index < m_summary_nav->GetCount())
        return m_summary_nav->GetAtIndex(index);
    else
        return m_regex_summary_nav->GetAtIndex(index-m_summary_nav->GetCount());
}

TypeCategoryImpl::FilterNavigator::MapValueType
TypeCategoryImpl::GetFilterAtIndex (size_t index)
{
    if (index < m_filter_nav->GetCount())
        return m_filter_nav->GetAtIndex(index);
    else
        return m_regex_filter_nav->GetAtIndex(index-m_filter_nav->GetCount());
}

lldb::TypeNameSpecifierImplSP
TypeCategoryImpl::GetTypeNameSpecifierForFilterAtIndex (size_t index)
{
    if (index < m_filter_nav->GetCount())
        return m_filter_nav->GetTypeNameSpecifierAtIndex(index);
    else
        return m_regex_filter_nav->GetTypeNameSpecifierAtIndex(index-m_filter_nav->GetCount());
}

#ifndef LLDB_DISABLE_PYTHON
TypeCategoryImpl::SynthNavigator::MapValueType
TypeCategoryImpl::GetSyntheticAtIndex (size_t index)
{
    if (index < m_synth_nav->GetCount())
        return m_synth_nav->GetAtIndex(index);
    else
        return m_regex_synth_nav->GetAtIndex(index-m_synth_nav->GetCount());
}

lldb::TypeNameSpecifierImplSP
TypeCategoryImpl::GetTypeNameSpecifierForSyntheticAtIndex (size_t index)
{
    if (index < m_synth_nav->GetCount())
        return m_synth_nav->GetTypeNameSpecifierAtIndex(index);
    else
        return m_regex_synth_nav->GetTypeNameSpecifierAtIndex(index - m_synth_nav->GetCount());
}
#endif

void
TypeCategoryImpl::Enable (bool value, uint32_t position)
{
    Mutex::Locker locker(m_mutex);
    m_enabled = value;
    m_enabled_position = position;
    if (m_change_listener)
        m_change_listener->Changed();
}
