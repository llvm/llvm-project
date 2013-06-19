//===-- LibCxxList.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "lldb/DataFormatters/CXXFormatterFunctions.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/ValueObject.h"
#include "lldb/Core/ValueObjectConstResult.h"
#include "lldb/Host/Endian.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

class MapEntry
{
public:
    MapEntry () {}
    MapEntry (ValueObjectSP entry_sp) : m_entry_sp(entry_sp) {}
    MapEntry (const MapEntry& rhs) : m_entry_sp(rhs.m_entry_sp) {}
    MapEntry (ValueObject* entry) : m_entry_sp(entry ? entry->GetSP() : ValueObjectSP()) {}
    
    ValueObjectSP
    left ()
    {
        if (!m_entry_sp)
            return m_entry_sp;
        return m_entry_sp->GetChildMemberWithName(ConstString("__left_"), true);
    }
    
    ValueObjectSP
    right ()
    {
        if (!m_entry_sp)
            return m_entry_sp;
        return m_entry_sp->GetChildMemberWithName(ConstString("__right_"), true);
    }
    
    ValueObjectSP
    parent ()
    {
        if (!m_entry_sp)
            return m_entry_sp;
        return m_entry_sp->GetChildMemberWithName(ConstString("__parent_"), true);
    }
    
    uint64_t
    value ()
    {
        if (!m_entry_sp)
            return 0;
        return m_entry_sp->GetValueAsUnsigned(0);
    }
    
    bool
    error ()
    {
        if (!m_entry_sp)
            return true;
        return m_entry_sp->GetError().Fail();
    }
    
    bool
    null()
    {
        return (value() == 0);
    }
    
    ValueObjectSP
    GetEntry ()
    {
        return m_entry_sp;
    }
    
    void
    SetEntry (ValueObjectSP entry)
    {
        m_entry_sp = entry;
    }
    
    bool
    operator == (const MapEntry& rhs) const
    {
        return (rhs.m_entry_sp.get() == m_entry_sp.get());
    }
    
private:
    ValueObjectSP m_entry_sp;
};

class MapIterator
{
public:
    MapIterator () {}
    MapIterator (MapEntry entry, size_t depth = 0) : m_entry(entry), m_max_depth(depth), m_error(false) {}
    MapIterator (ValueObjectSP entry, size_t depth = 0) : m_entry(entry), m_max_depth(depth), m_error(false) {}
    MapIterator (const MapIterator& rhs) : m_entry(rhs.m_entry),m_max_depth(rhs.m_max_depth), m_error(false) {}
    MapIterator (ValueObject* entry, size_t depth = 0) : m_entry(entry), m_max_depth(depth), m_error(false) {}
    
    ValueObjectSP
    value ()
    {
        return m_entry.GetEntry();
    }
    
    ValueObjectSP
    advance (size_t count)
    {
        if (m_error)
            return lldb::ValueObjectSP();
        if (count == 0)
            return m_entry.GetEntry();
        if (count == 1)
        {
            next ();
            return m_entry.GetEntry();
        }
        size_t steps = 0;
        while (count > 0)
        {
            if (m_error)
                return lldb::ValueObjectSP();
            next ();
            count--;
            if (m_entry.null())
                return lldb::ValueObjectSP();
            steps++;
            if (steps > m_max_depth)
                return lldb::ValueObjectSP();
        }
        return m_entry.GetEntry();
    }
protected:
    void
    next ()
    {
        m_entry.SetEntry(increment(m_entry.GetEntry()));
    }

private:
    ValueObjectSP
    tree_min (ValueObjectSP x_sp)
    {
        MapEntry x(x_sp);
        if (x.null())
            return ValueObjectSP();
        MapEntry left(x.left());
        size_t steps = 0;
        while (left.null() == false)
        {
            if (left.error())
            {
                m_error = true;
                return lldb::ValueObjectSP();
            }
            x.SetEntry(left.GetEntry());
            left.SetEntry(x.left());
            steps++;
            if (steps > m_max_depth)
                return lldb::ValueObjectSP();
        }
        return x.GetEntry();
    }
    
    ValueObjectSP
    tree_max (ValueObjectSP x_sp)
    {
        MapEntry x(x_sp);
        if (x.null())
            return ValueObjectSP();
        MapEntry right(x.right());
        size_t steps = 0;
        while (right.null() == false)
        {
            if (right.error())
                return lldb::ValueObjectSP();
            x.SetEntry(right.GetEntry());
            right.SetEntry(x.right());
            steps++;
            if (steps > m_max_depth)
                return lldb::ValueObjectSP();
        }
        return x.GetEntry();
    }
    
    bool
    is_left_child (ValueObjectSP x_sp)
    {
        MapEntry x(x_sp);
        if (x.null())
            return false;
        MapEntry rhs(x.parent());
        rhs.SetEntry(rhs.left());
        return x.value() == rhs.value();
    }
    
    ValueObjectSP
    increment (ValueObjectSP x_sp)
    {
        MapEntry node(x_sp);
        if (node.null())
            return ValueObjectSP();
        MapEntry right(node.right());
        if (right.null() == false)
            return tree_min(right.GetEntry());
        size_t steps = 0;
        while (!is_left_child(node.GetEntry()))
        {
            if (node.error())
            {
                m_error = true;
                return lldb::ValueObjectSP();
            }
            node.SetEntry(node.parent());
            steps++;
            if (steps > m_max_depth)
                return lldb::ValueObjectSP();
        }
        return node.parent();
    }
        
    MapEntry m_entry;
    size_t m_max_depth;
    bool m_error;
};

lldb_private::formatters::LibcxxStdMapSyntheticFrontEnd::LibcxxStdMapSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_tree(NULL),
m_root_node(NULL),
m_element_type(),
m_skip_size(UINT32_MAX),
m_count(UINT32_MAX),
m_children()
{
    if (valobj_sp)
        Update();
}

size_t
lldb_private::formatters::LibcxxStdMapSyntheticFrontEnd::CalculateNumChildren ()
{
    if (m_count != UINT32_MAX)
        return m_count;
    if (m_tree == NULL)
        return 0;
    ValueObjectSP m_item(m_tree->GetChildMemberWithName(ConstString("__pair3_"), true));
    if (!m_item)
        return 0;
    m_item = m_item->GetChildMemberWithName(ConstString("__first_"), true);
    if (!m_item)
        return 0;
    m_count = m_item->GetValueAsUnsigned(0);
    return m_count;
}

bool
lldb_private::formatters::LibcxxStdMapSyntheticFrontEnd::GetDataType()
{
    if (m_element_type.GetOpaqueQualType() && m_element_type.GetASTContext())
        return true;
    m_element_type.Clear();
    ValueObjectSP deref;
    Error error;
    deref = m_root_node->Dereference(error);
    if (!deref || error.Fail())
        return false;
    deref = deref->GetChildMemberWithName(ConstString("__value_"), true);
    if (!deref)
        return false;
    m_element_type.SetClangType(deref->GetClangAST(), deref->GetClangType());
    return true;
}

void
lldb_private::formatters::LibcxxStdMapSyntheticFrontEnd::GetValueOffset (const lldb::ValueObjectSP& node)
{
    if (m_skip_size != UINT32_MAX)
        return;
    if (!node)
        return;
    ClangASTType node_type(node->GetClangAST(),node->GetClangType());
    uint64_t bit_offset;
    if (ClangASTContext::GetIndexOfFieldWithName(node->GetClangAST(),node->GetClangType(),"__value_",NULL,&bit_offset) == UINT32_MAX)
        return;
    m_skip_size = bit_offset / 8u;
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxStdMapSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (idx >= CalculateNumChildren())
        return lldb::ValueObjectSP();
    if (m_tree == NULL || m_root_node == NULL)
        return lldb::ValueObjectSP();
    
    auto cached = m_children.find(idx);
    if (cached != m_children.end())
        return cached->second;
    
    bool need_to_skip = (idx > 0);
    MapIterator iterator(m_root_node, CalculateNumChildren());
    ValueObjectSP iterated_sp(iterator.advance(idx));
    if (iterated_sp.get() == NULL)
    {
        // this tree is garbage - stop
        m_tree = NULL; // this will stop all future searches until an Update() happens
        return iterated_sp;
    }
    if (GetDataType())
    {
        if (!need_to_skip)
        {
            Error error;
            iterated_sp = iterated_sp->Dereference(error);
            if (!iterated_sp || error.Fail())
            {
                m_tree = NULL;
                return lldb::ValueObjectSP();
            }
            GetValueOffset(iterated_sp);
            iterated_sp = iterated_sp->GetChildMemberWithName(ConstString("__value_"), true);
            if (!iterated_sp)
            {
                m_tree = NULL;
                return lldb::ValueObjectSP();
            }
        }
        else
        {
            // because of the way our debug info is made, we need to read item 0 first
            // so that we can cache information used to generate other elements
            if (m_skip_size == UINT32_MAX)
                GetChildAtIndex(0);
            if (m_skip_size == UINT32_MAX)
            {
                m_tree = NULL;
                return lldb::ValueObjectSP();
            }
            iterated_sp = iterated_sp->GetSyntheticChildAtOffset(m_skip_size, m_element_type, true);
            if (!iterated_sp)
            {
                m_tree = NULL;
                return lldb::ValueObjectSP();
            }
        }
    }
    else
    {
        m_tree = NULL;
        return lldb::ValueObjectSP();
    }
    // at this point we have a valid 
    // we need to copy current_sp into a new object otherwise we will end up with all items named __value_
    DataExtractor data;
    iterated_sp->GetData(data);
    StreamString name;
    name.Printf("[%zu]",idx);
    return (m_children[idx] = ValueObject::CreateValueObjectFromData(name.GetData(), data, m_backend.GetExecutionContextRef(), m_element_type));
}

bool
lldb_private::formatters::LibcxxStdMapSyntheticFrontEnd::Update()
{
    m_count = UINT32_MAX;
    m_tree = m_root_node = NULL;
    m_children.clear();
    m_tree = m_backend.GetChildMemberWithName(ConstString("__tree_"), true).get();
    if (!m_tree)
        return false;
    m_root_node = m_tree->GetChildMemberWithName(ConstString("__begin_node_"), true).get();
    return false;
}

bool
lldb_private::formatters::LibcxxStdMapSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibcxxStdMapSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    return ExtractIndexFromString(name.GetCString());
}

lldb_private::formatters::LibcxxStdMapSyntheticFrontEnd::~LibcxxStdMapSyntheticFrontEnd ()
{}

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibcxxStdMapSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    if (!valobj_sp)
        return NULL;
    return (new LibcxxStdMapSyntheticFrontEnd(valobj_sp));
}
