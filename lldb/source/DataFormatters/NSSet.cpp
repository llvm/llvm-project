//===-- NSSet.cpp ------------------------------------------------*- C++ -*-===//
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

template<bool cf_style>
bool
lldb_private::formatters::NSSetSummaryProvider (ValueObject& valobj, Stream& stream)
{
    ProcessSP process_sp = valobj.GetProcessSP();
    if (!process_sp)
        return false;
    
    ObjCLanguageRuntime* runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    
    if (!runtime)
        return false;
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(valobj));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return false;
    
    uint32_t ptr_size = process_sp->GetAddressByteSize();
    bool is_64bit = (ptr_size == 8);
    
    lldb::addr_t valobj_addr = valobj.GetValueAsUnsigned(0);
    
    if (!valobj_addr)
        return false;
    
    uint64_t value = 0;
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return false;
    
    if (!strcmp(class_name,"__NSSetI"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
        value &= (is_64bit ? ~0xFC00000000000000UL : ~0xFC000000U);
    }
    else if (!strcmp(class_name,"__NSSetM"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
        value &= (is_64bit ? ~0xFC00000000000000UL : ~0xFC000000U);
    }
    else if (!strcmp(class_name,"__NSCFSet"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + (is_64bit ? 20 : 12), 4, 0, error);
        if (error.Fail())
            return false;
        if (is_64bit)
            value &= ~0x1fff000000000000UL;
    }
    /*else if (!strcmp(class_name,"NSCountedSet"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
        value = process_sp->ReadUnsignedIntegerFromMemory(value + (is_64bit ? 20 : 12), 4, 0, error);
        if (error.Fail())
            return false;
        if (is_64bit)
            value &= ~0x1fff000000000000UL;
    }*/
    else
    {
        if (!ExtractValueFromObjCExpression(valobj, "int", "count", value))
            return false;
    }
    
    stream.Printf("%s%" PRIu64 " %s%s",
                  (cf_style ? "@\"" : ""),
                  value,
                  (cf_style ? (value == 1 ? "value" : "values") : (value == 1 ? "object" : "objects")),
                  (cf_style ? "\"" : ""));
    return true;
}

SyntheticChildrenFrontEnd* lldb_private::formatters::NSSetSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    lldb::ProcessSP process_sp (valobj_sp->GetProcessSP());
    if (!process_sp)
        return NULL;
    ObjCLanguageRuntime *runtime = (ObjCLanguageRuntime*)process_sp->GetLanguageRuntime(lldb::eLanguageTypeObjC);
    if (!runtime)
        return NULL;
    
    if (!valobj_sp->IsPointerType())
    {
        Error error;
        valobj_sp = valobj_sp->AddressOf(error);
        if (error.Fail() || !valobj_sp)
            return NULL;
    }
    
    ObjCLanguageRuntime::ClassDescriptorSP descriptor(runtime->GetClassDescriptor(*valobj_sp.get()));
    
    if (!descriptor.get() || !descriptor->IsValid())
        return NULL;
    
    const char* class_name = descriptor->GetClassName().GetCString();
    
    if (!class_name || !*class_name)
        return NULL;
    
    if (!strcmp(class_name,"__NSSetI"))
    {
        return (new NSSetISyntheticFrontEnd(valobj_sp));
    }
    else if (!strcmp(class_name,"__NSSetM"))
    {
        return (new NSSetMSyntheticFrontEnd(valobj_sp));
    }
    else if ((!strcmp(class_name,"__NSOrderedSetI")) || (!strcmp(class_name,"__NSOrderedSetM")))
    {
        return new NSOrderedSetSyntheticFrontEnd(valobj_sp); // this runs code
    }
    else
    {
        return /*(new NSSetCodeRunningSyntheticFrontEnd(valobj_sp))*/ NULL;
    }
}

lldb_private::formatters::NSSetISyntheticFrontEnd::NSSetISyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_ptr_size(8),
m_data_32(NULL),
m_data_64(NULL)
{
    if (valobj_sp)
        Update();
}

lldb_private::formatters::NSSetISyntheticFrontEnd::~NSSetISyntheticFrontEnd ()
{
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
}

size_t
lldb_private::formatters::NSSetISyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

size_t
lldb_private::formatters::NSSetISyntheticFrontEnd::CalculateNumChildren ()
{
    if (!m_data_32 && !m_data_64)
        return 0;
    return (m_data_32 ? m_data_32->_used : m_data_64->_used);
}

bool
lldb_private::formatters::NSSetISyntheticFrontEnd::Update()
{
    m_children.clear();
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
    m_ptr_size = 0;
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    Error error;
    if (valobj_sp->IsPointerType())
    {
        valobj_sp = valobj_sp->Dereference(error);
        if (error.Fail() || !valobj_sp)
            return false;
    }
    error.Clear();
    lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
    if (!process_sp)
        return false;
    m_ptr_size = process_sp->GetAddressByteSize();
    uint64_t data_location = valobj_sp->GetAddressOf() + m_ptr_size;
    if (m_ptr_size == 4)
    {
        m_data_32 = new DataDescriptor_32();
        process_sp->ReadMemory (data_location, m_data_32, sizeof(DataDescriptor_32), error);
    }
    else
    {
        m_data_64 = new DataDescriptor_64();
        process_sp->ReadMemory (data_location, m_data_64, sizeof(DataDescriptor_64), error);
    }
    if (error.Fail())
        return false;
    m_data_ptr = data_location + m_ptr_size;
    return false;
}

bool
lldb_private::formatters::NSSetISyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

lldb::ValueObjectSP
lldb_private::formatters::NSSetISyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    uint32_t num_children = CalculateNumChildren();
    
    if (idx >= num_children)
        return lldb::ValueObjectSP();
    
    if (m_children.empty())
    {
        // do the scan phase
        lldb::addr_t obj_at_idx = 0;
        
        uint32_t tries = 0;
        uint32_t test_idx = 0;
        
        while(tries < num_children)
        {
            obj_at_idx = m_data_ptr + (test_idx * m_ptr_size);
            ProcessSP process_sp = m_exe_ctx_ref.GetProcessSP();
            if (!process_sp)
                return lldb::ValueObjectSP();
            Error error;
            obj_at_idx = process_sp->ReadPointerFromMemory(obj_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            
            test_idx++;
            
            if (!obj_at_idx)
                continue;
            tries++;
            
            SetItemDescriptor descriptor = {obj_at_idx,lldb::ValueObjectSP()};
            
            m_children.push_back(descriptor);
        }
    }
    
    if (idx >= m_children.size()) // should never happen
        return lldb::ValueObjectSP();
    
    SetItemDescriptor &set_item = m_children[idx];
    if (!set_item.valobj_sp)
    {
        // make the new ValueObject
        StreamString expr;
        expr.Printf("(id)%" PRIu64,set_item.item_ptr);
        StreamString idx_name;
        idx_name.Printf("[%zu]",idx);
        set_item.valobj_sp = ValueObject::CreateValueObjectFromExpression(idx_name.GetData(), expr.GetData(), m_exe_ctx_ref);
    }
    return set_item.valobj_sp;
}

lldb_private::formatters::NSSetMSyntheticFrontEnd::NSSetMSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_ptr_size(8),
m_data_32(NULL),
m_data_64(NULL)
{
    if (valobj_sp)
        Update ();
}

lldb_private::formatters::NSSetMSyntheticFrontEnd::~NSSetMSyntheticFrontEnd ()
{
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
}

size_t
lldb_private::formatters::NSSetMSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

size_t
lldb_private::formatters::NSSetMSyntheticFrontEnd::CalculateNumChildren ()
{
    if (!m_data_32 && !m_data_64)
        return 0;
    return (m_data_32 ? m_data_32->_used : m_data_64->_used);
}

bool
lldb_private::formatters::NSSetMSyntheticFrontEnd::Update()
{
    m_children.clear();
    ValueObjectSP valobj_sp = m_backend.GetSP();
    m_ptr_size = 0;
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
    if (!valobj_sp)
        return false;
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    Error error;
    if (valobj_sp->IsPointerType())
    {
        valobj_sp = valobj_sp->Dereference(error);
        if (error.Fail() || !valobj_sp)
            return false;
    }
    error.Clear();
    lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
    if (!process_sp)
        return false;
    m_ptr_size = process_sp->GetAddressByteSize();
    uint64_t data_location = valobj_sp->GetAddressOf() + m_ptr_size;
    if (m_ptr_size == 4)
    {
        m_data_32 = new DataDescriptor_32();
        process_sp->ReadMemory (data_location, m_data_32, sizeof(DataDescriptor_32), error);
    }
    else
    {
        m_data_64 = new DataDescriptor_64();
        process_sp->ReadMemory (data_location, m_data_64, sizeof(DataDescriptor_64), error);
    }
    if (error.Fail())
        return false;
    return false;
}

bool
lldb_private::formatters::NSSetMSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

lldb::ValueObjectSP
lldb_private::formatters::NSSetMSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    lldb::addr_t m_objs_addr = (m_data_32 ? m_data_32->_objs_addr : m_data_64->_objs_addr);

    uint32_t num_children = CalculateNumChildren();
    
    if (idx >= num_children)
        return lldb::ValueObjectSP();
    
    if (m_children.empty())
    {
        // do the scan phase
        lldb::addr_t obj_at_idx = 0;
        
        uint32_t tries = 0;
        uint32_t test_idx = 0;
        
        while(tries < num_children)
        {
            obj_at_idx = m_objs_addr + (test_idx * m_ptr_size);
            ProcessSP process_sp = m_exe_ctx_ref.GetProcessSP();
            if (!process_sp)
                return lldb::ValueObjectSP();
            Error error;
            obj_at_idx = process_sp->ReadPointerFromMemory(obj_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            
            test_idx++;
            
            if (!obj_at_idx)
                continue;
            tries++;
            
            SetItemDescriptor descriptor = {obj_at_idx,lldb::ValueObjectSP()};
            
            m_children.push_back(descriptor);
        }
    }
    
    if (idx >= m_children.size()) // should never happen
        return lldb::ValueObjectSP();
    
    SetItemDescriptor &set_item = m_children[idx];
    if (!set_item.valobj_sp)
    {
        // make the new ValueObject
        StreamString expr;
        expr.Printf("(id)%" PRIu64,set_item.item_ptr);
        StreamString idx_name;
        idx_name.Printf("[%zu]",idx);
        set_item.valobj_sp = ValueObject::CreateValueObjectFromExpression(idx_name.GetData(), expr.GetData(), m_exe_ctx_ref);
    }
    return set_item.valobj_sp;
}

lldb_private::formatters::NSOrderedSetSyntheticFrontEnd::NSOrderedSetSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_count(UINT32_MAX),
m_children()
{}

size_t
lldb_private::formatters::NSOrderedSetSyntheticFrontEnd::CalculateNumChildren ()
{
    if (m_count != UINT32_MAX)
        return m_count;
    uint64_t count_temp;
    if (ExtractValueFromObjCExpression(m_backend,"unsigned int","count",count_temp))
        return (m_count = count_temp);
    return (m_count = 0);
}

lldb::ValueObjectSP
lldb_private::formatters::NSOrderedSetSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    auto iter = m_children.find(idx);
    if (iter == m_children.end())
    {
        lldb::ValueObjectSP retval_sp;
        if (idx <= m_count)
        {
            retval_sp = CallSelectorOnObject(m_backend, "id", "objectAtIndex", idx);
            if (retval_sp)
            {
                StreamString idx_name;
                idx_name.Printf("[%zu]",idx);
                retval_sp->SetName(ConstString(idx_name.GetData()));
            }
            m_children[idx] = retval_sp;
        }
        return retval_sp;
    }
    else
        return iter->second;
}

bool
lldb_private::formatters::NSOrderedSetSyntheticFrontEnd::Update()
{
    return false;
}

bool
lldb_private::formatters::NSOrderedSetSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::NSOrderedSetSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

lldb_private::formatters::NSOrderedSetSyntheticFrontEnd::~NSOrderedSetSyntheticFrontEnd ()
{
}

template bool
lldb_private::formatters::NSSetSummaryProvider<true> (ValueObject& valobj, Stream& stream);

template bool
lldb_private::formatters::NSSetSummaryProvider<false> (ValueObject& valobj, Stream& stream);
