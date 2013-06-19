//===-- LibCxx.cpp ----------------------------------------------*- C++ -*-===//
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
#include "lldb/Core/Debugger.h"
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

lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::LibcxxVectorBoolSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_count(0),
m_base_data_address(0),
m_options()
{
    if (valobj_sp)
        Update();
    m_options.SetCoerceToId(false)
    .SetUnwindOnError(true)
    .SetKeepInMemory(true)
    .SetUseDynamic(lldb::eDynamicCanRunTarget);
}

size_t
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::CalculateNumChildren ()
{
    return m_count;
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (idx >= m_count)
        return ValueObjectSP();
    if (m_base_data_address == 0 || m_count == 0)
        return ValueObjectSP();
    size_t byte_idx = (idx >> 3); // divide by 8 to get byte index
    size_t bit_index = (idx & 7); // efficient idx % 8 for bit index
    lldb::addr_t byte_location = m_base_data_address + byte_idx;
    ProcessSP process_sp(m_exe_ctx_ref.GetProcessSP());
    if (!process_sp)
        return ValueObjectSP();
    uint8_t byte = 0;
    uint8_t mask = 0;
    Error err;
    size_t bytes_read = process_sp->ReadMemory(byte_location, &byte, 1, err);
    if (err.Fail() || bytes_read == 0)
        return ValueObjectSP();
    switch (bit_index)
    {
        case 0:
            mask = 1; break;
        case 1:
            mask = 2; break;
        case 2:
            mask = 4; break;
        case 3:
            mask = 8; break;
        case 4:
            mask = 16; break;
        case 5:
            mask = 32; break;
        case 6:
            mask = 64; break;
        case 7:
            mask = 128; break;
        default:
            return ValueObjectSP();
    }
    bool bit_set = ((byte & mask) != 0);
    Target& target(process_sp->GetTarget());
    ValueObjectSP retval_sp;
    if (bit_set)
        target.EvaluateExpression("(bool)true", NULL, retval_sp);
    else
        target.EvaluateExpression("(bool)false", NULL, retval_sp);
    StreamString name; name.Printf("[%zu]",idx);
    if (retval_sp)
        retval_sp->SetName(ConstString(name.GetData()));
    return retval_sp;
}

/*(std::__1::vector<std::__1::allocator<bool> >) vBool = {
 __begin_ = 0x00000001001000e0
 __size_ = 56
 __cap_alloc_ = {
 std::__1::__libcpp_compressed_pair_imp<unsigned long, std::__1::allocator<unsigned long> > = {
 __first_ = 1
 }
 }
 }*/

bool
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::Update()
{
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    ValueObjectSP size_sp(valobj_sp->GetChildMemberWithName(ConstString("__size_"), true));
    if (!size_sp)
        return false;
    m_count = size_sp->GetValueAsUnsigned(0);
    if (!m_count)
        return true;
    ValueObjectSP begin_sp(valobj_sp->GetChildMemberWithName(ConstString("__begin_"), true));
    if (!begin_sp)
    {
        m_count = 0;
        return false;
    }
    m_base_data_address = begin_sp->GetValueAsUnsigned(0);
    if (!m_base_data_address)
    {
        m_count = 0;
        return false;
    }
    return false;
}

bool
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (!m_count || !m_base_data_address)
        return UINT32_MAX;
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEnd::~LibcxxVectorBoolSyntheticFrontEnd ()
{}

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibcxxVectorBoolSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    if (!valobj_sp)
        return NULL;
    return (new LibcxxVectorBoolSyntheticFrontEnd(valobj_sp));
}

/*
 (lldb) fr var ibeg --raw --ptr-depth 1
 (std::__1::__map_iterator<std::__1::__tree_iterator<std::__1::pair<int, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, std::__1::__tree_node<std::__1::pair<int, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > >, void *> *, long> >) ibeg = {
 __i_ = {
 __ptr_ = 0x0000000100103870 {
 std::__1::__tree_node_base<void *> = {
 std::__1::__tree_end_node<std::__1::__tree_node_base<void *> *> = {
 __left_ = 0x0000000000000000
 }
 __right_ = 0x0000000000000000
 __parent_ = 0x00000001001038b0
 __is_black_ = true
 }
 __value_ = {
 first = 0
 second = { std::string }
 */

lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::LibCxxMapIteratorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_pair_ptr()
{
    if (valobj_sp)
        Update();
}

bool
lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::Update()
{
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    
    TargetSP target_sp(valobj_sp->GetTargetSP());
    
    if (!target_sp)
        return false;

    if (!valobj_sp)
        return false;

    // this must be a ValueObject* because it is a child of the ValueObject we are producing children for
    // it if were a ValueObjectSP, we would end up with a loop (iterator -> synthetic -> child -> parent == iterator)
    // and that would in turn leak memory by never allowing the ValueObjects to die and free their memory
    m_pair_ptr = valobj_sp->GetValueForExpressionPath(".__i_.__ptr_->__value_",
                                                     NULL,
                                                     NULL,
                                                     NULL,
                                                     ValueObject::GetValueForExpressionPathOptions().DontCheckDotVsArrowSyntax().DontAllowSyntheticChildren(),
                                                     NULL).get();
    
    return false;
}

size_t
lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::CalculateNumChildren ()
{
    return 2;
}

lldb::ValueObjectSP
lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (!m_pair_ptr)
        return lldb::ValueObjectSP();
    return m_pair_ptr->GetChildAtIndex(idx, true);
}

bool
lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (name == ConstString("first"))
        return 0;
    if (name == ConstString("second"))
        return 1;
    return UINT32_MAX;
}

lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEnd::~LibCxxMapIteratorSyntheticFrontEnd ()
{
    // this will be deleted when its parent dies (since it's a child object)
    //delete m_pair_ptr;
}

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibCxxMapIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    if (!valobj_sp)
        return NULL;
    return (new LibCxxMapIteratorSyntheticFrontEnd(valobj_sp));
}

/*
 (lldb) fr var ibeg --raw --ptr-depth 1 -T
 (std::__1::__wrap_iter<int *>) ibeg = {
 (std::__1::__wrap_iter<int *>::iterator_type) __i = 0x00000001001037a0 {
 (int) *__i = 1
 }
 }
*/

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibCxxVectorIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    static ConstString g_item_name;
    if (!g_item_name)
        g_item_name.SetCString("__i");
    if (!valobj_sp)
        return NULL;
    return (new VectorIteratorSyntheticFrontEnd(valobj_sp,g_item_name));
}

lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::LibcxxSharedPtrSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_cntrl(NULL),
m_count_sp(),
m_weak_count_sp(),
m_ptr_size(0),
m_byte_order(lldb::eByteOrderInvalid)
{
    if (valobj_sp)
        Update();
}

size_t
lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::CalculateNumChildren ()
{
    return (m_cntrl ? 1 : 0);
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (!m_cntrl)
        return lldb::ValueObjectSP();

    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return lldb::ValueObjectSP();

    if (idx == 0)
        return valobj_sp->GetChildMemberWithName(ConstString("__ptr_"), true);

    if (idx > 2)
        return lldb::ValueObjectSP();

    if (idx == 1)
    {
        if (!m_count_sp)
        {
            ValueObjectSP shared_owners_sp(m_cntrl->GetChildMemberWithName(ConstString("__shared_owners_"),true));
            if (!shared_owners_sp)
                return lldb::ValueObjectSP();
            uint64_t count = 1 + shared_owners_sp->GetValueAsUnsigned(0);
            DataExtractor data(&count, 8, m_byte_order, m_ptr_size);
            m_count_sp = ValueObject::CreateValueObjectFromData("count", data, valobj_sp->GetExecutionContextRef(), ClangASTType(shared_owners_sp->GetClangAST(), shared_owners_sp->GetClangType()));
        }
        return m_count_sp;
    }
    else /* if (idx == 2) */
    {
        if (!m_weak_count_sp)
        {
            ValueObjectSP shared_weak_owners_sp(m_cntrl->GetChildMemberWithName(ConstString("__shared_weak_owners_"),true));
            if (!shared_weak_owners_sp)
                return lldb::ValueObjectSP();
            uint64_t count = 1 + shared_weak_owners_sp->GetValueAsUnsigned(0);
            DataExtractor data(&count, 8, m_byte_order, m_ptr_size);
            m_weak_count_sp = ValueObject::CreateValueObjectFromData("count", data, valobj_sp->GetExecutionContextRef(), ClangASTType(shared_weak_owners_sp->GetClangAST(), shared_weak_owners_sp->GetClangType()));
        }
        return m_weak_count_sp;
    }
}

bool
lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::Update()
{
    m_count_sp.reset();
    m_weak_count_sp.reset();
    m_cntrl = NULL;
    
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    
    TargetSP target_sp(valobj_sp->GetTargetSP());
    if (!target_sp)
        return false;
    
    m_byte_order = target_sp->GetArchitecture().GetByteOrder();
    m_ptr_size = target_sp->GetArchitecture().GetAddressByteSize();
    
    lldb::ValueObjectSP cntrl_sp(valobj_sp->GetChildMemberWithName(ConstString("__cntrl_"),true));
    
    m_cntrl = cntrl_sp.get(); // need to store the raw pointer to avoid a circular dependency
    return false;
}

bool
lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (name == ConstString("__ptr_"))
        return 0;
    if (name == ConstString("count"))
        return 1;
    if (name == ConstString("weak_count"))
        return 2;
    return UINT32_MAX;
}

lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEnd::~LibcxxSharedPtrSyntheticFrontEnd ()
{}

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibcxxSharedPtrSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    if (!valobj_sp)
        return NULL;
    return (new LibcxxSharedPtrSyntheticFrontEnd(valobj_sp));
}

lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::LibcxxStdVectorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
    SyntheticChildrenFrontEnd(*valobj_sp.get()),
    m_start(NULL),
    m_finish(NULL),
    m_element_type(),
    m_element_size(0),
    m_children()
{
    if (valobj_sp)
        Update();
}

size_t
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::CalculateNumChildren ()
{
    if (!m_start || !m_finish)
        return 0;
    uint64_t start_val = m_start->GetValueAsUnsigned(0);
    uint64_t finish_val = m_finish->GetValueAsUnsigned(0);

    if (start_val == 0 || finish_val == 0)
        return 0;
    
    if (start_val >= finish_val)
        return 0;
    
    size_t num_children = (finish_val - start_val);
    if (num_children % m_element_size)
        return 0;
    return num_children/m_element_size;
}

lldb::ValueObjectSP
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (!m_start || !m_finish)
        return lldb::ValueObjectSP();
    
    auto cached = m_children.find(idx);
    if (cached != m_children.end())
        return cached->second;
    
    uint64_t offset = idx * m_element_size;
    offset = offset + m_start->GetValueAsUnsigned(0);
    StreamString name;
    name.Printf("[%zu]",idx);
    ValueObjectSP child_sp = ValueObject::CreateValueObjectFromAddress(name.GetData(), offset, m_backend.GetExecutionContextRef(), m_element_type);
    m_children[idx] = child_sp;
    return child_sp;
}

bool
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::Update()
{
    m_start = m_finish = NULL;
    m_children.clear();
    ValueObjectSP data_type_finder_sp(m_backend.GetChildMemberWithName(ConstString("__end_cap_"),true));
    if (!data_type_finder_sp)
        return false;
    data_type_finder_sp = data_type_finder_sp->GetChildMemberWithName(ConstString("__first_"),true);
    if (!data_type_finder_sp)
        return false;
    m_element_type = ClangASTType(data_type_finder_sp->GetClangAST(),data_type_finder_sp->GetClangType());
    m_element_type.SetClangType(m_element_type.GetASTContext(), m_element_type.GetPointeeType());
    m_element_size = m_element_type.GetTypeByteSize();
    
    if (m_element_size > 0)
    {
        // store raw pointers or end up with a circular dependency
        m_start = m_backend.GetChildMemberWithName(ConstString("__begin_"),true).get();
        m_finish = m_backend.GetChildMemberWithName(ConstString("__end_"),true).get();
    }
    return false;
}

bool
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (!m_start || !m_finish)
        return UINT32_MAX;
    return ExtractIndexFromString(name.GetCString());
}

lldb_private::formatters::LibcxxStdVectorSyntheticFrontEnd::~LibcxxStdVectorSyntheticFrontEnd ()
{
    // these need to stay around because they are child objects who will follow their parent's life cycle
    // delete m_start;
    // delete m_finish;
}

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibcxxStdVectorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    if (!valobj_sp)
        return NULL;
    return (new LibcxxStdVectorSyntheticFrontEnd(valobj_sp));
}

bool
lldb_private::formatters::LibcxxContainerSummaryProvider (ValueObject& valobj, Stream& stream)
{
    if (valobj.IsPointerType())
    {
        uint64_t value = valobj.GetValueAsUnsigned(0);
        if (!value)
            return false;
        stream.Printf("0x%016" PRIx64 " ", value);
    }
    return Debugger::FormatPrompt("size=${svar%#}", NULL, NULL, NULL, stream, &valobj);
}
