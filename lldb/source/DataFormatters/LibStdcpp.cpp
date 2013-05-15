//===-- LibStdcpp.cpp ---------------------------------------------*- C++ -*-===//
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

lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::LibstdcppVectorBoolSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
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
lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::CalculateNumChildren ()
{
    return m_count;
}

lldb::ValueObjectSP
lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::GetChildAtIndex (size_t idx)
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

/*((std::vector<std::allocator<bool> >) vBool = {
 (std::_Bvector_base<std::allocator<bool> >) std::_Bvector_base<std::allocator<bool> > = {
 (std::_Bvector_base<std::allocator<bool> >::_Bvector_impl) _M_impl = {
 (std::_Bit_iterator) _M_start = {
 (std::_Bit_iterator_base) std::_Bit_iterator_base = {
 (_Bit_type *) _M_p = 0x0016b160
 (unsigned int) _M_offset = 0
 }
 }
 (std::_Bit_iterator) _M_finish = {
 (std::_Bit_iterator_base) std::_Bit_iterator_base = {
 (_Bit_type *) _M_p = 0x0016b16c
 (unsigned int) _M_offset = 16
 }
 }
 (_Bit_type *) _M_end_of_storage = 0x0016b170
 }
 }
 }
 */

bool
lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::Update()
{
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    
    ValueObjectSP m_impl_sp(valobj_sp->GetChildMemberWithName(ConstString("_M_impl"), true));
    if (!m_impl_sp)
        return false;
    
    ValueObjectSP m_start_sp(m_impl_sp->GetChildMemberWithName(ConstString("_M_start"), true));
    ValueObjectSP m_finish_sp(m_impl_sp->GetChildMemberWithName(ConstString("_M_finish"), true));
    
    ValueObjectSP start_p_sp, finish_p_sp, finish_offset_sp;
    
    if (!m_start_sp || !m_finish_sp)
        return false;
    
    start_p_sp = m_start_sp->GetChildMemberWithName(ConstString("_M_p"), true);
    finish_p_sp = m_finish_sp->GetChildMemberWithName(ConstString("_M_p"), true);
    finish_offset_sp = m_finish_sp->GetChildMemberWithName(ConstString("_M_offset"), true);
    
    if (!start_p_sp || !finish_offset_sp || !finish_p_sp)
        return false;
    
    m_base_data_address = start_p_sp->GetValueAsUnsigned(0);
    if (!m_base_data_address)
        return false;
    
    lldb::addr_t end_data_address(finish_p_sp->GetValueAsUnsigned(0));
    if (!end_data_address)
        return false;
    
    if (end_data_address < m_base_data_address)
        return false;
    else
        m_count = finish_offset_sp->GetValueAsUnsigned(0) + (end_data_address-m_base_data_address)*8;
    
    return true;
}

bool
lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (!m_count || !m_base_data_address)
        return UINT32_MAX;
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEnd::~LibstdcppVectorBoolSyntheticFrontEnd ()
{}

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibstdcppVectorBoolSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    if (!valobj_sp)
        return NULL;
    return (new LibstdcppVectorBoolSyntheticFrontEnd(valobj_sp));
}

/*
 (std::_Rb_tree_iterator<std::pair<const int, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > >) ibeg = {
 (_Base_ptr) _M_node = 0x0000000100103910 {
 (std::_Rb_tree_color) _M_color = _S_black
 (std::_Rb_tree_node_base::_Base_ptr) _M_parent = 0x00000001001038c0
 (std::_Rb_tree_node_base::_Base_ptr) _M_left = 0x0000000000000000
 (std::_Rb_tree_node_base::_Base_ptr) _M_right = 0x0000000000000000
 }
 }
 */

lldb_private::formatters::LibstdcppMapIteratorSyntheticFrontEnd::LibstdcppMapIteratorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_pair_address(0),
m_pair_type(),
m_options(),
m_pair_sp()
{
    if (valobj_sp)
        Update();
    m_options.SetCoerceToId(false)
    .SetUnwindOnError(true)
    .SetKeepInMemory(true)
    .SetUseDynamic(lldb::eDynamicCanRunTarget);
}

bool
lldb_private::formatters::LibstdcppMapIteratorSyntheticFrontEnd::Update()
{
    ValueObjectSP valobj_sp = m_backend.GetSP();
    if (!valobj_sp)
        return false;
    
    TargetSP target_sp(valobj_sp->GetTargetSP());
    
    if (!target_sp)
        return false;
    
    bool is_64bit = (target_sp->GetArchitecture().GetAddressByteSize() == 8);
    
    if (!valobj_sp)
        return false;
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    
    ValueObjectSP _M_node_sp(valobj_sp->GetChildMemberWithName(ConstString("_M_node"), true));
    if (!_M_node_sp)
        return false;
    
    m_pair_address = _M_node_sp->GetValueAsUnsigned(0);
    if (m_pair_address == 0)
        return false;
    
    m_pair_address += (is_64bit ? 32 : 16);
    
    ClangASTType my_type(valobj_sp->GetClangAST(),valobj_sp->GetClangType());
    if (ClangASTContext::GetNumTemplateArguments(valobj_sp->GetClangAST(),valobj_sp->GetClangType()) >= 1)
    {
        TemplateArgumentKind kind;
        clang_type_t pair_type = ClangASTContext::GetTemplateArgument(valobj_sp->GetClangAST(),valobj_sp->GetClangType(), 0, kind);
        if (kind != eTemplateArgumentKindType && kind != eTemplateArgumentKindTemplate && kind != eTemplateArgumentKindTemplateExpansion)
            return false;
        m_pair_type = ClangASTType(valobj_sp->GetClangAST(),pair_type);
    }
    else
        return false;
    
    return true;
}

size_t
lldb_private::formatters::LibstdcppMapIteratorSyntheticFrontEnd::CalculateNumChildren ()
{
    return 2;
}

lldb::ValueObjectSP
lldb_private::formatters::LibstdcppMapIteratorSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    if (m_pair_address == 0)
        return lldb::ValueObjectSP();
    if (m_pair_type.GetASTContext() == NULL ||
        m_pair_type.GetOpaqueQualType() == NULL)
        return lldb::ValueObjectSP();
    if (!m_pair_sp)
        m_pair_sp = ValueObject::CreateValueObjectFromAddress("pair", m_pair_address, m_exe_ctx_ref, m_pair_type);
    if (m_pair_sp)
        return m_pair_sp->GetChildAtIndex(idx, true);
    return lldb::ValueObjectSP();
}

bool
lldb_private::formatters::LibstdcppMapIteratorSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::LibstdcppMapIteratorSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    if (name == ConstString("first"))
        return 0;
    if (name == ConstString("second"))
        return 1;
    return UINT32_MAX;
}

lldb_private::formatters::LibstdcppMapIteratorSyntheticFrontEnd::~LibstdcppMapIteratorSyntheticFrontEnd ()
{}

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibstdcppMapIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    if (!valobj_sp)
        return NULL;
    return (new LibstdcppMapIteratorSyntheticFrontEnd(valobj_sp));
}

/*
 (lldb) fr var ibeg --ptr-depth 1
 (__gnu_cxx::__normal_iterator<int *, std::vector<int, std::allocator<int> > >) ibeg = {
 _M_current = 0x00000001001037a0 {
 *_M_current = 1
 }
 }
 */

SyntheticChildrenFrontEnd*
lldb_private::formatters::LibStdcppVectorIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
{
    static ConstString g_item_name;
    if (!g_item_name)
        g_item_name.SetCString("_M_current");
    if (!valobj_sp)
        return NULL;
    return (new VectorIteratorSyntheticFrontEnd(valobj_sp,g_item_name));
}
