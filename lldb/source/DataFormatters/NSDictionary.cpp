//===-- NSDictionary.cpp ------------------------------------------*- C++ -*-===//
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

#include "clang/AST/DeclCXX.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;

static ClangASTType
GetLLDBNSPairType (TargetSP target_sp)
{
    ClangASTType clang_type = ClangASTType();

    ClangASTContext *target_ast_context = target_sp->GetScratchClangASTContext();

    if (!target_ast_context)
        return clang_type;
    
    const char* type_name = "__lldb_autogen_nspair";
    
    clang::IdentifierInfo &myIdent = target_ast_context->getASTContext()->Idents.get(type_name);
    clang::DeclarationName myName = target_ast_context->getASTContext()->DeclarationNames.getIdentifier(&myIdent);

    clang::DeclContext::lookup_const_result result = target_ast_context->getASTContext()->getTranslationUnitDecl()->lookup(myName);

    clang_type_t opaque_type = NULL;
    
    for (clang::NamedDecl *named_decl : result)
    {
        if (const clang::CXXRecordDecl *record_decl = llvm::dyn_cast<clang::CXXRecordDecl>(named_decl))
        {
            opaque_type = clang::QualType(record_decl->getTypeForDecl(), 0).getAsOpaquePtr();
            break;
        }
        else
        {
            // somebody else (the user?) has defined a type with the magic name already - fail!!!
            return clang_type;
        }
    }

    if (!opaque_type)
    {
        opaque_type = target_ast_context->CreateRecordType(NULL, lldb::eAccessPublic, type_name, clang::TTK_Struct, lldb::eLanguageTypeC);
        
        if (!opaque_type)
            return clang_type;
        
        target_ast_context->StartTagDeclarationDefinition(opaque_type);
        
        target_ast_context->AddFieldToRecordType(opaque_type, "key", target_ast_context->GetBuiltInType_objc_id(), lldb::eAccessPublic, 0);
        target_ast_context->AddFieldToRecordType(opaque_type, "value", target_ast_context->GetBuiltInType_objc_id(), lldb::eAccessPublic, 0);
        
        target_ast_context->CompleteTagDeclarationDefinition(opaque_type);
    }
    
    clang_type.SetClangType(target_ast_context->getASTContext(), opaque_type);
    return clang_type;
}

template<bool name_entries>
bool
lldb_private::formatters::NSDictionarySummaryProvider (ValueObject& valobj, Stream& stream)
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
    
    if (!strcmp(class_name,"__NSDictionaryI"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
        value &= (is_64bit ? ~0xFC00000000000000UL : ~0xFC000000U);
    }
    else if (!strcmp(class_name,"__NSDictionaryM"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + ptr_size, ptr_size, 0, error);
        if (error.Fail())
            return false;
        value &= (is_64bit ? ~0xFC00000000000000UL : ~0xFC000000U);
    }
    /*else if (!strcmp(class_name,"__NSCFDictionary"))
    {
        Error error;
        value = process_sp->ReadUnsignedIntegerFromMemory(valobj_addr + (is_64bit ? 20 : 12), 4, 0, error);
        if (error.Fail())
            return false;
        if (is_64bit)
            value &= ~0x0f1f000000000000UL;
    }*/
    else
    {
        if (!ExtractValueFromObjCExpression(valobj, "int", "count", value))
            return false;
    }
    
    stream.Printf("%s%" PRIu64 " %s%s",
                  (name_entries ? "@\"" : ""),
                  value,
                  (name_entries ? (value == 1 ? "entry" : "entries") : (value == 1 ? "key/value pair" : "key/value pairs")),
                  (name_entries ? "\"" : ""));
    return true;
}

SyntheticChildrenFrontEnd* lldb_private::formatters::NSDictionarySyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP valobj_sp)
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
    
    if (!strcmp(class_name,"__NSDictionaryI"))
    {
        return (new NSDictionaryISyntheticFrontEnd(valobj_sp));
    }
    else if (!strcmp(class_name,"__NSDictionaryM"))
    {
        return (new NSDictionaryMSyntheticFrontEnd(valobj_sp));
    }
    else
    {
        return (new NSDictionaryCodeRunningSyntheticFrontEnd(valobj_sp));
    }
}

lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::NSDictionaryCodeRunningSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get())
{}

size_t
lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::CalculateNumChildren ()
{
    uint64_t count = 0;
    if (ExtractValueFromObjCExpression(m_backend, "int", "count", count))
        return count;
    return 0;
}

lldb::ValueObjectSP
lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    StreamString idx_name;
    idx_name.Printf("[%zu]",idx);
    StreamString key_fetcher_expr;
    key_fetcher_expr.Printf("(id)[(NSArray*)[(id)0x%" PRIx64 " allKeys] objectAtIndex:%zu]",m_backend.GetPointerValue(),idx);
    StreamString value_fetcher_expr;
    value_fetcher_expr.Printf("(id)[(id)0x%" PRIx64 " objectForKey:(%s)]",m_backend.GetPointerValue(),key_fetcher_expr.GetData());
    StreamString object_fetcher_expr;
    object_fetcher_expr.Printf("struct __lldb_autogen_nspair { id key; id value; } _lldb_valgen_item; _lldb_valgen_item.key = %s; _lldb_valgen_item.value = %s; _lldb_valgen_item;",key_fetcher_expr.GetData(),value_fetcher_expr.GetData());
    lldb::ValueObjectSP child_sp;
    m_backend.GetTargetSP()->EvaluateExpression(object_fetcher_expr.GetData(), m_backend.GetFrameSP().get(), child_sp,
                                                EvaluateExpressionOptions().SetKeepInMemory(true));
    if (child_sp)
        child_sp->SetName(ConstString(idx_name.GetData()));
    return child_sp;
}

bool
lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::Update()
{
    return false;
}

bool
lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

size_t
lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    return 0;
}

lldb_private::formatters::NSDictionaryCodeRunningSyntheticFrontEnd::~NSDictionaryCodeRunningSyntheticFrontEnd ()
{}

lldb_private::formatters::NSDictionaryISyntheticFrontEnd::NSDictionaryISyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_ptr_size(8),
m_order(lldb::eByteOrderInvalid),
m_data_32(NULL),
m_data_64(NULL),
m_pair_type()
{
}

lldb_private::formatters::NSDictionaryISyntheticFrontEnd::~NSDictionaryISyntheticFrontEnd ()
{
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
}

size_t
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

size_t
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::CalculateNumChildren ()
{
    if (!m_data_32 && !m_data_64)
        return 0;
    return (m_data_32 ? m_data_32->_used : m_data_64->_used);
}

bool
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::Update()
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
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    Error error;
    error.Clear();
    lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
    if (!process_sp)
        return false;
    m_ptr_size = process_sp->GetAddressByteSize();
    m_order = process_sp->GetByteOrder();
    uint64_t data_location = valobj_sp->GetValueAsUnsigned(0) + m_ptr_size;
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
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

lldb::ValueObjectSP
lldb_private::formatters::NSDictionaryISyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    uint32_t num_children = CalculateNumChildren();
    
    if (idx >= num_children)
        return lldb::ValueObjectSP();
    
    if (m_children.empty())
    {
        // do the scan phase
        lldb::addr_t key_at_idx = 0, val_at_idx = 0;
        
        uint32_t tries = 0;
        uint32_t test_idx = 0;
        
        while(tries < num_children)
        {
            key_at_idx = m_data_ptr + (2*test_idx * m_ptr_size);
            val_at_idx = key_at_idx + m_ptr_size;
            ProcessSP process_sp = m_exe_ctx_ref.GetProcessSP();
            if (!process_sp)
                return lldb::ValueObjectSP();
            Error error;
            key_at_idx = process_sp->ReadPointerFromMemory(key_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            val_at_idx = process_sp->ReadPointerFromMemory(val_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            
            test_idx++;
            
            if (!key_at_idx || !val_at_idx)
                continue;
            tries++;
            
            DictionaryItemDescriptor descriptor = {key_at_idx,val_at_idx,lldb::ValueObjectSP()};
            
            m_children.push_back(descriptor);
        }
    }
    
    if (idx >= m_children.size()) // should never happen
        return lldb::ValueObjectSP();
    
    DictionaryItemDescriptor &dict_item = m_children[idx];
    if (!dict_item.valobj_sp)
    {
        if (!m_pair_type.IsValid())
        {
            TargetSP target_sp(m_backend.GetTargetSP());
            if (!target_sp)
                return ValueObjectSP();
            m_pair_type = GetLLDBNSPairType(target_sp);
        }
        if (!m_pair_type.IsValid())
            return ValueObjectSP();
        
        DataBufferSP buffer_sp(new DataBufferHeap(2*m_ptr_size,0));
        
        if (m_ptr_size == 8)
        {
            uint64_t *data_ptr = (uint64_t *)buffer_sp->GetBytes();
            *data_ptr = dict_item.key_ptr;
            *(data_ptr+1) = dict_item.val_ptr;
        }
        else
        {
            uint32_t *data_ptr = (uint32_t *)buffer_sp->GetBytes();
            *data_ptr = dict_item.key_ptr;
            *(data_ptr+1) = dict_item.val_ptr;
        }
        
        StreamString idx_name;
        idx_name.Printf("[%zu]",idx);
        DataExtractor data(buffer_sp, m_order, m_ptr_size);
        dict_item.valobj_sp = ValueObject::CreateValueObjectFromData(idx_name.GetData(), data, m_exe_ctx_ref, m_pair_type);
    }
    return dict_item.valobj_sp;
}

lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::NSDictionaryMSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp) :
SyntheticChildrenFrontEnd(*valobj_sp.get()),
m_exe_ctx_ref(),
m_ptr_size(8),
m_order(lldb::eByteOrderInvalid),
m_data_32(NULL),
m_data_64(NULL),
m_pair_type()
{
}

lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::~NSDictionaryMSyntheticFrontEnd ()
{
    delete m_data_32;
    m_data_32 = NULL;
    delete m_data_64;
    m_data_64 = NULL;
}

size_t
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::GetIndexOfChildWithName (const ConstString &name)
{
    const char* item_name = name.GetCString();
    uint32_t idx = ExtractIndexFromString(item_name);
    if (idx < UINT32_MAX && idx >= CalculateNumChildren())
        return UINT32_MAX;
    return idx;
}

size_t
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::CalculateNumChildren ()
{
    if (!m_data_32 && !m_data_64)
        return 0;
    return (m_data_32 ? m_data_32->_used : m_data_64->_used);
}

bool
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::Update()
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
    m_exe_ctx_ref = valobj_sp->GetExecutionContextRef();
    Error error;
    error.Clear();
    lldb::ProcessSP process_sp(valobj_sp->GetProcessSP());
    if (!process_sp)
        return false;
    m_ptr_size = process_sp->GetAddressByteSize();
    m_order = process_sp->GetByteOrder();
    uint64_t data_location = valobj_sp->GetValueAsUnsigned(0) + m_ptr_size;
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
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::MightHaveChildren ()
{
    return true;
}

lldb::ValueObjectSP
lldb_private::formatters::NSDictionaryMSyntheticFrontEnd::GetChildAtIndex (size_t idx)
{
    lldb::addr_t m_keys_ptr = (m_data_32 ? m_data_32->_keys_addr : m_data_64->_keys_addr);
    lldb::addr_t m_values_ptr = (m_data_32 ? m_data_32->_objs_addr : m_data_64->_objs_addr);
    
    uint32_t num_children = CalculateNumChildren();
    
    if (idx >= num_children)
        return lldb::ValueObjectSP();
    
    if (m_children.empty())
    {
        // do the scan phase
        lldb::addr_t key_at_idx = 0, val_at_idx = 0;
        
        uint32_t tries = 0;
        uint32_t test_idx = 0;
        
        while(tries < num_children)
        {
            key_at_idx = m_keys_ptr + (test_idx * m_ptr_size);
            val_at_idx = m_values_ptr + (test_idx * m_ptr_size);;
            ProcessSP process_sp = m_exe_ctx_ref.GetProcessSP();
            if (!process_sp)
                return lldb::ValueObjectSP();
            Error error;
            key_at_idx = process_sp->ReadPointerFromMemory(key_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            val_at_idx = process_sp->ReadPointerFromMemory(val_at_idx, error);
            if (error.Fail())
                return lldb::ValueObjectSP();
            
            test_idx++;
            
            if (!key_at_idx || !val_at_idx)
                continue;
            tries++;
            
            DictionaryItemDescriptor descriptor = {key_at_idx,val_at_idx,lldb::ValueObjectSP()};
            
            m_children.push_back(descriptor);
        }
    }
    
    if (idx >= m_children.size()) // should never happen
        return lldb::ValueObjectSP();
    
    DictionaryItemDescriptor &dict_item = m_children[idx];
    if (!dict_item.valobj_sp)
    {
        if (!m_pair_type.IsValid())
        {
            TargetSP target_sp(m_backend.GetTargetSP());
            if (!target_sp)
                return ValueObjectSP();
            m_pair_type = GetLLDBNSPairType(target_sp);
        }
        if (!m_pair_type.IsValid())
            return ValueObjectSP();
        
        DataBufferSP buffer_sp(new DataBufferHeap(2*m_ptr_size,0));
        
        if (m_ptr_size == 8)
        {
            uint64_t *data_ptr = (uint64_t *)buffer_sp->GetBytes();
            *data_ptr = dict_item.key_ptr;
            *(data_ptr+1) = dict_item.val_ptr;
        }
        else
        {
            uint32_t *data_ptr = (uint32_t *)buffer_sp->GetBytes();
            *data_ptr = dict_item.key_ptr;
            *(data_ptr+1) = dict_item.val_ptr;
        }
        
        StreamString idx_name;
        idx_name.Printf("[%zu]",idx);
        DataExtractor data(buffer_sp, m_order, m_ptr_size);
        dict_item.valobj_sp = ValueObject::CreateValueObjectFromData(idx_name.GetData(), data, m_exe_ctx_ref, m_pair_type);
    }
    return dict_item.valobj_sp;
}

template bool
lldb_private::formatters::NSDictionarySummaryProvider<true> (ValueObject&, Stream&) ;

template bool
lldb_private::formatters::NSDictionarySummaryProvider<false> (ValueObject&, Stream&) ;
