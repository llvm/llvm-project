//===-- ValueObjectConstResult.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ValueObjectConstResult_h_
#define liblldb_ValueObjectConstResult_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/ValueObject.h"

#include "lldb/Core/ValueObjectConstResultImpl.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A frozen ValueObject copied into host memory
//----------------------------------------------------------------------
class ValueObjectConstResult : public ValueObject
{
public:
    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope,
            lldb::ByteOrder byte_order, 
            uint32_t addr_byte_size,
            lldb::addr_t address = LLDB_INVALID_ADDRESS);

    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope,
            clang::ASTContext *clang_ast,
            void *clang_type,
            const ConstString &name,
            const DataExtractor &data,
            lldb::addr_t address = LLDB_INVALID_ADDRESS);

    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope,
            clang::ASTContext *clang_ast,
            void *clang_type,
            const ConstString &name,
            const lldb::DataBufferSP &result_data_sp,
            lldb::ByteOrder byte_order, 
            uint32_t addr_size,
            lldb::addr_t address = LLDB_INVALID_ADDRESS);

    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope,
            clang::ASTContext *clang_ast,
            void *clang_type,
            const ConstString &name,
            lldb::addr_t address,
            AddressType address_type,
            uint32_t addr_byte_size);

    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope,
            clang::ASTContext *clang_ast,
            Value &value,
            const ConstString &name);

    // When an expression fails to evaluate, we return an error
    static lldb::ValueObjectSP
    Create (ExecutionContextScope *exe_scope,
            const Error& error);

    virtual ~ValueObjectConstResult();

    virtual uint64_t
    GetByteSize();

    virtual lldb::ValueType
    GetValueType() const;

    virtual size_t
    CalculateNumChildren();

    virtual ConstString
    GetTypeName();

    virtual bool
    IsInScope ();

    virtual bool
    SetClangAST (clang::ASTContext *ast)
    {
        m_clang_ast = ast;
        return true;
    }

    void
    SetByteSize (size_t size);
    
    virtual lldb::ValueObjectSP
    Dereference (Error &error);
    
    virtual ValueObject *
    CreateChildAtIndex (size_t idx, bool synthetic_array_member, int32_t synthetic_index);
    
    virtual lldb::ValueObjectSP
    GetSyntheticChildAtOffset(uint32_t offset, const ClangASTType& type, bool can_create);
    
    virtual lldb::ValueObjectSP
    AddressOf (Error &error);
    
    virtual lldb::addr_t
    GetAddressOf (bool scalar_is_load_address = true,
                  AddressType *address_type = NULL);
    
    virtual size_t
    GetPointeeData (DataExtractor& data,
                    uint32_t item_idx = 0,
					uint32_t item_count = 1);
    
    virtual lldb::addr_t
    GetLiveAddress()
    {
        return m_impl.GetLiveAddress();
    }
    
    virtual void
    SetLiveAddress(lldb::addr_t addr = LLDB_INVALID_ADDRESS,
                   AddressType address_type = eAddressTypeLoad)
    {
        m_impl.SetLiveAddress(addr,
                              address_type);
    }
    
    virtual lldb::ValueObjectSP
    GetDynamicValue (lldb::DynamicValueType valueType);

protected:
    virtual bool
    UpdateValue ();
    
    virtual clang::ASTContext *
    GetClangASTImpl ();
    
    virtual lldb::clang_type_t
    GetClangTypeImpl ();

    clang::ASTContext *m_clang_ast; // The clang AST that the clang type comes from
    ConstString m_type_name;
    uint64_t m_byte_size;
    
    ValueObjectConstResultImpl m_impl;

private:
    friend class ValueObjectConstResultImpl;
    ValueObjectConstResult (ExecutionContextScope *exe_scope,
                            lldb::ByteOrder byte_order, 
                            uint32_t addr_byte_size,
                            lldb::addr_t address);

    ValueObjectConstResult (ExecutionContextScope *exe_scope,
                            clang::ASTContext *clang_ast,
                            void *clang_type,
                            const ConstString &name,
                            const DataExtractor &data,
                            lldb::addr_t address);

    ValueObjectConstResult (ExecutionContextScope *exe_scope,
                            clang::ASTContext *clang_ast,
                            void *clang_type,
                            const ConstString &name,
                            const lldb::DataBufferSP &result_data_sp,
                            lldb::ByteOrder byte_order, 
                            uint32_t addr_size,
                            lldb::addr_t address);

    ValueObjectConstResult (ExecutionContextScope *exe_scope,
                            clang::ASTContext *clang_ast,
                            void *clang_type,
                            const ConstString &name,
                            lldb::addr_t address,
                            AddressType address_type,
                            uint32_t addr_byte_size);

    ValueObjectConstResult (ExecutionContextScope *exe_scope,
                            clang::ASTContext *clang_ast,
                            const Value &value,
                            const ConstString &name);

    ValueObjectConstResult (ExecutionContextScope *exe_scope,
                            const Error& error);

    DISALLOW_COPY_AND_ASSIGN (ValueObjectConstResult);
};

} // namespace lldb_private

#endif  // liblldb_ValueObjectConstResult_h_
