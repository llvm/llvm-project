//===-- ClangASTType.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangASTType_h_
#define liblldb_ClangASTType_h_

#include <string>
#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"

namespace lldb_private {

//----------------------------------------------------------------------
// A class that can carry around a clang ASTContext and a opaque clang 
// QualType. A clang::QualType can be easily reconstructed from an
// opaque clang type and often the ASTContext is needed when doing 
// various type related tasks, so this class allows both items to travel
// in a single very lightweight class that can be used. There are many
// static equivalents of the member functions that allow the ASTContext
// and the opaque clang QualType to be specified for ease of use and
// to avoid code duplication.
//----------------------------------------------------------------------
class ClangASTType
{
public:
    
    ClangASTType (clang::ASTContext *ast_context, lldb::clang_type_t type) :
        m_type (type),
        m_ast  (ast_context) 
    {
    }
    
    ClangASTType (const ClangASTType &tw) :
        m_type (tw.m_type),
        m_ast  (tw.m_ast)
    {
    }
    
    ClangASTType () :
        m_type (0),
        m_ast  (0)
    {
    }
    
    virtual ~ClangASTType();
    
    const ClangASTType &
    operator= (const ClangASTType &atb)
    {
        m_type = atb.m_type;
        m_ast = atb.m_ast;
        return *this;
    }
    
    bool
    IsValid () const
    {
        return m_type != NULL && m_ast != NULL;
    }

    lldb::clang_type_t
    GetOpaqueQualType() const
    { 
        return m_type; 
    }
    
    clang::ASTContext *
    GetASTContext() const
    { 
        return m_ast; 
    }

    ConstString
    GetConstTypeName ();
    
    ConstString
    GetConstQualifiedTypeName ();

    static ConstString
    GetConstTypeName (clang::ASTContext *ast,
                      lldb::clang_type_t clang_type);
    
    static ConstString
    GetConstQualifiedTypeName (clang::ASTContext *ast,
                               lldb::clang_type_t clang_type);

    static std::string
    GetTypeNameForQualType (clang::ASTContext *ast,
                            clang::QualType qual_type);

    static std::string
    GetTypeNameForOpaqueQualType (clang::ASTContext *ast,
                                  lldb::clang_type_t opaque_qual_type);

    uint32_t
    GetClangTypeBitWidth ();

    static uint32_t
    GetClangTypeBitWidth (clang::ASTContext *ast_context, lldb::clang_type_t opaque_clang_qual_type);

    size_t
    GetTypeBitAlign ();
    
    static size_t
    GetTypeBitAlign (clang::ASTContext *ast_context, lldb::clang_type_t clang_type);

    lldb::LanguageType
    GetMinimumLanguage ();

    static lldb::LanguageType
    GetMinimumLanguage (clang::ASTContext *ctx,
                        lldb::clang_type_t clang_type);

    static lldb::TypeClass
    GetTypeClass (clang::ASTContext *ast_context, 
                  lldb::clang_type_t clang_type);

    void
    DumpValue (ExecutionContext *exe_ctx,
               Stream *s,
               lldb::Format format,
               const DataExtractor &data,
               uint32_t data_offset,
               size_t data_byte_size,
               uint32_t bitfield_bit_size,
               uint32_t bitfield_bit_offset,
               bool show_types,
               bool show_summary,
               bool verbose,
               uint32_t depth);

    static void
    DumpValue (clang::ASTContext *ast_context,
               lldb::clang_type_t opaque_clang_qual_type,
               ExecutionContext *exe_ctx,
               Stream *s,
               lldb::Format format,
               const DataExtractor &data,
               uint32_t data_offset,
               size_t data_byte_size,
               uint32_t bitfield_bit_size,
               uint32_t bitfield_bit_offset,
               bool show_types,
               bool show_summary,
               bool verbose,
               uint32_t depth);

    bool
    DumpTypeValue (Stream *s,
                   lldb::Format format,
                   const DataExtractor &data,
                   uint32_t data_offset,
                   size_t data_byte_size,
                   uint32_t bitfield_bit_size,
                   uint32_t bitfield_bit_offset,
                   ExecutionContextScope *exe_scope);
    
    
    static bool
    DumpTypeValue (clang::ASTContext *ast_context,
                   lldb::clang_type_t opaque_clang_qual_type,
                   Stream *s,
                   lldb::Format format,
                   const DataExtractor &data,
                   uint32_t data_offset,
                   size_t data_byte_size,
                   uint32_t bitfield_bit_size,
                   uint32_t bitfield_bit_offset,
                   ExecutionContextScope *exe_scope);

    void
    DumpSummary (ExecutionContext *exe_ctx,
                 Stream *s,
                 const DataExtractor &data,
                 uint32_t data_offset,
                 size_t data_byte_size);
                 
    
    static void
    DumpSummary (clang::ASTContext *ast_context,
                 lldb::clang_type_t opaque_clang_qual_type,
                 ExecutionContext *exe_ctx,
                 Stream *s,
                 const DataExtractor &data,
                 uint32_t data_offset,
                 size_t data_byte_size);
    
    void
    DumpTypeDescription (Stream *s);
    
    static void
    DumpTypeDescription (clang::ASTContext *ast_context,
                         lldb::clang_type_t opaque_clang_qual_type,
                         Stream *s);
    
    void DumpTypeCode (Stream *s);
    
    static void
    DumpTypeCode (void *type,
                  Stream *s);
                         
    lldb::Encoding
    GetEncoding (uint32_t &count);                 

    static lldb::Encoding
    GetEncoding (lldb::clang_type_t opaque_clang_qual_type, uint32_t &count);

    lldb::Format
    GetFormat ();
                 
    static lldb::Format
    GetFormat (lldb::clang_type_t opaque_clang_qual_type);
    
    uint32_t
    GetTypeByteSize() const;
    
    static uint32_t
    GetTypeByteSize(clang::ASTContext *ast_context,
                    lldb::clang_type_t opaque_clang_qual_type);

    bool
    GetValueAsScalar (const DataExtractor &data,
                      uint32_t data_offset,
                      size_t data_byte_size,
                      Scalar &value);

    static bool
    GetValueAsScalar (clang::ASTContext *ast_context,
                      lldb::clang_type_t opaque_clang_qual_type,
                      const DataExtractor &data,
                      uint32_t data_offset,
                      size_t data_byte_size,
                      Scalar &value);


    bool
    IsDefined();

    static bool
    IsDefined (lldb::clang_type_t opaque_clang_qual_type);

    bool
    IsConst();
    
    static bool
    IsConst (lldb::clang_type_t opaque_clang_qual_type);
    
    bool
    SetValueFromScalar (const Scalar &value,
                        Stream &strm);

    static bool
    SetValueFromScalar (clang::ASTContext *ast_context,
                        lldb::clang_type_t opaque_clang_qual_type,
                        const Scalar &value,
                        Stream &strm);

    void
    SetClangType (clang::ASTContext *ast, lldb::clang_type_t type)
    {
        m_type = type;
        m_ast = ast;
    }

    bool
    ReadFromMemory (ExecutionContext *exe_ctx,
                    lldb::addr_t addr,
                    AddressType address_type,
                    DataExtractor &data);

    static bool
    ReadFromMemory (clang::ASTContext *ast_context,
                    lldb::clang_type_t opaque_clang_qual_type,
                    ExecutionContext *exe_ctx,
                    lldb::addr_t addr,
                    AddressType address_type,
                    DataExtractor &data);

    bool
    WriteToMemory (ExecutionContext *exe_ctx,
                   lldb::addr_t addr,
                   AddressType address_type,
                   StreamString &new_value);

    static bool
    WriteToMemory (clang::ASTContext *ast_context,
                   lldb::clang_type_t opaque_clang_qual_type,
                   ExecutionContext *exe_ctx,
                   lldb::addr_t addr,
                   AddressType address_type,
                   StreamString &new_value);

    lldb::clang_type_t
    GetPointeeType ();

    static lldb::clang_type_t
    GetPointeeType (lldb::clang_type_t opaque_clang_qual_type);
    
    lldb::clang_type_t
    GetArrayElementType (uint32_t& stride);
    
    static lldb::clang_type_t
    GetArrayElementType (clang::ASTContext* ast,
                         lldb::clang_type_t opaque_clang_qual_type,
						 uint32_t& stride);
    
    lldb::clang_type_t
    GetPointerType () const;
    
    static lldb::clang_type_t
    GetPointerType (clang::ASTContext *ast_context,
                    lldb::clang_type_t opaque_clang_qual_type);

    static lldb::clang_type_t
    RemoveFastQualifiers (lldb::clang_type_t);

    static clang::CXXRecordDecl *
    GetAsCXXRecordDecl (lldb::clang_type_t opaque_clang_qual_type);

    void
    Clear()
    {
        m_type = NULL;
        m_ast = NULL;
    }

private:
    lldb::clang_type_t m_type;
    clang::ASTContext *m_ast;
};
    
bool operator == (const ClangASTType &lhs, const ClangASTType &rhs);
bool operator != (const ClangASTType &lhs, const ClangASTType &rhs);

    
} // namespace lldb_private

#endif // #ifndef liblldb_ClangASTType_h_
