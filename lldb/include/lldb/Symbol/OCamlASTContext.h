//===-- OCamlASTContext.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_OCamlASTContext_h_
#define liblldb_OCamlASTContext_h_

// C Includes
// C++ Includes
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/ConstString.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Symbol/CompilerType.h"

namespace lldb_private
{

class OCamlASTContext : public TypeSystem
{
  public:
    class OCamlType;
    typedef std::map<ConstString, std::unique_ptr<OCamlType>> OCamlTypeMap;

    OCamlASTContext();
    ~OCamlASTContext() override;

    ConstString
    GetPluginName() override;

    uint32_t
    GetPluginVersion() override;

    static ConstString
    GetPluginNameStatic ();

    static lldb::TypeSystemSP
    CreateInstance (lldb::LanguageType language, Module *module, Target *target,
                    const char *compiler_options);

    static void
    EnumerateSupportedLanguages(std::set<lldb::LanguageType> &languages_for_types,
                                std::set<lldb::LanguageType> &languages_for_expressions);

    static void
    Initialize ();

    static void
    Terminate ();

    DWARFASTParser *GetDWARFParser() override;

    void
    SetAddressByteSize(int byte_size)
    {
        m_pointer_byte_size = byte_size;
    }

    static bool classof(const TypeSystem *ts)
    {
        return ts->getKind() == TypeSystem::eKindOCaml;
    }

    ConstString
    DeclGetName (void *opaque_decl) override
    {
        return ConstString();
    }

    bool
    DeclContextIsStructUnionOrClass(void *opaque_decl_ctx) override
    {
        return false;
    }

    ConstString
    DeclContextGetName(void *opaque_decl_ctx) override
    {
        return ConstString();
    }

    ConstString
    DeclContextGetScopeQualifiedName(void *opaque_decl_ctx) override
    {
        return ConstString();
    }

    bool
    DeclContextIsClassMethod(void *opaque_decl_ctx, lldb::LanguageType *language_ptr,
                             bool *is_instance_method_ptr,
                             ConstString *language_object_name_ptr) override
    {
        return false;
    }

    bool SupportsLanguage (lldb::LanguageType language) override;
    uint32_t GetPointerByteSize() override;

    bool
    IsArrayType(lldb::opaque_compiler_type_t type, CompilerType *element_type, 
                uint64_t *size, bool *is_incomplete) override;

    bool IsAggregateType(lldb::opaque_compiler_type_t type) override;

    bool IsCharType(lldb::opaque_compiler_type_t type) override;

    bool IsCompleteType(lldb::opaque_compiler_type_t type) override;

    bool IsDefined(lldb::opaque_compiler_type_t type) override;

    bool IsFloatingPointType(lldb::opaque_compiler_type_t type, uint32_t &count, bool &is_complex) override;

    bool IsFunctionType(lldb::opaque_compiler_type_t type, bool *is_variadic_ptr = nullptr) override;

    size_t GetNumberOfFunctionArguments(lldb::opaque_compiler_type_t type) override;

    CompilerType GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type, const size_t index) override;

    bool IsFunctionPointerType(lldb::opaque_compiler_type_t type) override;

    bool IsBlockPointerType (lldb::opaque_compiler_type_t type, 
                             CompilerType *function_pointer_type_ptr) override;

    bool 
    IsIntegerType(lldb::opaque_compiler_type_t type, bool &is_signed) override;

    bool
    IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                          CompilerType *target_type, // Can pass NULL
                          bool check_cplusplus, bool check_objc,
                          bool check_swift) override;

    bool IsPointerType(lldb::opaque_compiler_type_t type, CompilerType *pointee_type = nullptr) override;

    bool IsScalarType(lldb::opaque_compiler_type_t type) override;

    bool IsVoidType(lldb::opaque_compiler_type_t type) override;

    bool GetCompleteType(lldb::opaque_compiler_type_t type) override;

    ConstString GetTypeName(lldb::opaque_compiler_type_t type) override;

    uint32_t
    GetTypeInfo(lldb::opaque_compiler_type_t type,
                CompilerType *pointee_or_element_compiler_type = nullptr) override;

    lldb::LanguageType GetMinimumLanguage(lldb::opaque_compiler_type_t type) override;

    lldb::TypeClass GetTypeClass(lldb::opaque_compiler_type_t type) override;

    CompilerType
    GetArrayElementType(lldb::opaque_compiler_type_t type, uint64_t *stride = nullptr) override;

    CompilerType GetCanonicalType(lldb::opaque_compiler_type_t type) override;

    CompilerType
    GetInstanceType(lldb::opaque_compiler_type_t type) override;

    int GetFunctionArgumentCount(lldb::opaque_compiler_type_t type) override;

    CompilerType
    GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type, size_t idx) override;

    CompilerType
    GetFunctionReturnType(lldb::opaque_compiler_type_t type) override;

    size_t
    GetNumMemberFunctions(lldb::opaque_compiler_type_t type) override;

    TypeMemberFunctionImpl
    GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type, size_t idx) override;

    CompilerType
    GetPointeeType(lldb::opaque_compiler_type_t type) override;

    CompilerType
    GetPointerType(lldb::opaque_compiler_type_t type) override;

    uint64_t
    GetBitSize(lldb::opaque_compiler_type_t type, ExecutionContextScope *exe_scope) override;

    uint64_t
    GetByteStride (lldb::opaque_compiler_type_t type) override;

    lldb::Encoding
    GetEncoding(lldb::opaque_compiler_type_t type, uint64_t &count) override;

    lldb::Format
    GetFormat(lldb::opaque_compiler_type_t type) override;

    uint32_t
    GetNumChildren(lldb::opaque_compiler_type_t type, bool omit_empty_base_classes) override;

    lldb::BasicType
    GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) override;

    CompilerType
    GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding, size_t bit_size) override;

    uint32_t
    GetNumFields(lldb::opaque_compiler_type_t type) override;

    CompilerType
    GetFieldAtIndex(lldb::opaque_compiler_type_t type, size_t idx, std::string &name, uint64_t *bit_offset_ptr, 
                    uint32_t *bitfield_bit_size_ptr, bool *is_bitfield_ptr) override;

    uint32_t
    GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) override
    {
        return 0;
    }

    uint32_t
    GetNumVirtualBaseClasses(lldb::opaque_compiler_type_t type) override
    {
        return 0;
    }

    CompilerType
    GetDirectBaseClassAtIndex(lldb::opaque_compiler_type_t type, size_t idx, uint32_t *bit_offset_ptr) override
    {
        return CompilerType();
    }

    CompilerType
    GetVirtualBaseClassAtIndex(lldb::opaque_compiler_type_t type, size_t idx, uint32_t *bit_offset_ptr) override
    {
        return CompilerType();
    }

    CompilerType 
    GetChildCompilerTypeAtIndex(lldb::opaque_compiler_type_t type,
                                ExecutionContext *exe_ctx, size_t idx,
                                bool transparent_pointers, bool omit_empty_base_classes,
                                bool ignore_array_bounds, std::string &child_name,
                                uint32_t &child_byte_size, int32_t &child_byte_offset,
                                uint32_t &child_bitfield_bit_size,
                                uint32_t &child_bitfield_bit_offset, bool &child_is_base_class,
                                bool &child_is_deref_of_parent, ValueObject *valobj,
                                uint64_t &language_flags) override;

    uint32_t 
    GetIndexOfChildWithName(lldb::opaque_compiler_type_t type, const char *name, bool omit_empty_base_classes) override;

    size_t
    GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type,
                                  const char *name, bool omit_empty_base_classes,
                                  std::vector<uint32_t> &child_indexes) override;

    size_t
    GetNumTemplateArguments(lldb::opaque_compiler_type_t type) override
    {
        return 0;
    }

    CompilerType
    GetTemplateArgument(lldb::opaque_compiler_type_t type,
                        size_t idx, lldb::TemplateArgumentKind &kind) override
    {
        return CompilerType();
    }

    void DumpValue(lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx,
                   Stream *s, lldb::Format format, const DataExtractor &data, 
                   lldb::offset_t data_offset, size_t data_byte_size,
                   uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
                   bool show_types, bool show_summary,
                   bool verbose, uint32_t depth) override;

    bool DumpTypeValue(lldb::opaque_compiler_type_t type, Stream *s,
                       lldb::Format format, const DataExtractor &data,
                       lldb::offset_t data_offset, size_t data_byte_size,
                       uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
                       ExecutionContextScope *exe_scope,
                       bool is_base_class) override;

    void
    DumpTypeDescription(lldb::opaque_compiler_type_t type) override;

    void
    DumpTypeDescription(lldb::opaque_compiler_type_t type, Stream *s) override;

    bool IsRuntimeGeneratedType(lldb::opaque_compiler_type_t type) override;

    void 
    DumpSummary(lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx,
                Stream *s, const DataExtractor &data,
                lldb::offset_t data_offset, size_t data_byte_size) override;

    size_t 
    ConvertStringToFloatValue(lldb::opaque_compiler_type_t type, const char *s,
                              uint8_t *dst, size_t dst_size) override;

    bool IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                                  CompilerType *pointee_type = nullptr) override;

    unsigned GetTypeQualifiers(lldb::opaque_compiler_type_t type) override;

    bool IsCStringType(lldb::opaque_compiler_type_t type, uint32_t &length) override;

    size_t GetTypeBitAlign(lldb::opaque_compiler_type_t type) override;

    CompilerType GetBasicTypeFromAST(lldb::BasicType basic_type) override;

    bool IsBeingDefined(lldb::opaque_compiler_type_t type) override;

    bool IsConst(lldb::opaque_compiler_type_t type) override;

    uint32_t
    IsHomogeneousAggregate(lldb::opaque_compiler_type_t type, CompilerType *base_type_ptr) override;

    bool IsPolymorphicClass(lldb::opaque_compiler_type_t type) override;

    bool IsTypedefType(lldb::opaque_compiler_type_t type) override;

    CompilerType GetTypedefedType(lldb::opaque_compiler_type_t type) override;

    CompilerType
    GetUnboundType(lldb::opaque_compiler_type_t type) override;

    bool
    IsVectorType(lldb::opaque_compiler_type_t type, CompilerType *element_type, 
                 uint64_t *size) override;

    CompilerType GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) override;

    CompilerType GetNonReferenceType(lldb::opaque_compiler_type_t type) override;

    bool
    IsReferenceType(lldb::opaque_compiler_type_t type, 
                    CompilerType *pointee_type = nullptr, 
                    bool *is_rvalue = nullptr) override;

    CompilerType CreateBaseType(const ConstString &name, uint64_t);

  private:
    int m_pointer_byte_size;
    int m_int_byte_size;
    std::unique_ptr<DWARFASTParser> m_dwarf_ast_parser_ap;
    OCamlTypeMap m_base_type_map;

    OCamlASTContext(const OCamlASTContext &) = delete;
    const OCamlASTContext &operator=(const OCamlASTContext &) = delete;
};

class OCamlASTContextForExpr : public OCamlASTContext
{
    public:
        OCamlASTContextForExpr(lldb::TargetSP target) : m_target_wp(target) {}

    private:
        lldb::TargetWP m_target_wp;
};

}
#endif // liblldb_OCamlASTContext_h_
