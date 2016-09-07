//===-- JavaASTContext.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_JavaASTContext_h_
#define liblldb_JavaASTContext_h_

// C Includes
// C++ Includes
#include <map>
#include <memory>
#include <set>

// Other libraries and framework includes
// Project includes
#include "lldb/Core/ConstString.h"
#include "lldb/Symbol/TypeSystem.h"

namespace lldb_private {

class JavaASTContext : public TypeSystem {
public:
  class JavaType;
  typedef std::map<ConstString, std::unique_ptr<JavaType>> JavaTypeMap;

  JavaASTContext(const ArchSpec &arch);
  ~JavaASTContext() override;

  //------------------------------------------------------------------
  // PluginInterface functions
  //------------------------------------------------------------------
  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  static ConstString GetPluginNameStatic();

  static lldb::TypeSystemSP CreateInstance(lldb::LanguageType language,
                                           Module *module, Target *target);

  static void EnumerateSupportedLanguages(
      std::set<lldb::LanguageType> &languages_for_types,
      std::set<lldb::LanguageType> &languages_for_expressions);

  static void Initialize();

  static void Terminate();

  DWARFASTParser *GetDWARFParser() override;

  uint32_t GetPointerByteSize() override;

  //----------------------------------------------------------------------
  // CompilerDecl functions
  //----------------------------------------------------------------------
  ConstString DeclGetName(void *opaque_decl) override;

  //----------------------------------------------------------------------
  // CompilerDeclContext functions
  //----------------------------------------------------------------------

  std::vector<CompilerDecl>
  DeclContextFindDeclByName(void *opaque_decl_ctx, ConstString name,
                            const bool ignore_imported_decls) override;

  bool DeclContextIsStructUnionOrClass(void *opaque_decl_ctx) override;

  ConstString DeclContextGetName(void *opaque_decl_ctx) override;

  bool DeclContextIsClassMethod(void *opaque_decl_ctx,
                                lldb::LanguageType *language_ptr,
                                bool *is_instance_method_ptr,
                                ConstString *language_object_name_ptr) override;

  //----------------------------------------------------------------------
  // Tests
  //----------------------------------------------------------------------

  bool IsArrayType(lldb::opaque_compiler_type_t type,
                   CompilerType *element_type, uint64_t *size,
                   bool *is_incomplete) override;

  bool IsAggregateType(lldb::opaque_compiler_type_t type) override;

  bool IsCharType(lldb::opaque_compiler_type_t type) override;

  bool IsFloatingPointType(lldb::opaque_compiler_type_t type, uint32_t &count,
                           bool &is_complex) override;

  bool IsFunctionType(lldb::opaque_compiler_type_t type,
                      bool *is_variadic_ptr = nullptr) override;

  size_t
  GetNumberOfFunctionArguments(lldb::opaque_compiler_type_t type) override;

  CompilerType GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                                          const size_t index) override;

  bool IsFunctionPointerType(lldb::opaque_compiler_type_t type) override;

  bool IsBlockPointerType(lldb::opaque_compiler_type_t type,
                          CompilerType *function_pointer_type_ptr) override;

  bool IsIntegerType(lldb::opaque_compiler_type_t type,
                     bool &is_signed) override;

  bool IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                             CompilerType *target_type, bool check_cplusplus,
                             bool check_objc) override;

  bool IsPointerType(lldb::opaque_compiler_type_t type,
                     CompilerType *pointee_type = nullptr) override;

  bool IsReferenceType(lldb::opaque_compiler_type_t type,
                       CompilerType *pointee_type = nullptr,
                       bool *is_rvalue = nullptr) override;

  bool IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                                CompilerType *pointee_type = nullptr) override;

  bool IsScalarType(lldb::opaque_compiler_type_t type) override;

  bool IsVoidType(lldb::opaque_compiler_type_t type) override;

  bool IsCStringType(lldb::opaque_compiler_type_t type,
                     uint32_t &length) override;

  bool IsRuntimeGeneratedType(lldb::opaque_compiler_type_t type) override;

  bool IsTypedefType(lldb::opaque_compiler_type_t type) override;

  bool IsVectorType(lldb::opaque_compiler_type_t type,
                    CompilerType *element_type, uint64_t *size) override;

  bool IsPolymorphicClass(lldb::opaque_compiler_type_t type) override;

  bool IsCompleteType(lldb::opaque_compiler_type_t type) override;

  bool IsConst(lldb::opaque_compiler_type_t type) override;

  bool IsBeingDefined(lldb::opaque_compiler_type_t type) override;

  bool IsDefined(lldb::opaque_compiler_type_t type) override;

  uint32_t IsHomogeneousAggregate(lldb::opaque_compiler_type_t type,
                                  CompilerType *base_type_ptr) override;

  bool SupportsLanguage(lldb::LanguageType language) override;

  bool GetCompleteType(lldb::opaque_compiler_type_t type) override;

  ConstString GetTypeName(lldb::opaque_compiler_type_t type) override;

  uint32_t GetTypeInfo(
      lldb::opaque_compiler_type_t type,
      CompilerType *pointee_or_element_compiler_type = nullptr) override;

  lldb::TypeClass GetTypeClass(lldb::opaque_compiler_type_t type) override;

  lldb::LanguageType
  GetMinimumLanguage(lldb::opaque_compiler_type_t type) override;

  CompilerType GetArrayElementType(lldb::opaque_compiler_type_t type,
                                   uint64_t *stride = nullptr) override;

  CompilerType GetPointeeType(lldb::opaque_compiler_type_t type) override;

  CompilerType GetPointerType(lldb::opaque_compiler_type_t type) override;

  CompilerType GetCanonicalType(lldb::opaque_compiler_type_t type) override;

  CompilerType
  GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) override;

  CompilerType GetNonReferenceType(lldb::opaque_compiler_type_t type) override;

  CompilerType GetTypedefedType(lldb::opaque_compiler_type_t type) override;

  CompilerType GetBasicTypeFromAST(lldb::BasicType basic_type) override;

  CompilerType GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                                   size_t bit_size) override;

  size_t GetTypeBitAlign(lldb::opaque_compiler_type_t type) override;

  lldb::BasicType
  GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) override;

  uint64_t GetBitSize(lldb::opaque_compiler_type_t type,
                      ExecutionContextScope *exe_scope) override;

  lldb::Encoding GetEncoding(lldb::opaque_compiler_type_t type,
                             uint64_t &count) override;

  lldb::Format GetFormat(lldb::opaque_compiler_type_t type) override;

  unsigned GetTypeQualifiers(lldb::opaque_compiler_type_t type) override;

  size_t GetNumTemplateArguments(lldb::opaque_compiler_type_t type) override;

  CompilerType GetTemplateArgument(lldb::opaque_compiler_type_t type,
                                   size_t idx,
                                   lldb::TemplateArgumentKind &kind) override;

  int GetFunctionArgumentCount(lldb::opaque_compiler_type_t type) override;

  CompilerType GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type,
                                              size_t idx) override;

  CompilerType
  GetFunctionReturnType(lldb::opaque_compiler_type_t type) override;

  size_t GetNumMemberFunctions(lldb::opaque_compiler_type_t type) override;

  TypeMemberFunctionImpl
  GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                           size_t idx) override;

  uint32_t GetNumFields(lldb::opaque_compiler_type_t type) override;

  CompilerType GetFieldAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                               std::string &name, uint64_t *bit_offset_ptr,
                               uint32_t *bitfield_bit_size_ptr,
                               bool *is_bitfield_ptr) override;

  uint32_t GetNumChildren(lldb::opaque_compiler_type_t type,
                          bool omit_empty_base_classes) override;

  uint32_t GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) override;

  uint32_t GetNumVirtualBaseClasses(lldb::opaque_compiler_type_t type) override;

  CompilerType GetDirectBaseClassAtIndex(lldb::opaque_compiler_type_t type,
                                         size_t idx,
                                         uint32_t *bit_offset_ptr) override;

  CompilerType GetVirtualBaseClassAtIndex(lldb::opaque_compiler_type_t type,
                                          size_t idx,
                                          uint32_t *bit_offset_ptr) override;

  size_t ConvertStringToFloatValue(lldb::opaque_compiler_type_t type,
                                   const char *s, uint8_t *dst,
                                   size_t dst_size) override;

  void DumpValue(lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx,
                 Stream *s, lldb::Format format, const DataExtractor &data,
                 lldb::offset_t data_offset, size_t data_byte_size,
                 uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
                 bool show_types, bool show_summary, bool verbose,
                 uint32_t depth) override;

  bool DumpTypeValue(lldb::opaque_compiler_type_t type, Stream *s,
                     lldb::Format format, const DataExtractor &data,
                     lldb::offset_t data_offset, size_t data_byte_size,
                     uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
                     ExecutionContextScope *exe_scope) override;

  void DumpTypeDescription(lldb::opaque_compiler_type_t type) override;

  void DumpTypeDescription(lldb::opaque_compiler_type_t type,
                           Stream *s) override;

  void DumpSummary(lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx,
                   Stream *s, const DataExtractor &data,
                   lldb::offset_t data_offset, size_t data_byte_size) override;

  CompilerType GetChildCompilerTypeAtIndex(
      lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
      bool transparent_pointers, bool omit_empty_base_classes,
      bool ignore_array_bounds, std::string &child_name,
      uint32_t &child_byte_size, int32_t &child_byte_offset,
      uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
      bool &child_is_base_class, bool &child_is_deref_of_parent,
      ValueObject *valobj, uint64_t &language_flags) override;

  uint32_t GetIndexOfChildWithName(lldb::opaque_compiler_type_t type,
                                   const char *name,
                                   bool omit_empty_base_classes) override;

  size_t
  GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type,
                                const char *name, bool omit_empty_base_classes,
                                std::vector<uint32_t> &child_indexes) override;

  CompilerType
  GetLValueReferenceType(lldb::opaque_compiler_type_t type) override;

  ConstString DeclContextGetScopeQualifiedName(
      lldb::opaque_compiler_type_t opaque_decl_ctx) override;

  CompilerType CreateBaseType(const ConstString &name);

  CompilerType CreateObjectType(const ConstString &name,
                                const ConstString &linkage_name,
                                uint32_t byte_size);

  CompilerType CreateArrayType(const ConstString &linkage_name,
                               const CompilerType &element_type,
                               const DWARFExpression &length_expression,
                               const lldb::addr_t data_offset);

  CompilerType CreateReferenceType(const CompilerType &pointee_type);

  void CompleteObjectType(const CompilerType &object_type);

  void AddBaseClassToObject(const CompilerType &object_type,
                            const CompilerType &member_type,
                            uint32_t member_offset);

  void AddMemberToObject(const CompilerType &object_type,
                         const ConstString &name,
                         const CompilerType &member_type,
                         uint32_t member_offset);

  void SetDynamicTypeId(const CompilerType &type,
                        const DWARFExpression &type_id);

  static uint64_t CalculateDynamicTypeId(ExecutionContext *exe_ctx,
                                         const CompilerType &type,
                                         ValueObject &in_value);

  static ConstString GetLinkageName(const CompilerType &type);

  static uint32_t CalculateArraySize(const CompilerType &type,
                                     ValueObject &in_value);

  static uint64_t CalculateArrayElementOffset(const CompilerType &type,
                                              size_t index);

  //------------------------------------------------------------------
  // llvm casting support
  //------------------------------------------------------------------
  static bool classof(const TypeSystem *ts) {
    return ts->getKind() == TypeSystem::eKindJava;
  }

private:
  uint32_t m_pointer_byte_size;
  std::unique_ptr<DWARFASTParser> m_dwarf_ast_parser_ap;
  JavaTypeMap m_array_type_map;
  JavaTypeMap m_base_type_map;
  JavaTypeMap m_reference_type_map;
  JavaTypeMap m_object_type_map;

  JavaASTContext(const JavaASTContext &) = delete;
  const JavaASTContext &operator=(const JavaASTContext &) = delete;
};
}
#endif // liblldb_JavaASTContext_h_
