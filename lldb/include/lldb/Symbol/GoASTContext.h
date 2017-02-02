//===-- GoASTContext.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_GoASTContext_h_
#define liblldb_GoASTContext_h_

// C Includes
// C++ Includes
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Utility/ConstString.h"

namespace lldb_private {

class Declaration;
class GoType;

class GoASTContext : public TypeSystem {
public:
  GoASTContext();
  ~GoASTContext() override;

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

  void SetAddressByteSize(int byte_size) { m_pointer_byte_size = byte_size; }

  //------------------------------------------------------------------
  // llvm casting support
  //------------------------------------------------------------------
  static bool classof(const TypeSystem *ts) {
    return ts->getKind() == TypeSystem::eKindGo;
  }

  //----------------------------------------------------------------------
  // CompilerDecl functions
  //----------------------------------------------------------------------
  ConstString DeclGetName(void *opaque_decl) override { return ConstString(); }

  //----------------------------------------------------------------------
  // CompilerDeclContext functions
  //----------------------------------------------------------------------

  bool DeclContextIsStructUnionOrClass(void *opaque_decl_ctx) override {
    return false;
  }

  ConstString DeclContextGetName(void *opaque_decl_ctx) override {
    return ConstString();
  }

  ConstString DeclContextGetScopeQualifiedName(void *opaque_decl_ctx) override {
    return ConstString();
  }

  bool
  DeclContextIsClassMethod(void *opaque_decl_ctx,
                           lldb::LanguageType *language_ptr,
                           bool *is_instance_method_ptr,
                           ConstString *language_object_name_ptr) override {
    return false;
  }

  //----------------------------------------------------------------------
  // Creating Types
  //----------------------------------------------------------------------

  CompilerType CreateArrayType(const ConstString &name,
                               const CompilerType &element_type,
                               uint64_t length);

  CompilerType CreateBaseType(int go_kind,
                              const ConstString &type_name_const_str,
                              uint64_t byte_size);

  // For interface, map, chan.
  CompilerType CreateTypedefType(int kind, const ConstString &name,
                                 CompilerType impl);

  CompilerType CreateVoidType(const ConstString &name);
  CompilerType CreateFunctionType(const lldb_private::ConstString &name,
                                  CompilerType *params, size_t params_count,
                                  bool is_variadic);

  CompilerType CreateStructType(int kind, const ConstString &name,
                                uint32_t byte_size);

  void CompleteStructType(const CompilerType &type);

  void AddFieldToStruct(const CompilerType &struct_type,
                        const ConstString &name, const CompilerType &field_type,
                        uint32_t byte_offset);

  //----------------------------------------------------------------------
  // Tests
  //----------------------------------------------------------------------

  static bool IsGoString(const CompilerType &type);
  static bool IsGoSlice(const CompilerType &type);
  static bool IsGoInterface(const CompilerType &type);
  static bool IsDirectIface(uint8_t kind);
  static bool IsPointerKind(uint8_t kind);

  bool IsArrayType(lldb::opaque_compiler_type_t type,
                   CompilerType *element_type, uint64_t *size,
                   bool *is_incomplete) override;

  bool IsAggregateType(lldb::opaque_compiler_type_t type) override;

  bool IsCharType(lldb::opaque_compiler_type_t type) override;

  bool IsCompleteType(lldb::opaque_compiler_type_t type) override;

  bool IsDefined(lldb::opaque_compiler_type_t type) override;

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
                             CompilerType *target_type, // Can pass nullptr
                             bool check_cplusplus, bool check_objc) override;

  bool IsPointerType(lldb::opaque_compiler_type_t type,
                     CompilerType *pointee_type = nullptr) override;

  bool IsScalarType(lldb::opaque_compiler_type_t type) override;

  bool IsVoidType(lldb::opaque_compiler_type_t type) override;

  bool SupportsLanguage(lldb::LanguageType language) override;

  //----------------------------------------------------------------------
  // Type Completion
  //----------------------------------------------------------------------

  bool GetCompleteType(lldb::opaque_compiler_type_t type) override;

  //----------------------------------------------------------------------
  // AST related queries
  //----------------------------------------------------------------------

  uint32_t GetPointerByteSize() override;

  //----------------------------------------------------------------------
  // Accessors
  //----------------------------------------------------------------------

  ConstString GetTypeName(lldb::opaque_compiler_type_t type) override;

  uint32_t GetTypeInfo(
      lldb::opaque_compiler_type_t type,
      CompilerType *pointee_or_element_compiler_type = nullptr) override;

  lldb::LanguageType
  GetMinimumLanguage(lldb::opaque_compiler_type_t type) override;

  lldb::TypeClass GetTypeClass(lldb::opaque_compiler_type_t type) override;

  //----------------------------------------------------------------------
  // Creating related types
  //----------------------------------------------------------------------

  CompilerType GetArrayElementType(lldb::opaque_compiler_type_t type,
                                   uint64_t *stride = nullptr) override;

  CompilerType GetCanonicalType(lldb::opaque_compiler_type_t type) override;

  // Returns -1 if this isn't a function of if the function doesn't have a
  // prototype
  // Returns a value >= 0 if there is a prototype.
  int GetFunctionArgumentCount(lldb::opaque_compiler_type_t type) override;

  CompilerType GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type,
                                              size_t idx) override;

  CompilerType
  GetFunctionReturnType(lldb::opaque_compiler_type_t type) override;

  size_t GetNumMemberFunctions(lldb::opaque_compiler_type_t type) override;

  TypeMemberFunctionImpl
  GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type,
                           size_t idx) override;

  CompilerType GetPointeeType(lldb::opaque_compiler_type_t type) override;

  CompilerType GetPointerType(lldb::opaque_compiler_type_t type) override;

  //----------------------------------------------------------------------
  // Exploring the type
  //----------------------------------------------------------------------

  uint64_t GetBitSize(lldb::opaque_compiler_type_t type,
                      ExecutionContextScope *exe_scope) override;

  lldb::Encoding GetEncoding(lldb::opaque_compiler_type_t type,
                             uint64_t &count) override;

  lldb::Format GetFormat(lldb::opaque_compiler_type_t type) override;

  uint32_t GetNumChildren(lldb::opaque_compiler_type_t type,
                          bool omit_empty_base_classes) override;

  lldb::BasicType
  GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) override;

  CompilerType GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                                   size_t bit_size) override;

  uint32_t GetNumFields(lldb::opaque_compiler_type_t type) override;

  CompilerType GetFieldAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                               std::string &name, uint64_t *bit_offset_ptr,
                               uint32_t *bitfield_bit_size_ptr,
                               bool *is_bitfield_ptr) override;

  uint32_t GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  uint32_t
  GetNumVirtualBaseClasses(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  CompilerType GetDirectBaseClassAtIndex(lldb::opaque_compiler_type_t type,
                                         size_t idx,
                                         uint32_t *bit_offset_ptr) override {
    return CompilerType();
  }

  CompilerType GetVirtualBaseClassAtIndex(lldb::opaque_compiler_type_t type,
                                          size_t idx,
                                          uint32_t *bit_offset_ptr) override {
    return CompilerType();
  }

  CompilerType GetChildCompilerTypeAtIndex(
      lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
      bool transparent_pointers, bool omit_empty_base_classes,
      bool ignore_array_bounds, std::string &child_name,
      uint32_t &child_byte_size, int32_t &child_byte_offset,
      uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
      bool &child_is_base_class, bool &child_is_deref_of_parent,
      ValueObject *valobj, uint64_t &language_flags) override;

  // Lookup a child given a name. This function will match base class names
  // and member member names in "clang_type" only, not descendants.
  uint32_t GetIndexOfChildWithName(lldb::opaque_compiler_type_t type,
                                   const char *name,
                                   bool omit_empty_base_classes) override;

  // Lookup a child member given a name. This function will match member names
  // only and will descend into "clang_type" children in search for the first
  // member in this class, or any base class that matches "name".
  // TODO: Return all matches for a given name by returning a
  // vector<vector<uint32_t>>
  // so we catch all names that match a given child name, not just the first.
  size_t
  GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type,
                                const char *name, bool omit_empty_base_classes,
                                std::vector<uint32_t> &child_indexes) override;

  size_t GetNumTemplateArguments(lldb::opaque_compiler_type_t type) override {
    return 0;
  }

  CompilerType GetTemplateArgument(lldb::opaque_compiler_type_t type,
                                   size_t idx,
                                   lldb::TemplateArgumentKind &kind) override {
    return CompilerType();
  }

  //----------------------------------------------------------------------
  // Dumping types
  //----------------------------------------------------------------------
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

  void DumpTypeDescription(
      lldb::opaque_compiler_type_t type) override; // Dump to stdout

  void DumpTypeDescription(lldb::opaque_compiler_type_t type,
                           Stream *s) override;

  //----------------------------------------------------------------------
  // TODO: These methods appear unused. Should they be removed?
  //----------------------------------------------------------------------

  bool IsRuntimeGeneratedType(lldb::opaque_compiler_type_t type) override;

  void DumpSummary(lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx,
                   Stream *s, const DataExtractor &data,
                   lldb::offset_t data_offset, size_t data_byte_size) override;

  // Converts "s" to a floating point value and place resulting floating
  // point bytes in the "dst" buffer.
  size_t ConvertStringToFloatValue(lldb::opaque_compiler_type_t type,
                                   const char *s, uint8_t *dst,
                                   size_t dst_size) override;

  //----------------------------------------------------------------------
  // TODO: Determine if these methods should move to ClangASTContext.
  //----------------------------------------------------------------------

  bool IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                                CompilerType *pointee_type = nullptr) override;

  unsigned GetTypeQualifiers(lldb::opaque_compiler_type_t type) override;

  bool IsCStringType(lldb::opaque_compiler_type_t type,
                     uint32_t &length) override;

  size_t GetTypeBitAlign(lldb::opaque_compiler_type_t type) override;

  CompilerType GetBasicTypeFromAST(lldb::BasicType basic_type) override;

  bool IsBeingDefined(lldb::opaque_compiler_type_t type) override;

  bool IsConst(lldb::opaque_compiler_type_t type) override;

  uint32_t IsHomogeneousAggregate(lldb::opaque_compiler_type_t type,
                                  CompilerType *base_type_ptr) override;

  bool IsPolymorphicClass(lldb::opaque_compiler_type_t type) override;

  bool IsTypedefType(lldb::opaque_compiler_type_t type) override;

  // If the current object represents a typedef type, get the underlying type
  CompilerType GetTypedefedType(lldb::opaque_compiler_type_t type) override;

  bool IsVectorType(lldb::opaque_compiler_type_t type,
                    CompilerType *element_type, uint64_t *size) override;

  CompilerType
  GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) override;

  CompilerType GetNonReferenceType(lldb::opaque_compiler_type_t type) override;

  bool IsReferenceType(lldb::opaque_compiler_type_t type,
                       CompilerType *pointee_type = nullptr,
                       bool *is_rvalue = nullptr) override;

private:
  typedef std::map<ConstString, std::unique_ptr<GoType>> TypeMap;
  int m_pointer_byte_size;
  int m_int_byte_size;
  std::unique_ptr<TypeMap> m_types;
  std::unique_ptr<DWARFASTParser> m_dwarf_ast_parser_ap;

  GoASTContext(const GoASTContext &) = delete;
  const GoASTContext &operator=(const GoASTContext &) = delete;
};

class GoASTContextForExpr : public GoASTContext {
public:
  GoASTContextForExpr(lldb::TargetSP target) : m_target_wp(target) {}
  UserExpression *
  GetUserExpression(llvm::StringRef expr, llvm::StringRef prefix,
                    lldb::LanguageType language,
                    Expression::ResultType desired_type,
                    const EvaluateExpressionOptions &options) override;

private:
  lldb::TargetWP m_target_wp;
};
}
#endif // liblldb_GoASTContext_h_
