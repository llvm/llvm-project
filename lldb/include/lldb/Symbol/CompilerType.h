//===-- CompilerType.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CompilerType_h_
#define liblldb_CompilerType_h_

#include <functional>
#include <string>
#include <vector>

#include "lldb/lldb-private.h"
#include "llvm/ADT/APSInt.h"

namespace lldb_private {

class DataExtractor;

/// Represents a generic type in a programming language.
///
/// This class serves as an abstraction for a type inside one of the TypeSystems
/// implemented by the language plugins. It does not have any actual logic in it
/// but only stores an opaque pointer and a pointer to the TypeSystem that
/// gives meaning to this opaque pointer. All methods of this class should call
/// their respective method in the TypeSystem interface and pass the opaque
/// pointer along.
///
/// \see lldb_private::TypeSystem
class CompilerType {
public:
  /// Creates a CompilerType with the given TypeSystem and opaque compiler type.
  ///
  /// This constructor should only be called from the respective TypeSystem
  /// implementation.
  ///
  /// \see lldb_private::TypeSystemClang::GetType(clang::QualType)
  CompilerType(TypeSystem *type_system, lldb::opaque_compiler_type_t type)
      : m_type(type), m_type_system(type_system) {}

  CompilerType(const CompilerType &rhs)
      : m_type(rhs.m_type), m_type_system(rhs.m_type_system) {}

  CompilerType() = default;

  // Operators

  const CompilerType &operator=(const CompilerType &rhs) {
    m_type = rhs.m_type;
    m_type_system = rhs.m_type_system;
    return *this;
  }

  // Tests

  explicit operator bool() const {
    return m_type != nullptr && m_type_system != nullptr;
  }

  bool operator<(const CompilerType &rhs) const {
    if (m_type_system == rhs.m_type_system)
      return m_type < rhs.m_type;
    return m_type_system < rhs.m_type_system;
  }

  bool IsValid() const { return m_type != nullptr && m_type_system != nullptr; }

  bool IsArrayType(CompilerType *element_type, uint64_t *size,
                   bool *is_incomplete) const;

  bool IsVectorType(CompilerType *element_type, uint64_t *size) const;

  bool IsArrayOfScalarType() const;

  bool IsAggregateType() const;

  bool IsAnonymousType() const;

  bool IsBeingDefined() const;

  bool IsCharType() const;

  bool IsCompleteType() const;

  bool IsConst() const;

  bool IsCStringType(uint32_t &length) const;

  bool IsDefined() const;

  bool IsFloatingPointType(uint32_t &count, bool &is_complex) const;

  bool IsFunctionType(bool *is_variadic_ptr = nullptr) const;

  uint32_t IsHomogeneousAggregate(CompilerType *base_type_ptr) const;

  size_t GetNumberOfFunctionArguments() const;

  CompilerType GetFunctionArgumentAtIndex(const size_t index) const;

  bool IsVariadicFunctionType() const;

  bool IsFunctionPointerType() const;

  bool IsBlockPointerType(CompilerType *function_pointer_type_ptr) const;

  bool IsIntegerType(bool &is_signed) const;

  bool IsEnumerationType(bool &is_signed) const;

  bool IsIntegerOrEnumerationType(bool &is_signed) const;

  bool IsPolymorphicClass() const;

  bool IsPossibleDynamicType(CompilerType *target_type, // Can pass nullptr
                             bool check_cplusplus, bool check_objc) const;

  bool IsPointerToScalarType() const;

  bool IsRuntimeGeneratedType() const;

  bool IsPointerType(CompilerType *pointee_type = nullptr) const;

  bool IsPointerOrReferenceType(CompilerType *pointee_type = nullptr) const;

  bool IsReferenceType(CompilerType *pointee_type = nullptr,
                       bool *is_rvalue = nullptr) const;

  bool ShouldTreatScalarValueAsAddress() const;

  bool IsScalarType() const;

  bool IsTypedefType() const;

  bool IsVoidType() const;

  // Type Completion

  bool GetCompleteType() const;

  // AST related queries

  size_t GetPointerByteSize() const;

  // Accessors

  TypeSystem *GetTypeSystem() const { return m_type_system; }

  ConstString GetConstQualifiedTypeName() const;

  ConstString GetConstTypeName() const;

  ConstString GetTypeName() const;

  ConstString GetDisplayTypeName() const;

  uint32_t
  GetTypeInfo(CompilerType *pointee_or_element_compiler_type = nullptr) const;

  lldb::LanguageType GetMinimumLanguage();

  lldb::opaque_compiler_type_t GetOpaqueQualType() const { return m_type; }

  lldb::TypeClass GetTypeClass() const;

  void SetCompilerType(TypeSystem *type_system,
                       lldb::opaque_compiler_type_t type);

  unsigned GetTypeQualifiers() const;

  // Creating related types

  CompilerType GetArrayElementType(uint64_t *stride = nullptr) const;

  CompilerType GetArrayType(uint64_t size) const;

  CompilerType GetCanonicalType() const;

  CompilerType GetFullyUnqualifiedType() const;

  // Returns -1 if this isn't a function of if the function doesn't have a
  // prototype Returns a value >= 0 if there is a prototype.
  int GetFunctionArgumentCount() const;

  CompilerType GetFunctionArgumentTypeAtIndex(size_t idx) const;

  CompilerType GetFunctionReturnType() const;

  size_t GetNumMemberFunctions() const;

  TypeMemberFunctionImpl GetMemberFunctionAtIndex(size_t idx);

  // If this type is a reference to a type (L value or R value reference),
  // return a new type with the reference removed, else return the current type
  // itself.
  CompilerType GetNonReferenceType() const;

  // If this type is a pointer type, return the type that the pointer points
  // to, else return an invalid type.
  CompilerType GetPointeeType() const;

  // Return a new CompilerType that is a pointer to this type
  CompilerType GetPointerType() const;

  // Return a new CompilerType that is a L value reference to this type if this
  // type is valid and the type system supports L value references, else return
  // an invalid type.
  CompilerType GetLValueReferenceType() const;

  // Return a new CompilerType that is a R value reference to this type if this
  // type is valid and the type system supports R value references, else return
  // an invalid type.
  CompilerType GetRValueReferenceType() const;

  // Return a new CompilerType adds a const modifier to this type if this type
  // is valid and the type system supports const modifiers, else return an
  // invalid type.
  CompilerType AddConstModifier() const;

  // Return a new CompilerType adds a volatile modifier to this type if this
  // type is valid and the type system supports volatile modifiers, else return
  // an invalid type.
  CompilerType AddVolatileModifier() const;

  // Return a new CompilerType that is the atomic type of this type. If this
  // type is not valid or the type system doesn't support atomic types, this
  // returns an invalid type.
  CompilerType GetAtomicType() const;

  // Return a new CompilerType adds a restrict modifier to this type if this
  // type is valid and the type system supports restrict modifiers, else return
  // an invalid type.
  CompilerType AddRestrictModifier() const;

  // Create a typedef to this type using "name" as the name of the typedef this
  // type is valid and the type system supports typedefs, else return an
  // invalid type.
  CompilerType CreateTypedef(const char *name,
                             const CompilerDeclContext &decl_ctx) const;

  // If the current object represents a typedef type, get the underlying type
  CompilerType GetTypedefedType() const;

  // Create related types using the current type's AST
  CompilerType GetBasicTypeFromAST(lldb::BasicType basic_type) const;

  // Exploring the type

  struct IntegralTemplateArgument;

  /// Return the size of the type in bytes.
  llvm::Optional<uint64_t> GetByteSize(ExecutionContextScope *exe_scope) const;
  /// Return the size of the type in bits.
  llvm::Optional<uint64_t> GetBitSize(ExecutionContextScope *exe_scope) const;

  lldb::Encoding GetEncoding(uint64_t &count) const;

  lldb::Format GetFormat() const;

  llvm::Optional<size_t> GetTypeBitAlign(ExecutionContextScope *exe_scope) const;

  uint32_t GetNumChildren(bool omit_empty_base_classes,
                          const ExecutionContext *exe_ctx) const;

  lldb::BasicType GetBasicTypeEnumeration() const;

  static lldb::BasicType GetBasicTypeEnumeration(ConstString name);

  // If this type is an enumeration, iterate through all of its enumerators
  // using a callback. If the callback returns true, keep iterating, else abort
  // the iteration.
  void ForEachEnumerator(
      std::function<bool(const CompilerType &integer_type,
                         ConstString name,
                         const llvm::APSInt &value)> const &callback) const;

  uint32_t GetNumFields() const;

  CompilerType GetFieldAtIndex(size_t idx, std::string &name,
                               uint64_t *bit_offset_ptr,
                               uint32_t *bitfield_bit_size_ptr,
                               bool *is_bitfield_ptr) const;

  uint32_t GetNumDirectBaseClasses() const;

  uint32_t GetNumVirtualBaseClasses() const;

  CompilerType GetDirectBaseClassAtIndex(size_t idx,
                                         uint32_t *bit_offset_ptr) const;

  CompilerType GetVirtualBaseClassAtIndex(size_t idx,
                                          uint32_t *bit_offset_ptr) const;

  uint32_t GetIndexOfFieldWithName(const char *name,
                                   CompilerType *field_compiler_type = nullptr,
                                   uint64_t *bit_offset_ptr = nullptr,
                                   uint32_t *bitfield_bit_size_ptr = nullptr,
                                   bool *is_bitfield_ptr = nullptr) const;

  CompilerType GetChildCompilerTypeAtIndex(
      ExecutionContext *exe_ctx, size_t idx, bool transparent_pointers,
      bool omit_empty_base_classes, bool ignore_array_bounds,
      std::string &child_name, uint32_t &child_byte_size,
      int32_t &child_byte_offset, uint32_t &child_bitfield_bit_size,
      uint32_t &child_bitfield_bit_offset, bool &child_is_base_class,
      bool &child_is_deref_of_parent, ValueObject *valobj,
      uint64_t &language_flags) const;

  // Lookup a child given a name. This function will match base class names and
  // member member names in "clang_type" only, not descendants.
  uint32_t GetIndexOfChildWithName(const char *name,
                                   bool omit_empty_base_classes) const;

  // Lookup a child member given a name. This function will match member names
  // only and will descend into "clang_type" children in search for the first
  // member in this class, or any base class that matches "name".
  // TODO: Return all matches for a given name by returning a
  // vector<vector<uint32_t>>
  // so we catch all names that match a given child name, not just the first.
  size_t
  GetIndexOfChildMemberWithName(const char *name, bool omit_empty_base_classes,
                                std::vector<uint32_t> &child_indexes) const;

  size_t GetNumTemplateArguments() const;

  lldb::TemplateArgumentKind GetTemplateArgumentKind(size_t idx) const;
  CompilerType GetTypeTemplateArgument(size_t idx) const;

  // Returns the value of the template argument and its type.
  llvm::Optional<IntegralTemplateArgument>
  GetIntegralTemplateArgument(size_t idx) const;

  CompilerType GetTypeForFormatters() const;

  LazyBool ShouldPrintAsOneLiner(ValueObject *valobj) const;

  bool IsMeaninglessWithoutDynamicResolution() const;

  // Dumping types

#ifndef NDEBUG
  /// Convenience LLVM-style dump method for use in the debugger only.
  /// Don't call this function from actual code.
  LLVM_DUMP_METHOD void dump() const;
#endif

  void DumpValue(ExecutionContext *exe_ctx, Stream *s, lldb::Format format,
                 const DataExtractor &data, lldb::offset_t data_offset,
                 size_t data_byte_size, uint32_t bitfield_bit_size,
                 uint32_t bitfield_bit_offset, bool show_types,
                 bool show_summary, bool verbose, uint32_t depth);

  bool DumpTypeValue(Stream *s, lldb::Format format, const DataExtractor &data,
                     lldb::offset_t data_offset, size_t data_byte_size,
                     uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
                     ExecutionContextScope *exe_scope);

  void DumpSummary(ExecutionContext *exe_ctx, Stream *s,
                   const DataExtractor &data, lldb::offset_t data_offset,
                   size_t data_byte_size);

  void DumpTypeDescription() const; // Dump to stdout

  void DumpTypeDescription(Stream *s) const;

  bool GetValueAsScalar(const DataExtractor &data, lldb::offset_t data_offset,
                        size_t data_byte_size, Scalar &value) const;

  void Clear() {
    m_type = nullptr;
    m_type_system = nullptr;
  }

private:
  lldb::opaque_compiler_type_t m_type = nullptr;
  TypeSystem *m_type_system = nullptr;
};

bool operator==(const CompilerType &lhs, const CompilerType &rhs);
bool operator!=(const CompilerType &lhs, const CompilerType &rhs);

struct CompilerType::IntegralTemplateArgument {
  llvm::APSInt value;
  CompilerType type;
};

} // namespace lldb_private

#endif // liblldb_CompilerType_h_
