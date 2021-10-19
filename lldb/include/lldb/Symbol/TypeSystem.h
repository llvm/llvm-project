//===-- TypeSystem.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SYMBOL_TYPESYSTEM_H
#define LLDB_SYMBOL_TYPESYSTEM_H

#include <functional>
#include <map>
#include <mutex>
#include <string>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"

#include "lldb/Core/PluginInterface.h"
#include "lldb/Expression/Expression.h"
#include "lldb/Symbol/CompilerDecl.h"
#include "lldb/Symbol/CompilerDeclContext.h"
#include "lldb/lldb-private.h"

class DWARFDIE;
class DWARFASTParser;
class PDBASTParser;

namespace lldb_private {

/// A SmallBitVector that represents a set of source languages (\p
/// lldb::LanguageType).  Each lldb::LanguageType is represented by
/// the bit with the position of its enumerator. The largest
/// LanguageType is < 64, so this is space-efficient and on 64-bit
/// architectures a LanguageSet can be completely stack-allocated.
struct LanguageSet {
  llvm::SmallBitVector bitvector;
  LanguageSet();

  /// If the set contains a single language only, return it.
  llvm::Optional<lldb::LanguageType> GetSingularLanguage();
  void Insert(lldb::LanguageType language);
  bool Empty() const;
  size_t Size() const;
  bool operator[](unsigned i) const;
};

/// Interface for representing a type system.
///
/// Implemented by language plugins to define the type system for a given
/// language.
///
/// This interface extensively used opaque pointers to prevent that generic
/// LLDB code has dependencies on language plugins. The type and semantics of
/// these opaque pointers are defined by the TypeSystem implementation inside
/// the respective language plugin. Opaque pointers from one TypeSystem
/// instance should never be passed to a different TypeSystem instance (even
/// when the language plugin for both TypeSystem instances is the same).
///
/// Most of the functions in this class should not be called directly but only
/// called by their respective counterparts in CompilerType, CompilerDecl and
/// CompilerDeclContext.
///
/// \see lldb_private::CompilerType
/// \see lldb_private::CompilerDecl
/// \see lldb_private::CompilerDeclContext
class TypeSystem : public PluginInterface {
public:
  // Constructors and Destructors
  ~TypeSystem() override;

  // LLVM RTTI support
  virtual bool isA(const void *ClassID) const = 0;

  static lldb::TypeSystemSP CreateInstance(lldb::LanguageType language,
                                           Module *module);

  static lldb::TypeSystemSP CreateInstance(lldb::LanguageType language,
                                           Target *target);

  // Free up any resources associated with this TypeSystem.  Done before
  // removing all the TypeSystems from the TypeSystemMap.
  virtual void Finalize() {}

  virtual DWARFASTParser *GetDWARFParser() { return nullptr; }
  virtual PDBASTParser *GetPDBParser() { return nullptr; }

  virtual SymbolFile *GetSymbolFile() const { return m_sym_file; }

  // Returns true if the symbol file changed during the set accessor.
  virtual void SetSymbolFile(SymbolFile *sym_file) { m_sym_file = sym_file; }

  // CompilerDecl functions
  virtual ConstString DeclGetName(void *opaque_decl) = 0;

  virtual ConstString DeclGetMangledName(void *opaque_decl);

  virtual CompilerDeclContext DeclGetDeclContext(void *opaque_decl);

  virtual CompilerType DeclGetFunctionReturnType(void *opaque_decl);

  virtual size_t DeclGetFunctionNumArguments(void *opaque_decl);

  virtual CompilerType DeclGetFunctionArgumentType(void *opaque_decl,
                                                   size_t arg_idx);

  virtual CompilerType GetTypeForDecl(void *opaque_decl) = 0;

  // CompilerDeclContext functions

  virtual std::vector<CompilerDecl>
  DeclContextFindDeclByName(void *opaque_decl_ctx, ConstString name,
                            const bool ignore_imported_decls);

  virtual ConstString DeclContextGetName(void *opaque_decl_ctx) = 0;

  virtual ConstString
  DeclContextGetScopeQualifiedName(void *opaque_decl_ctx) = 0;

  virtual bool DeclContextIsClassMethod(
      void *opaque_decl_ctx, lldb::LanguageType *language_ptr,
      bool *is_instance_method_ptr, ConstString *language_object_name_ptr) = 0;

  virtual bool DeclContextIsContainedInLookup(void *opaque_decl_ctx,
                                              void *other_opaque_decl_ctx) = 0;

  // Tests
#ifndef NDEBUG
  /// Verify the integrity of the type to catch CompilerTypes that mix
  /// and match invalid TypeSystem/Opaque type pairs.
  virtual bool Verify(lldb::opaque_compiler_type_t type) = 0;
#endif

  virtual bool IsArrayType(lldb::opaque_compiler_type_t type,
                           CompilerType *element_type, uint64_t *size,
                           bool *is_incomplete) = 0;

  virtual bool IsAggregateType(lldb::opaque_compiler_type_t type) = 0;

  virtual bool IsAnonymousType(lldb::opaque_compiler_type_t type);

  virtual bool IsCharType(lldb::opaque_compiler_type_t type) = 0;

  virtual bool IsCompleteType(lldb::opaque_compiler_type_t type) = 0;

  virtual bool IsDefined(lldb::opaque_compiler_type_t type) = 0;

  virtual bool IsFloatingPointType(lldb::opaque_compiler_type_t type,
                                   uint32_t &count, bool &is_complex) = 0;

  virtual bool IsFunctionType(lldb::opaque_compiler_type_t type) = 0;

  virtual size_t
  GetNumberOfFunctionArguments(lldb::opaque_compiler_type_t type) = 0;

  virtual CompilerType
  GetFunctionArgumentAtIndex(lldb::opaque_compiler_type_t type,
                             const size_t index) = 0;

  virtual bool IsFunctionPointerType(lldb::opaque_compiler_type_t type) = 0;

  virtual bool IsBlockPointerType(lldb::opaque_compiler_type_t type,
                                  CompilerType *function_pointer_type_ptr) = 0;

  virtual bool IsIntegerType(lldb::opaque_compiler_type_t type,
                             bool &is_signed) = 0;

  virtual bool IsEnumerationType(lldb::opaque_compiler_type_t type,
                                 bool &is_signed) {
    is_signed = false;
    return false;
  }

  virtual bool IsScopedEnumerationType(lldb::opaque_compiler_type_t type) = 0;

  virtual bool IsPossibleDynamicType(lldb::opaque_compiler_type_t type,
                                     CompilerType *target_type, // Can pass NULL
                                     bool check_cplusplus, bool check_objc) = 0;

  virtual bool IsPointerType(lldb::opaque_compiler_type_t type,
                             CompilerType *pointee_type) = 0;

  virtual bool IsScalarType(lldb::opaque_compiler_type_t type) = 0;

  virtual bool IsVoidType(lldb::opaque_compiler_type_t type) = 0;

  virtual bool CanPassInRegisters(const CompilerType &type) = 0;

  // TypeSystems can support more than one language
  virtual bool SupportsLanguage(lldb::LanguageType language) = 0;

  // Type Completion

  virtual bool GetCompleteType(lldb::opaque_compiler_type_t type) = 0;

  // AST related queries

  virtual uint32_t GetPointerByteSize() = 0;

  // Accessors

  virtual ConstString GetTypeName(lldb::opaque_compiler_type_t type) = 0;

  virtual ConstString GetDisplayTypeName(lldb::opaque_compiler_type_t type) = 0;

  virtual uint32_t
  GetTypeInfo(lldb::opaque_compiler_type_t type,
              CompilerType *pointee_or_element_compiler_type) = 0;

  virtual lldb::LanguageType
  GetMinimumLanguage(lldb::opaque_compiler_type_t type) = 0;

  virtual lldb::TypeClass GetTypeClass(lldb::opaque_compiler_type_t type) = 0;

  // Creating related types

  virtual CompilerType
  GetArrayElementType(lldb::opaque_compiler_type_t type,
                      ExecutionContextScope *exe_scope) = 0;

  virtual CompilerType GetArrayType(lldb::opaque_compiler_type_t type,
                                    uint64_t size);

  virtual CompilerType GetCanonicalType(lldb::opaque_compiler_type_t type) = 0;

  virtual CompilerType
  GetEnumerationIntegerType(lldb::opaque_compiler_type_t type) = 0;

  // Returns -1 if this isn't a function of if the function doesn't have a
  // prototype Returns a value >= 0 if there is a prototype.
  virtual int GetFunctionArgumentCount(lldb::opaque_compiler_type_t type) = 0;

  virtual CompilerType
  GetFunctionArgumentTypeAtIndex(lldb::opaque_compiler_type_t type,
                                 size_t idx) = 0;

  virtual CompilerType
  GetFunctionReturnType(lldb::opaque_compiler_type_t type) = 0;

  virtual size_t GetNumMemberFunctions(lldb::opaque_compiler_type_t type) = 0;

  virtual TypeMemberFunctionImpl
  GetMemberFunctionAtIndex(lldb::opaque_compiler_type_t type, size_t idx) = 0;

  virtual CompilerType GetPointeeType(lldb::opaque_compiler_type_t type) = 0;

  virtual CompilerType GetPointerType(lldb::opaque_compiler_type_t type) = 0;

  virtual CompilerType
  GetLValueReferenceType(lldb::opaque_compiler_type_t type);

  virtual CompilerType
  GetRValueReferenceType(lldb::opaque_compiler_type_t type);

  virtual CompilerType GetAtomicType(lldb::opaque_compiler_type_t type);

  virtual CompilerType AddConstModifier(lldb::opaque_compiler_type_t type);

  virtual CompilerType AddVolatileModifier(lldb::opaque_compiler_type_t type);

  virtual CompilerType AddRestrictModifier(lldb::opaque_compiler_type_t type);

  /// \param opaque_payload      The m_payload field of Type, which may
  /// carry TypeSystem-specific extra information.
  virtual CompilerType CreateTypedef(lldb::opaque_compiler_type_t type,
                                     const char *name,
                                     const CompilerDeclContext &decl_ctx,
                                     uint32_t opaque_payload);

  // Exploring the type

  virtual const llvm::fltSemantics &GetFloatTypeSemantics(size_t byte_size) = 0;

  virtual llvm::Optional<uint64_t>
  GetBitSize(lldb::opaque_compiler_type_t type,
             ExecutionContextScope *exe_scope) = 0;

  virtual lldb::Encoding GetEncoding(lldb::opaque_compiler_type_t type,
                                     uint64_t &count) = 0;

  virtual lldb::Format GetFormat(lldb::opaque_compiler_type_t type) = 0;

  virtual uint32_t GetNumChildren(lldb::opaque_compiler_type_t type,
                                  bool omit_empty_base_classes,
                                  const ExecutionContext *exe_ctx) = 0;

  virtual CompilerType GetBuiltinTypeByName(ConstString name);

  virtual lldb::BasicType
  GetBasicTypeEnumeration(lldb::opaque_compiler_type_t type) = 0;

  virtual void ForEachEnumerator(
      lldb::opaque_compiler_type_t type,
      std::function<bool(const CompilerType &integer_type,
                         ConstString name,
                         const llvm::APSInt &value)> const &callback) {}

  virtual uint32_t GetNumFields(lldb::opaque_compiler_type_t type) = 0;

  virtual CompilerType GetFieldAtIndex(lldb::opaque_compiler_type_t type,
                                       size_t idx, std::string &name,
                                       uint64_t *bit_offset_ptr,
                                       uint32_t *bitfield_bit_size_ptr,
                                       bool *is_bitfield_ptr) = 0;

  virtual uint32_t
  GetNumDirectBaseClasses(lldb::opaque_compiler_type_t type) = 0;

  virtual uint32_t
  GetNumVirtualBaseClasses(lldb::opaque_compiler_type_t type) = 0;

  virtual CompilerType
  GetDirectBaseClassAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                            uint32_t *bit_offset_ptr) = 0;

  virtual CompilerType
  GetVirtualBaseClassAtIndex(lldb::opaque_compiler_type_t type, size_t idx,
                             uint32_t *bit_offset_ptr) = 0;

  virtual CompilerType GetChildCompilerTypeAtIndex(
      lldb::opaque_compiler_type_t type, ExecutionContext *exe_ctx, size_t idx,
      bool transparent_pointers, bool omit_empty_base_classes,
      bool ignore_array_bounds, std::string &child_name,
      uint32_t &child_byte_size, int32_t &child_byte_offset,
      uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
      bool &child_is_base_class, bool &child_is_deref_of_parent,
      ValueObject *valobj, uint64_t &language_flags) = 0;

  // Lookup a child given a name. This function will match base class names and
  // member member names in "clang_type" only, not descendants.
  virtual uint32_t GetIndexOfChildWithName(lldb::opaque_compiler_type_t type,
                                           const char *name,
                                           bool omit_empty_base_classes) = 0;

  // Lookup a child member given a name. This function will match member names
  // only and will descend into "clang_type" children in search for the first
  // member in this class, or any base class that matches "name".
  // TODO: Return all matches for a given name by returning a
  // vector<vector<uint32_t>>
  // so we catch all names that match a given child name, not just the first.
  virtual size_t
  GetIndexOfChildMemberWithName(lldb::opaque_compiler_type_t type,
                                const char *name, bool omit_empty_base_classes,
                                std::vector<uint32_t> &child_indexes) = 0;

  virtual size_t GetNumTemplateArguments(lldb::opaque_compiler_type_t type);

  virtual lldb::TemplateArgumentKind
  GetTemplateArgumentKind(lldb::opaque_compiler_type_t type, size_t idx);
  virtual CompilerType GetTypeTemplateArgument(lldb::opaque_compiler_type_t type,
                                           size_t idx);
  virtual llvm::Optional<CompilerType::IntegralTemplateArgument>
  GetIntegralTemplateArgument(lldb::opaque_compiler_type_t type, size_t idx);

  // Dumping types

#ifndef NDEBUG
  /// Convenience LLVM-style dump method for use in the debugger only.
  LLVM_DUMP_METHOD virtual void
  dump(lldb::opaque_compiler_type_t type) const = 0;
#endif
  
  virtual void DumpValue(lldb::opaque_compiler_type_t type,
                         ExecutionContext *exe_ctx, Stream *s,
                         lldb::Format format, const DataExtractor &data,
                         lldb::offset_t data_offset, size_t data_byte_size,
                         uint32_t bitfield_bit_size,
                         uint32_t bitfield_bit_offset, bool show_types,
                         bool show_summary, bool verbose, uint32_t depth) = 0;

  virtual bool DumpTypeValue(lldb::opaque_compiler_type_t type, Stream *s,
                             lldb::Format format, const DataExtractor &data,
                             lldb::offset_t data_offset, size_t data_byte_size,
                             uint32_t bitfield_bit_size,
                             uint32_t bitfield_bit_offset,
                             ExecutionContextScope *exe_scope) = 0;

  /// Dump the type to stdout.
  virtual void DumpTypeDescription(
      lldb::opaque_compiler_type_t type,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) = 0;

  /// Print a description of the type to a stream. The exact implementation
  /// varies, but the expectation is that eDescriptionLevelFull returns a
  /// source-like representation of the type, whereas eDescriptionLevelVerbose
  /// does a dump of the underlying AST if applicable.
  virtual void DumpTypeDescription(
      lldb::opaque_compiler_type_t type, Stream *s,
      lldb::DescriptionLevel level = lldb::eDescriptionLevelFull) = 0;

  /// Dump a textual representation of the internal TypeSystem state to the
  /// given stream.
  ///
  /// This should not modify the state of the TypeSystem if possible.
  virtual void Dump(llvm::raw_ostream &output) = 0;

  // TODO: These methods appear unused. Should they be removed?

  virtual bool IsRuntimeGeneratedType(lldb::opaque_compiler_type_t type) = 0;

  virtual void DumpSummary(lldb::opaque_compiler_type_t type,
                           ExecutionContext *exe_ctx, Stream *s,
                           const DataExtractor &data,
                           lldb::offset_t data_offset,
                           size_t data_byte_size) = 0;

  // TODO: Determine if these methods should move to TypeSystemClang.

  virtual bool IsPointerOrReferenceType(lldb::opaque_compiler_type_t type,
                                        CompilerType *pointee_type) = 0;

  virtual unsigned GetTypeQualifiers(lldb::opaque_compiler_type_t type) = 0;

  virtual bool IsCStringType(lldb::opaque_compiler_type_t type,
                             uint32_t &length) = 0;

  virtual llvm::Optional<size_t>
  GetTypeBitAlign(lldb::opaque_compiler_type_t type,
                  ExecutionContextScope *exe_scope) = 0;

  virtual CompilerType GetBasicTypeFromAST(lldb::BasicType basic_type) = 0;

  virtual CompilerType
  GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                      size_t bit_size) = 0;

  virtual bool IsBeingDefined(lldb::opaque_compiler_type_t type) = 0;

  virtual bool IsConst(lldb::opaque_compiler_type_t type) = 0;

  virtual uint32_t IsHomogeneousAggregate(lldb::opaque_compiler_type_t type,
                                          CompilerType *base_type_ptr) = 0;

  virtual bool IsPolymorphicClass(lldb::opaque_compiler_type_t type) = 0;

  virtual bool IsTypedefType(lldb::opaque_compiler_type_t type) = 0;

  // If the current object represents a typedef type, get the underlying type
  virtual CompilerType GetTypedefedType(lldb::opaque_compiler_type_t type) = 0;

  virtual bool IsVectorType(lldb::opaque_compiler_type_t type,
                            CompilerType *element_type, uint64_t *size) = 0;

  virtual CompilerType
  GetFullyUnqualifiedType(lldb::opaque_compiler_type_t type) = 0;

  virtual CompilerType
  GetNonReferenceType(lldb::opaque_compiler_type_t type) = 0;

  virtual bool IsReferenceType(lldb::opaque_compiler_type_t type,
                               CompilerType *pointee_type, bool *is_rvalue) = 0;

  virtual bool
  ShouldTreatScalarValueAsAddress(lldb::opaque_compiler_type_t type) {
    return IsPointerOrReferenceType(type, nullptr);
  }

  virtual UserExpression *
  GetUserExpression(llvm::StringRef expr, llvm::StringRef prefix,
                    lldb::LanguageType language,
                    Expression::ResultType desired_type,
                    const EvaluateExpressionOptions &options,
                    ValueObject *ctx_obj) {
    return nullptr;
  }

  virtual FunctionCaller *GetFunctionCaller(const CompilerType &return_type,
                                            const Address &function_address,
                                            const ValueList &arg_value_list,
                                            const char *name) {
    return nullptr;
  }

  virtual std::unique_ptr<UtilityFunction>
  CreateUtilityFunction(std::string text, std::string name);

  virtual PersistentExpressionState *GetPersistentExpressionState() {
    return nullptr;
  }

  virtual CompilerType GetTypeForFormatters(void *type);

  virtual LazyBool ShouldPrintAsOneLiner(void *type, ValueObject *valobj);

  // Type systems can have types that are placeholder types, which are meant to
  // indicate the presence of a type, but offer no actual information about
  // said types, and leave the burden of actually figuring type information out
  // to dynamic type resolution. For instance a language with a generics
  // system, can use placeholder types to indicate "type argument goes here",
  // without promising uniqueness of the placeholder, nor attaching any
  // actually idenfiable information to said placeholder. This API allows type
  // systems to tell LLDB when such a type has been encountered In response,
  // the debugger can react by not using this type as a cache entry in any
  // type-specific way For instance, LLDB will currently not cache any
  // formatters that are discovered on such a type as attributable to the
  // meaningless type itself, instead preferring to use the dynamic type
  virtual bool IsMeaninglessWithoutDynamicResolution(void *type);

protected:
  SymbolFile *m_sym_file = nullptr;
};

class TypeSystemMap {
public:
  TypeSystemMap();
  ~TypeSystemMap();

  // Clear calls Finalize on all the TypeSystems managed by this map, and then
  // empties the map.
  void Clear();

  // Iterate through all of the type systems that are created. Return true from
  // callback to keep iterating, false to stop iterating.
  void ForEach(std::function<bool(TypeSystem *)> const &callback);

  llvm::Expected<TypeSystem &>
  GetTypeSystemForLanguage(lldb::LanguageType language, Module *module,
                           bool can_create);

  llvm::Expected<TypeSystem &>
  GetTypeSystemForLanguage(lldb::LanguageType language, Target *target,
                           bool can_create);

protected:
  typedef std::map<lldb::LanguageType, lldb::TypeSystemSP> collection;
  mutable std::mutex m_mutex; ///< A mutex to keep this object happy in
                              ///multi-threaded environments.
  collection m_map;
  bool m_clear_in_progress = false;

private:
  typedef llvm::function_ref<lldb::TypeSystemSP()> CreateCallback;
  /// Finds the type system for the given language. If no type system could be
  /// found for a language and a CreateCallback was provided, the value returned
  /// by the callback will be treated as the TypeSystem for the language.
  ///
  /// \param language The language for which the type system should be found.
  /// \param create_callback A callback that will be called if no previously
  ///                        created TypeSystem that fits the given language
  ///                        could found. Can be omitted if a non-existent
  ///                        type system should be treated as an error instead.
  /// \return The found type system or an error.
  llvm::Expected<TypeSystem &> GetTypeSystemForLanguage(
      lldb::LanguageType language,
      llvm::Optional<CreateCallback> create_callback = llvm::None);
};

} // namespace lldb_private

#endif // LLDB_SYMBOL_TYPESYSTEM_H
