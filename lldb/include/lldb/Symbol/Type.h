//===-- Type.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SYMBOL_TYPE_H
#define LLDB_SYMBOL_TYPE_H

#include "lldb/Symbol/CompilerDecl.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/Declaration.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/UserID.h"
#include "lldb/lldb-private.h"

#include "llvm/ADT/APSInt.h"

#include <set>

namespace lldb_private {

/// CompilerContext allows an array of these items to be passed to perform
/// detailed lookups in SymbolVendor and SymbolFile functions.
struct CompilerContext {
  CompilerContext(CompilerContextKind t, ConstString n) : kind(t), name(n) {}

  bool operator==(const CompilerContext &rhs) const {
    return kind == rhs.kind && name == rhs.name;
  }
  bool operator!=(const CompilerContext &rhs) const { return !(*this == rhs); }

  void Dump() const;

  CompilerContextKind kind;
  ConstString name;
};

/// Match \p context_chain against \p pattern, which may contain "Any"
/// kinds. The \p context_chain should *not* contain any "Any" kinds.
bool contextMatches(llvm::ArrayRef<CompilerContext> context_chain,
                    llvm::ArrayRef<CompilerContext> pattern);

class SymbolFileType : public std::enable_shared_from_this<SymbolFileType>,
                       public UserID {
public:
  SymbolFileType(SymbolFile &symbol_file, lldb::user_id_t uid)
      : UserID(uid), m_symbol_file(symbol_file) {}

  SymbolFileType(SymbolFile &symbol_file, const lldb::TypeSP &type_sp);

  ~SymbolFileType() {}

  Type *operator->() { return GetType(); }

  Type *GetType();
  SymbolFile &GetSymbolFile() const { return m_symbol_file; }

protected:
  SymbolFile &m_symbol_file;
  lldb::TypeSP m_type_sp;
};

class Type : public std::enable_shared_from_this<Type>, public UserID {
public:
  enum EncodingDataType {
    eEncodingInvalid,
    eEncodingIsUID,      ///< This type is the type whose UID is m_encoding_uid
    eEncodingIsConstUID, ///< This type is the type whose UID is m_encoding_uid
                         /// with the const qualifier added
    eEncodingIsRestrictUID, ///< This type is the type whose UID is
                            /// m_encoding_uid with the restrict qualifier added
    eEncodingIsVolatileUID, ///< This type is the type whose UID is
                            /// m_encoding_uid with the volatile qualifier added
    eEncodingIsTypedefUID,  ///< This type is pointer to a type whose UID is
                            /// m_encoding_uid
    eEncodingIsPointerUID,  ///< This type is pointer to a type whose UID is
                            /// m_encoding_uid
    eEncodingIsLValueReferenceUID, ///< This type is L value reference to a type
                                   /// whose UID is m_encoding_uid
    eEncodingIsRValueReferenceUID, ///< This type is R value reference to a type
                                   /// whose UID is m_encoding_uid,
    eEncodingIsAtomicUID,          ///< This type is the type whose UID is
                                   /// m_encoding_uid as an atomic type.
    eEncodingIsSyntheticUID
  };

  enum class ResolveState : unsigned char {
    Unresolved = 0,
    Forward = 1,
    Layout = 2,
    Full = 3
  };

  Type(lldb::user_id_t uid, SymbolFile *symbol_file, ConstString name,
       llvm::Optional<uint64_t> byte_size, SymbolContextScope *context,
       lldb::user_id_t encoding_uid, EncodingDataType encoding_uid_type,
       const Declaration &decl, const CompilerType &compiler_qual_type,
       ResolveState compiler_type_resolve_state, uint32_t opaque_payload = 0);

  // This makes an invalid type.  Used for functions that return a Type when
  // they get an error.
  Type();

  void Dump(Stream *s, bool show_context,
            lldb::DescriptionLevel level = lldb::eDescriptionLevelFull);

  void DumpTypeName(Stream *s);

  /// Since Type instances only keep a "SymbolFile *" internally, other classes
  /// like TypeImpl need make sure the module is still around before playing
  /// with
  /// Type instances. They can store a weak pointer to the Module;
  lldb::ModuleSP GetModule();

  /// GetModule may return module for compile unit's object file.
  /// GetExeModule returns module for executable object file that contains
  /// compile unit where type was actualy defined.
  /// GetModule and GetExeModule may return the same value.
  lldb::ModuleSP GetExeModule();

  void GetDescription(Stream *s, lldb::DescriptionLevel level, bool show_name,
                      ExecutionContextScope *exe_scope);

  SymbolFile *GetSymbolFile() { return m_symbol_file; }
  const SymbolFile *GetSymbolFile() const { return m_symbol_file; }

  ConstString GetName();

  llvm::Optional<uint64_t> GetByteSize(ExecutionContextScope *exe_scope);

  uint32_t GetNumChildren(bool omit_empty_base_classes);

  bool IsAggregateType();

  bool IsValidType() { return m_encoding_uid_type != eEncodingInvalid; }

  bool IsTypedef() { return m_encoding_uid_type == eEncodingIsTypedefUID; }

  lldb::TypeSP GetTypedefType();

  ConstString GetName() const { return m_name; }

  ConstString GetQualifiedName();

  void DumpValue(ExecutionContext *exe_ctx, Stream *s,
                 const DataExtractor &data, uint32_t data_offset,
                 bool show_type, bool show_summary, bool verbose,
                 lldb::Format format = lldb::eFormatDefault);

  bool DumpValueInMemory(ExecutionContext *exe_ctx, Stream *s,
                         lldb::addr_t address, AddressType address_type,
                         bool show_types, bool show_summary, bool verbose);

  bool ReadFromMemory(ExecutionContext *exe_ctx, lldb::addr_t address,
                      AddressType address_type, DataExtractor &data);

  bool WriteToMemory(ExecutionContext *exe_ctx, lldb::addr_t address,
                     AddressType address_type, DataExtractor &data);

  bool GetIsDeclaration() const;

  void SetIsDeclaration(bool b);

  bool GetIsExternal() const;

  void SetIsExternal(bool b);

  lldb::Format GetFormat();

  lldb::Encoding GetEncoding(uint64_t &count);

  SymbolContextScope *GetSymbolContextScope() { return m_context; }
  const SymbolContextScope *GetSymbolContextScope() const { return m_context; }
  void SetSymbolContextScope(SymbolContextScope *context) {
    m_context = context;
  }

  const lldb_private::Declaration &GetDeclaration() const;

  // Get the clang type, and resolve definitions for any
  // class/struct/union/enum types completely.
  CompilerType GetFullCompilerType();

  // Get the clang type, and resolve definitions enough so that the type could
  // have layout performed. This allows ptrs and refs to
  // class/struct/union/enum types remain forward declarations.
  CompilerType GetLayoutCompilerType();

  // Get the clang type and leave class/struct/union/enum types as forward
  // declarations if they haven't already been fully defined.
  CompilerType GetForwardCompilerType();

  static int Compare(const Type &a, const Type &b);

  // From a fully qualified typename, split the type into the type basename and
  // the remaining type scope (namespaces/classes).
  static bool GetTypeScopeAndBasename(const llvm::StringRef& name,
                                      llvm::StringRef &scope,
                                      llvm::StringRef &basename,
                                      lldb::TypeClass &type_class);
  void SetEncodingType(Type *encoding_type) { m_encoding_type = encoding_type; }

  uint32_t GetEncodingMask();

  typedef uint32_t Payload;
  /// Return the language-specific payload.
  Payload GetPayload() { return m_payload; }
  /// Return the language-specific payload.
  void SetPayload(Payload opaque_payload) { m_payload = opaque_payload; }

protected:
  ConstString m_name;
  SymbolFile *m_symbol_file;
  /// The symbol context in which this type is defined.
  SymbolContextScope *m_context;
  Type *m_encoding_type;
  lldb::user_id_t m_encoding_uid;
  EncodingDataType m_encoding_uid_type;
  uint64_t m_byte_size : 63;
  uint64_t m_byte_size_has_value : 1;
  Declaration m_decl;
  CompilerType m_compiler_type;
  ResolveState m_compiler_type_resolve_state;
  /// Language-specific flags.
  Payload m_payload;

  Type *GetEncodingType();

  bool ResolveCompilerType(ResolveState compiler_type_resolve_state);
};

// the two classes here are used by the public API as a backend to the SBType
// and SBTypeList classes

class TypeImpl {
public:
  TypeImpl() = default;

  ~TypeImpl() {}

  TypeImpl(const lldb::TypeSP &type_sp);

  TypeImpl(const CompilerType &compiler_type);

  TypeImpl(const lldb::TypeSP &type_sp, const CompilerType &dynamic);

  TypeImpl(const CompilerType &compiler_type, const CompilerType &dynamic);

  void SetType(const lldb::TypeSP &type_sp);

  void SetType(const CompilerType &compiler_type);

  void SetType(const lldb::TypeSP &type_sp, const CompilerType &dynamic);

  void SetType(const CompilerType &compiler_type, const CompilerType &dynamic);

  bool operator==(const TypeImpl &rhs) const;

  bool operator!=(const TypeImpl &rhs) const;

  bool IsValid() const;

  explicit operator bool() const;

  void Clear();

  lldb::ModuleSP GetModule() const;

  ConstString GetName() const;

  ConstString GetDisplayTypeName() const;

  TypeImpl GetPointerType() const;

  TypeImpl GetPointeeType() const;

  TypeImpl GetReferenceType() const;

  TypeImpl GetTypedefedType() const;

  TypeImpl GetDereferencedType() const;

  TypeImpl GetUnqualifiedType() const;

  TypeImpl GetCanonicalType() const;

  CompilerType GetCompilerType(bool prefer_dynamic);

  TypeSystem *GetTypeSystem(bool prefer_dynamic);

  bool GetDescription(lldb_private::Stream &strm,
                      lldb::DescriptionLevel description_level);

private:
  bool CheckModule(lldb::ModuleSP &module_sp) const;
  bool CheckExeModule(lldb::ModuleSP &module_sp) const;
  bool CheckModuleCommon(const lldb::ModuleWP &input_module_wp,
                         lldb::ModuleSP &module_sp) const;

  lldb::ModuleWP m_module_wp;
  lldb::ModuleWP m_exe_module_wp;
  CompilerType m_static_type;
  CompilerType m_dynamic_type;
};

class TypeListImpl {
public:
  TypeListImpl() : m_content() {}

  void Append(const lldb::TypeImplSP &type) { m_content.push_back(type); }

  class AppendVisitor {
  public:
    AppendVisitor(TypeListImpl &type_list) : m_type_list(type_list) {}

    void operator()(const lldb::TypeImplSP &type) { m_type_list.Append(type); }

  private:
    TypeListImpl &m_type_list;
  };

  void Append(const lldb_private::TypeList &type_list);

  lldb::TypeImplSP GetTypeAtIndex(size_t idx) {
    lldb::TypeImplSP type_sp;
    if (idx < GetSize())
      type_sp = m_content[idx];
    return type_sp;
  }

  size_t GetSize() { return m_content.size(); }

private:
  std::vector<lldb::TypeImplSP> m_content;
};

class TypeMemberImpl {
public:
  TypeMemberImpl()
      : m_type_impl_sp(), m_bit_offset(0), m_name(), m_bitfield_bit_size(0),
        m_is_bitfield(false)

  {}

  TypeMemberImpl(const lldb::TypeImplSP &type_impl_sp, uint64_t bit_offset,
                 ConstString name, uint32_t bitfield_bit_size = 0,
                 bool is_bitfield = false)
      : m_type_impl_sp(type_impl_sp), m_bit_offset(bit_offset), m_name(name),
        m_bitfield_bit_size(bitfield_bit_size), m_is_bitfield(is_bitfield) {}

  TypeMemberImpl(const lldb::TypeImplSP &type_impl_sp, uint64_t bit_offset)
      : m_type_impl_sp(type_impl_sp), m_bit_offset(bit_offset), m_name(),
        m_bitfield_bit_size(0), m_is_bitfield(false) {
    if (m_type_impl_sp)
      m_name = m_type_impl_sp->GetName();
  }

  const lldb::TypeImplSP &GetTypeImpl() { return m_type_impl_sp; }

  ConstString GetName() const { return m_name; }

  uint64_t GetBitOffset() const { return m_bit_offset; }

  uint32_t GetBitfieldBitSize() const { return m_bitfield_bit_size; }

  void SetBitfieldBitSize(uint32_t bitfield_bit_size) {
    m_bitfield_bit_size = bitfield_bit_size;
  }

  bool GetIsBitfield() const { return m_is_bitfield; }

  void SetIsBitfield(bool is_bitfield) { m_is_bitfield = is_bitfield; }

protected:
  lldb::TypeImplSP m_type_impl_sp;
  uint64_t m_bit_offset;
  ConstString m_name;
  uint32_t m_bitfield_bit_size; // Bit size for bitfield members only
  bool m_is_bitfield;
};

///
/// Sometimes you can find the name of the type corresponding to an object, but
/// we don't have debug
/// information for it.  If that is the case, you can return one of these
/// objects, and then if it
/// has a full type, you can use that, but if not at least you can print the
/// name for informational
/// purposes.
///

class TypeAndOrName {
public:
  TypeAndOrName() = default;
  TypeAndOrName(lldb::TypeSP &type_sp);
  TypeAndOrName(const CompilerType &compiler_type);
  TypeAndOrName(const char *type_str);
  TypeAndOrName(ConstString &type_const_string);

  bool operator==(const TypeAndOrName &other) const;

  bool operator!=(const TypeAndOrName &other) const;

  ConstString GetName() const;

  CompilerType GetCompilerType() const { return m_compiler_type; }

  void SetName(ConstString type_name);

  void SetName(const char *type_name_cstr);

  void SetTypeSP(lldb::TypeSP type_sp);

  void SetCompilerType(CompilerType compiler_type);

  bool IsEmpty() const;

  bool HasName() const;

  bool HasCompilerType() const;

  bool HasType() const { return HasCompilerType(); }

  void Clear();

  explicit operator bool() { return !IsEmpty(); }

private:
  CompilerType m_compiler_type;
  ConstString m_type_name;
};

class TypeMemberFunctionImpl {
public:
  TypeMemberFunctionImpl()
      : m_type(), m_decl(), m_name(), m_kind(lldb::eMemberFunctionKindUnknown) {
  }

  TypeMemberFunctionImpl(const CompilerType &type, const CompilerDecl &decl,
                         const std::string &name,
                         const lldb::MemberFunctionKind &kind)
      : m_type(type), m_decl(decl), m_name(name), m_kind(kind) {}

  bool IsValid();

  ConstString GetName() const;

  ConstString GetMangledName() const;

  CompilerType GetType() const;

  CompilerType GetReturnType() const;

  size_t GetNumArguments() const;

  CompilerType GetArgumentAtIndex(size_t idx) const;

  lldb::MemberFunctionKind GetKind() const;

  bool GetDescription(Stream &stream);

protected:
  std::string GetPrintableTypeName();

private:
  CompilerType m_type;
  CompilerDecl m_decl;
  ConstString m_name;
  lldb::MemberFunctionKind m_kind;
};

class TypeEnumMemberImpl {
public:
  TypeEnumMemberImpl()
      : m_integer_type_sp(), m_name("<invalid>"), m_value(), m_valid(false) {}

  TypeEnumMemberImpl(const lldb::TypeImplSP &integer_type_sp,
                     ConstString name, const llvm::APSInt &value);

  TypeEnumMemberImpl(const TypeEnumMemberImpl &rhs) = default;

  TypeEnumMemberImpl &operator=(const TypeEnumMemberImpl &rhs);

  bool IsValid() { return m_valid; }

  ConstString GetName() const { return m_name; }

  const lldb::TypeImplSP &GetIntegerType() const { return m_integer_type_sp; }

  uint64_t GetValueAsUnsigned() const { return m_value.getZExtValue(); }

  int64_t GetValueAsSigned() const { return m_value.getSExtValue(); }

protected:
  lldb::TypeImplSP m_integer_type_sp;
  ConstString m_name;
  llvm::APSInt m_value;
  bool m_valid;
};

class TypeEnumMemberListImpl {
public:
  TypeEnumMemberListImpl() : m_content() {}

  void Append(const lldb::TypeEnumMemberImplSP &type) {
    m_content.push_back(type);
  }

  void Append(const lldb_private::TypeEnumMemberListImpl &type_list);

  lldb::TypeEnumMemberImplSP GetTypeEnumMemberAtIndex(size_t idx) {
    lldb::TypeEnumMemberImplSP enum_member;
    if (idx < GetSize())
      enum_member = m_content[idx];
    return enum_member;
  }

  size_t GetSize() { return m_content.size(); }

private:
  std::vector<lldb::TypeEnumMemberImplSP> m_content;
};

} // namespace lldb_private

#endif // LLDB_SYMBOL_TYPE_H
