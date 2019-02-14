//===-- SwiftASTContext.h ---------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2016 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SwiftASTContext_h_
#define liblldb_SwiftASTContext_h_

#include "Plugins/ExpressionParser/Swift/SwiftPersistentExpressionState.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/ThreadSafeDenseMap.h"
#include "lldb/Core/ThreadSafeDenseSet.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/lldb-private.h"

#include "lldb/Utility/Either.h"
#include "lldb/Utility/Status.h"

#include "llvm/ADT/Optional.h"
#include "llvm/Support/Threading.h"

#include <map>
#include <set>

namespace swift {
enum class IRGenDebugInfoLevel : unsigned;
class CanType;
class DWARFImporter;
class IRGenOptions;
class NominalTypeDecl;
class SILModule;
class VarDecl;
class ModuleDecl;
struct PrintOptions;
namespace irgen {
class FixedTypeInfo;
class TypeInfo;
}
namespace serialization {
struct ValidationInfo;
class ExtendedValidationInfo;
}
}

class DWARFASTParser;
class SwiftEnumDescriptor;

namespace lldb_private {

class SwiftASTContext : public TypeSystem {
public:
  typedef lldb_utility::Either<CompilerType, swift::Decl *> TypeOrDecl;

private:
  struct EitherComparator {
    bool operator()(const TypeOrDecl &r1, const TypeOrDecl &r2) const {
      auto r1_as1 = r1.GetAs<CompilerType>();
      auto r1_as2 = r1.GetAs<swift::Decl *>();

      auto r2_as1 = r2.GetAs<CompilerType>();
      auto r2_as2 = r2.GetAs<swift::Decl *>();

      if (r1_as1.hasValue() && r2_as1.hasValue())
        return r1_as1.getValue() < r2_as1.getValue();

      if (r1_as2.hasValue() && r2_as2.hasValue())
        return r1_as2.getValue() < r2_as2.getValue();

      if (r1_as1.hasValue() && r2_as2.hasValue())
        return (void *)r1_as1->GetOpaqueQualType() < (void *)r2_as2.getValue();

      if (r1_as2.hasValue() && r2_as1.hasValue())
        return (void *)r1_as2.getValue() < (void *)r2_as1->GetOpaqueQualType();

      return false;
    }
  };

public:
  typedef std::set<TypeOrDecl, EitherComparator> TypesOrDecls;

  class LanguageFlags {
  public:
    enum : uint64_t {
      eIsIndirectEnumCase = 0x1ULL,
      eIgnoreInstancePointerness = 0x2ULL
    };

  private:
    LanguageFlags() = delete;
  };

  //------------------------------------------------------------------
  // llvm casting support
  //------------------------------------------------------------------
  static bool classof(const TypeSystem *ts) {
    return ts->getKind() == TypeSystem::eKindSwift;
  }

  /// Provide the global LLVMContext.
  static llvm::LLVMContext &GetGlobalLLVMContext();

  //------------------------------------------------------------------
  // Constructors and destructors
  //------------------------------------------------------------------
  SwiftASTContext(const char *triple = NULL, Target *target = NULL);

  SwiftASTContext(const SwiftASTContext &rhs);

  ~SwiftASTContext();

  //------------------------------------------------------------------
  // PluginInterface functions
  //------------------------------------------------------------------
  ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  static ConstString GetPluginNameStatic();

  /// Create a SwiftASTContext from a Module.  This context is used
  /// for frame variable ans uses ClangImporter options specific to
  /// this lldb::Module.  The optional target is to create a
  /// module-specific scract context.
  static lldb::TypeSystemSP CreateInstance(lldb::LanguageType language,
                                           Module &module,
                                           Target *target = nullptr);
  /// Create a SwiftASTContext from a Target.  This context is global
  /// and used for the expression evaluator.
  static lldb::TypeSystemSP CreateInstance(lldb::LanguageType language,
                                           Target &target,
                                           const char *extra_options);

  static void EnumerateSupportedLanguages(
      std::set<lldb::LanguageType> &languages_for_types,
      std::set<lldb::LanguageType> &languages_for_expressions);

  static void Initialize();

  static void Terminate();

  bool SupportsLanguage(lldb::LanguageType language) override;

  Status IsCompatible() override;

  swift::SourceManager &GetSourceManager();

  swift::LangOptions &GetLanguageOptions();

  swift::DiagnosticEngine &GetDiagnosticEngine();

  swift::SearchPathOptions &GetSearchPathOptions();

  swift::ClangImporterOptions &GetClangImporterOptions();

  swift::CompilerInvocation &GetCompilerInvocation();

  swift::SILOptions &GetSILOptions();

  swift::ASTContext *GetASTContext();

  swift::IRGenDebugInfoLevel GetGenerateDebugInfo();

  static swift::PrintOptions
  GetUserVisibleTypePrintingOptions(bool print_help_if_available);

  void SetGenerateDebugInfo(swift::IRGenDebugInfoLevel b);

  bool AddModuleSearchPath(const char *path);

  bool AddFrameworkSearchPath(const char *path);

  bool AddClangArgument(std::string arg, bool unique = true);

  bool AddClangArgumentPair(const char *arg1, const char *arg2);

  /// Add a list of Clang arguments to the ClangImporter options and
  /// apply the working directory to any relative paths.
  void AddExtraClangArgs(std::vector<std::string> ExtraArgs);

  const char *GetPlatformSDKPath() const {
    if (m_platform_sdk_path.empty())
      return NULL;
    return m_platform_sdk_path.c_str();
  }

  void SetPlatformSDKPath(std::string &&sdk_path) {
    m_platform_sdk_path = sdk_path;
  }

  void SetPlatformSDKPath(const char *path) {
    if (path)
      m_platform_sdk_path = path;
    else
      m_platform_sdk_path.clear();
  }

  size_t GetNumModuleSearchPaths() const;

  const char *GetModuleSearchPathAtIndex(size_t idx) const;

  size_t GetNumFrameworkSearchPaths() const;

  const char *GetFrameworkSearchPathAtIndex(size_t idx) const;

  /// \return the ExtraArgs of the ClangImporterOptions.
  const std::vector<std::string> &GetClangArguments();

  swift::ModuleDecl *CreateModule(const ConstString &module_basename,
                                  Status &error);

  // This function should only be called when all search paths
  // for all items in a swift::ASTContext have been setup to
  // allow for imports to happen correctly. Use with caution,
  // or use the GetModule() call that takes a FileSpec.
  swift::ModuleDecl *GetModule(const ConstString &module_name, Status &error);

  swift::ModuleDecl *GetModule(const FileSpec &module_spec, Status &error);

  void CacheModule(swift::ModuleDecl *module);

  // Call this after the search paths are set up, it will find the module given
  // by module, load the module into the AST context, and also load any
  // "LinkLibraries" that the module requires.

  swift::ModuleDecl *FindAndLoadModule(const ConstString &module_basename,
                                       Process &process, Status &error);

  swift::ModuleDecl *FindAndLoadModule(const FileSpec &module_spec,
                                       Process &process, Status &error);

  void LoadModule(swift::ModuleDecl *swift_module, Process &process,
                  Status &error);

  bool RegisterSectionModules(Module &module,
                              std::vector<std::string> &module_names);

  void ValidateSectionModules(Module &module, // this is used to print errors
                              const std::vector<std::string> &module_names);

  // Swift modules that are backed by dylibs (libFoo.dylib) rather than
  // frameworks don't actually record the library dependencies in the module.
  // This will hand load any libraries that are on the IRGen LinkLibraries list
  // using the compiler's search paths.
  // It doesn't do frameworks since frameworks don't need it and this is kind of
  // a hack anyway.

  void LoadExtraDylibs(Process &process, Status &error);

  swift::Identifier GetIdentifier(const char *name);

  swift::Identifier GetIdentifier(const llvm::StringRef &name);

  // Find a type by a fully qualified name that includes the module name
  // (the format being "<module_name>.<type_name>").
  CompilerType FindQualifiedType(const char *qualified_name);

  CompilerType FindType(const char *name, swift::ModuleDecl *swift_module);

  llvm::Optional<SwiftASTContext::TypeOrDecl>
  FindTypeOrDecl(const char *name, swift::ModuleDecl *swift_module);

  size_t FindTypes(const char *name, swift::ModuleDecl *swift_module,
                   std::set<CompilerType> &results, bool append = true);

  size_t FindTypesOrDecls(const char *name, swift::ModuleDecl *swift_module,
                          TypesOrDecls &results, bool append = true);

  size_t FindContainedTypeOrDecl(llvm::StringRef name,
                                 TypeOrDecl container_type_or_decl,
                                 TypesOrDecls &results, bool append = true);

  size_t FindType(const char *name, std::set<CompilerType> &results,
                  bool append = true);

  CompilerType FindFirstType(const char *name, const ConstString &module_name);

  CompilerType GetTypeFromMangledTypename(const char *mangled_typename,
                                          Status &error);

  // Retrieve the Swift.AnyObject type.
  CompilerType GetAnyObjectType();

  // Get a function type that returns nothing and take no parameters
  CompilerType GetVoidFunctionType();

  static SwiftASTContext *GetSwiftASTContext(swift::ASTContext *ast);

  swift::irgen::IRGenerator &GetIRGenerator(swift::IRGenOptions &opts,
                                            swift::SILModule &module);

  swift::irgen::IRGenModule &GetIRGenModule();

  std::string GetTriple() const;

  bool SetTriple(const char *triple, lldb_private::Module *module = NULL);

  uint32_t GetPointerBitAlignment();

  // Imports the type from the passed in type into this SwiftASTContext. The
  // type must be a Swift type. If the type can be imported, returns the
  // CompilerType for the imported type.
  // If it cannot be, returns an invalid CompilerType, and sets the error to
  // indicate what went wrong.
  CompilerType ImportType(CompilerType &type, Status &error);

  swift::ClangImporter *GetClangImporter();

  // ***********************************************************
  //  these calls create non-nominal types which are given in
  //  metadata just in terms of their building blocks and for
  //  which there is no one basic type to compose from
  // ***********************************************************
  CompilerType CreateTupleType(const std::vector<CompilerType> &elements);

  struct TupleElement {
    ConstString element_name;
    CompilerType element_type;
  };

  CompilerType CreateTupleType(const std::vector<TupleElement> &elements);

  CompilerType GetErrorType();

  CompilerType GetNSErrorType(Status &error);

  CompilerType CreateMetatypeType(CompilerType instance_type);

  bool HasErrors();

  // NEVER call this without checking HasFatalErrors() first.
  // This clears the fatal-error state which is terrible.
  // We will assert if you clear an actual fatal error.
  void ClearDiagnostics();

  bool SetColorizeDiagnostics(bool b);

  void PrintDiagnostics(DiagnosticManager &diagnostic_manager,
                        uint32_t bufferID = UINT32_MAX, uint32_t first_line = 0,
                        uint32_t last_line = UINT32_MAX,
                        uint32_t line_offset = 0);

  ConstString GetMangledTypeName(swift::TypeBase *);

  swift::IRGenOptions &GetIRGenOptions();

  void ModulesDidLoad(ModuleList &module_list);

  void ClearModuleDependentCaches();

  void DumpConfiguration(Log *log);

  bool HasTarget() const;

  bool CheckProcessChanged();

  // FIXME: this should be removed once we figure out who should really own the
  // DebuggerClient's that we are sticking into the Swift Modules.
  void AddDebuggerClient(swift::DebuggerClient *debugger_client);

  typedef llvm::DenseMap<const char *, swift::ModuleDecl *> SwiftModuleMap;

  const SwiftModuleMap &GetModuleCache() { return m_swift_module_cache; }

  static bool HasFatalErrors(swift::ASTContext *ast_context);

  bool HasFatalErrors() const {
    return m_fatal_errors.Fail() || HasFatalErrors(m_ast_context_ap.get());
  }

  Status GetFatalErrors();

  union ExtraTypeInformation {
    uint64_t m_intValue;
    struct ExtraTypeInformationFlags {
      ExtraTypeInformationFlags(bool is_trivial_option_set)
          : m_is_trivial_option_set(is_trivial_option_set) {}

      bool m_is_trivial_option_set : 1;
    } m_flags;

    ExtraTypeInformation();

    ExtraTypeInformation(swift::CanType);
  };

  const swift::irgen::TypeInfo *GetSwiftTypeInfo(void *type);

  const swift::irgen::FixedTypeInfo *GetSwiftFixedTypeInfo(void *type);

  DWARFASTParser *GetDWARFParser() override;

  //----------------------------------------------------------------------
  // CompilerDecl functions
  //----------------------------------------------------------------------
  ConstString DeclGetName(void *opaque_decl) override {
    return ConstString("");
  }

  //----------------------------------------------------------------------
  // CompilerDeclContext functions
  //----------------------------------------------------------------------

  std::vector<CompilerDecl>
  DeclContextFindDeclByName(void *opaque_decl_ctx, ConstString name,
                            const bool ignore_imported_decls) override {
    return {};
  }

  bool DeclContextIsStructUnionOrClass(void *opaque_decl_ctx) override;

  ConstString DeclContextGetName(void *opaque_decl_ctx) override;

  ConstString DeclContextGetScopeQualifiedName(void *opaque_decl_ctx) override;

  bool DeclContextIsClassMethod(void *opaque_decl_ctx,
                                lldb::LanguageType *language_ptr,
                                bool *is_instance_method_ptr,
                                ConstString *language_object_name_ptr) override;

  //----------------------------------------------------------------------
  // Tests
  //----------------------------------------------------------------------

  bool IsArrayType(void *type, CompilerType *element_type, uint64_t *size,
                   bool *is_incomplete) override;

  bool IsAggregateType(void *type) override;

  bool IsCharType(void *type) override;

  bool IsCompleteType(void *type) override;

  bool IsDefined(void *type) override;

  bool IsFloatingPointType(void *type, uint32_t &count,
                           bool &is_complex) override;

  bool IsFunctionType(void *type, bool *is_variadic_ptr) override;

  size_t GetNumberOfFunctionArguments(void *type) override;

  CompilerType GetFunctionArgumentAtIndex(void *type,
                                          const size_t index) override;

  bool IsFunctionPointerType(void *type) override;

  bool IsBlockPointerType(void *type,
                          CompilerType *function_pointer_type_ptr) override;

  bool IsIntegerType(void *type, bool &is_signed) override;

  bool IsPossibleDynamicType(void *type,
                             CompilerType *target_type, // Can pass NULL
                             bool check_cplusplus, bool check_objc,
                             bool check_swift) override;

  bool IsPointerType(void *type, CompilerType *pointee_type) override;

  bool IsScalarType(void *type) override;

  bool IsVoidType(void *type) override;

  static bool IsGenericType(const CompilerType &compiler_type);

  static bool IsSelfArchetypeType(const CompilerType &compiler_type);

  bool IsTrivialOptionSetType(const CompilerType &compiler_type);

  bool IsErrorType(const CompilerType &compiler_type);

  static bool IsFullyRealized(const CompilerType &compiler_type);

  struct ProtocolInfo {
    uint32_t m_num_protocols;
    uint32_t m_num_payload_words;
    uint32_t m_num_storage_words;
    bool m_is_class_only;
    bool m_is_objc;
    bool m_is_anyobject;
    bool m_is_errortype;

    /// The superclass bound, which can only be non-null when this is
    /// a class-bound existential.
    CompilerType m_superclass;

    /// The member index for the error value within an error
    /// existential.
    static constexpr uint32_t error_instance_index = 0;

    /// Retrieve the index at which the instance type occurs.
    uint32_t GetInstanceTypeIndex() const { return m_num_payload_words; }
  };

  static bool GetProtocolTypeInfo(const CompilerType &type,
                                  ProtocolInfo &protocol_info);

  enum class TypeAllocationStrategy { eInline, ePointer, eDynamic, eUnknown };

  static TypeAllocationStrategy GetAllocationStrategy(const CompilerType &type);

  enum class NonTriviallyManagedReferenceStrategy {
    eWeak,
    eUnowned,
    eUnmanaged
  };

  static bool IsNonTriviallyManagedReferenceType(
      const CompilerType &type, NonTriviallyManagedReferenceStrategy &strategy,
      CompilerType *underlying_type = nullptr);

  bool IsObjCObjectPointerType(const CompilerType &type,
                               CompilerType *class_type_ptr);

  //----------------------------------------------------------------------
  // Type Completion
  //----------------------------------------------------------------------

  bool GetCompleteType(void *type) override;

  //----------------------------------------------------------------------
  // AST related queries
  //----------------------------------------------------------------------

  uint32_t GetPointerByteSize() override;

  //----------------------------------------------------------------------
  // Accessors
  //----------------------------------------------------------------------

  ConstString GetTypeName(void *type) override;

  ConstString GetDisplayTypeName(void *type) override;

  ConstString GetTypeSymbolName(void *type) override;

  ConstString GetMangledTypeName(void *type) override;

  uint32_t GetTypeInfo(void *type,
                       CompilerType *pointee_or_element_clang_type) override;

  lldb::LanguageType GetMinimumLanguage(void *type) override;

  lldb::TypeClass GetTypeClass(void *type) override;

  //----------------------------------------------------------------------
  // Creating related types
  //----------------------------------------------------------------------

  CompilerType GetArrayElementType(void *type, uint64_t *stride) override;

  CompilerType GetCanonicalType(void *type) override;

  CompilerType GetInstanceType(void *type) override;

  // Returns -1 if this isn't a function of if the function doesn't have a
  // prototype. Returns a value >override if there is a prototype.
  int GetFunctionArgumentCount(void *type) override;

  CompilerType GetFunctionArgumentTypeAtIndex(void *type, size_t idx) override;

  CompilerType GetFunctionReturnType(void *type) override;

  size_t GetNumMemberFunctions(void *type) override;

  TypeMemberFunctionImpl GetMemberFunctionAtIndex(void *type,
                                                  size_t idx) override;

  CompilerType GetPointeeType(void *type) override;

  CompilerType GetPointerType(void *type) override;

  //----------------------------------------------------------------------
  // Exploring the type
  //----------------------------------------------------------------------

  llvm::Optional<uint64_t>
  GetBitSize(lldb::opaque_compiler_type_t type,
             ExecutionContextScope *exe_scope) override;

  uint64_t GetByteStride(lldb::opaque_compiler_type_t type) override;

  lldb::Encoding GetEncoding(void *type, uint64_t &count) override;

  lldb::Format GetFormat(void *type) override;

  uint32_t GetNumChildren(void *type, bool omit_empty_base_classes,
                          const ExecutionContext *exe_ctx) override;

  lldb::BasicType GetBasicTypeEnumeration(void *type) override;

  uint32_t GetNumFields(void *type) override;

  CompilerType GetFieldAtIndex(void *type, size_t idx, std::string &name,
                               uint64_t *bit_offset_ptr,
                               uint32_t *bitfield_bit_size_ptr,
                               bool *is_bitfield_ptr) override;

  CompilerType GetChildCompilerTypeAtIndex(
      void *type, ExecutionContext *exe_ctx, size_t idx,
      bool transparent_pointers, bool omit_empty_base_classes,
      bool ignore_array_bounds, std::string &child_name,
      uint32_t &child_byte_size, int32_t &child_byte_offset,
      uint32_t &child_bitfield_bit_size, uint32_t &child_bitfield_bit_offset,
      bool &child_is_base_class, bool &child_is_deref_of_parent,
      ValueObject *valobj, uint64_t &language_flags) override;

  // Lookup a child given a name. This function will match base class names
  // and member names in "clang_type" only, not descendants.
  uint32_t GetIndexOfChildWithName(void *type, const char *name,
                                   bool omit_empty_base_classes) override;

  // Lookup a child member given a name. This function will match member names
  // only and will descend into "clang_type" children in search for the first
  // member in this class, or any base class that matches "name".
  // TODO: Return all matches for a given name by returning a
  // vector<vector<uint32_t>> so we catch all names that match a given child
  // name, not just the first.
  size_t
  GetIndexOfChildMemberWithName(void *type, const char *name,
                                bool omit_empty_base_classes,
                                std::vector<uint32_t> &child_indexes) override;

  size_t GetNumTemplateArguments(void *type) override;

  lldb::GenericKind GetGenericArgumentKind(void *type, size_t idx) override;
  CompilerType GetUnboundGenericType(void *type, size_t idx);
  CompilerType GetBoundGenericType(void *type, size_t idx);
  CompilerType GetGenericArgumentType(void *type, size_t idx) override;

  CompilerType GetTypeForFormatters(void *type) override;

  LazyBool ShouldPrintAsOneLiner(void *type, ValueObject *valobj) override;

  bool IsMeaninglessWithoutDynamicResolution(void *type) override;

  static bool GetSelectedEnumCase(const CompilerType &type,
                                  const DataExtractor &data, ConstString *name,
                                  bool *has_payload, CompilerType *payload,
                                  bool *is_indirect);

  //----------------------------------------------------------------------
  // Dumping types
  //----------------------------------------------------------------------
  void DumpValue(void *type, ExecutionContext *exe_ctx, Stream *s,
                 lldb::Format format, const DataExtractor &data,
                 lldb::offset_t data_offset, size_t data_byte_size,
                 uint32_t bitfield_bit_size, uint32_t bitfield_bit_offset,
                 bool show_types, bool show_summary, bool verbose,
                 uint32_t depth) override;

  bool DumpTypeValue(void *type, Stream *s, lldb::Format format,
                     const DataExtractor &data, lldb::offset_t data_offset,
                     size_t data_byte_size, uint32_t bitfield_bit_size,
                     uint32_t bitfield_bit_offset,
                     ExecutionContextScope *exe_scope,
                     bool is_base_class) override;

  void DumpTypeDescription(void *type) override; // Dump to stdout

  void DumpTypeDescription(void *type, Stream *s) override;

  void DumpTypeDescription(void *type, bool print_help_if_available,
                           bool print_extensions_if_available);

  void DumpTypeDescription(void *type, Stream *s, bool print_help_if_available,
                           bool print_extensions_if_available);

  //----------------------------------------------------------------------
  // TODO: These methods appear unused. Should they be removed?
  //----------------------------------------------------------------------

  bool IsRuntimeGeneratedType(void *type) override;

  void DumpSummary(void *type, ExecutionContext *exe_ctx, Stream *s,
                   const DataExtractor &data, lldb::offset_t data_offset,
                   size_t data_byte_size) override;

  // Converts "s" to a floating point value and place resulting floating
  // point bytes in the "dst" buffer.
  size_t ConvertStringToFloatValue(void *type, const char *s, uint8_t *dst,
                                   size_t dst_size) override;

  //----------------------------------------------------------------------
  // TODO: Determine if these methods should move to ClangASTContext.
  //----------------------------------------------------------------------

  bool IsPointerOrReferenceType(void *type,
                                CompilerType *pointee_type) override;

  unsigned GetTypeQualifiers(void *type) override;

  bool IsCStringType(void *type, uint32_t &length) override;

  size_t GetTypeBitAlign(void *type) override;

  CompilerType GetBasicTypeFromAST(lldb::BasicType basic_type) override;

  CompilerType GetBuiltinTypeForEncodingAndBitSize(lldb::Encoding encoding,
                                                   size_t bit_size) override {
    return CompilerType();
  }

  bool IsBeingDefined(void *type) override;

  bool IsConst(void *type) override;

  uint32_t IsHomogeneousAggregate(void *type,
                                  CompilerType *base_type_ptr) override;

  bool IsPolymorphicClass(void *type) override;

  bool IsTypedefType(void *type) override;

  // If the current object represents a typedef type, get the underlying type
  CompilerType GetTypedefedType(void *type) override;

  CompilerType GetUnboundType(lldb::opaque_compiler_type_t type) override;
  CompilerType MapIntoContext(lldb::StackFrameSP &frame_sp,
                              lldb::opaque_compiler_type_t type) override;


  bool IsVectorType(void *type, CompilerType *element_type,
                    uint64_t *size) override;

  CompilerType GetFullyUnqualifiedType(void *type) override;

  CompilerType GetNonReferenceType(void *type) override;

  CompilerType GetLValueReferenceType(void *type) override;

  CompilerType GetRValueReferenceType(void *opaque_type) override;

  uint32_t GetNumDirectBaseClasses(void *opaque_type) override;

  uint32_t GetNumVirtualBaseClasses(void *opaque_type) override;

  CompilerType GetDirectBaseClassAtIndex(void *opaque_type, size_t idx,
                                         uint32_t *bit_offset_ptr) override;

  CompilerType GetVirtualBaseClassAtIndex(void *opaque_type, size_t idx,
                                          uint32_t *bit_offset_ptr) override;

  bool IsReferenceType(void *type, CompilerType *pointee_type,
                       bool *is_rvalue) override;

  bool
  ShouldTreatScalarValueAsAddress(lldb::opaque_compiler_type_t type) override;

  uint32_t GetNumPointeeChildren(void *type);

  static bool IsImportedType(const CompilerType &type,
                             CompilerType *original_type);

  static bool IsImportedObjectiveCType(const CompilerType &type,
                                       CompilerType *original_type);

  CompilerType GetReferentType(const CompilerType &compiler_type);

  lldb::TypeSP GetCachedType(const ConstString &mangled);

  void SetCachedType(const ConstString &mangled, const lldb::TypeSP &type_sp);

  static bool PerformUserImport(SwiftASTContext &swift_ast_context,
                                SymbolContext &sc,
                                ExecutionContextScope &exe_scope,
                                lldb::StackFrameWP &stack_frame_wp,
                                swift::SourceFile &source_file, Status &error);

  static bool PerformAutoImport(SwiftASTContext &swift_ast_context,
                                SymbolContext &sc,
                                lldb::StackFrameWP &stack_frame_wp,
                                swift::SourceFile *source_file, Status &error);

protected:
  // This map uses the string value of ConstStrings as the key, and the TypeBase
  // * as the value. Since the ConstString strings are uniqued, we can use
  // pointer equality for string value equality.
  typedef llvm::DenseMap<const char *, swift::TypeBase *>
      SwiftTypeFromMangledNameMap;
  // Similar logic applies to this "reverse" map
  typedef llvm::DenseMap<swift::TypeBase *, const char *>
      SwiftMangledNameFromTypeMap;

  llvm::TargetOptions *getTargetOptions();

  swift::ModuleDecl *GetScratchModule();

  swift::SILModule *GetSILModule();

  swift::SerializedModuleLoader *GetSerializeModuleLoader();

  swift::ModuleDecl *GetCachedModule(const ConstString &module_name);

  void CacheDemangledType(const char *, swift::TypeBase *);

  void CacheDemangledTypeFailure(const char *);

  bool LoadOneImage(Process &process, FileSpec &link_lib_spec, Status &error);

  bool LoadLibraryUsingPaths(Process &process, llvm::StringRef library_name,
                             std::vector<std::string> &search_paths,
                             bool check_rpath, StreamString &all_dlopen_errors);

  bool TargetHasNoSDK();

  std::vector<lldb::DataBufferSP> &GetASTVectorForModule(const Module *module);

  /// Data members.
  /// @{
  std::unique_ptr<swift::CompilerInvocation> m_compiler_invocation_ap;
  std::unique_ptr<swift::SourceManager> m_source_manager_ap;
  std::unique_ptr<swift::DiagnosticEngine> m_diagnostic_engine_ap;
  // CompilerInvocation, SourceMgr, and DiagEngine must come
  // before the ASTContext, so they get deallocated *after* the
  // ASTContext.
  std::unique_ptr<swift::ASTContext> m_ast_context_ap;
  std::unique_ptr<llvm::TargetOptions> m_target_options_ap;
  std::unique_ptr<swift::irgen::IRGenerator> m_ir_generator_ap;
  std::unique_ptr<swift::irgen::IRGenModule> m_ir_gen_module_ap;
  llvm::once_flag m_ir_gen_module_once;
  std::unique_ptr<swift::DiagnosticConsumer> m_diagnostic_consumer_ap;
  std::unique_ptr<DWARFASTParser> m_dwarf_ast_parser_ap;
  Status m_error; // Any errors that were found while creating or using the AST
                 // context
  swift::ModuleDecl *m_scratch_module = nullptr;
  std::unique_ptr<swift::SILModule> m_sil_module_ap;
  /// Owned by the AST.
  swift::SerializedModuleLoader *m_serialized_module_loader = nullptr;
  swift::ClangImporter *m_clang_importer = nullptr;
  swift::DWARFImporter *m_dwarf_importer = nullptr;
  SwiftModuleMap m_swift_module_cache;
  SwiftTypeFromMangledNameMap m_mangled_name_to_type_map;
  SwiftMangledNameFromTypeMap m_type_to_mangled_name_map;
  uint32_t m_pointer_byte_size = 0;
  uint32_t m_pointer_bit_align = 0;
  CompilerType m_void_function_type;
  /// Only if this AST belongs to a target will this contain a valid
  /// target weak pointer.
  lldb::TargetWP m_target_wp;
  /// Only if this AST belongs to a target, and an expression has been
  /// evaluated will the target's process pointer be filled in
  lldb_private::Process *m_process = nullptr;
  std::string m_platform_sdk_path;

  typedef std::map<Module *, std::vector<lldb::DataBufferSP>> ASTFileDataMap;
  ASTFileDataMap m_ast_file_data_map;
  // FIXME: this vector is needed because the LLDBNameLookup debugger clients
  // are being put into the Module for the SourceFile that we compile the
  // expression into, and so have to live as long as the Module. But it's too
  // late to change swift to get it to take ownership of these DebuggerClients.
  // Since we use the same Target SwiftASTContext for all our compilations,
  // holding them here will keep them alive as long as we need.
  std::vector<std::unique_ptr<swift::DebuggerClient>> m_debugger_clients;
  bool m_initialized_language_options = false;
  bool m_initialized_search_path_options = false;
  bool m_initialized_clang_importer_options = false;
  bool m_reported_fatal_error = false;
  Status m_fatal_errors;

  typedef ThreadSafeDenseSet<const char *> SwiftMangledNameSet;
  SwiftMangledNameSet m_negative_type_cache;

  typedef ThreadSafeDenseMap<void *, ExtraTypeInformation>
      ExtraTypeInformationMap;
  ExtraTypeInformationMap m_extra_type_info_cache;

  typedef ThreadSafeDenseMap<const char *, lldb::TypeSP> SwiftTypeMap;
  SwiftTypeMap m_swift_type_map;
  /// @}

  ExtraTypeInformation GetExtraTypeInformation(void *type);

  /// Record the set of stored properties for each nominal type declaration
  /// for which we've asked this question.
  ///
  /// All of the information in this DenseMap is easily re-constructed
  /// with NominalTypeDecl::getStoredProperties(), but we cache the
  /// result to provide constant-time indexed access.
  llvm::DenseMap<swift::NominalTypeDecl *, std::vector<swift::VarDecl *>>
    m_stored_properties;

  /// Retrieve the stored properties for the given nominal type declaration.
  llvm::ArrayRef<swift::VarDecl *> GetStoredProperties(
                                               swift::NominalTypeDecl *nominal);

  SwiftEnumDescriptor *GetCachedEnumInfo(void *type);

  friend class CompilerType;

  /// Apply a PathMappingList dictionary on all search paths in the
  /// ClangImporterOptions.
  void RemapClangImporterOptions(const PathMappingList &path_map);

  /// Infer the appropriate Swift resource directory for a target triple.
  llvm::StringRef GetResourceDir(const llvm::Triple &target);

  /// Implementation of \c GetResourceDir.
  static std::string GetResourceDir(llvm::StringRef platform_sdk_path,
                                    llvm::StringRef swift_stdlib_os_dir,
                                    std::string swift_dir,
                                    std::string xcode_contents_path,
                                    std::string toolchain_path,
                                    std::string cl_tools_path);

  /// Return the name of the OS-specific subdirectory containing the
  /// Swift stdlib needed for \p target.
  static llvm::StringRef GetSwiftStdlibOSDir(const llvm::Triple &target,
                                             const llvm::Triple &host);
};

class SwiftASTContextForExpressions : public SwiftASTContext {
public:
  SwiftASTContextForExpressions(Target &target);

  virtual ~SwiftASTContextForExpressions() {}

  UserExpression *
  GetUserExpression(llvm::StringRef expr, llvm::StringRef prefix,
                    lldb::LanguageType language,
                    Expression::ResultType desired_type,
                    const EvaluateExpressionOptions &options) override;

  PersistentExpressionState *GetPersistentExpressionState() override;

private:
  std::unique_ptr<SwiftPersistentExpressionState> m_persistent_state_up;
};

void printASTValidationInfo(
    const swift::serialization::ValidationInfo &ast_info,
    const swift::serialization::ExtendedValidationInfo &ext_ast_info,
    const Module &module, llvm::StringRef module_buf);

}

#endif // #ifndef liblldb_SwiftASTContext_h_
