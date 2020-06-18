//===-- SymbolFile.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SYMBOL_SYMBOLFILE_H
#define LLDB_SYMBOL_SYMBOLFILE_H

#include "lldb/Core/PluginInterface.h"
#include "lldb/Symbol/CompilerDecl.h"
#include "lldb/Symbol/CompilerDeclContext.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/Function.h"
#include "lldb/Symbol/SourceModule.h"
#include "lldb/Symbol/Type.h"
#include "lldb/Symbol/TypeList.h"
#include "lldb/Symbol/TypeSystem.h"
#include "lldb/Utility/XcodeSDK.h"
#include "lldb/lldb-private.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Errc.h"

#include <mutex>

#if defined(LLDB_CONFIGURATION_DEBUG)
#define ASSERT_MODULE_LOCK(expr) (expr->AssertModuleLock())
#else
#define ASSERT_MODULE_LOCK(expr) ((void)0)
#endif

namespace lldb_private {

class SymbolFile : public PluginInterface {
  /// LLVM RTTI support.
  static char ID;

public:
  /// LLVM RTTI support.
  /// \{
  virtual bool isA(const void *ClassID) const { return ClassID == &ID; }
  static bool classof(const SymbolFile *obj) { return obj->isA(&ID); }
  /// \}

  // Symbol file ability bits.
  //
  // Each symbol file can claim to support one or more symbol file abilities.
  // These get returned from SymbolFile::GetAbilities(). These help us to
  // determine which plug-in will be best to load the debug information found
  // in files.
  enum Abilities {
    CompileUnits = (1u << 0),
    LineTables = (1u << 1),
    Functions = (1u << 2),
    Blocks = (1u << 3),
    GlobalVariables = (1u << 4),
    LocalVariables = (1u << 5),
    VariableTypes = (1u << 6),
    kAllAbilities = ((1u << 7) - 1u)
  };

  static SymbolFile *FindPlugin(lldb::ObjectFileSP objfile_sp);

  // Constructors and Destructors
  SymbolFile(lldb::ObjectFileSP objfile_sp)
      : m_objfile_sp(std::move(objfile_sp)), m_abilities(0),
        m_calculated_abilities(false) {}

  ~SymbolFile() override {}

  /// Get a mask of what this symbol file supports for the object file
  /// that it was constructed with.
  ///
  /// Each symbol file gets to respond with a mask of abilities that
  /// it supports for each object file. This happens when we are
  /// trying to figure out which symbol file plug-in will get used
  /// for a given object file. The plug-in that responds with the
  /// best mix of "SymbolFile::Abilities" bits set, will get chosen to
  /// be the symbol file parser. This allows each plug-in to check for
  /// sections that contain data a symbol file plug-in would need. For
  /// example the DWARF plug-in requires DWARF sections in a file that
  /// contain debug information. If the DWARF plug-in doesn't find
  /// these sections, it won't respond with many ability bits set, and
  /// we will probably fall back to the symbol table SymbolFile plug-in
  /// which uses any information in the symbol table. Also, plug-ins
  /// might check for some specific symbols in a symbol table in the
  /// case where the symbol table contains debug information (STABS
  /// and COFF). Not a lot of work should happen in these functions
  /// as the plug-in might not get selected due to another plug-in
  /// having more abilities. Any initialization work should be saved
  /// for "void SymbolFile::InitializeObject()" which will get called
  /// on the SymbolFile object with the best set of abilities.
  ///
  /// \return
  ///     A uint32_t mask containing bits from the SymbolFile::Abilities
  ///     enumeration. Any bits that are set represent an ability that
  ///     this symbol plug-in can parse from the object file.
  uint32_t GetAbilities() {
    if (!m_calculated_abilities) {
      m_abilities = CalculateAbilities();
      m_calculated_abilities = true;
    }

    return m_abilities;
  }

  virtual uint32_t CalculateAbilities() = 0;

  /// Symbols file subclasses should override this to return the Module that
  /// owns the TypeSystem that this symbol file modifies type information in.
  virtual std::recursive_mutex &GetModuleMutex() const;

  /// Initialize the SymbolFile object.
  ///
  /// The SymbolFile object with the best set of abilities (detected
  /// in "uint32_t SymbolFile::GetAbilities()) will have this function
  /// called if it is chosen to parse an object file. More complete
  /// initialization can happen in this function which will get called
  /// prior to any other functions in the SymbolFile protocol.
  virtual void InitializeObject() {}

  // Compile Unit function calls
  // Approach 1 - iterator
  uint32_t GetNumCompileUnits();
  lldb::CompUnitSP GetCompileUnitAtIndex(uint32_t idx);

  Symtab *GetSymtab();

  virtual lldb::LanguageType ParseLanguage(CompileUnit &comp_unit) = 0;
  /// Return the Xcode SDK comp_unit was compiled against.
  virtual XcodeSDK ParseXcodeSDK(CompileUnit &comp_unit) { return {}; }
  virtual size_t ParseFunctions(CompileUnit &comp_unit) = 0;
  virtual bool ParseLineTable(CompileUnit &comp_unit) = 0;
  virtual bool ParseDebugMacros(CompileUnit &comp_unit) = 0;

  /// Apply a lambda to each external lldb::Module referenced by this
  /// \p comp_unit. Recursively also descends into the referenced external
  /// modules of any encountered compilation unit.
  ///
  /// This function can be used to traverse Clang -gmodules debug
  /// information, which is stored in DWARF files separate from the
  /// object files.
  ///
  /// \param comp_unit
  ///     When this SymbolFile consists of multiple auxilliary
  ///     SymbolFiles, for example, a Darwin debug map that references
  ///     multiple .o files, comp_unit helps choose the auxilliary
  ///     file. In most other cases comp_unit's symbol file is
  ///     identical with *this.
  ///
  /// \param[in] lambda
  ///     The lambda that should be applied to every function. The lambda can
  ///     return true if the iteration should be aborted earlier.
  ///
  /// \param visited_symbol_files
  ///     A set of SymbolFiles that were already visited to avoid
  ///     visiting one file more than once.
  ///
  /// \return
  ///     If the lambda early-exited, this function returns true to
  ///     propagate the early exit.
  virtual bool ForEachExternalModule(
      lldb_private::CompileUnit &comp_unit,
      llvm::DenseSet<lldb_private::SymbolFile *> &visited_symbol_files,
      llvm::function_ref<bool(Module &)> lambda) {
    return false;
  }
  virtual bool ParseSupportFiles(CompileUnit &comp_unit,
                                 FileSpecList &support_files) = 0;
  virtual size_t ParseTypes(CompileUnit &comp_unit) = 0;
  virtual bool ParseIsOptimized(CompileUnit &comp_unit) { return false; }

  virtual bool
  ParseImportedModules(const SymbolContext &sc,
                       std::vector<SourceModule> &imported_modules) = 0;
  virtual size_t ParseBlocksRecursive(Function &func) = 0;
  virtual size_t ParseVariablesForContext(const SymbolContext &sc) = 0;
  virtual Type *ResolveTypeUID(lldb::user_id_t type_uid) = 0;


  /// The characteristics of an array type.
  struct ArrayInfo {
    int64_t first_index = 0;
    llvm::SmallVector<uint64_t, 1> element_orders;
    uint32_t byte_stride = 0;
    uint32_t bit_stride = 0;
  };
  /// If \c type_uid points to an array type, return its characteristics.
  /// To support variable-length array types, this function takes an
  /// optional \p ExecutionContext. If \c exe_ctx is non-null, the
  /// dynamic characteristics for that context are returned.
  virtual llvm::Optional<ArrayInfo>
  GetDynamicArrayInfoForUID(lldb::user_id_t type_uid,
                            const lldb_private::ExecutionContext *exe_ctx) = 0;

  virtual bool CompleteType(CompilerType &compiler_type) = 0;
  virtual void ParseDeclsForContext(CompilerDeclContext decl_ctx) {}
  virtual CompilerDecl GetDeclForUID(lldb::user_id_t uid) {
    return CompilerDecl();
  }
  virtual CompilerDeclContext GetDeclContextForUID(lldb::user_id_t uid) {
    return CompilerDeclContext();
  }
  virtual CompilerDeclContext GetDeclContextContainingUID(lldb::user_id_t uid) {
    return CompilerDeclContext();
  }
  virtual uint32_t ResolveSymbolContext(const Address &so_addr,
                                        lldb::SymbolContextItem resolve_scope,
                                        SymbolContext &sc) = 0;
  virtual uint32_t ResolveSymbolContext(const FileSpec &file_spec,
                                        uint32_t line, bool check_inlines,
                                        lldb::SymbolContextItem resolve_scope,
                                        SymbolContextList &sc_list);

  virtual void DumpClangAST(Stream &s) {}
  virtual void FindGlobalVariables(ConstString name,
                                   const CompilerDeclContext &parent_decl_ctx,
                                   uint32_t max_matches,
                                   VariableList &variables);
  virtual void FindGlobalVariables(const RegularExpression &regex,
                                   uint32_t max_matches,
                                   VariableList &variables);
  virtual void FindFunctions(ConstString name,
                             const CompilerDeclContext &parent_decl_ctx,
                             lldb::FunctionNameType name_type_mask,
                             bool include_inlines, SymbolContextList &sc_list);
  virtual void FindFunctions(const RegularExpression &regex,
                             bool include_inlines, SymbolContextList &sc_list);
  virtual void
  FindTypes(ConstString name, const CompilerDeclContext &parent_decl_ctx,
            uint32_t max_matches,
            llvm::DenseSet<lldb_private::SymbolFile *> &searched_symbol_files,
            TypeMap &types);

  /// Find types specified by a CompilerContextPattern.
  /// \param languages
  ///     Only return results in these languages.
  /// \param searched_symbol_files
  ///     Prevents one file from being visited multiple times.
  virtual void
  FindTypes(llvm::ArrayRef<CompilerContext> pattern, LanguageSet languages,
            llvm::DenseSet<lldb_private::SymbolFile *> &searched_symbol_files,
            TypeMap &types);

  virtual void
  GetMangledNamesForFunction(const std::string &scope_qualified_name,
                             std::vector<ConstString> &mangled_names);

  virtual void GetTypes(lldb_private::SymbolContextScope *sc_scope,
                        lldb::TypeClass type_mask,
                        lldb_private::TypeList &type_list) = 0;

  virtual void PreloadSymbols();

  virtual llvm::Expected<lldb_private::TypeSystem &>
  GetTypeSystemForLanguage(lldb::LanguageType language);

  virtual CompilerDeclContext
  FindNamespace(ConstString name, const CompilerDeclContext &parent_decl_ctx) {
    return CompilerDeclContext();
  }

  ObjectFile *GetObjectFile() { return m_objfile_sp.get(); }
  const ObjectFile *GetObjectFile() const { return m_objfile_sp.get(); }
  ObjectFile *GetMainObjectFile();

  virtual std::vector<std::unique_ptr<CallEdge>>
  ParseCallEdgesInFunction(UserID func_id) {
    return {};
  }

  virtual void AddSymbols(Symtab &symtab) {}

  /// Notify the SymbolFile that the file addresses in the Sections
  /// for this module have been changed.
  virtual void SectionFileAddressesChanged();

  struct RegisterInfoResolver {
    virtual ~RegisterInfoResolver(); // anchor

    virtual const RegisterInfo *ResolveName(llvm::StringRef name) const = 0;
    virtual const RegisterInfo *ResolveNumber(lldb::RegisterKind kind,
                                              uint32_t number) const = 0;
  };
  virtual lldb::UnwindPlanSP
  GetUnwindPlan(const Address &address, const RegisterInfoResolver &resolver) {
    return nullptr;
  }

  /// Return the number of stack bytes taken up by the parameters to this
  /// function.
  virtual llvm::Expected<lldb::addr_t> GetParameterStackSize(Symbol &symbol) {
    return llvm::createStringError(make_error_code(llvm::errc::not_supported),
                                   "Operation not supported.");
  }

  virtual void Dump(Stream &s);

protected:
  void AssertModuleLock();
  virtual uint32_t CalculateNumCompileUnits() = 0;
  virtual lldb::CompUnitSP ParseCompileUnitAtIndex(uint32_t idx) = 0;
  virtual TypeList &GetTypeList() { return m_type_list; }

  void SetCompileUnitAtIndex(uint32_t idx, const lldb::CompUnitSP &cu_sp);

  lldb::ObjectFileSP m_objfile_sp; // Keep a reference to the object file in
                                   // case it isn't the same as the module
                                   // object file (debug symbols in a separate
                                   // file)
  llvm::Optional<std::vector<lldb::CompUnitSP>> m_compile_units;
  TypeList m_type_list;
  Symtab *m_symtab = nullptr;
  uint32_t m_abilities;
  bool m_calculated_abilities;

private:
  SymbolFile(const SymbolFile &) = delete;
  const SymbolFile &operator=(const SymbolFile &) = delete;
};

} // namespace lldb_private

#endif // LLDB_SYMBOL_SYMBOLFILE_H
