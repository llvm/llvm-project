//===-- SymbolFile.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolFile_h_
#define liblldb_SymbolFile_h_

#include "lldb/Core/PluginInterface.h"
#include "lldb/Symbol/CompilerDecl.h"
#include "lldb/Symbol/CompilerDeclContext.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Symbol/Type.h"
#include "lldb/lldb-private.h"

#include "llvm/ADT/DenseSet.h"

namespace lldb_private {

class SymbolFile : public PluginInterface {
public:
  //------------------------------------------------------------------
  // Symbol file ability bits.
  //
  // Each symbol file can claim to support one or more symbol file
  // abilities. These get returned from SymbolFile::GetAbilities().
  // These help us to determine which plug-in will be best to load
  // the debug information found in files.
  //------------------------------------------------------------------
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

  static SymbolFile *FindPlugin(ObjectFile *obj_file);

  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  SymbolFile(ObjectFile *obj_file)
      : m_obj_file(obj_file), m_abilities(0), m_calculated_abilities(false) {}

  ~SymbolFile() override {}

  //------------------------------------------------------------------
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
  /// @return
  ///     A uint32_t mask containing bits from the SymbolFile::Abilities
  ///     enumeration. Any bits that are set represent an ability that
  ///     this symbol plug-in can parse from the object file.
  ///------------------------------------------------------------------
  uint32_t GetAbilities() {
    if (!m_calculated_abilities) {
      m_abilities = CalculateAbilities();
      m_calculated_abilities = true;
    }

    return m_abilities;
  }

  virtual uint32_t CalculateAbilities() = 0;

  //------------------------------------------------------------------
  /// Initialize the SymbolFile object.
  ///
  /// The SymbolFile object with the best set of abilities (detected
  /// in "uint32_t SymbolFile::GetAbilities()) will have this function
  /// called if it is chosen to parse an object file. More complete
  /// initialization can happen in this function which will get called
  /// prior to any other functions in the SymbolFile protocol.
  //------------------------------------------------------------------
  virtual void InitializeObject() {}

  //------------------------------------------------------------------
  // Compile Unit function calls
  //------------------------------------------------------------------
  // Approach 1 - iterator
  virtual uint32_t GetNumCompileUnits() = 0;
  virtual lldb::CompUnitSP ParseCompileUnitAtIndex(uint32_t index) = 0;

  virtual lldb::LanguageType
  ParseCompileUnitLanguage(const SymbolContext &sc) = 0;
  virtual size_t ParseCompileUnitFunctions(const SymbolContext &sc) = 0;
  virtual bool ParseCompileUnitLineTable(const SymbolContext &sc) = 0;
  virtual bool ParseCompileUnitDebugMacros(const SymbolContext &sc) = 0;
  virtual bool ParseCompileUnitSupportFiles(const SymbolContext &sc,
                                            FileSpecList &support_files) = 0;
  virtual bool
  ParseCompileUnitIsOptimized(const lldb_private::SymbolContext &sc) {
    return false;
  }
  virtual bool
  ParseImportedModules(const SymbolContext &sc,
                       std::vector<ConstString> &imported_modules) = 0;
  virtual size_t ParseFunctionBlocks(const SymbolContext &sc) = 0;
  virtual size_t ParseTypes(const SymbolContext &sc) = 0;
  virtual size_t ParseVariablesForContext(const SymbolContext &sc) = 0;
  virtual Type *ResolveTypeUID(lldb::user_id_t type_uid) = 0;
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
                                        uint32_t resolve_scope,
                                        SymbolContext &sc) = 0;
  virtual uint32_t ResolveSymbolContext(const FileSpec &file_spec,
                                        uint32_t line, bool check_inlines,
                                        uint32_t resolve_scope,
                                        SymbolContextList &sc_list);
  virtual uint32_t
  FindGlobalVariables(const ConstString &name,
                      const CompilerDeclContext *parent_decl_ctx, bool append,
                      uint32_t max_matches, VariableList &variables);
  virtual uint32_t FindGlobalVariables(const RegularExpression &regex,
                                       bool append, uint32_t max_matches,
                                       VariableList &variables);
  virtual uint32_t FindFunctions(const ConstString &name,
                                 const CompilerDeclContext *parent_decl_ctx,
                                 uint32_t name_type_mask, bool include_inlines,
                                 bool append, SymbolContextList &sc_list);
  virtual uint32_t FindFunctions(const RegularExpression &regex,
                                 bool include_inlines, bool append,
                                 SymbolContextList &sc_list);
  virtual uint32_t
  FindTypes(const SymbolContext &sc, const ConstString &name,
            const CompilerDeclContext *parent_decl_ctx, bool append,
            uint32_t max_matches,
            llvm::DenseSet<lldb_private::SymbolFile *> &searched_symbol_files,
            TypeMap &types);
  virtual size_t FindTypes(const std::vector<CompilerContext> &context,
                           bool append, TypeMap &types);

  virtual void
  GetMangledNamesForFunction(const std::string &scope_qualified_name,
                             std::vector<ConstString> &mangled_names);
  //  virtual uint32_t        FindTypes (const SymbolContext& sc, const
  //  RegularExpression& regex, bool append, uint32_t max_matches, TypeList&
  //  types) = 0;
  virtual TypeList *GetTypeList();
  virtual size_t GetTypes(lldb_private::SymbolContextScope *sc_scope,
                          uint32_t type_mask,
                          lldb_private::TypeList &type_list) = 0;

  virtual lldb_private::TypeSystem *
  GetTypeSystemForLanguage(lldb::LanguageType language);

  virtual CompilerDeclContext
  FindNamespace(const SymbolContext &sc, const ConstString &name,
                const CompilerDeclContext *parent_decl_ctx) {
    return CompilerDeclContext();
  }

  ObjectFile *GetObjectFile() { return m_obj_file; }
  const ObjectFile *GetObjectFile() const { return m_obj_file; }

  //------------------------------------------------------------------
  /// Notify the SymbolFile that the file addresses in the Sections
  /// for this module have been changed.
  //------------------------------------------------------------------
  virtual void SectionFileAddressesChanged() {}

  virtual bool GetCompileOption(const char *option, std::string &value,
                                CompileUnit *cu = nullptr) {
    value.clear();
    return false;
  }

  virtual int GetCompileOptions(const char *option,
                                std::vector<std::string> &values,
                                CompileUnit *cu = nullptr) {
    values.clear();
    return false;
  }

  virtual void GetLoadedModules(lldb::LanguageType language,
                                FileSpecList &modules) {}

  //------------------------------------------------------------------
  // Some symbol files might know if we should always check for inline
  // source file and line entries. This virtual function lets
  // SymbolFile subclasses control that, but a default implementation
  // is supplied.
  //------------------------------------------------------------------
  virtual bool ForceInlineSourceFileCheck();

  //------------------------------------------------------------------
  // Symbol files can store AST data for any language that wants to
  // store the native AST format supported by the current compiler.
  // This information is often only usable by a compiler that is in
  // sync with the compiler sources that were used to build LLDB so
  // any data should be versioned appropriately so the compiler can
  // try to load the data and know if the data will be able to be
  // used.
  //------------------------------------------------------------------
  virtual lldb::DataBufferSP GetASTData(lldb::LanguageType language);

  // Used for the REPL to limit source file ranges that are valid within "file".
  // Since
  // breakpoint setting call fall through, we need to stop the fall through from
  // happening
  virtual bool SetLimitSourceFileRange(const FileSpec &file,
                                       uint32_t first_line, uint32_t last_line);

  virtual bool SymbolContextShouldBeExcluded(const SymbolContext &sc,
                                             uint32_t actual_line);

protected:
  class SourceRange {
  public:
    SourceRange(const FileSpec &f, uint32_t first, uint32_t last)
        : file(f), first_line(first), last_line(last) {}
    FileSpec file;
    uint32_t first_line;
    uint32_t last_line;
  };

  ObjectFile *m_obj_file; // The object file that symbols can be extracted from.
  uint32_t m_abilities;
  bool m_calculated_abilities;
  std::vector<SourceRange> m_limit_source_ranges;

private:
  DISALLOW_COPY_AND_ASSIGN(SymbolFile);
};

} // namespace lldb_private

#endif // liblldb_SymbolFile_h_
