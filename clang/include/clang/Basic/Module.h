//===--- Module.h - Describe a module ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the clang::Module class, which describes a module in the
/// source code.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_BASIC_MODULE_H
#define LLVM_CLANG_BASIC_MODULE_H

#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SetVector.h"
#include <string>
#include <utility>
#include <vector>

namespace llvm {
  class raw_ostream;
}

namespace clang {
  
class DirectoryEntry;
class FileEntry;
class LangOptions;
class TargetInfo;
  
/// \brief Describes the name of a module.
typedef llvm::SmallVector<std::pair<std::string, SourceLocation>, 2>
  ModuleId;
  
/// \brief Describes a module or submodule.
class Module {
public:
  /// \brief The name of this module.
  std::string Name;
  
  /// \brief The location of the module definition.
  SourceLocation DefinitionLoc;
  
  /// \brief The parent of this module. This will be NULL for the top-level
  /// module.
  Module *Parent;
  
  /// \brief The umbrella header or directory.
  llvm::PointerUnion<const DirectoryEntry *, const FileEntry *> Umbrella;
  
private:
  /// \brief The submodules of this module, indexed by name.
  std::vector<Module *> SubModules;
  
  /// \brief A mapping from the submodule name to the index into the 
  /// \c SubModules vector at which that submodule resides.
  llvm::StringMap<unsigned> SubModuleIndex;

  /// \brief The AST file if this is a top-level module which has a
  /// corresponding serialized AST file, or null otherwise.
  const FileEntry *ASTFile;
  
public:
  /// \brief The headers that are part of this module.
  llvm::SmallVector<const FileEntry *, 2> Headers;

  /// \brief The headers that are explicitly excluded from this module.
  llvm::SmallVector<const FileEntry *, 2> ExcludedHeaders;

  /// \brief The top-level headers associated with this module.
  llvm::SmallSetVector<const FileEntry *, 2> TopHeaders;

  /// \brief The set of language features required to use this module.
  ///
  /// If any of these features is not present, the \c IsAvailable bit
  /// will be false to indicate that this (sub)module is not
  /// available.
  llvm::SmallVector<std::string, 2> Requires;

  /// \brief Whether this module is available in the current
  /// translation unit.
  unsigned IsAvailable : 1;

  /// \brief Whether this module was loaded from a module file.
  unsigned IsFromModuleFile : 1;
  
  /// \brief Whether this is a framework module.
  unsigned IsFramework : 1;
  
  /// \brief Whether this is an explicit submodule.
  unsigned IsExplicit : 1;
  
  /// \brief Whether this is a "system" module (which assumes that all
  /// headers in it are system headers).
  unsigned IsSystem : 1;
  
  /// \brief Whether we should infer submodules for this module based on 
  /// the headers.
  ///
  /// Submodules can only be inferred for modules with an umbrella header.
  unsigned InferSubmodules : 1;
  
  /// \brief Whether, when inferring submodules, the inferred submodules
  /// should be explicit.
  unsigned InferExplicitSubmodules : 1;
  
  /// \brief Whether, when inferring submodules, the inferr submodules should
  /// export all modules they import (e.g., the equivalent of "export *").
  unsigned InferExportWildcard : 1;
  
  /// \brief Describes the visibility of the various names within a
  /// particular module.
  enum NameVisibilityKind {
    /// \brief All of the names in this module are hidden.
    ///
    Hidden,
    /// \brief Only the macro names in this module are visible.
    MacrosVisible,
    /// \brief All of the names in this module are visible.
    AllVisible
  };  
  
  ///\ brief The visibility of names within this particular module.
  NameVisibilityKind NameVisibility;

  /// \brief The location of the inferred submodule.
  SourceLocation InferredSubmoduleLoc;

  /// \brief The set of modules imported by this module, and on which this
  /// module depends.
  llvm::SmallVector<Module *, 2> Imports;
  
  /// \brief Describes an exported module.
  ///
  /// The pointer is the module being re-exported, while the bit will be true
  /// to indicate that this is a wildcard export.
  typedef llvm::PointerIntPair<Module *, 1, bool> ExportDecl;
  
  /// \brief The set of export declarations.
  llvm::SmallVector<ExportDecl, 2> Exports;
  
  /// \brief Describes an exported module that has not yet been resolved
  /// (perhaps because the module it refers to has not yet been loaded).
  struct UnresolvedExportDecl {
    /// \brief The location of the 'export' keyword in the module map file.
    SourceLocation ExportLoc;
    
    /// \brief The name of the module.
    ModuleId Id;
    
    /// \brief Whether this export declaration ends in a wildcard, indicating
    /// that all of its submodules should be exported (rather than the named
    /// module itself).
    bool Wildcard;
  };
  
  /// \brief The set of export declarations that have yet to be resolved.
  llvm::SmallVector<UnresolvedExportDecl, 2> UnresolvedExports;
  
  /// \brief Construct a top-level module.
  explicit Module(StringRef Name, SourceLocation DefinitionLoc,
                  bool IsFramework)
    : Name(Name), DefinitionLoc(DefinitionLoc), Parent(0),Umbrella(),ASTFile(0),
      IsAvailable(true), IsFromModuleFile(false), IsFramework(IsFramework), 
      IsExplicit(false), IsSystem(false),
      InferSubmodules(false), InferExplicitSubmodules(false),
      InferExportWildcard(false), NameVisibility(Hidden) { }
  
  /// \brief Construct a new module or submodule.
  Module(StringRef Name, SourceLocation DefinitionLoc, Module *Parent, 
         bool IsFramework, bool IsExplicit);
  
  ~Module();
  
  /// \brief Determine whether this module is available for use within the
  /// current translation unit.
  bool isAvailable() const { return IsAvailable; }

  /// \brief Determine whether this module is available for use within the
  /// current translation unit.
  ///
  /// \param LangOpts The language options used for the current
  /// translation unit.
  ///
  /// \param Target The target options used for the current translation unit.
  ///
  /// \param Feature If this module is unavailable, this parameter
  /// will be set to one of the features that is required for use of
  /// this module (but is not available).
  bool isAvailable(const LangOptions &LangOpts, 
                   const TargetInfo &Target,
                   StringRef &Feature) const;

  /// \brief Determine whether this module is a submodule.
  bool isSubModule() const { return Parent != 0; }
  
  /// \brief Determine whether this module is a submodule of the given other
  /// module.
  bool isSubModuleOf(Module *Other) const;
  
  /// \brief Determine whether this module is a part of a framework,
  /// either because it is a framework module or because it is a submodule
  /// of a framework module.
  bool isPartOfFramework() const {
    for (const Module *Mod = this; Mod; Mod = Mod->Parent) 
      if (Mod->IsFramework)
        return true;
    
    return false;
  }
  
  /// \brief Retrieve the full name of this module, including the path from
  /// its top-level module.
  std::string getFullModuleName() const;

  /// \brief Retrieve the top-level module for this (sub)module, which may
  /// be this module.
  Module *getTopLevelModule() {
    return const_cast<Module *>(
             const_cast<const Module *>(this)->getTopLevelModule());
  }

  /// \brief Retrieve the top-level module for this (sub)module, which may
  /// be this module.
  const Module *getTopLevelModule() const;
  
  /// \brief Retrieve the name of the top-level module.
  ///
  StringRef getTopLevelModuleName() const {
    return getTopLevelModule()->Name;
  }

  /// \brief The serialized AST file for this module, if one was created.
  const FileEntry *getASTFile() const {
    return getTopLevelModule()->ASTFile;
  }

  /// \brief Set the serialized AST file for the top-level module of this module.
  void setASTFile(const FileEntry *File) {
    assert((getASTFile() == 0 || getASTFile() == File) && "file path changed");
    getTopLevelModule()->ASTFile = File;
  }

  /// \brief Retrieve the directory for which this module serves as the
  /// umbrella.
  const DirectoryEntry *getUmbrellaDir() const;

  /// \brief Retrieve the header that serves as the umbrella header for this
  /// module.
  const FileEntry *getUmbrellaHeader() const {
    return Umbrella.dyn_cast<const FileEntry *>();
  }

  /// \brief Determine whether this module has an umbrella directory that is
  /// not based on an umbrella header.
  bool hasUmbrellaDir() const {
    return Umbrella && Umbrella.is<const DirectoryEntry *>();
  }

  /// \brief Add the given feature requirement to the list of features
  /// required by this module.
  ///
  /// \param Feature The feature that is required by this module (and
  /// its submodules).
  ///
  /// \param LangOpts The set of language options that will be used to
  /// evaluate the availability of this feature.
  ///
  /// \param Target The target options that will be used to evaluate the
  /// availability of this feature.
  void addRequirement(StringRef Feature, const LangOptions &LangOpts,
                      const TargetInfo &Target);

  /// \brief Find the submodule with the given name.
  ///
  /// \returns The submodule if found, or NULL otherwise.
  Module *findSubmodule(StringRef Name) const;
  
  typedef std::vector<Module *>::iterator submodule_iterator;
  typedef std::vector<Module *>::const_iterator submodule_const_iterator;
  
  submodule_iterator submodule_begin() { return SubModules.begin(); }
  submodule_const_iterator submodule_begin() const {return SubModules.begin();}
  submodule_iterator submodule_end()   { return SubModules.end(); }
  submodule_const_iterator submodule_end() const { return SubModules.end(); }
  
  static StringRef getModuleInputBufferName() {
    return "<module-includes>";
  }

  /// \brief Print the module map for this module to the given stream. 
  ///
  void print(llvm::raw_ostream &OS, unsigned Indent = 0) const;
  
  /// \brief Dump the contents of this module to the given output stream.
  void dump() const;
};

} // end namespace clang


#endif // LLVM_CLANG_BASIC_MODULE_H
