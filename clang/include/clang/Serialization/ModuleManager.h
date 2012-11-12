//===--- ModuleManager.cpp - Module Manager ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ModuleManager class, which manages a set of loaded
//  modules for the ASTReader.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SERIALIZATION_MODULE_MANAGER_H
#define LLVM_CLANG_SERIALIZATION_MODULE_MANAGER_H

#include "clang/Serialization/Module.h"
#include "clang/Basic/FileManager.h"
#include "llvm/ADT/DenseMap.h"

namespace clang { 

namespace serialization {
  
/// \brief Manages the set of modules loaded by an AST reader.
class ModuleManager {
  /// \brief The chain of AST files. The first entry is the one named by the
  /// user, the last one is the one that doesn't depend on anything further.
  llvm::SmallVector<ModuleFile*, 2> Chain;
  
  /// \brief All loaded modules, indexed by name.
  llvm::DenseMap<const FileEntry *, ModuleFile *> Modules;
  
  /// \brief FileManager that handles translating between filenames and
  /// FileEntry *.
  FileManager &FileMgr;
  
  /// \brief A lookup of in-memory (virtual file) buffers
  llvm::DenseMap<const FileEntry *, llvm::MemoryBuffer *> InMemoryBuffers;
  
public:
  typedef SmallVector<ModuleFile*, 2>::iterator ModuleIterator;
  typedef SmallVector<ModuleFile*, 2>::const_iterator ModuleConstIterator;
  typedef SmallVector<ModuleFile*, 2>::reverse_iterator ModuleReverseIterator;
  typedef std::pair<uint32_t, StringRef> ModuleOffset;
  
  explicit ModuleManager(FileManager &FileMgr);
  ~ModuleManager();
  
  /// \brief Forward iterator to traverse all loaded modules.  This is reverse
  /// source-order.
  ModuleIterator begin() { return Chain.begin(); }
  /// \brief Forward iterator end-point to traverse all loaded modules
  ModuleIterator end() { return Chain.end(); }
  
  /// \brief Const forward iterator to traverse all loaded modules.  This is 
  /// in reverse source-order.
  ModuleConstIterator begin() const { return Chain.begin(); }
  /// \brief Const forward iterator end-point to traverse all loaded modules
  ModuleConstIterator end() const { return Chain.end(); }
  
  /// \brief Reverse iterator to traverse all loaded modules.  This is in 
  /// source order.
  ModuleReverseIterator rbegin() { return Chain.rbegin(); }
  /// \brief Reverse iterator end-point to traverse all loaded modules.
  ModuleReverseIterator rend() { return Chain.rend(); }
  
  /// \brief Returns the primary module associated with the manager, that is,
  /// the first module loaded
  ModuleFile &getPrimaryModule() { return *Chain[0]; }
  
  /// \brief Returns the primary module associated with the manager, that is,
  /// the first module loaded.
  ModuleFile &getPrimaryModule() const { return *Chain[0]; }
  
  /// \brief Returns the module associated with the given index
  ModuleFile &operator[](unsigned Index) const { return *Chain[Index]; }
  
  /// \brief Returns the module associated with the given name
  ModuleFile *lookup(StringRef Name);
  
  /// \brief Returns the in-memory (virtual file) buffer with the given name
  llvm::MemoryBuffer *lookupBuffer(StringRef Name);
  
  /// \brief Number of modules loaded
  unsigned size() const { return Chain.size(); }
  /// \brief Attempts to create a new module and add it to the list of known
  /// modules.
  ///
  /// \param FileName The file name of the module to be loaded.
  ///
  /// \param Type The kind of module being loaded.
  ///
  /// \param ImportedBy The module that is importing this module, or NULL if
  /// this module is imported directly by the user.
  ///
  /// \param Generation The generation in which this module was loaded.
  ///
  /// \param ErrorStr Will be set to a non-empty string if any errors occurred
  /// while trying to load the module.
  ///
  /// \return A pointer to the module that corresponds to this file name,
  /// and a boolean indicating whether the module was newly added.
  std::pair<ModuleFile *, bool> 
  addModule(StringRef FileName, ModuleKind Type, ModuleFile *ImportedBy,
            unsigned Generation, std::string &ErrorStr);

  /// \brief Remove the given set of modules.
  void removeModules(ModuleIterator first, ModuleIterator last);

  /// \brief Add an in-memory buffer the list of known buffers
  void addInMemoryBuffer(StringRef FileName, llvm::MemoryBuffer *Buffer);
  
  /// \brief Visit each of the modules.
  ///
  /// This routine visits each of the modules, starting with the
  /// "root" modules that no other loaded modules depend on, and
  /// proceeding to the leaf modules, visiting each module only once
  /// during the traversal.
  ///
  /// This traversal is intended to support various "lookup"
  /// operations that can find data in any of the loaded modules.
  ///
  /// \param Visitor A visitor function that will be invoked with each
  /// module and the given user data pointer. The return value must be
  /// convertible to bool; when false, the visitation continues to
  /// modules that the current module depends on. When true, the
  /// visitation skips any modules that the current module depends on.
  ///
  /// \param UserData User data associated with the visitor object, which
  /// will be passed along to the visitor.
  void visit(bool (*Visitor)(ModuleFile &M, void *UserData), void *UserData);
  
  /// \brief Visit each of the modules with a depth-first traversal.
  ///
  /// This routine visits each of the modules known to the module
  /// manager using a depth-first search, starting with the first
  /// loaded module. The traversal invokes the callback both before
  /// traversing the children (preorder traversal) and after
  /// traversing the children (postorder traversal).
  ///
  /// \param Visitor A visitor function that will be invoked with each
  /// module and given a \c Preorder flag that indicates whether we're
  /// visiting the module before or after visiting its children.  The
  /// visitor may return true at any time to abort the depth-first
  /// visitation.
  ///
  /// \param UserData User data ssociated with the visitor object,
  /// which will be passed along to the user.
  void visitDepthFirst(bool (*Visitor)(ModuleFile &M, bool Preorder, 
                                       void *UserData), 
                       void *UserData);
  
  /// \brief View the graphviz representation of the module graph.
  void viewGraph();
};

} } // end namespace clang::serialization

#endif
