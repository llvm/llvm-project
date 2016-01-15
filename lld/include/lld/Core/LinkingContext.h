//===- lld/Core/LinkingContext.h - Linker Target Info Interface -----------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_CORE_LINKING_CONTEXT_H
#define LLD_CORE_LINKING_CONTEXT_H

#include "lld/Core/Error.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Node.h"
#include "lld/Core/Parallel.h"
#include "lld/Core/Reference.h"
#include "lld/Core/range.h"
#include "lld/Core/Reader.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace lld {
class PassManager;
class File;
class Writer;
class Node;
class SharedLibraryFile;

/// \brief The LinkingContext class encapsulates "what and how" to link.
///
/// The base class LinkingContext contains the options needed by core linking.
/// Subclasses of LinkingContext have additional options needed by specific
/// Writers. For example, ELFLinkingContext has methods that supplies
/// options to the ELF Writer and ELF Passes.
class LinkingContext {
public:
  /// \brief The types of output file that the linker creates.
  enum class OutputFileType : uint8_t {
    Default, // The default output type for this target
    YAML,    // The output type is set to YAML
  };

  virtual ~LinkingContext();

  /// \name Methods needed by core linking
  /// @{

  /// Name of symbol linker should use as "entry point" to program,
  /// usually "main" or "start".
  virtual StringRef entrySymbolName() const { return _entrySymbolName; }

  /// Whether core linking should remove Atoms not reachable by following
  /// References from the entry point Atom or from all global scope Atoms
  /// if globalsAreDeadStripRoots() is true.
  bool deadStrip() const { return _deadStrip; }

  /// Only used if deadStrip() returns true.  Means all global scope Atoms
  /// should be marked live (along with all Atoms they reference).  Usually
  /// this method returns false for main executables, but true for dynamic
  /// shared libraries.
  bool globalsAreDeadStripRoots() const { return _globalsAreDeadStripRoots; }

  /// Only used if deadStrip() returns true.  This method returns the names
  /// of DefinedAtoms that should be marked live (along with all Atoms they
  /// reference). Only Atoms with scope scopeLinkageUnit or scopeGlobal can
  /// be kept live using this method.
  const std::vector<StringRef> &deadStripRoots() const {
    return _deadStripRoots;
  }

  /// Add the given symbol name to the dead strip root set. Only used if
  /// deadStrip() returns true.
  void addDeadStripRoot(StringRef symbolName) {
    assert(!symbolName.empty() && "Empty symbol cannot be a dead strip root");
    _deadStripRoots.push_back(symbolName);
  }

  /// Archive files (aka static libraries) are normally lazily loaded.  That is,
  /// object files within an archive are only loaded and linked in, if the
  /// object file contains a DefinedAtom which will replace an existing
  /// UndefinedAtom.  If this method returns true, core linking will also look
  /// for archive members to replace existing tentative definitions in addition
  /// to replacing undefines. Note: a "tentative definition" (also called a
  /// "common" symbols) is a C (but not C++) concept. They are modeled in lld
  /// as a DefinedAtom with merge() of mergeAsTentative.
  bool searchArchivesToOverrideTentativeDefinitions() const {
    return _searchArchivesToOverrideTentativeDefinitions;
  }

  /// Normally core linking will turn a tentative definition into a real
  /// definition if not replaced by a real DefinedAtom from some object file.
  /// If this method returns true, core linking will search all supplied
  /// dynamic shared libraries for symbol names that match remaining tentative
  /// definitions.  If any are found, the corresponding tentative definition
  /// atom is replaced with SharedLibraryAtom.
  bool searchSharedLibrariesToOverrideTentativeDefinitions() const {
    return _searchSharedLibrariesToOverrideTentativeDefinitions;
  }

  /// Normally, every UndefinedAtom must be replaced by a DefinedAtom or a
  /// SharedLibraryAtom for the link to be successful.  This method controls
  /// whether core linking prints out a list of remaining UndefinedAtoms.
  ///
  /// \todo This should be a method core linking calls with a list of the
  /// UndefinedAtoms so that different drivers can format the error message
  /// as needed.
  bool printRemainingUndefines() const { return _printRemainingUndefines; }

  /// Normally, every UndefinedAtom must be replaced by a DefinedAtom or a
  /// SharedLibraryAtom for the link to be successful.  This method controls
  /// whether core linking considers remaining undefines to be an error.
  bool allowRemainingUndefines() const { return _allowRemainingUndefines; }

  /// In the lld model, a SharedLibraryAtom is a proxy atom for something
  /// that will be found in a dynamic shared library when the program runs.
  /// A SharedLibraryAtom optionally contains the name of the shared library
  /// in which to find the symbol name at runtime.  Core linking may merge
  /// two SharedLibraryAtom with the same name.  If this method returns true,
  /// when merging core linking will also verify that they both have the same
  /// loadName() and if not print a warning.
  ///
  /// \todo This should be a method core linking calls so that drivers can
  /// format the warning as needed.
  bool warnIfCoalesableAtomsHaveDifferentLoadName() const {
    return _warnIfCoalesableAtomsHaveDifferentLoadName;
  }

  /// In C/C++ you can mark a function's prototype with
  /// __attribute__((weak_import)) or __attribute__((weak)) to say the function
  /// may not be available at runtime and/or build time and in which case its
  /// address will evaluate to NULL. In lld this is modeled using the
  /// UndefinedAtom::canBeNull() method.  During core linking, UndefinedAtom
  /// with the same name are automatically merged.  If this method returns
  /// true, core link also verfies that the canBeNull() value for merged
  /// UndefinedAtoms are the same and warns if not.
  ///
  /// \todo This should be a method core linking calls so that drivers can
  /// format the warning as needed.
  bool warnIfCoalesableAtomsHaveDifferentCanBeNull() const {
    return _warnIfCoalesableAtomsHaveDifferentCanBeNull;
  }

  /// Normally, every UndefinedAtom must be replaced by a DefinedAtom or a
  /// SharedLibraryAtom for the link to be successful.  This method controls
  /// whether core linking considers remaining undefines from the shared library
  /// to be an error.
  bool allowShlibUndefines() const { return _allowShlibUndefines; }

  /// If true, core linking will write the path to each input file to stdout
  /// (i.e. llvm::outs()) as it is used.  This is used to implement the -t
  /// linker option.
  ///
  /// \todo This should be a method core linking calls so that drivers can
  /// format the line as needed.
  bool logInputFiles() const { return _logInputFiles; }

  /// Parts of LLVM use global variables which are bound to command line
  /// options (see llvm::cl::Options). This method returns "command line"
  /// options which are used to configure LLVM's command line settings.
  /// For instance the -debug-only XXX option can be used to dynamically
  /// trace different parts of LLVM and lld.
  const std::vector<const char *> &llvmOptions() const { return _llvmOptions; }

  /// \name Methods used by Drivers to configure TargetInfo
  /// @{
  void setOutputPath(StringRef str) { _outputPath = str; }

  // Set the entry symbol name. You may also need to call addDeadStripRoot() for
  // the symbol if your platform supports dead-stripping, so that the symbol
  // will not be removed from the output.
  void setEntrySymbolName(StringRef name) {
    _entrySymbolName = name;
  }

  void setDeadStripping(bool enable) { _deadStrip = enable; }
  void setAllowDuplicates(bool enable) { _allowDuplicates = enable; }
  void setGlobalsAreDeadStripRoots(bool v) { _globalsAreDeadStripRoots = v; }
  void setSearchArchivesToOverrideTentativeDefinitions(bool search) {
    _searchArchivesToOverrideTentativeDefinitions = search;
  }
  void setSearchSharedLibrariesToOverrideTentativeDefinitions(bool search) {
    _searchSharedLibrariesToOverrideTentativeDefinitions = search;
  }
  void setWarnIfCoalesableAtomsHaveDifferentCanBeNull(bool warn) {
    _warnIfCoalesableAtomsHaveDifferentCanBeNull = warn;
  }
  void setWarnIfCoalesableAtomsHaveDifferentLoadName(bool warn) {
    _warnIfCoalesableAtomsHaveDifferentLoadName = warn;
  }
  void setPrintRemainingUndefines(bool print) {
    _printRemainingUndefines = print;
  }
  void setAllowRemainingUndefines(bool allow) {
    _allowRemainingUndefines = allow;
  }
  void setAllowShlibUndefines(bool allow) { _allowShlibUndefines = allow; }
  void setLogInputFiles(bool log) { _logInputFiles = log; }

  // Returns true if multiple definitions should not be treated as a
  // fatal error.
  bool getAllowDuplicates() const { return _allowDuplicates; }

  void appendLLVMOption(const char *opt) { _llvmOptions.push_back(opt); }

  void addAlias(StringRef from, StringRef to) { _aliasSymbols[from] = to; }
  const std::map<std::string, std::string> &getAliases() const {
    return _aliasSymbols;
  }

  std::vector<std::unique_ptr<Node>> &getNodes() { return _nodes; }
  const std::vector<std::unique_ptr<Node>> &getNodes() const { return _nodes; }

  /// Notify the LinkingContext when the symbol table found a name collision.
  /// The useNew parameter specifies which the symbol table plans to keep,
  /// but that can be changed by the LinkingContext.  This is also an
  /// opportunity for flavor specific processing.
  virtual void notifySymbolTableCoalesce(const Atom *existingAtom,
                                         const Atom *newAtom, bool &useNew) {}

  /// This method adds undefined symbols specified by the -u option to the to
  /// the list of undefined symbols known to the linker. This option essentially
  /// forces an undefined symbol to be created. You may also need to call
  /// addDeadStripRoot() for the symbol if your platform supports dead
  /// stripping, so that the symbol will not be removed from the output.
  void addInitialUndefinedSymbol(StringRef symbolName) {
    _initialUndefinedSymbols.push_back(symbolName);
  }

  /// Iterators for symbols that appear on the command line.
  typedef std::vector<StringRef> StringRefVector;
  typedef StringRefVector::iterator StringRefVectorIter;
  typedef StringRefVector::const_iterator StringRefVectorConstIter;

  /// Create linker internal files containing atoms for the linker to include
  /// during link. Flavors can override this function in their LinkingContext
  /// to add more internal files. These internal files are positioned before
  /// the actual input files.
  virtual void createInternalFiles(std::vector<std::unique_ptr<File> > &) const;

  /// Return the list of undefined symbols that are specified in the
  /// linker command line, using the -u option.
  range<const StringRef *> initialUndefinedSymbols() const {
    return _initialUndefinedSymbols;
  }

  /// After all set* methods are called, the Driver calls this method
  /// to validate that there are no missing options or invalid combinations
  /// of options.  If there is a problem, a description of the problem
  /// is written to the supplied stream.
  ///
  /// \returns true if there is an error with the current settings.
  bool validate(raw_ostream &diagnostics);

  /// Formats symbol name for use in error messages.
  virtual std::string demangle(StringRef symbolName) const {
    return symbolName;
  }

  /// @}
  /// \name Methods used by Driver::link()
  /// @{

  /// Returns the file system path to which the linked output should be written.
  ///
  /// \todo To support in-memory linking, we need an abstraction that allows
  /// the linker to write to an in-memory buffer.
  StringRef outputPath() const { return _outputPath; }

  /// Set the various output file types that the linker would
  /// create
  bool setOutputFileType(StringRef outputFileType) {
    if (outputFileType.equals_lower("yaml")) {
      _outputFileType = OutputFileType::YAML;
      return true;
    }
    return false;
  }

  /// Returns the output file type that that the linker needs to create.
  OutputFileType outputFileType() const { return _outputFileType; }

  /// Accessor for Register object embedded in LinkingContext.
  const Registry &registry() const { return _registry; }
  Registry &registry() { return _registry; }

  /// This method is called by core linking to give the Writer a chance
  /// to add file format specific "files" to set of files to be linked. This is
  /// how file format specific atoms can be added to the link.
  virtual void createImplicitFiles(std::vector<std::unique_ptr<File>> &);

  /// This method is called by core linking to build the list of Passes to be
  /// run on the merged/linked graph of all input files.
  virtual void addPasses(PassManager &pm);

  /// Calls through to the writeFile() method on the specified Writer.
  ///
  /// \param linkedFile This is the merged/linked graph of all input file Atoms.
  virtual std::error_code writeFile(const File &linkedFile) const;

  /// Return the next ordinal and Increment it.
  virtual uint64_t getNextOrdinalAndIncrement() const { return _nextOrdinal++; }

  // This function is called just before the Resolver kicks in.
  // Derived classes may use it to change the list of input files.
  virtual void finalizeInputFiles() {}

  /// Callback invoked for each file the Resolver decides we are going to load.
  /// This can be used to update context state based on the file, and emit
  /// errors for any differences between the context state and a loaded file.
  /// For example, we can error if we try to load a file which is a different
  /// arch from that being linked.
  virtual std::error_code handleLoadedFile(File &file) {
    return std::error_code();
  }

  TaskGroup &getTaskGroup() { return _taskGroup; }

  /// @}
protected:
  LinkingContext(); // Must be subclassed

  /// Abstract method to lazily instantiate the Writer.
  virtual Writer &writer() const = 0;

  /// Method to create an internal file for the entry symbol
  virtual std::unique_ptr<File> createEntrySymbolFile() const;
  std::unique_ptr<File> createEntrySymbolFile(StringRef filename) const;

  /// Method to create an internal file for an undefined symbol
  virtual std::unique_ptr<File> createUndefinedSymbolFile() const;
  std::unique_ptr<File> createUndefinedSymbolFile(StringRef filename) const;

  /// Method to create an internal file for alias symbols
  std::unique_ptr<File> createAliasSymbolFile() const;

  StringRef _outputPath;
  StringRef _entrySymbolName;
  bool _deadStrip;
  bool _allowDuplicates;
  bool _globalsAreDeadStripRoots;
  bool _searchArchivesToOverrideTentativeDefinitions;
  bool _searchSharedLibrariesToOverrideTentativeDefinitions;
  bool _warnIfCoalesableAtomsHaveDifferentCanBeNull;
  bool _warnIfCoalesableAtomsHaveDifferentLoadName;
  bool _printRemainingUndefines;
  bool _allowRemainingUndefines;
  bool _logInputFiles;
  bool _allowShlibUndefines;
  OutputFileType _outputFileType;
  std::vector<StringRef> _deadStripRoots;
  std::map<std::string, std::string> _aliasSymbols;
  std::vector<const char *> _llvmOptions;
  StringRefVector _initialUndefinedSymbols;
  std::vector<std::unique_ptr<Node>> _nodes;
  mutable llvm::BumpPtrAllocator _allocator;
  mutable uint64_t _nextOrdinal;
  Registry _registry;

private:
  /// Validate the subclass bits. Only called by validate.
  virtual bool validateImpl(raw_ostream &diagnostics) = 0;
  TaskGroup _taskGroup;
};

} // end namespace lld

#endif
