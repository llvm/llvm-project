//===- lld/ReaderWriter/ELFLinkingContext.h -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_ELF_LINKER_CONTEXT_H
#define LLD_READER_WRITER_ELF_LINKER_CONTEXT_H

#include "lld/Core/LinkingContext.h"
#include "lld/Core/Pass.h"
#include "lld/Core/PassManager.h"
#include "lld/Core/STDExtras.h"
#include "lld/Core/range.h"
#include "lld/Core/Reader.h"
#include "lld/Core/Writer.h"
#include "lld/ReaderWriter/LinkerScript.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"
#include <map>
#include <memory>
#include <set>

namespace llvm {
class FileOutputBuffer;
}

namespace lld {
struct AtomLayout;
class File;
class Reference;

namespace elf {
using llvm::object::ELF32LE;
using llvm::object::ELF32BE;
using llvm::object::ELF64LE;
using llvm::object::ELF64BE;

class ELFWriter;

std::unique_ptr<ELFLinkingContext> createAArch64LinkingContext(llvm::Triple);
std::unique_ptr<ELFLinkingContext> createARMLinkingContext(llvm::Triple);
std::unique_ptr<ELFLinkingContext> createExampleLinkingContext(llvm::Triple);
std::unique_ptr<ELFLinkingContext> createHexagonLinkingContext(llvm::Triple);
std::unique_ptr<ELFLinkingContext> createMipsLinkingContext(llvm::Triple);
std::unique_ptr<ELFLinkingContext> createX86LinkingContext(llvm::Triple);
std::unique_ptr<ELFLinkingContext> createX86_64LinkingContext(llvm::Triple);

class TargetRelocationHandler {
public:
  virtual ~TargetRelocationHandler() {}

  virtual std::error_code applyRelocation(ELFWriter &, llvm::FileOutputBuffer &,
                                          const lld::AtomLayout &,
                                          const Reference &) const = 0;
};

} // namespace elf

/// \brief TargetHandler contains all the information responsible to handle a
/// a particular target on ELF. A target might wish to override implementation
/// of creating atoms and how the atoms are written to the output file.
class TargetHandler {
public:
  virtual ~TargetHandler() {}

  /// Determines how relocations need to be applied.
  virtual const elf::TargetRelocationHandler &getRelocationHandler() const = 0;

  /// Returns a reader for object files.
  virtual std::unique_ptr<Reader> getObjReader() = 0;

  /// Returns a reader for .so files.
  virtual std::unique_ptr<Reader> getDSOReader() = 0;

  /// Returns a writer to write an ELF file.
  virtual std::unique_ptr<Writer> getWriter() = 0;
};

class ELFLinkingContext : public LinkingContext {
public:
  /// \brief The type of ELF executable that the linker
  /// creates.
  enum class OutputMagic : uint8_t {
    // The default mode, no specific magic set
    DEFAULT,
    // Disallow shared libraries and don't align sections
    // PageAlign Data, Mark Text Segment/Data segment RW
    NMAGIC,
    // Disallow shared libraries and don't align sections,
    // Mark Text Segment/Data segment RW
    OMAGIC,
  };

  /// \brief ELF DT_FLAGS.
  enum DTFlag : uint32_t {
    DT_NOW = 1 << 1,
    DT_ORIGIN = 1 << 2,
  };

  llvm::Triple getTriple() const { return _triple; }

  uint64_t getPageSize() const { return _maxPageSize; }
  void setMaxPageSize(uint64_t v) { _maxPageSize = v; }

  OutputMagic getOutputMagic() const { return _outputMagic; }
  uint16_t getOutputELFType() const { return _outputELFType; }
  uint16_t getOutputMachine() const;
  bool mergeCommonStrings() const { return _mergeCommonStrings; }
  virtual int getMachineType() const = 0;
  virtual uint64_t getBaseAddress() const { return _baseAddress; }
  virtual void setBaseAddress(uint64_t address) { _baseAddress = address; }

  void notifySymbolTableCoalesce(const Atom *existingAtom, const Atom *newAtom,
                                 bool &useNew) override;

  /// This controls if undefined atoms need to be created for undefines that are
  /// present in a SharedLibrary. If this option is set, undefined atoms are
  /// created for every undefined symbol that are present in the dynamic table
  /// in the shared library
  bool useShlibUndefines() const { return _useShlibUndefines; }

  /// \brief Returns true if a given relocation should be added to the
  /// dynamic relocation table.
  ///
  /// This table is evaluated at loadtime by the dynamic loader and is
  /// referenced by the DT_RELA{,ENT,SZ} entries in the dynamic table.
  /// Relocations that return true will be added to the dynamic relocation
  /// table.
  virtual bool isDynamicRelocation(const Reference &) const { return false; }

  /// \brief Returns true if a given reference is a copy relocation.
  ///
  /// If this is a copy relocation, its target must be an ObjectAtom. We must
  /// include in DT_NEEDED the name of the library where this object came from.
  virtual bool isCopyRelocation(const Reference &) const { return false; }

  bool validateImpl(raw_ostream &diagnostics) override;

  /// \brief Returns true if the linker allows dynamic libraries to be
  /// linked with.
  ///
  /// This is true when the output mode of the executable is set to be
  /// having NMAGIC/OMAGIC
  bool allowLinkWithDynamicLibraries() const {
    if (_outputMagic == OutputMagic::NMAGIC ||
        _outputMagic == OutputMagic::OMAGIC || _noAllowDynamicLibraries)
      return false;
    return true;
  }

  /// \brief Use Elf_Rela format to output relocation tables.
  virtual bool isRelaOutputFormat() const { return true; }

  /// \brief Returns true if a given relocation should be added to PLT.
  ///
  /// This table holds all of the relocations used for delayed symbol binding.
  /// It will be evaluated at load time if LD_BIND_NOW is set. It is referenced
  /// by the DT_{JMPREL,PLTRELSZ} entries in the dynamic table.
  /// Relocations that return true will be added to the dynamic plt relocation
  /// table.
  virtual bool isPLTRelocation(const Reference &) const { return false; }

  /// \brief The path to the dynamic interpreter
  virtual StringRef getDefaultInterpreter() const {
    return "/lib64/ld-linux-x86-64.so.2";
  }

  /// \brief The dynamic linker path set by the --dynamic-linker option
  StringRef getInterpreter() const {
    if (_dynamicLinkerPath.hasValue())
      return _dynamicLinkerPath.getValue();
    return getDefaultInterpreter();
  }

  /// \brief Returns true if the output have dynamic sections.
  bool isDynamic() const;

  /// \brief Returns true if we are creating a shared library.
  bool isDynamicLibrary() const { return _outputELFType == llvm::ELF::ET_DYN; }

  /// \brief Returns true if a given relocation is a relative relocation.
  virtual bool isRelativeReloc(const Reference &r) const;

  TargetHandler &getTargetHandler() const {
    assert(_targetHandler && "Got null TargetHandler!");
    return *_targetHandler;
  }

  virtual void registerRelocationNames(Registry &) = 0;

  void addPasses(PassManager &pm) override;

  void setTriple(llvm::Triple trip) { _triple = trip; }
  void setNoInhibitExec(bool v) { _noInhibitExec = v; }
  void setExportDynamic(bool v) { _exportDynamic = v; }
  void setIsStaticExecutable(bool v) { _isStaticExecutable = v; }
  void setMergeCommonStrings(bool v) { _mergeCommonStrings = v; }
  void setUseShlibUndefines(bool use) { _useShlibUndefines = use; }
  void setOutputELFType(uint32_t type) { _outputELFType = type; }

  bool shouldExportDynamic() const { return _exportDynamic; }

  void createInternalFiles(std::vector<std::unique_ptr<File>> &) const override;

  void finalizeInputFiles() override;

  /// \brief Set the dynamic linker path
  void setInterpreter(StringRef s) { _dynamicLinkerPath = s; }

  /// \brief Set NMAGIC output kind when the linker specifies --nmagic
  /// or -n in the command line
  /// Set OMAGIC output kind when the linker specifies --omagic
  /// or -N in the command line
  void setOutputMagic(OutputMagic magic) { _outputMagic = magic; }

  /// \brief Disallow dynamic libraries during linking
  void setNoAllowDynamicLibraries() { _noAllowDynamicLibraries = true; }

  /// Searches directories for a match on the input File
  ErrorOr<StringRef> searchLibrary(StringRef libName) const;

  /// \brief Searches directories for a match on the input file.
  /// If \p fileName is an absolute path and \p isSysRooted is true, check
  /// the file under sysroot directory. If \p fileName is a relative path
  /// and is not in the current directory, search the file through library
  /// search directories.
  ErrorOr<StringRef> searchFile(StringRef fileName, bool isSysRooted) const;

  /// Get the entry symbol name
  StringRef entrySymbolName() const override;

  /// \brief Set new initializer function
  void setInitFunction(StringRef name) { _initFunction = name; }

  /// \brief Return an initializer function name.
  /// Either default "_init" or configured by the -init command line option.
  StringRef initFunction() const { return _initFunction; }

  /// \brief Set new finalizer function
  void setFiniFunction(StringRef name) { _finiFunction = name; }

  /// \brief Return a finalizer function name.
  /// Either default "_fini" or configured by the -fini command line option.
  StringRef finiFunction() const { return _finiFunction; }

  /// Add an absolute symbol. Used for --defsym.
  void addInitialAbsoluteSymbol(StringRef name, uint64_t addr) {
    _absoluteSymbols[name] = addr;
  }

  StringRef sharedObjectName() const { return _soname; }
  void setSharedObjectName(StringRef soname) { _soname = soname; }

  StringRef getSysroot() const { return _sysrootPath; }
  void setSysroot(StringRef path) { _sysrootPath = path; }

  void addRpath(StringRef path) { _rpathList.push_back(path); }
  range<const StringRef *> getRpathList() const { return _rpathList; }

  void addRpathLink(StringRef path) { _rpathLinkList.push_back(path); }
  range<const StringRef *> getRpathLinkList() const { return _rpathLinkList; }

  const std::map<std::string, uint64_t> &getAbsoluteSymbols() const {
    return _absoluteSymbols;
  }

  /// \brief Helper function to allocate strings.
  StringRef allocateString(StringRef ref) const {
    char *x = _allocator.Allocate<char>(ref.size() + 1);
    memcpy(x, ref.data(), ref.size());
    x[ref.size()] = '\0';
    return x;
  }

  // add search path to list.
  void addSearchPath(StringRef ref) { _inputSearchPaths.push_back(ref); }

  // Retrieve search path list.
  StringRefVector getSearchPaths() { return _inputSearchPaths; };

  // By default, the linker would merge sections that are read only with
  // segments that have read and execute permissions. When the user specifies a
  // flag --rosegment, a separate segment needs to be created.
  bool mergeRODataToTextSegment() const { return _mergeRODataToTextSegment; }

  void setCreateSeparateROSegment() { _mergeRODataToTextSegment = false; }

  bool isDynamicallyExportedSymbol(StringRef name) const {
    return _dynamicallyExportedSymbols.count(name) != 0;
  }

  /// \brief Demangle symbols.
  std::string demangle(StringRef symbolName) const override;
  bool demangleSymbols() const { return _demangle; }
  void setDemangleSymbols(bool d) { _demangle = d; }

  /// \brief Align segments.
  bool alignSegments() const { return _alignSegments; }
  void setAlignSegments(bool align) { _alignSegments = align; }

  /// \brief Enable new dtags.
  /// If this flag is set lld emits DT_RUNPATH instead of
  /// DT_RPATH. They are functionally equivalent except for
  /// the following two differences:
  /// - DT_RUNPATH is searched after LD_LIBRARY_PATH, while
  /// DT_RPATH is searched before.
  /// - DT_RUNPATH is used only to search for direct dependencies
  /// of the object it's contained in, while DT_RPATH is used
  /// for indirect dependencies as well.
  bool getEnableNewDtags() const { return _enableNewDtags; }
  void setEnableNewDtags(bool e) { _enableNewDtags = e; }

  /// \brief Discard local symbols.
  bool discardLocals() const { return _discardLocals; }
  void setDiscardLocals(bool d) { _discardLocals = d; }

  /// \brief Discard temprorary local symbols.
  bool discardTempLocals() const { return _discardTempLocals; }
  void setDiscardTempLocals(bool d) { _discardTempLocals = d; }

  /// \brief Strip symbols.
  bool stripSymbols() const { return _stripSymbols; }
  void setStripSymbols(bool strip) { _stripSymbols = strip; }

  /// \brief Collect statistics.
  bool collectStats() const { return _collectStats; }
  void setCollectStats(bool s) { _collectStats = s; }

  // --wrap option.
  void addWrapForSymbol(StringRef sym) { _wrapCalls.insert(sym); }

  // \brief Set DT_FLAGS flag.
  void setDTFlag(DTFlag f) { _dtFlags |= f; };
  bool getDTFlag(DTFlag f) { return (_dtFlags & f); };

  const llvm::StringSet<> &wrapCalls() const { return _wrapCalls; }

  void setUndefinesResolver(std::unique_ptr<File> resolver);

  script::Sema &linkerScriptSema() { return _linkerScriptSema; }
  const script::Sema &linkerScriptSema() const { return _linkerScriptSema; }

  /// Notify the ELFLinkingContext when the new ELF section is read.
  void notifyInputSectionName(StringRef name);
  /// Encountered C-ident input section names.
  const llvm::StringSet<> &cidentSectionNames() const {
    return _cidentSections;
  }

  // Set R_ARM_TARGET1 relocation behaviour
  bool armTarget1Rel() const { return _armTarget1Rel; }
  void setArmTarget1Rel(bool value) { _armTarget1Rel = value; }

  // Set R_MIPS_EH relocation behaviour.
  bool mipsPcRelEhRel() const { return _mipsPcRelEhRel; }
  void setMipsPcRelEhRel(bool value) { _mipsPcRelEhRel = value; }

protected:
  ELFLinkingContext(llvm::Triple triple, std::unique_ptr<TargetHandler> handler)
      : _triple(triple), _targetHandler(std::move(handler)) {}

  Writer &writer() const override;

  /// Method to create a internal file for an undefined symbol
  std::unique_ptr<File> createUndefinedSymbolFile() const override;

  uint16_t _outputELFType = llvm::ELF::ET_EXEC;
  llvm::Triple _triple;
  std::unique_ptr<TargetHandler> _targetHandler;
  uint64_t _baseAddress = 0;
  bool _isStaticExecutable = false;
  bool _noInhibitExec = false;
  bool _exportDynamic = false;
  bool _mergeCommonStrings = false;
  bool _useShlibUndefines = true;
  bool _dynamicLinkerArg = false;
  bool _noAllowDynamicLibraries = false;
  bool _mergeRODataToTextSegment = true;
  bool _demangle = true;
  bool _discardTempLocals = false;
  bool _discardLocals = false;
  bool _stripSymbols = false;
  bool _alignSegments = true;
  bool _enableNewDtags = false;
  bool _collectStats = false;
  bool _armTarget1Rel = false;
  bool _mipsPcRelEhRel = false;
  uint64_t _maxPageSize = 0x1000;
  uint32_t _dtFlags = 0;

  OutputMagic _outputMagic = OutputMagic::DEFAULT;
  StringRefVector _inputSearchPaths;
  std::unique_ptr<Writer> _writer;
  llvm::Optional<StringRef> _dynamicLinkerPath;
  StringRef _initFunction = "_init";
  StringRef _finiFunction = "_fini";
  StringRef _sysrootPath = "";
  StringRef _soname;
  StringRefVector _rpathList;
  StringRefVector _rpathLinkList;
  llvm::StringSet<> _wrapCalls;
  std::map<std::string, uint64_t> _absoluteSymbols;
  llvm::StringSet<> _dynamicallyExportedSymbols;
  std::unique_ptr<File> _resolver;
  std::mutex _cidentMutex;
  llvm::StringSet<> _cidentSections;

  // The linker script semantic object, which owns all script ASTs, is stored
  // in the current linking context via _linkerScriptSema.
  script::Sema _linkerScriptSema;
};

} // end namespace lld

#endif
