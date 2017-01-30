//===--- Module.h - Module description --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the Module class, which describes a module that has
//  been loaded from an AST file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SERIALIZATION_MODULE_H
#define LLVM_CLANG_SERIALIZATION_MODULE_H

#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Serialization/ASTBitCodes.h"
#include "clang/Serialization/ContinuousRangeMap.h"
#include "clang/Serialization/ModuleFileExtension.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/Endian.h"
#include <memory>
#include <string>

namespace llvm {
template <typename Info> class OnDiskChainedHashTable;
template <typename Info> class OnDiskIterableChainedHashTable;
}

namespace clang {

class DeclContext;
class Module;

namespace serialization {

namespace reader {
  class ASTDeclContextNameLookupTrait;
}

/// \brief Specifies the kind of module that has been loaded.
enum ModuleKind {
  MK_ImplicitModule, ///< File is an implicitly-loaded module.
  MK_ExplicitModule, ///< File is an explicitly-loaded module.
  MK_PCH,            ///< File is a PCH file treated as such.
  MK_Preamble,       ///< File is a PCH file treated as the preamble.
  MK_MainFile,       ///< File is a PCH file treated as the actual main file.
  MK_PrebuiltModule  ///< File is from a prebuilt module path.
};

/// \brief The input file that has been loaded from this AST file, along with
/// bools indicating whether this was an overridden buffer or if it was
/// out-of-date or not-found.
class InputFile {
  enum {
    Overridden = 1,
    OutOfDate = 2,
    NotFound = 3
  };
  llvm::PointerIntPair<const FileEntry *, 2, unsigned> Val;

public:
  InputFile() {}
  InputFile(const FileEntry *File,
            bool isOverridden = false, bool isOutOfDate = false) {
    assert(!(isOverridden && isOutOfDate) &&
           "an overridden cannot be out-of-date");
    unsigned intVal = 0;
    if (isOverridden)
      intVal = Overridden;
    else if (isOutOfDate)
      intVal = OutOfDate;
    Val.setPointerAndInt(File, intVal);
  }

  static InputFile getNotFound() {
    InputFile File;
    File.Val.setInt(NotFound);
    return File;
  }

  const FileEntry *getFile() const { return Val.getPointer(); }
  bool isOverridden() const { return Val.getInt() == Overridden; }
  bool isOutOfDate() const { return Val.getInt() == OutOfDate; }
  bool isNotFound() const { return Val.getInt() == NotFound; }
};

typedef unsigned ASTFileSignature;

/// \brief Information about a module that has been loaded by the ASTReader.
///
/// Each instance of the Module class corresponds to a single AST file, which
/// may be a precompiled header, precompiled preamble, a module, or an AST file
/// of some sort loaded as the main file, all of which are specific formulations
/// of the general notion of a "module". A module may depend on any number of
/// other modules.
class ModuleFile {
public:
  ModuleFile(ModuleKind Kind, unsigned Generation)
      : Kind(Kind), Generation(Generation) {}
  ~ModuleFile();

  // === General information ===

  /// \brief The index of this module in the list of modules.
  unsigned Index = 0;

  /// \brief The type of this module.
  ModuleKind Kind;

  /// \brief The file name of the module file.
  std::string FileName;

  /// \brief The name of the module.
  std::string ModuleName;

  /// \brief The base directory of the module.
  std::string BaseDirectory;

  std::string getTimestampFilename() const {
    return FileName + ".timestamp";
  }

  /// \brief The original source file name that was used to build the
  /// primary AST file, which may have been modified for
  /// relocatable-pch support.
  std::string OriginalSourceFileName;

  /// \brief The actual original source file name that was used to
  /// build this AST file.
  std::string ActualOriginalSourceFileName;

  /// \brief The file ID for the original source file that was used to
  /// build this AST file.
  FileID OriginalSourceFileID;

  /// \brief The directory that the PCH was originally created in. Used to
  /// allow resolving headers even after headers+PCH was moved to a new path.
  std::string OriginalDir;

  std::string ModuleMapPath;

  /// \brief Whether this precompiled header is a relocatable PCH file.
  bool RelocatablePCH = false;

  /// \brief Whether timestamps are included in this module file.
  bool HasTimestamps = false;

  /// \brief The file entry for the module file.
  const FileEntry *File = nullptr;

  /// \brief The signature of the module file, which may be used along with size
  /// and modification time to identify this particular file.
  ASTFileSignature Signature = 0;

  /// \brief Whether this module has been directly imported by the
  /// user.
  bool DirectlyImported = false;

  /// \brief The generation of which this module file is a part.
  unsigned Generation;
  
  /// \brief The memory buffer that stores the data associated with
  /// this AST file.
  std::unique_ptr<llvm::MemoryBuffer> Buffer;

  /// \brief The size of this file, in bits.
  uint64_t SizeInBits = 0;

  /// \brief The global bit offset (or base) of this module
  uint64_t GlobalBitOffset = 0;

  /// \brief The serialized bitstream data for this file.
  StringRef Data;

  /// \brief The main bitstream cursor for the main block.
  llvm::BitstreamCursor Stream;

  /// \brief The source location where the module was explicitly or implicitly
  /// imported in the local translation unit.
  ///
  /// If module A depends on and imports module B, both modules will have the
  /// same DirectImportLoc, but different ImportLoc (B's ImportLoc will be a
  /// source location inside module A).
  ///
  /// WARNING: This is largely useless. It doesn't tell you when a module was
  /// made visible, just when the first submodule of that module was imported.
  SourceLocation DirectImportLoc;

  /// \brief The source location where this module was first imported.
  SourceLocation ImportLoc;

  /// \brief The first source location in this module.
  SourceLocation FirstLoc;

  /// The list of extension readers that are attached to this module
  /// file.
  std::vector<std::unique_ptr<ModuleFileExtensionReader>> ExtensionReaders;

  // === Input Files ===
  /// \brief The cursor to the start of the input-files block.
  llvm::BitstreamCursor InputFilesCursor;

  /// \brief Offsets for all of the input file entries in the AST file.
  const llvm::support::unaligned_uint64_t *InputFileOffsets = nullptr;

  /// \brief The input files that have been loaded from this AST file.
  std::vector<InputFile> InputFilesLoaded;

  // All user input files reside at the index range [0, NumUserInputFiles), and
  // system input files reside at [NumUserInputFiles, InputFilesLoaded.size()).
  unsigned NumUserInputFiles = 0;

  /// \brief If non-zero, specifies the time when we last validated input
  /// files.  Zero means we never validated them.
  ///
  /// The time is specified in seconds since the start of the Epoch.
  uint64_t InputFilesValidationTimestamp = 0;

  // === Source Locations ===

  /// \brief Cursor used to read source location entries.
  llvm::BitstreamCursor SLocEntryCursor;

  /// \brief The number of source location entries in this AST file.
  unsigned LocalNumSLocEntries = 0;

  /// \brief The base ID in the source manager's view of this module.
  int SLocEntryBaseID = 0;

  /// \brief The base offset in the source manager's view of this module.
  unsigned SLocEntryBaseOffset = 0;

  /// \brief Offsets for all of the source location entries in the
  /// AST file.
  const uint32_t *SLocEntryOffsets = nullptr;

  /// \brief SLocEntries that we're going to preload.
  SmallVector<uint64_t, 4> PreloadSLocEntries;

  /// \brief Remapping table for source locations in this module.
  ContinuousRangeMap<uint32_t, int, 2> SLocRemap;

  // === Identifiers ===

  /// \brief The number of identifiers in this AST file.
  unsigned LocalNumIdentifiers = 0;

  /// \brief Offsets into the identifier table data.
  ///
  /// This array is indexed by the identifier ID (-1), and provides
  /// the offset into IdentifierTableData where the string data is
  /// stored.
  const uint32_t *IdentifierOffsets = nullptr;

  /// \brief Base identifier ID for identifiers local to this module.
  serialization::IdentID BaseIdentifierID = 0;

  /// \brief Remapping table for identifier IDs in this module.
  ContinuousRangeMap<uint32_t, int, 2> IdentifierRemap;

  /// \brief Actual data for the on-disk hash table of identifiers.
  ///
  /// This pointer points into a memory buffer, where the on-disk hash
  /// table for identifiers actually lives.
  const char *IdentifierTableData = nullptr;

  /// \brief A pointer to an on-disk hash table of opaque type
  /// IdentifierHashTable.
  void *IdentifierLookupTable = nullptr;

  /// \brief Offsets of identifiers that we're going to preload within
  /// IdentifierTableData.
  std::vector<unsigned> PreloadIdentifierOffsets;

  // === Macros ===

  /// \brief The cursor to the start of the preprocessor block, which stores
  /// all of the macro definitions.
  llvm::BitstreamCursor MacroCursor;

  /// \brief The number of macros in this AST file.
  unsigned LocalNumMacros = 0;

  /// \brief Offsets of macros in the preprocessor block.
  ///
  /// This array is indexed by the macro ID (-1), and provides
  /// the offset into the preprocessor block where macro definitions are
  /// stored.
  const uint32_t *MacroOffsets = nullptr;

  /// \brief Base macro ID for macros local to this module.
  serialization::MacroID BaseMacroID = 0;

  /// \brief Remapping table for macro IDs in this module.
  ContinuousRangeMap<uint32_t, int, 2> MacroRemap;

  /// \brief The offset of the start of the set of defined macros.
  uint64_t MacroStartOffset = 0;

  // === Detailed PreprocessingRecord ===

  /// \brief The cursor to the start of the (optional) detailed preprocessing
  /// record block.
  llvm::BitstreamCursor PreprocessorDetailCursor;

  /// \brief The offset of the start of the preprocessor detail cursor.
  uint64_t PreprocessorDetailStartOffset = 0;

  /// \brief Base preprocessed entity ID for preprocessed entities local to
  /// this module.
  serialization::PreprocessedEntityID BasePreprocessedEntityID = 0;

  /// \brief Remapping table for preprocessed entity IDs in this module.
  ContinuousRangeMap<uint32_t, int, 2> PreprocessedEntityRemap;

  const PPEntityOffset *PreprocessedEntityOffsets = nullptr;
  unsigned NumPreprocessedEntities = 0;

  // === Header search information ===

  /// \brief The number of local HeaderFileInfo structures.
  unsigned LocalNumHeaderFileInfos = 0;

  /// \brief Actual data for the on-disk hash table of header file
  /// information.
  ///
  /// This pointer points into a memory buffer, where the on-disk hash
  /// table for header file information actually lives.
  const char *HeaderFileInfoTableData = nullptr;

  /// \brief The on-disk hash table that contains information about each of
  /// the header files.
  void *HeaderFileInfoTable = nullptr;

  // === Submodule information ===  
  /// \brief The number of submodules in this module.
  unsigned LocalNumSubmodules = 0;
  
  /// \brief Base submodule ID for submodules local to this module.
  serialization::SubmoduleID BaseSubmoduleID = 0;
  
  /// \brief Remapping table for submodule IDs in this module.
  ContinuousRangeMap<uint32_t, int, 2> SubmoduleRemap;
  
  // === Selectors ===

  /// \brief The number of selectors new to this file.
  ///
  /// This is the number of entries in SelectorOffsets.
  unsigned LocalNumSelectors = 0;

  /// \brief Offsets into the selector lookup table's data array
  /// where each selector resides.
  const uint32_t *SelectorOffsets = nullptr;

  /// \brief Base selector ID for selectors local to this module.
  serialization::SelectorID BaseSelectorID = 0;

  /// \brief Remapping table for selector IDs in this module.
  ContinuousRangeMap<uint32_t, int, 2> SelectorRemap;

  /// \brief A pointer to the character data that comprises the selector table
  ///
  /// The SelectorOffsets table refers into this memory.
  const unsigned char *SelectorLookupTableData = nullptr;

  /// \brief A pointer to an on-disk hash table of opaque type
  /// ASTSelectorLookupTable.
  ///
  /// This hash table provides the IDs of all selectors, and the associated
  /// instance and factory methods.
  void *SelectorLookupTable = nullptr;

  // === Declarations ===

  /// DeclsCursor - This is a cursor to the start of the DECLS_BLOCK block. It
  /// has read all the abbreviations at the start of the block and is ready to
  /// jump around with these in context.
  llvm::BitstreamCursor DeclsCursor;

  /// \brief The number of declarations in this AST file.
  unsigned LocalNumDecls = 0;

  /// \brief Offset of each declaration within the bitstream, indexed
  /// by the declaration ID (-1).
  const DeclOffset *DeclOffsets = nullptr;

  /// \brief Base declaration ID for declarations local to this module.
  serialization::DeclID BaseDeclID = 0;

  /// \brief Remapping table for declaration IDs in this module.
  ContinuousRangeMap<uint32_t, int, 2> DeclRemap;

  /// \brief Mapping from the module files that this module file depends on
  /// to the base declaration ID for that module as it is understood within this
  /// module.
  ///
  /// This is effectively a reverse global-to-local mapping for declaration
  /// IDs, so that we can interpret a true global ID (for this translation unit)
  /// as a local ID (for this module file).
  llvm::DenseMap<ModuleFile *, serialization::DeclID> GlobalToLocalDeclIDs;

  /// \brief Array of file-level DeclIDs sorted by file.
  const serialization::DeclID *FileSortedDecls = nullptr;
  unsigned NumFileSortedDecls = 0;

  /// \brief Array of category list location information within this 
  /// module file, sorted by the definition ID.
  const serialization::ObjCCategoriesInfo *ObjCCategoriesMap = nullptr;
  
  /// \brief The number of redeclaration info entries in ObjCCategoriesMap.
  unsigned LocalNumObjCCategoriesInMap = 0;
  
  /// \brief The Objective-C category lists for categories known to this
  /// module.
  SmallVector<uint64_t, 1> ObjCCategories;

  // === Types ===

  /// \brief The number of types in this AST file.
  unsigned LocalNumTypes = 0;

  /// \brief Offset of each type within the bitstream, indexed by the
  /// type ID, or the representation of a Type*.
  const uint32_t *TypeOffsets = nullptr;

  /// \brief Base type ID for types local to this module as represented in
  /// the global type ID space.
  serialization::TypeID BaseTypeIndex = 0;

  /// \brief Remapping table for type IDs in this module.
  ContinuousRangeMap<uint32_t, int, 2> TypeRemap;

  // === Miscellaneous ===

  /// \brief Diagnostic IDs and their mappings that the user changed.
  SmallVector<uint64_t, 8> PragmaDiagMappings;

  /// \brief List of modules which depend on this module
  llvm::SetVector<ModuleFile *> ImportedBy;

  /// \brief List of modules which this module depends on
  llvm::SetVector<ModuleFile *> Imports;

  /// \brief Determine whether this module was directly imported at
  /// any point during translation.
  bool isDirectlyImported() const { return DirectlyImported; }

  /// \brief Is this a module file for a module (rather than a PCH or similar).
  bool isModule() const {
    return Kind == MK_ImplicitModule || Kind == MK_ExplicitModule ||
           Kind == MK_PrebuiltModule;
  }

  /// \brief Dump debugging output for this module.
  void dump();
};

} // end namespace serialization

} // end namespace clang

#endif
