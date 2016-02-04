//===- ASTBitCodes.h - Enum values for the PCH bitcode format ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines Bitcode enum values for Clang serialized AST files.
//
// The enum values defined in this file should be considered permanent.  If
// new features are added, they should have values added at the end of the
// respective lists.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SERIALIZATION_ASTBITCODES_H
#define LLVM_CLANG_SERIALIZATION_ASTBITCODES_H

#include "clang/AST/DeclarationName.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Bitcode/BitCodes.h"
#include "llvm/Support/DataTypes.h"

namespace clang {
  namespace serialization {
    /// \brief AST file major version number supported by this version of
    /// Clang.
    ///
    /// Whenever the AST file format changes in a way that makes it
    /// incompatible with previous versions (such that a reader
    /// designed for the previous version could not support reading
    /// the new version), this number should be increased.
    ///
    /// Version 4 of AST files also requires that the version control branch and
    /// revision match exactly, since there is no backward compatibility of
    /// AST files at this time.
    const unsigned VERSION_MAJOR = 6;

    /// \brief AST file minor version number supported by this version of
    /// Clang.
    ///
    /// Whenever the AST format changes in a way that is still
    /// compatible with previous versions (such that a reader designed
    /// for the previous version could still support reading the new
    /// version by ignoring new kinds of subblocks), this number
    /// should be increased.
    const unsigned VERSION_MINOR = 0;

    /// \brief An ID number that refers to an identifier in an AST file.
    /// 
    /// The ID numbers of identifiers are consecutive (in order of discovery)
    /// and start at 1. 0 is reserved for NULL.
    typedef uint32_t IdentifierID;
    
    /// \brief An ID number that refers to a declaration in an AST file.
    ///
    /// The ID numbers of declarations are consecutive (in order of
    /// discovery), with values below NUM_PREDEF_DECL_IDS being reserved. 
    /// At the start of a chain of precompiled headers, declaration ID 1 is 
    /// used for the translation unit declaration.
    typedef uint32_t DeclID;

    // FIXME: Turn these into classes so we can have some type safety when
    // we go from local ID to global and vice-versa.
    typedef DeclID LocalDeclID;
    typedef DeclID GlobalDeclID;

    /// \brief An ID number that refers to a type in an AST file.
    ///
    /// The ID of a type is partitioned into two parts: the lower
    /// three bits are used to store the const/volatile/restrict
    /// qualifiers (as with QualType) and the upper bits provide a
    /// type index. The type index values are partitioned into two
    /// sets. The values below NUM_PREDEF_TYPE_IDs are predefined type
    /// IDs (based on the PREDEF_TYPE_*_ID constants), with 0 as a
    /// placeholder for "no type". Values from NUM_PREDEF_TYPE_IDs are
    /// other types that have serialized representations.
    typedef uint32_t TypeID;

    /// \brief A type index; the type ID with the qualifier bits removed.
    class TypeIdx {
      uint32_t Idx;
    public:
      TypeIdx() : Idx(0) { }
      explicit TypeIdx(uint32_t index) : Idx(index) { }

      uint32_t getIndex() const { return Idx; }
      TypeID asTypeID(unsigned FastQuals) const {
        if (Idx == uint32_t(-1))
          return TypeID(-1);
        
        return (Idx << Qualifiers::FastWidth) | FastQuals;
      }
      static TypeIdx fromTypeID(TypeID ID) {
        if (ID == TypeID(-1))
          return TypeIdx(-1);
        
        return TypeIdx(ID >> Qualifiers::FastWidth);
      }
    };

    /// A structure for putting "fast"-unqualified QualTypes into a
    /// DenseMap.  This uses the standard pointer hash function.
    struct UnsafeQualTypeDenseMapInfo {
      static inline bool isEqual(QualType A, QualType B) { return A == B; }
      static inline QualType getEmptyKey() {
        return QualType::getFromOpaquePtr((void*) 1);
      }
      static inline QualType getTombstoneKey() {
        return QualType::getFromOpaquePtr((void*) 2);
      }
      static inline unsigned getHashValue(QualType T) {
        assert(!T.getLocalFastQualifiers() && 
               "hash invalid for types with fast quals");
        uintptr_t v = reinterpret_cast<uintptr_t>(T.getAsOpaquePtr());
        return (unsigned(v) >> 4) ^ (unsigned(v) >> 9);
      }
    };

    /// \brief An ID number that refers to an identifier in an AST file.
    typedef uint32_t IdentID;

    /// \brief The number of predefined identifier IDs.
    const unsigned int NUM_PREDEF_IDENT_IDS = 1;

    /// \brief An ID number that refers to a macro in an AST file.
    typedef uint32_t MacroID;

    /// \brief A global ID number that refers to a macro in an AST file.
    typedef uint32_t GlobalMacroID;

    /// \brief A local to a module ID number that refers to a macro in an
    /// AST file.
    typedef uint32_t LocalMacroID;

    /// \brief The number of predefined macro IDs.
    const unsigned int NUM_PREDEF_MACRO_IDS = 1;

    /// \brief An ID number that refers to an ObjC selector in an AST file.
    typedef uint32_t SelectorID;

    /// \brief The number of predefined selector IDs.
    const unsigned int NUM_PREDEF_SELECTOR_IDS = 1;
    
    /// \brief An ID number that refers to a set of CXXBaseSpecifiers in an 
    /// AST file.
    typedef uint32_t CXXBaseSpecifiersID;

    /// \brief An ID number that refers to a list of CXXCtorInitializers in an
    /// AST file.
    typedef uint32_t CXXCtorInitializersID;

    /// \brief An ID number that refers to an entity in the detailed
    /// preprocessing record.
    typedef uint32_t PreprocessedEntityID;

    /// \brief An ID number that refers to a submodule in a module file.
    typedef uint32_t SubmoduleID;
    
    /// \brief The number of predefined submodule IDs.
    const unsigned int NUM_PREDEF_SUBMODULE_IDS = 1;

    /// \brief Source range/offset of a preprocessed entity.
    struct PPEntityOffset {
      /// \brief Raw source location of beginning of range.
      unsigned Begin;
      /// \brief Raw source location of end of range.
      unsigned End;
      /// \brief Offset in the AST file.
      uint32_t BitOffset;

      PPEntityOffset(SourceRange R, uint32_t BitOffset)
        : Begin(R.getBegin().getRawEncoding()),
          End(R.getEnd().getRawEncoding()),
          BitOffset(BitOffset) { }
    };

    /// \brief Source range/offset of a preprocessed entity.
    struct DeclOffset {
      /// \brief Raw source location.
      unsigned Loc;
      /// \brief Offset in the AST file.
      uint32_t BitOffset;

      DeclOffset() : Loc(0), BitOffset(0) { }
      DeclOffset(SourceLocation Loc, uint32_t BitOffset)
        : Loc(Loc.getRawEncoding()),
          BitOffset(BitOffset) { }
      void setLocation(SourceLocation L) {
        Loc = L.getRawEncoding();
      }
    };

    /// \brief The number of predefined preprocessed entity IDs.
    const unsigned int NUM_PREDEF_PP_ENTITY_IDS = 1;

    /// \brief Describes the various kinds of blocks that occur within
    /// an AST file.
    enum BlockIDs {
      /// \brief The AST block, which acts as a container around the
      /// full AST block.
      AST_BLOCK_ID = llvm::bitc::FIRST_APPLICATION_BLOCKID,

      /// \brief The block containing information about the source
      /// manager.
      SOURCE_MANAGER_BLOCK_ID,

      /// \brief The block containing information about the
      /// preprocessor.
      PREPROCESSOR_BLOCK_ID,

      /// \brief The block containing the definitions of all of the
      /// types and decls used within the AST file.
      DECLTYPES_BLOCK_ID,

      /// \brief The block containing the detailed preprocessing record.
      PREPROCESSOR_DETAIL_BLOCK_ID,
      
      /// \brief The block containing the submodule structure.
      SUBMODULE_BLOCK_ID,

      /// \brief The block containing comments.
      COMMENTS_BLOCK_ID,

      /// \brief The control block, which contains all of the
      /// information that needs to be validated prior to committing
      /// to loading the AST file.
      CONTROL_BLOCK_ID,

      /// \brief The block of input files, which were used as inputs
      /// to create this AST file.
      ///
      /// This block is part of the control block.
      INPUT_FILES_BLOCK_ID,

      /// \brief The block of configuration options, used to check that
      /// a module is being used in a configuration compatible with the
      /// configuration in which it was built.
      ///
      /// This block is part of the control block.
      OPTIONS_BLOCK_ID,

      /// \brief A block containing a module file extension.
      EXTENSION_BLOCK_ID,
    };

    /// \brief Record types that occur within the control block.
    enum ControlRecordTypes {
      /// \brief AST file metadata, including the AST file version number
      /// and information about the compiler used to build this AST file.
      METADATA = 1,

      /// \brief Record code for the list of other AST files imported by
      /// this AST file.
      IMPORTS,

      /// \brief Record code for the original file that was used to
      /// generate the AST file, including both its file ID and its
      /// name.
      ORIGINAL_FILE,
      
      /// \brief The directory that the PCH was originally created in.
      ORIGINAL_PCH_DIR,

      /// \brief Record code for file ID of the file or buffer that was used to
      /// generate the AST file.
      ORIGINAL_FILE_ID,

      /// \brief Offsets into the input-files block where input files
      /// reside.
      INPUT_FILE_OFFSETS,

      /// \brief Record code for the module name.
      MODULE_NAME,

      /// \brief Record code for the module map file that was used to build this
      /// AST file.
      MODULE_MAP_FILE,

      /// \brief Record code for the signature that identifiers this AST file.
      SIGNATURE,

      /// \brief Record code for the module build directory.
      MODULE_DIRECTORY,
    };

    /// \brief Record types that occur within the options block inside
    /// the control block.
    enum OptionsRecordTypes {
      /// \brief Record code for the language options table.
      ///
      /// The record with this code contains the contents of the
      /// LangOptions structure. We serialize the entire contents of
      /// the structure, and let the reader decide which options are
      /// actually important to check.
      LANGUAGE_OPTIONS = 1,

      /// \brief Record code for the target options table.
      TARGET_OPTIONS,

      /// \brief Record code for the diagnostic options table.
      DIAGNOSTIC_OPTIONS,

      /// \brief Record code for the filesystem options table.
      FILE_SYSTEM_OPTIONS,

      /// \brief Record code for the headers search options table.
      HEADER_SEARCH_OPTIONS,

      /// \brief Record code for the preprocessor options table.
      PREPROCESSOR_OPTIONS,
    };

    /// \brief Record code for extension blocks.
    enum ExtensionBlockRecordTypes {
      /// Metadata describing this particular extension.
      EXTENSION_METADATA = 1,

      /// The first record ID allocated to the extensions themselves.
      FIRST_EXTENSION_RECORD_ID = 4
    };

    /// \brief Record types that occur within the input-files block
    /// inside the control block.
    enum InputFileRecordTypes {
      /// \brief An input file.
      INPUT_FILE = 1
    };

    /// \brief Record types that occur within the AST block itself.
    enum ASTRecordTypes {
      /// \brief Record code for the offsets of each type.
      ///
      /// The TYPE_OFFSET constant describes the record that occurs
      /// within the AST block. The record itself is an array of offsets that
      /// point into the declarations and types block (identified by 
      /// DECLTYPES_BLOCK_ID). The index into the array is based on the ID
      /// of a type. For a given type ID @c T, the lower three bits of
      /// @c T are its qualifiers (const, volatile, restrict), as in
      /// the QualType class. The upper bits, after being shifted and
      /// subtracting NUM_PREDEF_TYPE_IDS, are used to index into the
      /// TYPE_OFFSET block to determine the offset of that type's
      /// corresponding record within the DECLTYPES_BLOCK_ID block.
      TYPE_OFFSET = 1,

      /// \brief Record code for the offsets of each decl.
      ///
      /// The DECL_OFFSET constant describes the record that occurs
      /// within the block identified by DECL_OFFSETS_BLOCK_ID within
      /// the AST block. The record itself is an array of offsets that
      /// point into the declarations and types block (identified by
      /// DECLTYPES_BLOCK_ID). The declaration ID is an index into this
      /// record, after subtracting one to account for the use of
      /// declaration ID 0 for a NULL declaration pointer. Index 0 is
      /// reserved for the translation unit declaration.
      DECL_OFFSET = 2,

      /// \brief Record code for the table of offsets of each
      /// identifier ID.
      ///
      /// The offset table contains offsets into the blob stored in
      /// the IDENTIFIER_TABLE record. Each offset points to the
      /// NULL-terminated string that corresponds to that identifier.
      IDENTIFIER_OFFSET = 3,

      /// \brief This is so that older clang versions, before the introduction
      /// of the control block, can read and reject the newer PCH format.
      /// *DON'T CHANGE THIS NUMBER*.
      METADATA_OLD_FORMAT = 4,

      /// \brief Record code for the identifier table.
      ///
      /// The identifier table is a simple blob that contains
      /// NULL-terminated strings for all of the identifiers
      /// referenced by the AST file. The IDENTIFIER_OFFSET table
      /// contains the mapping from identifier IDs to the characters
      /// in this blob. Note that the starting offsets of all of the
      /// identifiers are odd, so that, when the identifier offset
      /// table is loaded in, we can use the low bit to distinguish
      /// between offsets (for unresolved identifier IDs) and
      /// IdentifierInfo pointers (for already-resolved identifier
      /// IDs).
      IDENTIFIER_TABLE = 5,

      /// \brief Record code for the array of eagerly deserialized decls.
      ///
      /// The AST file contains a list of all of the declarations that should be
      /// eagerly deserialized present within the parsed headers, stored as an
      /// array of declaration IDs. These declarations will be
      /// reported to the AST consumer after the AST file has been
      /// read, since their presence can affect the semantics of the
      /// program (e.g., for code generation).
      EAGERLY_DESERIALIZED_DECLS = 6,

      /// \brief Record code for the set of non-builtin, special
      /// types.
      ///
      /// This record contains the type IDs for the various type nodes
      /// that are constructed during semantic analysis (e.g.,
      /// __builtin_va_list). The SPECIAL_TYPE_* constants provide
      /// offsets into this record.
      SPECIAL_TYPES = 7,

      /// \brief Record code for the extra statistics we gather while
      /// generating an AST file.
      STATISTICS = 8,

      /// \brief Record code for the array of tentative definitions.
      TENTATIVE_DEFINITIONS = 9,

      // ID 10 used to be for a list of extern "C" declarations.

      /// \brief Record code for the table of offsets into the
      /// Objective-C method pool.
      SELECTOR_OFFSETS = 11,

      /// \brief Record code for the Objective-C method pool,
      METHOD_POOL = 12,

      /// \brief The value of the next __COUNTER__ to dispense.
      /// [PP_COUNTER_VALUE, Val]
      PP_COUNTER_VALUE = 13,

      /// \brief Record code for the table of offsets into the block
      /// of source-location information.
      SOURCE_LOCATION_OFFSETS = 14,

      /// \brief Record code for the set of source location entries
      /// that need to be preloaded by the AST reader.
      ///
      /// This set contains the source location entry for the
      /// predefines buffer and for any file entries that need to be
      /// preloaded.
      SOURCE_LOCATION_PRELOADS = 15,

      /// \brief Record code for the set of ext_vector type names.
      EXT_VECTOR_DECLS = 16,
      
      /// \brief Record code for the array of unused file scoped decls.
      UNUSED_FILESCOPED_DECLS = 17,
      
      /// \brief Record code for the table of offsets to entries in the
      /// preprocessing record.
      PPD_ENTITIES_OFFSETS = 18,

      /// \brief Record code for the array of VTable uses.
      VTABLE_USES = 19,

      // ID 20 used to be for a list of dynamic classes.

      /// \brief Record code for referenced selector pool.
      REFERENCED_SELECTOR_POOL = 21,

      /// \brief Record code for an update to the TU's lexically contained
      /// declarations.
      TU_UPDATE_LEXICAL = 22,
      
      // ID 23 used to be for a list of local redeclarations.

      /// \brief Record code for declarations that Sema keeps references of.
      SEMA_DECL_REFS = 24,

      /// \brief Record code for weak undeclared identifiers.
      WEAK_UNDECLARED_IDENTIFIERS = 25,

      /// \brief Record code for pending implicit instantiations.
      PENDING_IMPLICIT_INSTANTIATIONS = 26,

      /// \brief Record code for a decl replacement block.
      ///
      /// If a declaration is modified after having been deserialized, and then
      /// written to a dependent AST file, its ID and offset must be added to
      /// the replacement block.
      DECL_REPLACEMENTS = 27,

      /// \brief Record code for an update to a decl context's lookup table.
      ///
      /// In practice, this should only be used for the TU and namespaces.
      UPDATE_VISIBLE = 28,

      /// \brief Record for offsets of DECL_UPDATES records for declarations
      /// that were modified after being deserialized and need updates.
      DECL_UPDATE_OFFSETS = 29,

      /// \brief Record of updates for a declaration that was modified after
      /// being deserialized.
      DECL_UPDATES = 30,
      
      /// \brief Record code for the table of offsets to CXXBaseSpecifier
      /// sets.
      CXX_BASE_SPECIFIER_OFFSETS = 31,

      /// \brief Record code for \#pragma diagnostic mappings.
      DIAG_PRAGMA_MAPPINGS = 32,

      /// \brief Record code for special CUDA declarations.
      CUDA_SPECIAL_DECL_REFS = 33,
      
      /// \brief Record code for header search information.
      HEADER_SEARCH_TABLE = 34,

      /// \brief Record code for floating point \#pragma options.
      FP_PRAGMA_OPTIONS = 35,

      /// \brief Record code for enabled OpenCL extensions.
      OPENCL_EXTENSIONS = 36,

      /// \brief The list of delegating constructor declarations.
      DELEGATING_CTORS = 37,

      /// \brief Record code for the set of known namespaces, which are used
      /// for typo correction.
      KNOWN_NAMESPACES = 38,

      /// \brief Record code for the remapping information used to relate
      /// loaded modules to the various offsets and IDs(e.g., source location 
      /// offests, declaration and type IDs) that are used in that module to
      /// refer to other modules.
      MODULE_OFFSET_MAP = 39,

      /// \brief Record code for the source manager line table information,
      /// which stores information about \#line directives.
      SOURCE_MANAGER_LINE_TABLE = 40,

      /// \brief Record code for map of Objective-C class definition IDs to the 
      /// ObjC categories in a module that are attached to that class.
      OBJC_CATEGORIES_MAP = 41,

      /// \brief Record code for a file sorted array of DeclIDs in a module.
      FILE_SORTED_DECLS = 42,
      
      /// \brief Record code for an array of all of the (sub)modules that were
      /// imported by the AST file.
      IMPORTED_MODULES = 43,
      
      // ID 44 used to be a table of merged canonical declarations.
      // ID 45 used to be a list of declaration IDs of local redeclarations.
      
      /// \brief Record code for the array of Objective-C categories (including
      /// extensions).
      ///
      /// This array can only be interpreted properly using the Objective-C
      /// categories map.
      OBJC_CATEGORIES = 46,

      /// \brief Record code for the table of offsets of each macro ID.
      ///
      /// The offset table contains offsets into the blob stored in
      /// the preprocessor block. Each offset points to the corresponding
      /// macro definition.
      MACRO_OFFSET = 47,

      /// \brief A list of "interesting" identifiers. Only used in C++ (where we
      /// don't normally do lookups into the serialized identifier table). These
      /// are eagerly deserialized.
      INTERESTING_IDENTIFIERS = 48,

      /// \brief Record code for undefined but used functions and variables that
      /// need a definition in this TU.
      UNDEFINED_BUT_USED = 49,

      /// \brief Record code for late parsed template functions.
      LATE_PARSED_TEMPLATE = 50,

      /// \brief Record code for \#pragma optimize options.
      OPTIMIZE_PRAGMA_OPTIONS = 51,

      /// \brief Record code for potentially unused local typedef names.
      UNUSED_LOCAL_TYPEDEF_NAME_CANDIDATES = 52,

      /// \brief Record code for the table of offsets to CXXCtorInitializers
      /// lists.
      CXX_CTOR_INITIALIZERS_OFFSETS = 53,

      /// \brief Delete expressions that will be analyzed later.
      DELETE_EXPRS_TO_ANALYZE = 54
    };

    /// \brief Record types used within a source manager block.
    enum SourceManagerRecordTypes {
      /// \brief Describes a source location entry (SLocEntry) for a
      /// file.
      SM_SLOC_FILE_ENTRY = 1,
      /// \brief Describes a source location entry (SLocEntry) for a
      /// buffer.
      SM_SLOC_BUFFER_ENTRY = 2,
      /// \brief Describes a blob that contains the data for a buffer
      /// entry. This kind of record always directly follows a
      /// SM_SLOC_BUFFER_ENTRY record or a SM_SLOC_FILE_ENTRY with an
      /// overridden buffer.
      SM_SLOC_BUFFER_BLOB = 3,
      /// \brief Describes a source location entry (SLocEntry) for a
      /// macro expansion.
      SM_SLOC_EXPANSION_ENTRY = 4
    };

    /// \brief Record types used within a preprocessor block.
    enum PreprocessorRecordTypes {
      // The macros in the PP section are a PP_MACRO_* instance followed by a
      // list of PP_TOKEN instances for each token in the definition.

      /// \brief An object-like macro definition.
      /// [PP_MACRO_OBJECT_LIKE, IdentInfoID, SLoc, IsUsed]
      PP_MACRO_OBJECT_LIKE = 1,

      /// \brief A function-like macro definition.
      /// [PP_MACRO_FUNCTION_LIKE, \<ObjectLikeStuff>, IsC99Varargs,
      /// IsGNUVarars, NumArgs, ArgIdentInfoID* ]
      PP_MACRO_FUNCTION_LIKE = 2,

      /// \brief Describes one token.
      /// [PP_TOKEN, SLoc, Length, IdentInfoID, Kind, Flags]
      PP_TOKEN = 3,

      /// \brief The macro directives history for a particular identifier.
      PP_MACRO_DIRECTIVE_HISTORY = 4,

      /// \brief A macro directive exported by a module.
      /// [PP_MODULE_MACRO, SubmoduleID, MacroID, (Overridden SubmoduleID)*]
      PP_MODULE_MACRO = 5,
    };

    /// \brief Record types used within a preprocessor detail block.
    enum PreprocessorDetailRecordTypes {
      /// \brief Describes a macro expansion within the preprocessing record.
      PPD_MACRO_EXPANSION = 0,
      
      /// \brief Describes a macro definition within the preprocessing record.
      PPD_MACRO_DEFINITION = 1,
      
      /// \brief Describes an inclusion directive within the preprocessing
      /// record.
      PPD_INCLUSION_DIRECTIVE = 2
    };
    
    /// \brief Record types used within a submodule description block.
    enum SubmoduleRecordTypes {
      /// \brief Metadata for submodules as a whole.
      SUBMODULE_METADATA = 0,
      /// \brief Defines the major attributes of a submodule, including its
      /// name and parent.
      SUBMODULE_DEFINITION = 1,
      /// \brief Specifies the umbrella header used to create this module,
      /// if any.
      SUBMODULE_UMBRELLA_HEADER = 2,
      /// \brief Specifies a header that falls into this (sub)module.
      SUBMODULE_HEADER = 3,
      /// \brief Specifies a top-level header that falls into this (sub)module.
      SUBMODULE_TOPHEADER = 4,
      /// \brief Specifies an umbrella directory.
      SUBMODULE_UMBRELLA_DIR = 5,
      /// \brief Specifies the submodules that are imported by this 
      /// submodule.
      SUBMODULE_IMPORTS = 6,
      /// \brief Specifies the submodules that are re-exported from this 
      /// submodule.
      SUBMODULE_EXPORTS = 7,
      /// \brief Specifies a required feature.
      SUBMODULE_REQUIRES = 8,
      /// \brief Specifies a header that has been explicitly excluded
      /// from this submodule.
      SUBMODULE_EXCLUDED_HEADER = 9,
      /// \brief Specifies a library or framework to link against.
      SUBMODULE_LINK_LIBRARY = 10,
      /// \brief Specifies a configuration macro for this module.
      SUBMODULE_CONFIG_MACRO = 11,
      /// \brief Specifies a conflict with another module.
      SUBMODULE_CONFLICT = 12,
      /// \brief Specifies a header that is private to this submodule.
      SUBMODULE_PRIVATE_HEADER = 13,
      /// \brief Specifies a header that is part of the module but must be
      /// textually included.
      SUBMODULE_TEXTUAL_HEADER = 14,
      /// \brief Specifies a header that is private to this submodule but
      /// must be textually included.
      SUBMODULE_PRIVATE_TEXTUAL_HEADER = 15,
    };

    /// \brief Record types used within a comments block.
    enum CommentRecordTypes {
      COMMENTS_RAW_COMMENT = 0
    };

    /// \defgroup ASTAST AST file AST constants
    ///
    /// The constants in this group describe various components of the
    /// abstract syntax tree within an AST file.
    ///
    /// @{

    /// \brief Predefined type IDs.
    ///
    /// These type IDs correspond to predefined types in the AST
    /// context, such as built-in types (int) and special place-holder
    /// types (the \<overload> and \<dependent> type markers). Such
    /// types are never actually serialized, since they will be built
    /// by the AST context when it is created.
    enum PredefinedTypeIDs {
      /// \brief The NULL type.
      PREDEF_TYPE_NULL_ID       = 0,
      /// \brief The void type.
      PREDEF_TYPE_VOID_ID       = 1,
      /// \brief The 'bool' or '_Bool' type.
      PREDEF_TYPE_BOOL_ID       = 2,
      /// \brief The 'char' type, when it is unsigned.
      PREDEF_TYPE_CHAR_U_ID     = 3,
      /// \brief The 'unsigned char' type.
      PREDEF_TYPE_UCHAR_ID      = 4,
      /// \brief The 'unsigned short' type.
      PREDEF_TYPE_USHORT_ID     = 5,
      /// \brief The 'unsigned int' type.
      PREDEF_TYPE_UINT_ID       = 6,
      /// \brief The 'unsigned long' type.
      PREDEF_TYPE_ULONG_ID      = 7,
      /// \brief The 'unsigned long long' type.
      PREDEF_TYPE_ULONGLONG_ID  = 8,
      /// \brief The 'char' type, when it is signed.
      PREDEF_TYPE_CHAR_S_ID     = 9,
      /// \brief The 'signed char' type.
      PREDEF_TYPE_SCHAR_ID      = 10,
      /// \brief The C++ 'wchar_t' type.
      PREDEF_TYPE_WCHAR_ID      = 11,
      /// \brief The (signed) 'short' type.
      PREDEF_TYPE_SHORT_ID      = 12,
      /// \brief The (signed) 'int' type.
      PREDEF_TYPE_INT_ID        = 13,
      /// \brief The (signed) 'long' type.
      PREDEF_TYPE_LONG_ID       = 14,
      /// \brief The (signed) 'long long' type.
      PREDEF_TYPE_LONGLONG_ID   = 15,
      /// \brief The 'float' type.
      PREDEF_TYPE_FLOAT_ID      = 16,
      /// \brief The 'double' type.
      PREDEF_TYPE_DOUBLE_ID     = 17,
      /// \brief The 'long double' type.
      PREDEF_TYPE_LONGDOUBLE_ID = 18,
      /// \brief The placeholder type for overloaded function sets.
      PREDEF_TYPE_OVERLOAD_ID   = 19,
      /// \brief The placeholder type for dependent types.
      PREDEF_TYPE_DEPENDENT_ID  = 20,
      /// \brief The '__uint128_t' type.
      PREDEF_TYPE_UINT128_ID    = 21,
      /// \brief The '__int128_t' type.
      PREDEF_TYPE_INT128_ID     = 22,
      /// \brief The type of 'nullptr'.
      PREDEF_TYPE_NULLPTR_ID    = 23,
      /// \brief The C++ 'char16_t' type.
      PREDEF_TYPE_CHAR16_ID     = 24,
      /// \brief The C++ 'char32_t' type.
      PREDEF_TYPE_CHAR32_ID     = 25,
      /// \brief The ObjC 'id' type.
      PREDEF_TYPE_OBJC_ID       = 26,
      /// \brief The ObjC 'Class' type.
      PREDEF_TYPE_OBJC_CLASS    = 27,
      /// \brief The ObjC 'SEL' type.
      PREDEF_TYPE_OBJC_SEL      = 28,
      /// \brief The 'unknown any' placeholder type.
      PREDEF_TYPE_UNKNOWN_ANY   = 29,
      /// \brief The placeholder type for bound member functions.
      PREDEF_TYPE_BOUND_MEMBER  = 30,
      /// \brief The "auto" deduction type.
      PREDEF_TYPE_AUTO_DEDUCT   = 31,
      /// \brief The "auto &&" deduction type.
      PREDEF_TYPE_AUTO_RREF_DEDUCT = 32,
      /// \brief The OpenCL 'half' / ARM NEON __fp16 type.
      PREDEF_TYPE_HALF_ID       = 33,
      /// \brief ARC's unbridged-cast placeholder type.
      PREDEF_TYPE_ARC_UNBRIDGED_CAST = 34,
      /// \brief The pseudo-object placeholder type.
      PREDEF_TYPE_PSEUDO_OBJECT = 35,
      /// \brief The placeholder type for builtin functions.
      PREDEF_TYPE_BUILTIN_FN = 36,
      /// \brief OpenCL 1d image type.
      PREDEF_TYPE_IMAGE1D_ID    = 37,
      /// \brief OpenCL 1d image array type.
      PREDEF_TYPE_IMAGE1D_ARR_ID = 38,
      /// \brief OpenCL 1d image buffer type.
      PREDEF_TYPE_IMAGE1D_BUFF_ID = 39,
      /// \brief OpenCL 2d image type.
      PREDEF_TYPE_IMAGE2D_ID    = 40,
      /// \brief OpenCL 2d image array type.
      PREDEF_TYPE_IMAGE2D_ARR_ID = 41,
      /// \brief OpenCL 2d image depth type.
      PREDEF_TYPE_IMAGE2D_DEP_ID = 42,
      /// \brief OpenCL 2d image array depth type.
      PREDEF_TYPE_IMAGE2D_ARR_DEP_ID = 43,
      /// \brief OpenCL 2d image MSAA type.
      PREDEF_TYPE_IMAGE2D_MSAA_ID = 44,
      /// \brief OpenCL 2d image array MSAA type.
      PREDEF_TYPE_IMAGE2D_ARR_MSAA_ID = 45,
      /// \brief OpenCL 2d image MSAA depth type.
      PREDEF_TYPE_IMAGE2D_MSAA_DEP_ID = 46,
      /// \brief OpenCL 2d image array MSAA depth type.
      PREDEF_TYPE_IMAGE2D_ARR_MSAA_DEPTH_ID = 47,
      /// \brief OpenCL 3d image type.
      PREDEF_TYPE_IMAGE3D_ID    = 48,
      /// \brief OpenCL event type.
      PREDEF_TYPE_EVENT_ID      = 49,
      /// \brief OpenCL clk event type.
      PREDEF_TYPE_CLK_EVENT_ID  = 50,
      /// \brief OpenCL sampler type.
      PREDEF_TYPE_SAMPLER_ID    = 51,
      /// \brief OpenCL queue type.
      PREDEF_TYPE_QUEUE_ID      = 52,
      /// \brief OpenCL ndrange type.
      PREDEF_TYPE_NDRANGE_ID    = 53,
      /// \brief OpenCL reserve_id type.
      PREDEF_TYPE_RESERVE_ID_ID = 54,
      /// \brief The placeholder type for OpenMP array section.
      PREDEF_TYPE_OMP_ARRAY_SECTION = 55
    };

    /// \brief The number of predefined type IDs that are reserved for
    /// the PREDEF_TYPE_* constants.
    ///
    /// Type IDs for non-predefined types will start at
    /// NUM_PREDEF_TYPE_IDs.
    const unsigned NUM_PREDEF_TYPE_IDS = 100;

    /// \brief Record codes for each kind of type.
    ///
    /// These constants describe the type records that can occur within a
    /// block identified by DECLTYPES_BLOCK_ID in the AST file. Each
    /// constant describes a record for a specific type class in the
    /// AST.
    enum TypeCode {
      /// \brief An ExtQualType record.
      TYPE_EXT_QUAL                 = 1,
      /// \brief A ComplexType record.
      TYPE_COMPLEX                  = 3,
      /// \brief A PointerType record.
      TYPE_POINTER                  = 4,
      /// \brief A BlockPointerType record.
      TYPE_BLOCK_POINTER            = 5,
      /// \brief An LValueReferenceType record.
      TYPE_LVALUE_REFERENCE         = 6,
      /// \brief An RValueReferenceType record.
      TYPE_RVALUE_REFERENCE         = 7,
      /// \brief A MemberPointerType record.
      TYPE_MEMBER_POINTER           = 8,
      /// \brief A ConstantArrayType record.
      TYPE_CONSTANT_ARRAY           = 9,
      /// \brief An IncompleteArrayType record.
      TYPE_INCOMPLETE_ARRAY         = 10,
      /// \brief A VariableArrayType record.
      TYPE_VARIABLE_ARRAY           = 11,
      /// \brief A VectorType record.
      TYPE_VECTOR                   = 12,
      /// \brief An ExtVectorType record.
      TYPE_EXT_VECTOR               = 13,
      /// \brief A FunctionNoProtoType record.
      TYPE_FUNCTION_NO_PROTO        = 14,
      /// \brief A FunctionProtoType record.
      TYPE_FUNCTION_PROTO           = 15,
      /// \brief A TypedefType record.
      TYPE_TYPEDEF                  = 16,
      /// \brief A TypeOfExprType record.
      TYPE_TYPEOF_EXPR              = 17,
      /// \brief A TypeOfType record.
      TYPE_TYPEOF                   = 18,
      /// \brief A RecordType record.
      TYPE_RECORD                   = 19,
      /// \brief An EnumType record.
      TYPE_ENUM                     = 20,
      /// \brief An ObjCInterfaceType record.
      TYPE_OBJC_INTERFACE           = 21,
      /// \brief An ObjCObjectPointerType record.
      TYPE_OBJC_OBJECT_POINTER      = 22,
      /// \brief a DecltypeType record.
      TYPE_DECLTYPE                 = 23,
      /// \brief An ElaboratedType record.
      TYPE_ELABORATED               = 24,
      /// \brief A SubstTemplateTypeParmType record.
      TYPE_SUBST_TEMPLATE_TYPE_PARM = 25,
      /// \brief An UnresolvedUsingType record.
      TYPE_UNRESOLVED_USING         = 26,
      /// \brief An InjectedClassNameType record.
      TYPE_INJECTED_CLASS_NAME      = 27,
      /// \brief An ObjCObjectType record.
      TYPE_OBJC_OBJECT              = 28,
      /// \brief An TemplateTypeParmType record.
      TYPE_TEMPLATE_TYPE_PARM       = 29,
      /// \brief An TemplateSpecializationType record.
      TYPE_TEMPLATE_SPECIALIZATION  = 30,
      /// \brief A DependentNameType record.
      TYPE_DEPENDENT_NAME           = 31,
      /// \brief A DependentTemplateSpecializationType record.
      TYPE_DEPENDENT_TEMPLATE_SPECIALIZATION = 32,
      /// \brief A DependentSizedArrayType record.
      TYPE_DEPENDENT_SIZED_ARRAY    = 33,
      /// \brief A ParenType record.
      TYPE_PAREN                    = 34,
      /// \brief A PackExpansionType record.
      TYPE_PACK_EXPANSION           = 35,
      /// \brief An AttributedType record.
      TYPE_ATTRIBUTED               = 36,
      /// \brief A SubstTemplateTypeParmPackType record.
      TYPE_SUBST_TEMPLATE_TYPE_PARM_PACK = 37,
      /// \brief A AutoType record.
      TYPE_AUTO                  = 38,
      /// \brief A UnaryTransformType record.
      TYPE_UNARY_TRANSFORM       = 39,
      /// \brief An AtomicType record.
      TYPE_ATOMIC                = 40,
      /// \brief A DecayedType record.
      TYPE_DECAYED               = 41,
      /// \brief An AdjustedType record.
      TYPE_ADJUSTED              = 42,
      /// \brief A PipeType record.
      TYPE_PIPE                  = 43
    };

    /// \brief The type IDs for special types constructed by semantic
    /// analysis.
    ///
    /// The constants in this enumeration are indices into the
    /// SPECIAL_TYPES record.
    enum SpecialTypeIDs {
      /// \brief CFConstantString type
      SPECIAL_TYPE_CF_CONSTANT_STRING          = 0,
      /// \brief C FILE typedef type
      SPECIAL_TYPE_FILE                        = 1,
      /// \brief C jmp_buf typedef type
      SPECIAL_TYPE_JMP_BUF                     = 2,
      /// \brief C sigjmp_buf typedef type
      SPECIAL_TYPE_SIGJMP_BUF                  = 3,
      /// \brief Objective-C "id" redefinition type
      SPECIAL_TYPE_OBJC_ID_REDEFINITION        = 4,
      /// \brief Objective-C "Class" redefinition type
      SPECIAL_TYPE_OBJC_CLASS_REDEFINITION     = 5,
      /// \brief Objective-C "SEL" redefinition type
      SPECIAL_TYPE_OBJC_SEL_REDEFINITION       = 6,
      /// \brief C ucontext_t typedef type
      SPECIAL_TYPE_UCONTEXT_T                  = 7
    };
    
    /// \brief The number of special type IDs.
    const unsigned NumSpecialTypeIDs = 8;

    /// \brief Predefined declaration IDs.
    ///
    /// These declaration IDs correspond to predefined declarations in the AST
    /// context, such as the NULL declaration ID. Such declarations are never
    /// actually serialized, since they will be built by the AST context when 
    /// it is created.
    enum PredefinedDeclIDs {
      /// \brief The NULL declaration.
      PREDEF_DECL_NULL_ID = 0,

      /// \brief The translation unit.
      PREDEF_DECL_TRANSLATION_UNIT_ID = 1,

      /// \brief The Objective-C 'id' type.
      PREDEF_DECL_OBJC_ID_ID = 2,

      /// \brief The Objective-C 'SEL' type.
      PREDEF_DECL_OBJC_SEL_ID = 3,

      /// \brief The Objective-C 'Class' type.
      PREDEF_DECL_OBJC_CLASS_ID = 4,

      /// \brief The Objective-C 'Protocol' type.
      PREDEF_DECL_OBJC_PROTOCOL_ID = 5,

      /// \brief The signed 128-bit integer type.
      PREDEF_DECL_INT_128_ID = 6,

      /// \brief The unsigned 128-bit integer type.
      PREDEF_DECL_UNSIGNED_INT_128_ID = 7,

      /// \brief The internal 'instancetype' typedef.
      PREDEF_DECL_OBJC_INSTANCETYPE_ID = 8,

      /// \brief The internal '__builtin_va_list' typedef.
      PREDEF_DECL_BUILTIN_VA_LIST_ID = 9,

      /// \brief The internal '__va_list_tag' struct, if any.
      PREDEF_DECL_VA_LIST_TAG = 10,

      /// \brief The internal '__builtin_ms_va_list' typedef.
      PREDEF_DECL_BUILTIN_MS_VA_LIST_ID = 11,

      /// \brief The extern "C" context.
      PREDEF_DECL_EXTERN_C_CONTEXT_ID = 12,

      /// \brief The internal '__make_integer_seq' template.
      PREDEF_DECL_MAKE_INTEGER_SEQ_ID = 13,

      /// \brief The internal '__NSConstantString' typedef.
      PREDEF_DECL_CF_CONSTANT_STRING_ID = 14,

      /// \brief The internal '__NSConstantString' tag type.
      PREDEF_DECL_CF_CONSTANT_STRING_TAG_ID = 15,
    };

    /// \brief The number of declaration IDs that are predefined.
    ///
    /// For more information about predefined declarations, see the
    /// \c PredefinedDeclIDs type and the PREDEF_DECL_*_ID constants.
    const unsigned int NUM_PREDEF_DECL_IDS = 16;

    /// \brief Record code for a list of local redeclarations of a declaration.
    const unsigned int LOCAL_REDECLARATIONS = 50;
    
    /// \brief Record codes for each kind of declaration.
    ///
    /// These constants describe the declaration records that can occur within
    /// a declarations block (identified by DECLS_BLOCK_ID). Each
    /// constant describes a record for a specific declaration class
    /// in the AST.
    enum DeclCode {
      /// \brief A TypedefDecl record.
      DECL_TYPEDEF = 51,
      /// \brief A TypeAliasDecl record.
      DECL_TYPEALIAS,
      /// \brief An EnumDecl record.
      DECL_ENUM,
      /// \brief A RecordDecl record.
      DECL_RECORD,
      /// \brief An EnumConstantDecl record.
      DECL_ENUM_CONSTANT,
      /// \brief A FunctionDecl record.
      DECL_FUNCTION,
      /// \brief A ObjCMethodDecl record.
      DECL_OBJC_METHOD,
      /// \brief A ObjCInterfaceDecl record.
      DECL_OBJC_INTERFACE,
      /// \brief A ObjCProtocolDecl record.
      DECL_OBJC_PROTOCOL,
      /// \brief A ObjCIvarDecl record.
      DECL_OBJC_IVAR,
      /// \brief A ObjCAtDefsFieldDecl record.
      DECL_OBJC_AT_DEFS_FIELD,
      /// \brief A ObjCCategoryDecl record.
      DECL_OBJC_CATEGORY,
      /// \brief A ObjCCategoryImplDecl record.
      DECL_OBJC_CATEGORY_IMPL,
      /// \brief A ObjCImplementationDecl record.
      DECL_OBJC_IMPLEMENTATION,
      /// \brief A ObjCCompatibleAliasDecl record.
      DECL_OBJC_COMPATIBLE_ALIAS,
      /// \brief A ObjCPropertyDecl record.
      DECL_OBJC_PROPERTY,
      /// \brief A ObjCPropertyImplDecl record.
      DECL_OBJC_PROPERTY_IMPL,
      /// \brief A FieldDecl record.
      DECL_FIELD,
      /// \brief A MSPropertyDecl record.
      DECL_MS_PROPERTY,
      /// \brief A VarDecl record.
      DECL_VAR,
      /// \brief An ImplicitParamDecl record.
      DECL_IMPLICIT_PARAM,
      /// \brief A ParmVarDecl record.
      DECL_PARM_VAR,
      /// \brief A FileScopeAsmDecl record.
      DECL_FILE_SCOPE_ASM,
      /// \brief A BlockDecl record.
      DECL_BLOCK,
      /// \brief A CapturedDecl record.
      DECL_CAPTURED,
      /// \brief A record that stores the set of declarations that are
      /// lexically stored within a given DeclContext.
      ///
      /// The record itself is a blob that is an array of declaration IDs,
      /// in the order in which those declarations were added to the
      /// declaration context. This data is used when iterating over
      /// the contents of a DeclContext, e.g., via
      /// DeclContext::decls_begin() and DeclContext::decls_end().
      DECL_CONTEXT_LEXICAL,
      /// \brief A record that stores the set of declarations that are
      /// visible from a given DeclContext.
      ///
      /// The record itself stores a set of mappings, each of which
      /// associates a declaration name with one or more declaration
      /// IDs. This data is used when performing qualified name lookup
      /// into a DeclContext via DeclContext::lookup.
      DECL_CONTEXT_VISIBLE,
      /// \brief A LabelDecl record.
      DECL_LABEL,
      /// \brief A NamespaceDecl record.
      DECL_NAMESPACE,
      /// \brief A NamespaceAliasDecl record.
      DECL_NAMESPACE_ALIAS,
      /// \brief A UsingDecl record.
      DECL_USING,
      /// \brief A UsingShadowDecl record.
      DECL_USING_SHADOW,
      /// \brief A UsingDirecitveDecl record.
      DECL_USING_DIRECTIVE,
      /// \brief An UnresolvedUsingValueDecl record.
      DECL_UNRESOLVED_USING_VALUE,
      /// \brief An UnresolvedUsingTypenameDecl record.
      DECL_UNRESOLVED_USING_TYPENAME,
      /// \brief A LinkageSpecDecl record.
      DECL_LINKAGE_SPEC,
      /// \brief A CXXRecordDecl record.
      DECL_CXX_RECORD,
      /// \brief A CXXMethodDecl record.
      DECL_CXX_METHOD,
      /// \brief A CXXConstructorDecl record.
      DECL_CXX_CONSTRUCTOR,
      /// \brief A CXXDestructorDecl record.
      DECL_CXX_DESTRUCTOR,
      /// \brief A CXXConversionDecl record.
      DECL_CXX_CONVERSION,
      /// \brief An AccessSpecDecl record.
      DECL_ACCESS_SPEC,

      /// \brief A FriendDecl record.
      DECL_FRIEND,
      /// \brief A FriendTemplateDecl record.
      DECL_FRIEND_TEMPLATE,
      /// \brief A ClassTemplateDecl record.
      DECL_CLASS_TEMPLATE,
      /// \brief A ClassTemplateSpecializationDecl record.
      DECL_CLASS_TEMPLATE_SPECIALIZATION,
      /// \brief A ClassTemplatePartialSpecializationDecl record.
      DECL_CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
      /// \brief A VarTemplateDecl record.
      DECL_VAR_TEMPLATE,
      /// \brief A VarTemplateSpecializationDecl record.
      DECL_VAR_TEMPLATE_SPECIALIZATION,
      /// \brief A VarTemplatePartialSpecializationDecl record.
      DECL_VAR_TEMPLATE_PARTIAL_SPECIALIZATION,
      /// \brief A FunctionTemplateDecl record.
      DECL_FUNCTION_TEMPLATE,
      /// \brief A TemplateTypeParmDecl record.
      DECL_TEMPLATE_TYPE_PARM,
      /// \brief A NonTypeTemplateParmDecl record.
      DECL_NON_TYPE_TEMPLATE_PARM,
      /// \brief A TemplateTemplateParmDecl record.
      DECL_TEMPLATE_TEMPLATE_PARM,
      /// \brief A TypeAliasTemplateDecl record.
      DECL_TYPE_ALIAS_TEMPLATE,
      /// \brief A StaticAssertDecl record.
      DECL_STATIC_ASSERT,
      /// \brief A record containing CXXBaseSpecifiers.
      DECL_CXX_BASE_SPECIFIERS,
      /// \brief A record containing CXXCtorInitializers.
      DECL_CXX_CTOR_INITIALIZERS,
      /// \brief A IndirectFieldDecl record.
      DECL_INDIRECTFIELD,
      /// \brief A NonTypeTemplateParmDecl record that stores an expanded
      /// non-type template parameter pack.
      DECL_EXPANDED_NON_TYPE_TEMPLATE_PARM_PACK,
      /// \brief A TemplateTemplateParmDecl record that stores an expanded
      /// template template parameter pack.
      DECL_EXPANDED_TEMPLATE_TEMPLATE_PARM_PACK,
      /// \brief A ClassScopeFunctionSpecializationDecl record a class scope
      /// function specialization. (Microsoft extension).
      DECL_CLASS_SCOPE_FUNCTION_SPECIALIZATION,
      /// \brief An ImportDecl recording a module import.
      DECL_IMPORT,
      /// \brief An OMPThreadPrivateDecl record.
      DECL_OMP_THREADPRIVATE,
      /// \brief An EmptyDecl record.
      DECL_EMPTY,
      /// \brief An ObjCTypeParamDecl record.
      DECL_OBJC_TYPE_PARAM,
    };

    /// \brief Record codes for each kind of statement or expression.
    ///
    /// These constants describe the records that describe statements
    /// or expressions. These records  occur within type and declarations
    /// block, so they begin with record values of 128.  Each constant 
    /// describes a record for a specific statement or expression class in the
    /// AST.
    enum StmtCode {
      /// \brief A marker record that indicates that we are at the end
      /// of an expression.
      STMT_STOP = 128,
      /// \brief A NULL expression.
      STMT_NULL_PTR,
      /// \brief A reference to a previously [de]serialized Stmt record.
      STMT_REF_PTR,
      /// \brief A NullStmt record.
      STMT_NULL,
      /// \brief A CompoundStmt record.
      STMT_COMPOUND,
      /// \brief A CaseStmt record.
      STMT_CASE,
      /// \brief A DefaultStmt record.
      STMT_DEFAULT,
      /// \brief A LabelStmt record.
      STMT_LABEL,
      /// \brief An AttributedStmt record.
      STMT_ATTRIBUTED,
      /// \brief An IfStmt record.
      STMT_IF,
      /// \brief A SwitchStmt record.
      STMT_SWITCH,
      /// \brief A WhileStmt record.
      STMT_WHILE,
      /// \brief A DoStmt record.
      STMT_DO,
      /// \brief A ForStmt record.
      STMT_FOR,
      /// \brief A GotoStmt record.
      STMT_GOTO,
      /// \brief An IndirectGotoStmt record.
      STMT_INDIRECT_GOTO,
      /// \brief A ContinueStmt record.
      STMT_CONTINUE,
      /// \brief A BreakStmt record.
      STMT_BREAK,
      /// \brief A ReturnStmt record.
      STMT_RETURN,
      /// \brief A DeclStmt record.
      STMT_DECL,
      /// \brief A CapturedStmt record.
      STMT_CAPTURED,
      /// \brief A GCC-style AsmStmt record.
      STMT_GCCASM,
      /// \brief A MS-style AsmStmt record.
      STMT_MSASM,
      /// \brief A PredefinedExpr record.
      EXPR_PREDEFINED,
      /// \brief A DeclRefExpr record.
      EXPR_DECL_REF,
      /// \brief An IntegerLiteral record.
      EXPR_INTEGER_LITERAL,
      /// \brief A FloatingLiteral record.
      EXPR_FLOATING_LITERAL,
      /// \brief An ImaginaryLiteral record.
      EXPR_IMAGINARY_LITERAL,
      /// \brief A StringLiteral record.
      EXPR_STRING_LITERAL,
      /// \brief A CharacterLiteral record.
      EXPR_CHARACTER_LITERAL,
      /// \brief A ParenExpr record.
      EXPR_PAREN,
      /// \brief A ParenListExpr record.
      EXPR_PAREN_LIST,
      /// \brief A UnaryOperator record.
      EXPR_UNARY_OPERATOR,
      /// \brief An OffsetOfExpr record.
      EXPR_OFFSETOF,
      /// \brief A SizefAlignOfExpr record.
      EXPR_SIZEOF_ALIGN_OF,
      /// \brief An ArraySubscriptExpr record.
      EXPR_ARRAY_SUBSCRIPT,
      /// \brief A CallExpr record.
      EXPR_CALL,
      /// \brief A MemberExpr record.
      EXPR_MEMBER,
      /// \brief A BinaryOperator record.
      EXPR_BINARY_OPERATOR,
      /// \brief A CompoundAssignOperator record.
      EXPR_COMPOUND_ASSIGN_OPERATOR,
      /// \brief A ConditionOperator record.
      EXPR_CONDITIONAL_OPERATOR,
      /// \brief An ImplicitCastExpr record.
      EXPR_IMPLICIT_CAST,
      /// \brief A CStyleCastExpr record.
      EXPR_CSTYLE_CAST,
      /// \brief A CompoundLiteralExpr record.
      EXPR_COMPOUND_LITERAL,
      /// \brief An ExtVectorElementExpr record.
      EXPR_EXT_VECTOR_ELEMENT,
      /// \brief An InitListExpr record.
      EXPR_INIT_LIST,
      /// \brief A DesignatedInitExpr record.
      EXPR_DESIGNATED_INIT,
      /// \brief A DesignatedInitUpdateExpr record.
      EXPR_DESIGNATED_INIT_UPDATE,
      /// \brief An ImplicitValueInitExpr record.
      EXPR_IMPLICIT_VALUE_INIT,
      /// \brief An NoInitExpr record.
      EXPR_NO_INIT,
      /// \brief A VAArgExpr record.
      EXPR_VA_ARG,
      /// \brief An AddrLabelExpr record.
      EXPR_ADDR_LABEL,
      /// \brief A StmtExpr record.
      EXPR_STMT,
      /// \brief A ChooseExpr record.
      EXPR_CHOOSE,
      /// \brief A GNUNullExpr record.
      EXPR_GNU_NULL,
      /// \brief A ShuffleVectorExpr record.
      EXPR_SHUFFLE_VECTOR,
      /// \brief A ConvertVectorExpr record.
      EXPR_CONVERT_VECTOR,
      /// \brief BlockExpr
      EXPR_BLOCK,
      /// \brief A GenericSelectionExpr record.
      EXPR_GENERIC_SELECTION,
      /// \brief A PseudoObjectExpr record.
      EXPR_PSEUDO_OBJECT,
      /// \brief An AtomicExpr record.
      EXPR_ATOMIC,

      // Objective-C

      /// \brief An ObjCStringLiteral record.
      EXPR_OBJC_STRING_LITERAL,

      EXPR_OBJC_BOXED_EXPRESSION,
      EXPR_OBJC_ARRAY_LITERAL,
      EXPR_OBJC_DICTIONARY_LITERAL,

    
      /// \brief An ObjCEncodeExpr record.
      EXPR_OBJC_ENCODE,
      /// \brief An ObjCSelectorExpr record.
      EXPR_OBJC_SELECTOR_EXPR,
      /// \brief An ObjCProtocolExpr record.
      EXPR_OBJC_PROTOCOL_EXPR,
      /// \brief An ObjCIvarRefExpr record.
      EXPR_OBJC_IVAR_REF_EXPR,
      /// \brief An ObjCPropertyRefExpr record.
      EXPR_OBJC_PROPERTY_REF_EXPR,
      /// \brief An ObjCSubscriptRefExpr record.
      EXPR_OBJC_SUBSCRIPT_REF_EXPR,
      /// \brief UNUSED
      EXPR_OBJC_KVC_REF_EXPR,
      /// \brief An ObjCMessageExpr record.
      EXPR_OBJC_MESSAGE_EXPR,
      /// \brief An ObjCIsa Expr record.
      EXPR_OBJC_ISA,
      /// \brief An ObjCIndirectCopyRestoreExpr record.
      EXPR_OBJC_INDIRECT_COPY_RESTORE,

      /// \brief An ObjCForCollectionStmt record.
      STMT_OBJC_FOR_COLLECTION,
      /// \brief An ObjCAtCatchStmt record.
      STMT_OBJC_CATCH,
      /// \brief An ObjCAtFinallyStmt record.
      STMT_OBJC_FINALLY,
      /// \brief An ObjCAtTryStmt record.
      STMT_OBJC_AT_TRY,
      /// \brief An ObjCAtSynchronizedStmt record.
      STMT_OBJC_AT_SYNCHRONIZED,
      /// \brief An ObjCAtThrowStmt record.
      STMT_OBJC_AT_THROW,
      /// \brief An ObjCAutoreleasePoolStmt record.
      STMT_OBJC_AUTORELEASE_POOL,
      /// \brief A ObjCBoolLiteralExpr record.
      EXPR_OBJC_BOOL_LITERAL,

      // C++
      
      /// \brief A CXXCatchStmt record.
      STMT_CXX_CATCH,
      /// \brief A CXXTryStmt record.
      STMT_CXX_TRY,
      /// \brief A CXXForRangeStmt record.
      STMT_CXX_FOR_RANGE,

      /// \brief A CXXOperatorCallExpr record.
      EXPR_CXX_OPERATOR_CALL,
      /// \brief A CXXMemberCallExpr record.
      EXPR_CXX_MEMBER_CALL,
      /// \brief A CXXConstructExpr record.
      EXPR_CXX_CONSTRUCT,
      /// \brief A CXXTemporaryObjectExpr record.
      EXPR_CXX_TEMPORARY_OBJECT,
      /// \brief A CXXStaticCastExpr record.
      EXPR_CXX_STATIC_CAST,
      /// \brief A CXXDynamicCastExpr record.
      EXPR_CXX_DYNAMIC_CAST,
      /// \brief A CXXReinterpretCastExpr record.
      EXPR_CXX_REINTERPRET_CAST,
      /// \brief A CXXConstCastExpr record.
      EXPR_CXX_CONST_CAST,
      /// \brief A CXXFunctionalCastExpr record.
      EXPR_CXX_FUNCTIONAL_CAST,
      /// \brief A UserDefinedLiteral record.
      EXPR_USER_DEFINED_LITERAL,
      /// \brief A CXXStdInitializerListExpr record.
      EXPR_CXX_STD_INITIALIZER_LIST,
      /// \brief A CXXBoolLiteralExpr record.
      EXPR_CXX_BOOL_LITERAL,
      EXPR_CXX_NULL_PTR_LITERAL,  // CXXNullPtrLiteralExpr
      EXPR_CXX_TYPEID_EXPR,       // CXXTypeidExpr (of expr).
      EXPR_CXX_TYPEID_TYPE,       // CXXTypeidExpr (of type).
      EXPR_CXX_THIS,              // CXXThisExpr
      EXPR_CXX_THROW,             // CXXThrowExpr
      EXPR_CXX_DEFAULT_ARG,       // CXXDefaultArgExpr
      EXPR_CXX_DEFAULT_INIT,      // CXXDefaultInitExpr
      EXPR_CXX_BIND_TEMPORARY,    // CXXBindTemporaryExpr

      EXPR_CXX_SCALAR_VALUE_INIT, // CXXScalarValueInitExpr
      EXPR_CXX_NEW,               // CXXNewExpr
      EXPR_CXX_DELETE,            // CXXDeleteExpr
      EXPR_CXX_PSEUDO_DESTRUCTOR, // CXXPseudoDestructorExpr
      
      EXPR_EXPR_WITH_CLEANUPS,    // ExprWithCleanups
      
      EXPR_CXX_DEPENDENT_SCOPE_MEMBER,   // CXXDependentScopeMemberExpr
      EXPR_CXX_DEPENDENT_SCOPE_DECL_REF, // DependentScopeDeclRefExpr
      EXPR_CXX_UNRESOLVED_CONSTRUCT,     // CXXUnresolvedConstructExpr
      EXPR_CXX_UNRESOLVED_MEMBER,        // UnresolvedMemberExpr
      EXPR_CXX_UNRESOLVED_LOOKUP,        // UnresolvedLookupExpr

      EXPR_CXX_EXPRESSION_TRAIT,  // ExpressionTraitExpr
      EXPR_CXX_NOEXCEPT,          // CXXNoexceptExpr

      EXPR_OPAQUE_VALUE,          // OpaqueValueExpr
      EXPR_BINARY_CONDITIONAL_OPERATOR,  // BinaryConditionalOperator
      EXPR_TYPE_TRAIT,            // TypeTraitExpr
      EXPR_ARRAY_TYPE_TRAIT,      // ArrayTypeTraitIntExpr
      
      EXPR_PACK_EXPANSION,        // PackExpansionExpr
      EXPR_SIZEOF_PACK,           // SizeOfPackExpr
      EXPR_SUBST_NON_TYPE_TEMPLATE_PARM, // SubstNonTypeTemplateParmExpr
      EXPR_SUBST_NON_TYPE_TEMPLATE_PARM_PACK,// SubstNonTypeTemplateParmPackExpr
      EXPR_FUNCTION_PARM_PACK,    // FunctionParmPackExpr
      EXPR_MATERIALIZE_TEMPORARY, // MaterializeTemporaryExpr
      EXPR_CXX_FOLD,              // CXXFoldExpr

      // CUDA
      EXPR_CUDA_KERNEL_CALL,       // CUDAKernelCallExpr      

      // OpenCL
      EXPR_ASTYPE,                 // AsTypeExpr

      // Microsoft
      EXPR_CXX_PROPERTY_REF_EXPR, // MSPropertyRefExpr
      EXPR_CXX_PROPERTY_SUBSCRIPT_EXPR, // MSPropertySubscriptExpr
      EXPR_CXX_UUIDOF_EXPR,       // CXXUuidofExpr (of expr).
      EXPR_CXX_UUIDOF_TYPE,       // CXXUuidofExpr (of type).
      STMT_SEH_LEAVE,             // SEHLeaveStmt
      STMT_SEH_EXCEPT,            // SEHExceptStmt
      STMT_SEH_FINALLY,           // SEHFinallyStmt
      STMT_SEH_TRY,               // SEHTryStmt

      // OpenMP directives
      STMT_OMP_PARALLEL_DIRECTIVE,
      STMT_OMP_SIMD_DIRECTIVE,
      STMT_OMP_FOR_DIRECTIVE,
      STMT_OMP_FOR_SIMD_DIRECTIVE,
      STMT_OMP_SECTIONS_DIRECTIVE,
      STMT_OMP_SECTION_DIRECTIVE,
      STMT_OMP_SINGLE_DIRECTIVE,
      STMT_OMP_MASTER_DIRECTIVE,
      STMT_OMP_CRITICAL_DIRECTIVE,
      STMT_OMP_PARALLEL_FOR_DIRECTIVE,
      STMT_OMP_PARALLEL_FOR_SIMD_DIRECTIVE,
      STMT_OMP_PARALLEL_SECTIONS_DIRECTIVE,
      STMT_OMP_TASK_DIRECTIVE,
      STMT_OMP_TASKYIELD_DIRECTIVE,
      STMT_OMP_BARRIER_DIRECTIVE,
      STMT_OMP_TASKWAIT_DIRECTIVE,
      STMT_OMP_FLUSH_DIRECTIVE,
      STMT_OMP_ORDERED_DIRECTIVE,
      STMT_OMP_ATOMIC_DIRECTIVE,
      STMT_OMP_TARGET_DIRECTIVE,
      STMT_OMP_TARGET_DATA_DIRECTIVE,
      STMT_OMP_TARGET_ENTER_DATA_DIRECTIVE,
      STMT_OMP_TARGET_EXIT_DATA_DIRECTIVE,
      STMT_OMP_TARGET_PARALLEL_DIRECTIVE,
      STMT_OMP_TARGET_PARALLEL_FOR_DIRECTIVE,
      STMT_OMP_TEAMS_DIRECTIVE,
      STMT_OMP_TASKGROUP_DIRECTIVE,
      STMT_OMP_CANCELLATION_POINT_DIRECTIVE,
      STMT_OMP_CANCEL_DIRECTIVE,
      STMT_OMP_TASKLOOP_DIRECTIVE,
      STMT_OMP_TASKLOOP_SIMD_DIRECTIVE,
      STMT_OMP_DISTRIBUTE_DIRECTIVE,
      EXPR_OMP_ARRAY_SECTION,

      // ARC
      EXPR_OBJC_BRIDGED_CAST,     // ObjCBridgedCastExpr
      
      STMT_MS_DEPENDENT_EXISTS,   // MSDependentExistsStmt
      EXPR_LAMBDA                 // LambdaExpr
    };

    /// \brief The kinds of designators that can occur in a
    /// DesignatedInitExpr.
    enum DesignatorTypes {
      /// \brief Field designator where only the field name is known.
      DESIG_FIELD_NAME  = 0,
      /// \brief Field designator where the field has been resolved to
      /// a declaration.
      DESIG_FIELD_DECL  = 1,
      /// \brief Array designator.
      DESIG_ARRAY       = 2,
      /// \brief GNU array range designator.
      DESIG_ARRAY_RANGE = 3
    };

    /// \brief The different kinds of data that can occur in a
    /// CtorInitializer.
    enum CtorInitializerType {
      CTOR_INITIALIZER_BASE,
      CTOR_INITIALIZER_DELEGATING,
      CTOR_INITIALIZER_MEMBER,
      CTOR_INITIALIZER_INDIRECT_MEMBER
    };

    /// \brief Describes the redeclarations of a declaration.
    struct LocalRedeclarationsInfo {
      DeclID FirstID;      // The ID of the first declaration
      unsigned Offset;     // Offset into the array of redeclaration chains.
      
      friend bool operator<(const LocalRedeclarationsInfo &X,
                            const LocalRedeclarationsInfo &Y) {
        return X.FirstID < Y.FirstID;
      }
      
      friend bool operator>(const LocalRedeclarationsInfo &X,
                            const LocalRedeclarationsInfo &Y) {
        return X.FirstID > Y.FirstID;
      }
      
      friend bool operator<=(const LocalRedeclarationsInfo &X,
                             const LocalRedeclarationsInfo &Y) {
        return X.FirstID <= Y.FirstID;
      }
      
      friend bool operator>=(const LocalRedeclarationsInfo &X,
                             const LocalRedeclarationsInfo &Y) {
        return X.FirstID >= Y.FirstID;
      }
    };

    /// \brief Describes the categories of an Objective-C class.
    struct ObjCCategoriesInfo {
      DeclID DefinitionID; // The ID of the definition
      unsigned Offset;     // Offset into the array of category lists.
      
      friend bool operator<(const ObjCCategoriesInfo &X,
                            const ObjCCategoriesInfo &Y) {
        return X.DefinitionID < Y.DefinitionID;
      }
      
      friend bool operator>(const ObjCCategoriesInfo &X,
                            const ObjCCategoriesInfo &Y) {
        return X.DefinitionID > Y.DefinitionID;
      }
      
      friend bool operator<=(const ObjCCategoriesInfo &X,
                             const ObjCCategoriesInfo &Y) {
        return X.DefinitionID <= Y.DefinitionID;
      }
      
      friend bool operator>=(const ObjCCategoriesInfo &X,
                             const ObjCCategoriesInfo &Y) {
        return X.DefinitionID >= Y.DefinitionID;
      }
    };

    /// \brief A key used when looking up entities by \ref DeclarationName.
    ///
    /// Different \ref DeclarationNames are mapped to different keys, but the
    /// same key can occasionally represent multiple names (for names that
    /// contain types, in particular).
    class DeclarationNameKey {
      typedef unsigned NameKind;

      NameKind Kind;
      uint64_t Data;

    public:
      DeclarationNameKey() : Kind(), Data() {}
      DeclarationNameKey(DeclarationName Name);

      DeclarationNameKey(NameKind Kind, uint64_t Data)
          : Kind(Kind), Data(Data) {}

      NameKind getKind() const { return Kind; }

      IdentifierInfo *getIdentifier() const {
        assert(Kind == DeclarationName::Identifier ||
               Kind == DeclarationName::CXXLiteralOperatorName);
        return (IdentifierInfo *)Data;
      }
      Selector getSelector() const {
        assert(Kind == DeclarationName::ObjCZeroArgSelector ||
               Kind == DeclarationName::ObjCOneArgSelector ||
               Kind == DeclarationName::ObjCMultiArgSelector);
        return Selector(Data);
      }
      OverloadedOperatorKind getOperatorKind() const {
        assert(Kind == DeclarationName::CXXOperatorName);
        return (OverloadedOperatorKind)Data;
      }

      /// Compute a fingerprint of this key for use in on-disk hash table.
      unsigned getHash() const;

      friend bool operator==(const DeclarationNameKey &A,
                             const DeclarationNameKey &B) {
        return A.Kind == B.Kind && A.Data == B.Data;
      }
    };

    /// @}
  }
} // end namespace clang

namespace llvm {
  template <> struct DenseMapInfo<clang::serialization::DeclarationNameKey> {
    static clang::serialization::DeclarationNameKey getEmptyKey() {
      return clang::serialization::DeclarationNameKey(-1, 1);
    }
    static clang::serialization::DeclarationNameKey getTombstoneKey() {
      return clang::serialization::DeclarationNameKey(-1, 2);
    }
    static unsigned
    getHashValue(const clang::serialization::DeclarationNameKey &Key) {
      return Key.getHash();
    }
    static bool isEqual(const clang::serialization::DeclarationNameKey &L,
                        const clang::serialization::DeclarationNameKey &R) {
      return L == R;
    }
  };
}

#endif
