//===--- APINotesFormat.h - The internals of API notes files ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Contains various constants and helper types to deal with API notes
/// files.
///
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_API_NOTES_FORMAT_H
#define LLVM_CLANG_API_NOTES_FORMAT_H

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/RecordLayout.h"

namespace clang {
namespace api_notes {

using namespace llvm;

/// Magic number for API notes files.
const unsigned char API_NOTES_SIGNATURE[] = { 0xE2, 0x9C, 0xA8, 0x01 };

/// API notes file major version number.
///
const uint16_t VERSION_MAJOR = 0;

/// API notes file minor version number.
///
/// When the format changes IN ANY WAY, this number should be incremented.
const uint16_t VERSION_MINOR = 13;  // Function/method parameters

using IdentifierID = PointerEmbeddedInt<unsigned, 31>;
using IdentifierIDField = BCVBR<16>;

using SelectorID = PointerEmbeddedInt<unsigned, 31>;
using SelectorIDField = BCVBR<16>;

using StoredContextID = PointerEmbeddedInt<unsigned, 31>;

/// The various types of blocks that can occur within a API notes file.
///
/// These IDs must \em not be renumbered or reordered without incrementing
/// VERSION_MAJOR.
enum BlockID {
  /// The control block, which contains all of the information that needs to
  /// be validated prior to committing to loading the API notes file.
  ///
  /// \sa control_block
  CONTROL_BLOCK_ID = llvm::bitc::FIRST_APPLICATION_BLOCKID,

  /// The identifier data block, which maps identifier strings to IDs.
  IDENTIFIER_BLOCK_ID,

  /// The Objective-C class data block, which maps Objective-C class
  /// names to information about the class.
  OBJC_CONTEXT_BLOCK_ID,

  /// The Objective-C property data block, which maps Objective-C
  /// (class name, property name) pairs to information about the
  /// property.
  OBJC_PROPERTY_BLOCK_ID,

  /// The Objective-C property data block, which maps Objective-C
  /// (class name, selector, is_instance_method) tuples to information
  /// about the method.
  OBJC_METHOD_BLOCK_ID,

  /// The Objective-C selector data block, which maps Objective-C
  /// selector names (# of pieces, identifier IDs) to the selector ID
  /// used in other tables.
  OBJC_SELECTOR_BLOCK_ID,

  /// The global variables data block, which maps global variable names to
  /// information about the global variable.
  GLOBAL_VARIABLE_BLOCK_ID,

  /// The (global) functions data block, which maps global function names to
  /// information about the global function.
  GLOBAL_FUNCTION_BLOCK_ID,

  /// The tag data block, which maps tag names to information about
  /// the tags.
  TAG_BLOCK_ID,

  /// The typedef data block, which maps typedef names to information about
  /// the typedefs.
  TYPEDEF_BLOCK_ID,

  /// The enum constant data block, which maps enumerator names to
  /// information about the enumerators.
  ENUM_CONSTANT_BLOCK_ID,
};

namespace control_block {
  // These IDs must \em not be renumbered or reordered without incrementing
  // VERSION_MAJOR.
  enum {
    METADATA = 1,
    MODULE_NAME = 2,
    MODULE_OPTIONS = 3
  };

  using MetadataLayout = BCRecordLayout<
    METADATA, // ID
    BCFixed<16>, // Module format major version
    BCFixed<16>  // Module format minor version
  >;

  using ModuleNameLayout = BCRecordLayout<
    MODULE_NAME,
    BCBlob       // Module name
  >;

  using ModuleOptionsLayout = BCRecordLayout<
    MODULE_OPTIONS,
    BCFixed<1> // SwiftInferImportAsMember
  >;
}

namespace identifier_block {
  enum {
    IDENTIFIER_DATA = 1,
  };

  using IdentifierDataLayout = BCRecordLayout<
    IDENTIFIER_DATA,  // record ID
    BCVBR<16>,  // table offset within the blob (see below)
    BCBlob  // map from identifier strings to decl kinds / decl IDs
  >;
}

namespace objc_context_block {
  enum {
    OBJC_CONTEXT_DATA = 1,
  };

  using ObjCContextDataLayout = BCRecordLayout<
    OBJC_CONTEXT_DATA,  // record ID
    BCVBR<16>,  // table offset within the blob (see below)
    BCBlob  // map from ObjC class names (as IDs) to ObjC class information
  >;
}

namespace objc_property_block {
  enum {
    OBJC_PROPERTY_DATA = 1,
  };

  using ObjCPropertyDataLayout = BCRecordLayout<
    OBJC_PROPERTY_DATA,  // record ID
    BCVBR<16>,  // table offset within the blob (see below)
    BCBlob  // map from ObjC (class name, property name) pairs to ObjC
            // property information
  >;
}

namespace objc_method_block {
  enum {
    OBJC_METHOD_DATA = 1,
  };

  using ObjCMethodDataLayout = BCRecordLayout<
    OBJC_METHOD_DATA,  // record ID
    BCVBR<16>,  // table offset within the blob (see below)
    BCBlob  // map from ObjC (class names, selector,
            // is-instance-method) tuples to ObjC method information
  >;
}

namespace objc_selector_block {
  enum {
    OBJC_SELECTOR_DATA = 1,
  };

  using ObjCSelectorDataLayout = BCRecordLayout<
    OBJC_SELECTOR_DATA,  // record ID
    BCVBR<16>,  // table offset within the blob (see below)
    BCBlob  // map from (# pieces, identifier IDs) to Objective-C selector ID.
  >;
}

namespace global_variable_block {
  enum {
    GLOBAL_VARIABLE_DATA = 1
  };

  using GlobalVariableDataLayout = BCRecordLayout<
    GLOBAL_VARIABLE_DATA,  // record ID
    BCVBR<16>,  // table offset within the blob (see below)
    BCBlob  // map from name to global variable information
  >;
}

namespace global_function_block {
  enum {
    GLOBAL_FUNCTION_DATA = 1
  };

  using GlobalFunctionDataLayout = BCRecordLayout<
    GLOBAL_FUNCTION_DATA,  // record ID
    BCVBR<16>,  // table offset within the blob (see below)
    BCBlob  // map from name to global function information
  >;
}

namespace tag_block {
  enum {
    TAG_DATA = 1
  };

  using TagDataLayout = BCRecordLayout<
    TAG_DATA,   // record ID
    BCVBR<16>,  // table offset within the blob (see below)
    BCBlob      // map from name to tag information
  >;
};

namespace typedef_block {
  enum {
    TYPEDEF_DATA = 1
  };

  using TypedefDataLayout = BCRecordLayout<
    TYPEDEF_DATA,   // record ID
    BCVBR<16>,  // table offset within the blob (see below)
    BCBlob      // map from name to typedef information
  >;
};

namespace enum_constant_block {
  enum {
    ENUM_CONSTANT_DATA = 1
  };

  using EnumConstantDataLayout = BCRecordLayout<
    ENUM_CONSTANT_DATA,  // record ID
    BCVBR<16>,           // table offset within the blob (see below)
    BCBlob               // map from name to enumerator information
  >;
}

/// A stored Objective-C selector.
struct StoredObjCSelector {
  unsigned NumPieces;
  llvm::SmallVector<IdentifierID, 2> Identifiers;
};

} // end namespace api_notes
} // end namespace clang

namespace llvm {
  template<>
  struct DenseMapInfo<clang::api_notes::StoredObjCSelector> {
    typedef DenseMapInfo<unsigned> UnsignedInfo;

    static inline clang::api_notes::StoredObjCSelector getEmptyKey() {
      return clang::api_notes::StoredObjCSelector{ 
               UnsignedInfo::getEmptyKey(), { } };
    }

    static inline clang::api_notes::StoredObjCSelector getTombstoneKey() {
      return clang::api_notes::StoredObjCSelector{ 
               UnsignedInfo::getTombstoneKey(), { } };
    }
    
    static unsigned getHashValue(
                      const clang::api_notes::StoredObjCSelector& value) {
      auto hash = llvm::hash_value(value.NumPieces);
      hash = hash_combine(hash, value.Identifiers.size());
      for (auto piece : value.Identifiers)
        hash = hash_combine(hash, static_cast<unsigned>(piece));
      // FIXME: Mix upper/lower 32-bit values together to produce
      // unsigned rather than truncating.
      return hash;
    }

    static bool isEqual(const clang::api_notes::StoredObjCSelector &lhs, 
                        const clang::api_notes::StoredObjCSelector &rhs) {
      return lhs.NumPieces == rhs.NumPieces && 
             lhs.Identifiers == rhs.Identifiers;
    }
  };
}

#endif // LLVM_CLANG_API_NOTES_FORMAT_H
