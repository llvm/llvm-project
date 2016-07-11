//===--- APINotesReader.cpp - Side Car Reader --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the \c APINotesReader class that reads source
// API notes data providing additional information about source code as
// a separate input, such as the non-nil/nilable annotations for
// method parameters.
//
//===----------------------------------------------------------------------===//
#include "clang/APINotes/APINotesReader.h"
#include "APINotesFormat.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/OnDiskHashTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace api_notes;
using namespace llvm::support;
using namespace llvm;

namespace {
  /// Read serialized CommonEntityInfo.
  void readCommonEntityInfo(const uint8_t *&data, CommonEntityInfo &info) {
    uint8_t unavailableBits = *data++;
    info.Unavailable = (unavailableBits >> 1) & 0x01;
    info.UnavailableInSwift = unavailableBits & 0x01;
    info.SwiftPrivate = (unavailableBits >> 2) & 0x01;

    unsigned msgLength = endian::readNext<uint16_t, little, unaligned>(data);
    info.UnavailableMsg
      = std::string(reinterpret_cast<const char *>(data),
                    reinterpret_cast<const char *>(data) + msgLength);
    data += msgLength;

    unsigned swiftNameLength
      = endian::readNext<uint16_t, little, unaligned>(data);
    info.SwiftName
      = std::string(reinterpret_cast<const char *>(data),
                    reinterpret_cast<const char *>(data) + swiftNameLength);
    data += swiftNameLength;
  }

  /// Read serialized CommonTypeInfo.
  void readCommonTypeInfo(const uint8_t *&data, CommonTypeInfo &info) {
    readCommonEntityInfo(data, info);

    unsigned swiftBridgeLength =
        endian::readNext<uint16_t, little, unaligned>(data);
    info.setSwiftBridge(
        StringRef(reinterpret_cast<const char *>(data), swiftBridgeLength));
    data += swiftBridgeLength;

    unsigned errorDomainLength =
      endian::readNext<uint16_t, little, unaligned>(data);
    info.setNSErrorDomain(
        StringRef(reinterpret_cast<const char *>(data), errorDomainLength));
    data += errorDomainLength;
  }

  /// Used to deserialize the on-disk identifier table.
  class IdentifierTableInfo {
  public:
    using internal_key_type = StringRef;
    using external_key_type = StringRef;
    using data_type = IdentifierID;
    using hash_value_type = uint32_t;
    using offset_type = unsigned;

    internal_key_type GetInternalKey(external_key_type key) {
      return key;
    }

    external_key_type GetExternalKey(internal_key_type key) {
      return key;
    }

    hash_value_type ComputeHash(internal_key_type key) {
      return llvm::HashString(key);
    }
    
    static bool EqualKey(internal_key_type lhs, internal_key_type rhs) {
      return lhs == rhs;
    }
    
    static std::pair<unsigned, unsigned> 
    ReadKeyDataLength(const uint8_t *&data) {
      unsigned keyLength = endian::readNext<uint16_t, little, unaligned>(data);
      unsigned dataLength = endian::readNext<uint16_t, little, unaligned>(data);
      return { keyLength, dataLength };
    }
    
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      return StringRef(reinterpret_cast<const char *>(data), length);
    }
    
    static data_type ReadData(internal_key_type key, const uint8_t *data,
                              unsigned length) {
      return endian::readNext<uint32_t, little, unaligned>(data);
    }
  };

  /// Used to deserialize the on-disk Objective-C class table.
  class ObjCContextTableInfo {
  public:
    // identifier ID, is-protocol
    using internal_key_type = std::pair<unsigned, char>;
    using external_key_type = internal_key_type;
    using data_type = std::pair<unsigned, ObjCContextInfo>;
    using hash_value_type = size_t;
    using offset_type = unsigned;

    internal_key_type GetInternalKey(external_key_type key) {
      return key;
    }

    external_key_type GetExternalKey(internal_key_type key) {
      return key;
    }

    hash_value_type ComputeHash(internal_key_type key) {
      return static_cast<size_t>(llvm::hash_value(key));
    }
    
    static bool EqualKey(internal_key_type lhs, internal_key_type rhs) {
      return lhs == rhs;
    }
    
    static std::pair<unsigned, unsigned> 
    ReadKeyDataLength(const uint8_t *&data) {
      unsigned keyLength = endian::readNext<uint16_t, little, unaligned>(data);
      unsigned dataLength = endian::readNext<uint16_t, little, unaligned>(data);
      return { keyLength, dataLength };
    }
    
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto nameID
        = endian::readNext<uint32_t, little, unaligned>(data);
      auto isProtocol = endian::readNext<uint8_t, little, unaligned>(data);
      return { nameID, isProtocol };
    }
    
    static data_type ReadData(internal_key_type key, const uint8_t *data,
                              unsigned length) {
      data_type result;
      result.first = endian::readNext<uint32_t, little, unaligned>(data);
      readCommonTypeInfo(data, result.second);
      if (*data++) {
        result.second.setDefaultNullability(static_cast<NullabilityKind>(*data));
      }
      ++data;
      result.second.setHasDesignatedInits(*data++);
                                             
      return result;
    }
  };

  /// Read serialized VariableInfo.
  void readVariableInfo(const uint8_t *&data, VariableInfo &info) {
    readCommonEntityInfo(data, info);
    if (*data++) {
      info.setNullabilityAudited(static_cast<NullabilityKind>(*data));
    }
    ++data;
  }

  /// Used to deserialize the on-disk Objective-C property table.
  class ObjCPropertyTableInfo {
  public:
    // (context ID, name ID)
    using internal_key_type = std::pair<unsigned, unsigned>; 
    using external_key_type = internal_key_type;
    using data_type = ObjCPropertyInfo;
    using hash_value_type = size_t;
    using offset_type = unsigned;

    internal_key_type GetInternalKey(external_key_type key) {
      return key;
    }

    external_key_type GetExternalKey(internal_key_type key) {
      return key;
    }

    hash_value_type ComputeHash(internal_key_type key) {
      return static_cast<size_t>(llvm::hash_value(key));
    }
    
    static bool EqualKey(internal_key_type lhs, internal_key_type rhs) {
      return lhs == rhs;
    }
    
    static std::pair<unsigned, unsigned> 
    ReadKeyDataLength(const uint8_t *&data) {
      unsigned keyLength = endian::readNext<uint16_t, little, unaligned>(data);
      unsigned dataLength = endian::readNext<uint16_t, little, unaligned>(data);
      return { keyLength, dataLength };
    }
    
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto classID = endian::readNext<uint32_t, little, unaligned>(data);
      auto nameID = endian::readNext<uint32_t, little, unaligned>(data);
      return { classID, nameID };
    }
    
    static data_type ReadData(internal_key_type key, const uint8_t *data,
                              unsigned length) {
      ObjCPropertyInfo info;
      readVariableInfo(data, info);
      return info;
    }
  };

  /// Read serialized FunctionInfo.
  void readFunctionInfo(const uint8_t *&data, FunctionInfo &info) {
    readCommonEntityInfo(data, info);
    info.NullabilityAudited
      = endian::readNext<uint8_t, little, unaligned>(data);
    info.NumAdjustedNullable
      = endian::readNext<uint8_t, little, unaligned>(data);
    info.NullabilityPayload
      = endian::readNext<uint64_t, little, unaligned>(data);

    unsigned numParams = endian::readNext<uint16_t, little, unaligned>(data);
    while (numParams > 0) {
      uint8_t payload = endian::readNext<uint8_t, little, unaligned>(data);

      ParamInfo pi;
      uint8_t nullabilityValue = payload & 0x3; payload >>= 2;
      if (payload & 0x01)
        pi.setNullabilityAudited(static_cast<NullabilityKind>(nullabilityValue));
      payload >>= 1;
      pi.setNoEscape(payload & 0x01);
      payload >>= 1; assert(payload == 0 && "Bad API notes");

      info.Params.push_back(pi);
      --numParams;
    }
  }

  /// Used to deserialize the on-disk Objective-C method table.
  class ObjCMethodTableInfo {
  public:
    // (class ID, selector ID, is-instance)
    using internal_key_type = std::tuple<unsigned, unsigned, char>; 
    using external_key_type = internal_key_type;
    using data_type = ObjCMethodInfo;
    using hash_value_type = size_t;
    using offset_type = unsigned;

    internal_key_type GetInternalKey(external_key_type key) {
      return key;
    }

    external_key_type GetExternalKey(internal_key_type key) {
      return key;
    }

    hash_value_type ComputeHash(internal_key_type key) {
      return llvm::hash_combine(std::get<0>(key), 
                                std::get<1>(key), 
                                std::get<2>(key));
    }
    
    static bool EqualKey(internal_key_type lhs, internal_key_type rhs) {
      return lhs == rhs;
    }
    
    static std::pair<unsigned, unsigned> 
    ReadKeyDataLength(const uint8_t *&data) {
      unsigned keyLength = endian::readNext<uint16_t, little, unaligned>(data);
      unsigned dataLength = endian::readNext<uint16_t, little, unaligned>(data);
      return { keyLength, dataLength };
    }
    
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto classID = endian::readNext<uint32_t, little, unaligned>(data);
      auto selectorID = endian::readNext<uint32_t, little, unaligned>(data);
      auto isInstance = endian::readNext<uint8_t, little, unaligned>(data);
      return internal_key_type{ classID, selectorID, isInstance };
    }
    
    static data_type ReadData(internal_key_type key, const uint8_t *data,
                              unsigned length) {
      ObjCMethodInfo info;
      readFunctionInfo(data, info);
      info.DesignatedInit = endian::readNext<uint8_t, little, unaligned>(data);
      info.FactoryAsInit = endian::readNext<uint8_t, little, unaligned>(data);
      info.Required = endian::readNext<uint8_t, little, unaligned>(data);
      return info;
    }
  };

  /// Used to deserialize the on-disk Objective-C selector table.
  class ObjCSelectorTableInfo {
  public:
    using internal_key_type = StoredObjCSelector; 
    using external_key_type = internal_key_type;
    using data_type = SelectorID;
    using hash_value_type = unsigned;
    using offset_type = unsigned;

    internal_key_type GetInternalKey(external_key_type key) {
      return key;
    }

    external_key_type GetExternalKey(internal_key_type key) {
      return key;
    }

    hash_value_type ComputeHash(internal_key_type key) {
      return llvm::DenseMapInfo<StoredObjCSelector>::getHashValue(key);
    }
    
    static bool EqualKey(internal_key_type lhs, internal_key_type rhs) {
      return llvm::DenseMapInfo<StoredObjCSelector>::isEqual(lhs, rhs);
    }
    
    static std::pair<unsigned, unsigned> 
    ReadKeyDataLength(const uint8_t *&data) {
      unsigned keyLength = endian::readNext<uint16_t, little, unaligned>(data);
      unsigned dataLength = endian::readNext<uint16_t, little, unaligned>(data);
      return { keyLength, dataLength };
    }
    
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      internal_key_type key;
      key.NumPieces = endian::readNext<uint16_t, little, unaligned>(data);
      unsigned numIdents = (length - sizeof(uint16_t)) / sizeof(uint32_t);
      for (unsigned i = 0; i != numIdents; ++i) {
        key.Identifiers.push_back(
          endian::readNext<uint32_t, little, unaligned>(data));
      }
      return key;
    }
    
    static data_type ReadData(internal_key_type key, const uint8_t *data,
                              unsigned length) {
      return endian::readNext<uint32_t, little, unaligned>(data);
    }
  };

  /// Used to deserialize the on-disk global variable table.
  class GlobalVariableTableInfo {
  public:
    using internal_key_type = unsigned; // name ID
    using external_key_type = internal_key_type;
    using data_type = GlobalVariableInfo;
    using hash_value_type = size_t;
    using offset_type = unsigned;

    internal_key_type GetInternalKey(external_key_type key) {
      return key;
    }

    external_key_type GetExternalKey(internal_key_type key) {
      return key;
    }

    hash_value_type ComputeHash(internal_key_type key) {
      return static_cast<size_t>(llvm::hash_value(key));
    }
    
    static bool EqualKey(internal_key_type lhs, internal_key_type rhs) {
      return lhs == rhs;
    }
    
    static std::pair<unsigned, unsigned> 
    ReadKeyDataLength(const uint8_t *&data) {
      unsigned keyLength = endian::readNext<uint16_t, little, unaligned>(data);
      unsigned dataLength = endian::readNext<uint16_t, little, unaligned>(data);
      return { keyLength, dataLength };
    }
    
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto nameID = endian::readNext<uint32_t, little, unaligned>(data);
      return nameID;
    }
    
    static data_type ReadData(internal_key_type key, const uint8_t *data,
                              unsigned length) {
      GlobalVariableInfo info;
      readVariableInfo(data, info);
      return info;
    }
  };

  /// Used to deserialize the on-disk global function table.
  class GlobalFunctionTableInfo {
  public:
    using internal_key_type = unsigned; // name ID
    using external_key_type = internal_key_type;
    using data_type = GlobalFunctionInfo;
    using hash_value_type = size_t;
    using offset_type = unsigned;

    internal_key_type GetInternalKey(external_key_type key) {
      return key;
    }

    external_key_type GetExternalKey(internal_key_type key) {
      return key;
    }

    hash_value_type ComputeHash(internal_key_type key) {
      return static_cast<size_t>(llvm::hash_value(key));
    }
    
    static bool EqualKey(internal_key_type lhs, internal_key_type rhs) {
      return lhs == rhs;
    }
    
    static std::pair<unsigned, unsigned> 
    ReadKeyDataLength(const uint8_t *&data) {
      unsigned keyLength = endian::readNext<uint16_t, little, unaligned>(data);
      unsigned dataLength = endian::readNext<uint16_t, little, unaligned>(data);
      return { keyLength, dataLength };
    }
    
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto nameID = endian::readNext<uint32_t, little, unaligned>(data);
      return nameID;
    }
    
    static data_type ReadData(internal_key_type key, const uint8_t *data,
                              unsigned length) {
      GlobalFunctionInfo info;
      readFunctionInfo(data, info);
      return info;
    }
  };

  /// Used to deserialize the on-disk enumerator table.
  class EnumConstantTableInfo {
  public:
    using internal_key_type = unsigned; // name ID
    using external_key_type = internal_key_type;
    using data_type = EnumConstantInfo;
    using hash_value_type = size_t;
    using offset_type = unsigned;

    internal_key_type GetInternalKey(external_key_type key) {
      return key;
    }

    external_key_type GetExternalKey(internal_key_type key) {
      return key;
    }

    hash_value_type ComputeHash(internal_key_type key) {
      return static_cast<size_t>(llvm::hash_value(key));
    }
    
    static bool EqualKey(internal_key_type lhs, internal_key_type rhs) {
      return lhs == rhs;
    }
    
    static std::pair<unsigned, unsigned> 
    ReadKeyDataLength(const uint8_t *&data) {
      unsigned keyLength = endian::readNext<uint16_t, little, unaligned>(data);
      unsigned dataLength = endian::readNext<uint16_t, little, unaligned>(data);
      return { keyLength, dataLength };
    }
    
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto nameID = endian::readNext<uint32_t, little, unaligned>(data);
      return nameID;
    }
    
    static data_type ReadData(internal_key_type key, const uint8_t *data,
                              unsigned length) {
      EnumConstantInfo info;
      readCommonEntityInfo(data, info);
      return info;
    }
  };

  /// Used to deserialize the on-disk tag table.
  class TagTableInfo {
  public:
    using internal_key_type = unsigned; // name ID
    using external_key_type = internal_key_type;
    using data_type = TagInfo;
    using hash_value_type = size_t;
    using offset_type = unsigned;

    internal_key_type GetInternalKey(external_key_type key) {
      return key;
    }

    external_key_type GetExternalKey(internal_key_type key) {
      return key;
    }

    hash_value_type ComputeHash(internal_key_type key) {
      return static_cast<size_t>(llvm::hash_value(key));
    }
    
    static bool EqualKey(internal_key_type lhs, internal_key_type rhs) {
      return lhs == rhs;
    }
    
    static std::pair<unsigned, unsigned> 
    ReadKeyDataLength(const uint8_t *&data) {
      unsigned keyLength = endian::readNext<uint16_t, little, unaligned>(data);
      unsigned dataLength = endian::readNext<uint16_t, little, unaligned>(data);
      return { keyLength, dataLength };
    }
    
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto nameID = endian::readNext<IdentifierID, little, unaligned>(data);
      return nameID;
    }
    
    static data_type ReadData(internal_key_type key, const uint8_t *data,
                              unsigned length) {
      TagInfo info;
      readCommonTypeInfo(data, info);
      return info;
    }
  };

  /// Used to deserialize the on-disk typedef table.
  class TypedefTableInfo {
  public:
    using internal_key_type = unsigned; // name ID
    using external_key_type = internal_key_type;
    using data_type = TypedefInfo;
    using hash_value_type = size_t;
    using offset_type = unsigned;

    internal_key_type GetInternalKey(external_key_type key) {
      return key;
    }

    external_key_type GetExternalKey(internal_key_type key) {
      return key;
    }

    hash_value_type ComputeHash(internal_key_type key) {
      return static_cast<size_t>(llvm::hash_value(key));
    }
    
    static bool EqualKey(internal_key_type lhs, internal_key_type rhs) {
      return lhs == rhs;
    }
    
    static std::pair<unsigned, unsigned> 
    ReadKeyDataLength(const uint8_t *&data) {
      unsigned keyLength = endian::readNext<uint16_t, little, unaligned>(data);
      unsigned dataLength = endian::readNext<uint16_t, little, unaligned>(data);
      return { keyLength, dataLength };
    }
    
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto nameID = endian::readNext<IdentifierID, little, unaligned>(data);
      return nameID;
    }
    
    static data_type ReadData(internal_key_type key, const uint8_t *data,
                              unsigned length) {
      TypedefInfo info;
      readCommonTypeInfo(data, info);
      return info;
    }
  };
} // end anonymous namespace

class APINotesReader::Implementation {
public:
  /// The input buffer for the API notes data.
  std::unique_ptr<llvm::MemoryBuffer> InputBuffer;

  /// The reader attached to \c InputBuffer.
  llvm::BitstreamReader InputReader;

  /// The name of the module that we read from the control block.
  std::string ModuleName;

  /// Various options and attributes for the module
  ModuleOptions ModuleOpts;

  using SerializedIdentifierTable =
      llvm::OnDiskIterableChainedHashTable<IdentifierTableInfo>;

  /// The identifier table.
  std::unique_ptr<SerializedIdentifierTable> IdentifierTable;

  using SerializedObjCContextTable =
      llvm::OnDiskIterableChainedHashTable<ObjCContextTableInfo>;

  /// The Objective-C context table.
  std::unique_ptr<SerializedObjCContextTable> ObjCContextTable;

  using SerializedObjCPropertyTable =
      llvm::OnDiskIterableChainedHashTable<ObjCPropertyTableInfo>;

  /// The Objective-C property table.
  std::unique_ptr<SerializedObjCPropertyTable> ObjCPropertyTable;

  using SerializedObjCMethodTable =
      llvm::OnDiskIterableChainedHashTable<ObjCMethodTableInfo>;

  /// The Objective-C method table.
  std::unique_ptr<SerializedObjCMethodTable> ObjCMethodTable;

  using SerializedObjCSelectorTable =
      llvm::OnDiskIterableChainedHashTable<ObjCSelectorTableInfo>;

  /// The Objective-C selector table.
  std::unique_ptr<SerializedObjCSelectorTable> ObjCSelectorTable;

  using SerializedGlobalVariableTable =
      llvm::OnDiskIterableChainedHashTable<GlobalVariableTableInfo>;

  /// The global variable table.
  std::unique_ptr<SerializedGlobalVariableTable> GlobalVariableTable;

  using SerializedGlobalFunctionTable =
      llvm::OnDiskIterableChainedHashTable<GlobalFunctionTableInfo>;

  /// The global function table.
  std::unique_ptr<SerializedGlobalFunctionTable> GlobalFunctionTable;

  using SerializedEnumConstantTable =
      llvm::OnDiskIterableChainedHashTable<EnumConstantTableInfo>;

  /// The enumerator table.
  std::unique_ptr<SerializedEnumConstantTable> EnumConstantTable;

  using SerializedTagTable =
      llvm::OnDiskIterableChainedHashTable<TagTableInfo>;

  /// The tag table.
  std::unique_ptr<SerializedTagTable> TagTable;

  using SerializedTypedefTable =
      llvm::OnDiskIterableChainedHashTable<TypedefTableInfo>;

  /// The typedef table.
  std::unique_ptr<SerializedTypedefTable> TypedefTable;

  /// Retrieve the identifier ID for the given string, or an empty
  /// optional if the string is unknown.
  Optional<IdentifierID> getIdentifier(StringRef str);

  /// Retrieve the selector ID for the given selector, or an empty
  /// optional if the string is unknown.
  Optional<SelectorID> getSelector(ObjCSelectorRef selector);

  bool readControlBlock(llvm::BitstreamCursor &cursor, 
                        SmallVectorImpl<uint64_t> &scratch);
  bool readIdentifierBlock(llvm::BitstreamCursor &cursor,
                           SmallVectorImpl<uint64_t> &scratch);
  bool readObjCContextBlock(llvm::BitstreamCursor &cursor,
                            SmallVectorImpl<uint64_t> &scratch);
  bool readObjCPropertyBlock(llvm::BitstreamCursor &cursor, 
                             SmallVectorImpl<uint64_t> &scratch);
  bool readObjCMethodBlock(llvm::BitstreamCursor &cursor, 
                             SmallVectorImpl<uint64_t> &scratch);
  bool readObjCSelectorBlock(llvm::BitstreamCursor &cursor, 
                             SmallVectorImpl<uint64_t> &scratch);
  bool readGlobalVariableBlock(llvm::BitstreamCursor &cursor,
                               SmallVectorImpl<uint64_t> &scratch);
  bool readGlobalFunctionBlock(llvm::BitstreamCursor &cursor,
                               SmallVectorImpl<uint64_t> &scratch);
  bool readEnumConstantBlock(llvm::BitstreamCursor &cursor,
                             SmallVectorImpl<uint64_t> &scratch);
  bool readTagBlock(llvm::BitstreamCursor &cursor,
                    SmallVectorImpl<uint64_t> &scratch);
  bool readTypedefBlock(llvm::BitstreamCursor &cursor,
                        SmallVectorImpl<uint64_t> &scratch);
};

Optional<IdentifierID> APINotesReader::Implementation::getIdentifier(
                         StringRef str) {
  if (!IdentifierTable)
    return None;

  if (str.empty())
    return IdentifierID(0);

  auto known = IdentifierTable->find(str);
  if (known == IdentifierTable->end())
    return None;

  return *known;
}

Optional<SelectorID> APINotesReader::Implementation::getSelector(
                       ObjCSelectorRef selector) {
  if (!ObjCSelectorTable || !IdentifierTable)
    return None;

  // Translate the identifiers.
  StoredObjCSelector key;
  key.NumPieces = selector.NumPieces;
  for (auto ident : selector.Identifiers) {
    if (auto identID = getIdentifier(ident)) {
      key.Identifiers.push_back(*identID);
    } else {
      return None;
    }
  }

  auto known = ObjCSelectorTable->find(key);
  if (known == ObjCSelectorTable->end())
    return None;

  return *known;

}

bool APINotesReader::Implementation::readControlBlock(
       llvm::BitstreamCursor &cursor,
       SmallVectorImpl<uint64_t> &scratch) {
  if (cursor.EnterSubBlock(CONTROL_BLOCK_ID))
    return true;

  bool sawMetadata = false;
  
  auto next = cursor.advance();
  while (next.Kind != llvm::BitstreamEntry::EndBlock) {
    if (next.Kind == llvm::BitstreamEntry::Error)
      return true;

    if (next.Kind == llvm::BitstreamEntry::SubBlock) {
      // Unknown metadata sub-block, possibly for use by a future version of the
      // API notes format.
      if (cursor.SkipBlock())
        return true;
      
      next = cursor.advance();
      continue;
    }

    scratch.clear();
    StringRef blobData;
    unsigned kind = cursor.readRecord(next.ID, scratch, &blobData);
    switch (kind) {
    case control_block::METADATA:
      // Already saw metadata.
      if (sawMetadata)
        return true;

      if (scratch[0] != VERSION_MAJOR || scratch[1] != VERSION_MINOR)
        return true;

      sawMetadata = true;
      break;

    case control_block::MODULE_NAME:
      ModuleName = blobData.str();
      break;

    case control_block::MODULE_OPTIONS:
      ModuleOpts.SwiftInferImportAsMember = (scratch.front() & 1) != 0;
      break;

    default:
      // Unknown metadata record, possibly for use by a future version of the
      // module format.
      break;
    }

    next = cursor.advance();
  }

  return !sawMetadata;
}

bool APINotesReader::Implementation::readIdentifierBlock(
       llvm::BitstreamCursor &cursor,
       SmallVectorImpl<uint64_t> &scratch) {
  if (cursor.EnterSubBlock(IDENTIFIER_BLOCK_ID))
    return true;

  auto next = cursor.advance();
  while (next.Kind != llvm::BitstreamEntry::EndBlock) {
    if (next.Kind == llvm::BitstreamEntry::Error)
      return true;

    if (next.Kind == llvm::BitstreamEntry::SubBlock) {
      // Unknown sub-block, possibly for use by a future version of the
      // API notes format.
      if (cursor.SkipBlock())
        return true;
      
      next = cursor.advance();
      continue;
    }

    scratch.clear();
    StringRef blobData;
    unsigned kind = cursor.readRecord(next.ID, scratch, &blobData);
    switch (kind) {
    case identifier_block::IDENTIFIER_DATA: {
      // Already saw identifier table.
      if (IdentifierTable)
        return true;

      uint32_t tableOffset;
      identifier_block::IdentifierDataLayout::readRecord(scratch, tableOffset);
      auto base = reinterpret_cast<const uint8_t *>(blobData.data());

      IdentifierTable.reset(
        SerializedIdentifierTable::Create(base + tableOffset,
                                          base + sizeof(uint32_t),
                                          base));
      break;
    }

    default:
      // Unknown record, possibly for use by a future version of the
      // module format.
      break;
    }

    next = cursor.advance();
  }

  return false;
}

bool APINotesReader::Implementation::readObjCContextBlock(
       llvm::BitstreamCursor &cursor,
       SmallVectorImpl<uint64_t> &scratch) {
  if (cursor.EnterSubBlock(OBJC_CONTEXT_BLOCK_ID))
    return true;

  auto next = cursor.advance();
  while (next.Kind != llvm::BitstreamEntry::EndBlock) {
    if (next.Kind == llvm::BitstreamEntry::Error)
      return true;

    if (next.Kind == llvm::BitstreamEntry::SubBlock) {
      // Unknown sub-block, possibly for use by a future version of the
      // API notes format.
      if (cursor.SkipBlock())
        return true;
      
      next = cursor.advance();
      continue;
    }

    scratch.clear();
    StringRef blobData;
    unsigned kind = cursor.readRecord(next.ID, scratch, &blobData);
    switch (kind) {
    case objc_context_block::OBJC_CONTEXT_DATA: {
      // Already saw Objective-C class table.
      if (ObjCContextTable)
        return true;

      uint32_t tableOffset;
      objc_context_block::ObjCContextDataLayout::readRecord(scratch, tableOffset);
      auto base = reinterpret_cast<const uint8_t *>(blobData.data());

      ObjCContextTable.reset(
        SerializedObjCContextTable::Create(base + tableOffset,
                                         base + sizeof(uint32_t),
                                         base));
      break;
    }

    default:
      // Unknown record, possibly for use by a future version of the
      // module format.
      break;
    }

    next = cursor.advance();
  }

  return false;
}

bool APINotesReader::Implementation::readObjCPropertyBlock(
       llvm::BitstreamCursor &cursor, 
       SmallVectorImpl<uint64_t> &scratch) {
  if (cursor.EnterSubBlock(OBJC_PROPERTY_BLOCK_ID))
    return true;

  auto next = cursor.advance();
  while (next.Kind != llvm::BitstreamEntry::EndBlock) {
    if (next.Kind == llvm::BitstreamEntry::Error)
      return true;

    if (next.Kind == llvm::BitstreamEntry::SubBlock) {
      // Unknown sub-block, possibly for use by a future version of the
      // API notes format.
      if (cursor.SkipBlock())
        return true;
      
      next = cursor.advance();
      continue;
    }

    scratch.clear();
    StringRef blobData;
    unsigned kind = cursor.readRecord(next.ID, scratch, &blobData);
    switch (kind) {
    case objc_property_block::OBJC_PROPERTY_DATA: {
      // Already saw Objective-C property table.
      if (ObjCPropertyTable)
        return true;

      uint32_t tableOffset;
      objc_property_block::ObjCPropertyDataLayout::readRecord(scratch, 
                                                              tableOffset);
      auto base = reinterpret_cast<const uint8_t *>(blobData.data());

      ObjCPropertyTable.reset(
        SerializedObjCPropertyTable::Create(base + tableOffset,
                                            base + sizeof(uint32_t),
                                            base));
      break;
    }

    default:
      // Unknown record, possibly for use by a future version of the
      // module format.
      break;
    }

    next = cursor.advance();
  }

  return false;
}

bool APINotesReader::Implementation::readObjCMethodBlock(
       llvm::BitstreamCursor &cursor, 
       SmallVectorImpl<uint64_t> &scratch) {
  if (cursor.EnterSubBlock(OBJC_METHOD_BLOCK_ID))
    return true;

  auto next = cursor.advance();
  while (next.Kind != llvm::BitstreamEntry::EndBlock) {
    if (next.Kind == llvm::BitstreamEntry::Error)
      return true;

    if (next.Kind == llvm::BitstreamEntry::SubBlock) {
      // Unknown sub-block, possibly for use by a future version of the
      // API notes format.
      if (cursor.SkipBlock())
        return true;
      
      next = cursor.advance();
      continue;
    }

    scratch.clear();
    StringRef blobData;
    unsigned kind = cursor.readRecord(next.ID, scratch, &blobData);
    switch (kind) {
    case objc_method_block::OBJC_METHOD_DATA: {
      // Already saw Objective-C method table.
      if (ObjCMethodTable)
        return true;

      uint32_t tableOffset;
      objc_method_block::ObjCMethodDataLayout::readRecord(scratch, tableOffset);
      auto base = reinterpret_cast<const uint8_t *>(blobData.data());

      ObjCMethodTable.reset(
        SerializedObjCMethodTable::Create(base + tableOffset,
                                          base + sizeof(uint32_t),
                                          base));
      break;
    }

    default:
      // Unknown record, possibly for use by a future version of the
      // module format.
      break;
    }

    next = cursor.advance();
  }

  return false;
}

bool APINotesReader::Implementation::readObjCSelectorBlock(
       llvm::BitstreamCursor &cursor, 
       SmallVectorImpl<uint64_t> &scratch) {
  if (cursor.EnterSubBlock(OBJC_SELECTOR_BLOCK_ID))
    return true;

  auto next = cursor.advance();
  while (next.Kind != llvm::BitstreamEntry::EndBlock) {
    if (next.Kind == llvm::BitstreamEntry::Error)
      return true;

    if (next.Kind == llvm::BitstreamEntry::SubBlock) {
      // Unknown sub-block, possibly for use by a future version of the
      // API notes format.
      if (cursor.SkipBlock())
        return true;
      
      next = cursor.advance();
      continue;
    }

    scratch.clear();
    StringRef blobData;
    unsigned kind = cursor.readRecord(next.ID, scratch, &blobData);
    switch (kind) {
    case objc_selector_block::OBJC_SELECTOR_DATA: {
      // Already saw Objective-C selector table.
      if (ObjCSelectorTable)
        return true;

      uint32_t tableOffset;
      objc_selector_block::ObjCSelectorDataLayout::readRecord(scratch, 
                                                              tableOffset);
      auto base = reinterpret_cast<const uint8_t *>(blobData.data());

      ObjCSelectorTable.reset(
        SerializedObjCSelectorTable::Create(base + tableOffset,
                                          base + sizeof(uint32_t),
                                          base));
      break;
    }

    default:
      // Unknown record, possibly for use by a future version of the
      // module format.
      break;
    }

    next = cursor.advance();
  }

  return false;
}

bool APINotesReader::Implementation::readGlobalVariableBlock(
       llvm::BitstreamCursor &cursor, 
       SmallVectorImpl<uint64_t> &scratch) {
  if (cursor.EnterSubBlock(GLOBAL_VARIABLE_BLOCK_ID))
    return true;

  auto next = cursor.advance();
  while (next.Kind != llvm::BitstreamEntry::EndBlock) {
    if (next.Kind == llvm::BitstreamEntry::Error)
      return true;

    if (next.Kind == llvm::BitstreamEntry::SubBlock) {
      // Unknown sub-block, possibly for use by a future version of the
      // API notes format.
      if (cursor.SkipBlock())
        return true;
      
      next = cursor.advance();
      continue;
    }

    scratch.clear();
    StringRef blobData;
    unsigned kind = cursor.readRecord(next.ID, scratch, &blobData);
    switch (kind) {
    case global_variable_block::GLOBAL_VARIABLE_DATA: {
      // Already saw global variable table.
      if (GlobalVariableTable)
        return true;

      uint32_t tableOffset;
      global_variable_block::GlobalVariableDataLayout::readRecord(scratch,
                                                                  tableOffset);
      auto base = reinterpret_cast<const uint8_t *>(blobData.data());

      GlobalVariableTable.reset(
        SerializedGlobalVariableTable::Create(base + tableOffset,
                                              base + sizeof(uint32_t),
                                              base));
      break;
    }

    default:
      // Unknown record, possibly for use by a future version of the
      // module format.
      break;
    }

    next = cursor.advance();
  }

  return false;
}

bool APINotesReader::Implementation::readGlobalFunctionBlock(
       llvm::BitstreamCursor &cursor, 
       SmallVectorImpl<uint64_t> &scratch) {
  if (cursor.EnterSubBlock(GLOBAL_FUNCTION_BLOCK_ID))
    return true;

  auto next = cursor.advance();
  while (next.Kind != llvm::BitstreamEntry::EndBlock) {
    if (next.Kind == llvm::BitstreamEntry::Error)
      return true;

    if (next.Kind == llvm::BitstreamEntry::SubBlock) {
      // Unknown sub-block, possibly for use by a future version of the
      // API notes format.
      if (cursor.SkipBlock())
        return true;
      
      next = cursor.advance();
      continue;
    }

    scratch.clear();
    StringRef blobData;
    unsigned kind = cursor.readRecord(next.ID, scratch, &blobData);
    switch (kind) {
    case global_function_block::GLOBAL_FUNCTION_DATA: {
      // Already saw global function table.
      if (GlobalFunctionTable)
        return true;

      uint32_t tableOffset;
      global_function_block::GlobalFunctionDataLayout::readRecord(scratch,
                                                                  tableOffset);
      auto base = reinterpret_cast<const uint8_t *>(blobData.data());

      GlobalFunctionTable.reset(
        SerializedGlobalFunctionTable::Create(base + tableOffset,
                                              base + sizeof(uint32_t),
                                              base));
      break;
    }

    default:
      // Unknown record, possibly for use by a future version of the
      // module format.
      break;
    }

    next = cursor.advance();
  }

  return false;
}

bool APINotesReader::Implementation::readEnumConstantBlock(
       llvm::BitstreamCursor &cursor, 
       SmallVectorImpl<uint64_t> &scratch) {
  if (cursor.EnterSubBlock(ENUM_CONSTANT_BLOCK_ID))
    return true;

  auto next = cursor.advance();
  while (next.Kind != llvm::BitstreamEntry::EndBlock) {
    if (next.Kind == llvm::BitstreamEntry::Error)
      return true;

    if (next.Kind == llvm::BitstreamEntry::SubBlock) {
      // Unknown sub-block, possibly for use by a future version of the
      // API notes format.
      if (cursor.SkipBlock())
        return true;
      
      next = cursor.advance();
      continue;
    }

    scratch.clear();
    StringRef blobData;
    unsigned kind = cursor.readRecord(next.ID, scratch, &blobData);
    switch (kind) {
    case enum_constant_block::ENUM_CONSTANT_DATA: {
      // Already saw enumerator table.
      if (EnumConstantTable)
        return true;

      uint32_t tableOffset;
      enum_constant_block::EnumConstantDataLayout::readRecord(scratch,
                                                              tableOffset);
      auto base = reinterpret_cast<const uint8_t *>(blobData.data());

      EnumConstantTable.reset(
        SerializedEnumConstantTable::Create(base + tableOffset,
                                            base + sizeof(uint32_t),
                                            base));
      break;
    }

    default:
      // Unknown record, possibly for use by a future version of the
      // module format.
      break;
    }

    next = cursor.advance();
  }

  return false;
}

bool APINotesReader::Implementation::readTagBlock(
       llvm::BitstreamCursor &cursor, 
       SmallVectorImpl<uint64_t> &scratch) {
  if (cursor.EnterSubBlock(TAG_BLOCK_ID))
    return true;

  auto next = cursor.advance();
  while (next.Kind != llvm::BitstreamEntry::EndBlock) {
    if (next.Kind == llvm::BitstreamEntry::Error)
      return true;

    if (next.Kind == llvm::BitstreamEntry::SubBlock) {
      // Unknown sub-block, possibly for use by a future version of the
      // API notes format.
      if (cursor.SkipBlock())
        return true;
      
      next = cursor.advance();
      continue;
    }

    scratch.clear();
    StringRef blobData;
    unsigned kind = cursor.readRecord(next.ID, scratch, &blobData);
    switch (kind) {
    case tag_block::TAG_DATA: {
      // Already saw tag table.
      if (TagTable)
        return true;

      uint32_t tableOffset;
      tag_block::TagDataLayout::readRecord(scratch, tableOffset);
      auto base = reinterpret_cast<const uint8_t *>(blobData.data());

      TagTable.reset(
        SerializedTagTable::Create(base + tableOffset,
                                   base + sizeof(uint32_t),
                                   base));
      break;
    }

    default:
      // Unknown record, possibly for use by a future version of the
      // module format.
      break;
    }

    next = cursor.advance();
  }

  return false;
}

bool APINotesReader::Implementation::readTypedefBlock(
       llvm::BitstreamCursor &cursor, 
       SmallVectorImpl<uint64_t> &scratch) {
  if (cursor.EnterSubBlock(TYPEDEF_BLOCK_ID))
    return true;

  auto next = cursor.advance();
  while (next.Kind != llvm::BitstreamEntry::EndBlock) {
    if (next.Kind == llvm::BitstreamEntry::Error)
      return true;

    if (next.Kind == llvm::BitstreamEntry::SubBlock) {
      // Unknown sub-block, possibly for use by a future version of the
      // API notes format.
      if (cursor.SkipBlock())
        return true;
      
      next = cursor.advance();
      continue;
    }

    scratch.clear();
    StringRef blobData;
    unsigned kind = cursor.readRecord(next.ID, scratch, &blobData);
    switch (kind) {
    case typedef_block::TYPEDEF_DATA: {
      // Already saw typedef table.
      if (TypedefTable)
        return true;

      uint32_t tableOffset;
      typedef_block::TypedefDataLayout::readRecord(scratch, tableOffset);
      auto base = reinterpret_cast<const uint8_t *>(blobData.data());

      TypedefTable.reset(
        SerializedTypedefTable::Create(base + tableOffset,
                                       base + sizeof(uint32_t),
                                       base));
      break;
    }

    default:
      // Unknown record, possibly for use by a future version of the
      // module format.
      break;
    }

    next = cursor.advance();
  }

  return false;
}

APINotesReader::APINotesReader(std::unique_ptr<llvm::MemoryBuffer> inputBuffer, 
                             bool &failed) 
  : Impl(*new Implementation)
{
  failed = false;

  // Initialize the input buffer.
  Impl.InputBuffer = std::move(inputBuffer);
  Impl.InputReader.init(
    reinterpret_cast<const uint8_t *>(Impl.InputBuffer->getBufferStart()), 
    reinterpret_cast<const uint8_t *>(Impl.InputBuffer->getBufferEnd()));
  llvm::BitstreamCursor cursor(Impl.InputReader);

  // Validate signature.
  for (auto byte : API_NOTES_SIGNATURE) {
    if (cursor.AtEndOfStream() || cursor.Read(8) != byte) {
      failed = true;
      return;
    }
  }

  // Look at all of the blocks.
  bool hasValidControlBlock = false;
  SmallVector<uint64_t, 64> scratch;
  auto topLevelEntry = cursor.advance();
  while (topLevelEntry.Kind == llvm::BitstreamEntry::SubBlock) {
    switch (topLevelEntry.ID) {
    case llvm::bitc::BLOCKINFO_BLOCK_ID:
      if (cursor.ReadBlockInfoBlock()) {
        failed = true;
        break;
      }
      break;

    case CONTROL_BLOCK_ID:
      // Only allow a single control block.
      if (hasValidControlBlock || Impl.readControlBlock(cursor, scratch)) {
        failed = true;
        return;
      }

      hasValidControlBlock = true;
      break;

    case IDENTIFIER_BLOCK_ID:
      if (!hasValidControlBlock || Impl.readIdentifierBlock(cursor, scratch)) {
        failed = true;
        return;
      }
      break;

    case OBJC_CONTEXT_BLOCK_ID:
      if (!hasValidControlBlock || Impl.readObjCContextBlock(cursor, scratch)) {
        failed = true;
        return;
      }

      break;

    case OBJC_PROPERTY_BLOCK_ID:
      if (!hasValidControlBlock || 
          Impl.readObjCPropertyBlock(cursor, scratch)) {
        failed = true;
        return;
      }
      break;

    case OBJC_METHOD_BLOCK_ID:
      if (!hasValidControlBlock || Impl.readObjCMethodBlock(cursor, scratch)) {
        failed = true;
        return;
      }
      break;

    case OBJC_SELECTOR_BLOCK_ID:
      if (!hasValidControlBlock || 
          Impl.readObjCSelectorBlock(cursor, scratch)) {
        failed = true;
        return;
      }
      break;

    case GLOBAL_VARIABLE_BLOCK_ID:
      if (!hasValidControlBlock || 
          Impl.readGlobalVariableBlock(cursor, scratch)) {
        failed = true;
        return;
      }
      break;

    case GLOBAL_FUNCTION_BLOCK_ID:
      if (!hasValidControlBlock || 
          Impl.readGlobalFunctionBlock(cursor, scratch)) {
        failed = true;
        return;
      }
      break;

    case ENUM_CONSTANT_BLOCK_ID:
      if (!hasValidControlBlock || 
          Impl.readEnumConstantBlock(cursor, scratch)) {
        failed = true;
        return;
      }
      break;

    case TAG_BLOCK_ID:
      if (!hasValidControlBlock || Impl.readTagBlock(cursor, scratch)) {
        failed = true;
        return;
      }
      break;

    case TYPEDEF_BLOCK_ID:
      if (!hasValidControlBlock || Impl.readTypedefBlock(cursor, scratch)) {
        failed = true;
        return;
      }
      break;

    default:
      // Unknown top-level block, possibly for use by a future version of the
      // module format.
      if (cursor.SkipBlock()) {
        failed = true;
        return;
      }
      break;
    }

    topLevelEntry = cursor.advance(llvm::BitstreamCursor::AF_DontPopBlockAtEnd);
  }

  if (topLevelEntry.Kind != llvm::BitstreamEntry::EndBlock) {
    failed = true;
    return;
  }
}

APINotesReader::~APINotesReader() {
  delete &Impl;
}

std::unique_ptr<APINotesReader> 
APINotesReader::get(std::unique_ptr<llvm::MemoryBuffer> inputBuffer) {
  bool failed = false;
  std::unique_ptr<APINotesReader> 
    reader(new APINotesReader(std::move(inputBuffer), failed));
  if (failed)
    return nullptr;

  return reader;
}

StringRef APINotesReader::getModuleName() const {
  return Impl.ModuleName;
}

ModuleOptions APINotesReader::getModuleOptions() const {
  return Impl.ModuleOpts;
}

auto APINotesReader::lookupObjCClass(StringRef name)
       -> Optional<std::pair<ContextID, ObjCContextInfo>> {
  if (!Impl.ObjCContextTable)
    return None;

  Optional<IdentifierID> classID = Impl.getIdentifier(name);
  if (!classID)
    return None;

  auto known = Impl.ObjCContextTable->find({*classID, '\0'});
  if (known == Impl.ObjCContextTable->end())
    return None;

  auto result = *known;
  return std::make_pair(ContextID(result.first), result.second);
}

auto APINotesReader::lookupObjCProtocol(StringRef name)
       -> Optional<std::pair<ContextID, ObjCContextInfo>> {
  if (!Impl.ObjCContextTable)
    return None;

  Optional<IdentifierID> classID = Impl.getIdentifier(name);
  if (!classID)
    return None;

  auto known = Impl.ObjCContextTable->find({*classID, '\1'});
  if (known == Impl.ObjCContextTable->end())
    return None;

  auto result = *known;
  return std::make_pair(ContextID(result.first), result.second);
}

Optional<ObjCPropertyInfo> APINotesReader::lookupObjCProperty(
                             ContextID contextID,
                             StringRef name) {
  if (!Impl.ObjCPropertyTable)
    return None;

  Optional<IdentifierID> propertyID = Impl.getIdentifier(name);
  if (!propertyID)
    return None;

  auto known = Impl.ObjCPropertyTable->find({contextID.Value, *propertyID});
  if (known == Impl.ObjCPropertyTable->end())
    return None;

  return *known;
}

Optional<ObjCMethodInfo> APINotesReader::lookupObjCMethod(
                           ContextID contextID,
                           ObjCSelectorRef selector,
                           bool isInstanceMethod) {
  if (!Impl.ObjCMethodTable)
    return None;

  Optional<SelectorID> selectorID = Impl.getSelector(selector);
  if (!selectorID)
    return None;

  auto known = Impl.ObjCMethodTable->find(
      ObjCMethodTableInfo::internal_key_type{
          contextID.Value, *selectorID, isInstanceMethod});
  if (known == Impl.ObjCMethodTable->end())
    return None;

  return *known;
}

Optional<GlobalVariableInfo> APINotesReader::lookupGlobalVariable(
                               StringRef name) {
  if (!Impl.GlobalVariableTable)
    return None;

  Optional<IdentifierID> nameID = Impl.getIdentifier(name);
  if (!nameID)
    return None;

  auto known = Impl.GlobalVariableTable->find(*nameID);
  if (known == Impl.GlobalVariableTable->end())
    return None;

  return *known;
}

Optional<GlobalFunctionInfo> APINotesReader::lookupGlobalFunction(
                               StringRef name) {
  if (!Impl.GlobalFunctionTable)
    return None;

  Optional<IdentifierID> nameID = Impl.getIdentifier(name);
  if (!nameID)
    return None;

  auto known = Impl.GlobalFunctionTable->find(*nameID);
  if (known == Impl.GlobalFunctionTable->end())
    return None;

  return *known;
}

Optional<EnumConstantInfo> APINotesReader::lookupEnumConstant(StringRef name) {
  if (!Impl.EnumConstantTable)
    return None;

  Optional<IdentifierID> nameID = Impl.getIdentifier(name);
  if (!nameID)
    return None;

  auto known = Impl.EnumConstantTable->find(*nameID);
  if (known == Impl.EnumConstantTable->end())
    return None;

  return *known;
}

Optional<TagInfo> APINotesReader::lookupTag(StringRef name) {
  if (!Impl.TagTable)
    return None;

  Optional<IdentifierID> nameID = Impl.getIdentifier(name);
  if (!nameID)
    return None;

  auto known = Impl.TagTable->find(*nameID);
  if (known == Impl.TagTable->end())
    return None;

  return *known;
}

Optional<TypedefInfo> APINotesReader::lookupTypedef(StringRef name) {
  if (!Impl.TypedefTable)
    return None;

  Optional<IdentifierID> nameID = Impl.getIdentifier(name);
  if (!nameID)
    return None;

  auto known = Impl.TypedefTable->find(*nameID);
  if (known == Impl.TypedefTable->end())
    return None;

  return *known;
}

APINotesReader::Visitor::~Visitor() { }

void APINotesReader::Visitor::visitObjCClass(ContextID contextID,
                                             StringRef name,
                                             const ObjCContextInfo &info) { }

void APINotesReader::Visitor::visitObjCProtocol(ContextID contextID,
                                                StringRef name,
                                                const ObjCContextInfo &info) { }

void APINotesReader::Visitor::visitObjCMethod(ContextID contextID,
                                              StringRef selector,
                                              bool isInstanceMethod,
                                              const ObjCMethodInfo &info) { }

void APINotesReader::Visitor::visitObjCProperty(ContextID contextID,
                                                StringRef name,
                                                const ObjCPropertyInfo &info) { }

void APINotesReader::Visitor::visitGlobalVariable(
       StringRef name,
       const GlobalVariableInfo &info) { }

void APINotesReader::Visitor::visitGlobalFunction(
       StringRef name,
       const GlobalFunctionInfo &info) { }

void APINotesReader::Visitor::visitEnumConstant(
       StringRef name,
       const EnumConstantInfo &info) { }

void APINotesReader::Visitor::visitTag(
       StringRef name,
       const TagInfo &info) { }

void APINotesReader::Visitor::visitTypedef(
       StringRef name,
       const TypedefInfo &info) { }

void APINotesReader::visit(Visitor &visitor) {
  // FIXME: All of these iterations would be significantly more efficient if we
  // could get the keys and data together, but OnDiskIterableHashTable doesn't
  // support that.

  // Build an identifier ID -> string mapping, which we'll need when visiting
  // any of the tables.
  llvm::DenseMap<unsigned, StringRef> identifiers;
  if (Impl.IdentifierTable) {
    for (auto key : Impl.IdentifierTable->keys()) {
      unsigned ID = *Impl.IdentifierTable->find(key);
      assert(identifiers.count(ID) == 0);
      identifiers[ID] = key;
    }
  }

  // Visit classes and protocols.
  if (Impl.ObjCContextTable) {
    for (auto key : Impl.ObjCContextTable->keys()) {
      auto name = identifiers[key.first];
      auto info = *Impl.ObjCContextTable->find(key);

      if (key.second)
        visitor.visitObjCProtocol(ContextID(info.first), name, info.second);
      else
        visitor.visitObjCClass(ContextID(info.first), name, info.second);
    }
  }

  // Build a selector ID -> stored Objective-C selector mapping, which we need
  // when visiting the method tables.
  llvm::DenseMap<unsigned, std::string> selectors;
  if (Impl.ObjCSelectorTable) {
    for (auto key : Impl.ObjCSelectorTable->keys()) {
      std::string selector;
      if (key.NumPieces == 0)
        selector = identifiers[key.Identifiers[0]];
      else {
        for (auto identID : key.Identifiers) {
          selector += identifiers[identID];
          selector += ':';
        }
      }

      unsigned selectorID = *Impl.ObjCSelectorTable->find(key);
      selectors[selectorID] = selector;
    }
  }

  // Visit methods.
  if (Impl.ObjCMethodTable) {
    for (auto key : Impl.ObjCMethodTable->keys()) {
      ContextID contextID(std::get<0>(key));
      const auto &selector = selectors[std::get<1>(key)];
      auto info = *Impl.ObjCMethodTable->find(key);
      visitor.visitObjCMethod(contextID, selector, std::get<2>(key), info);
    }
  }

  // Visit properties.
  if (Impl.ObjCPropertyTable) {
    for (auto key : Impl.ObjCPropertyTable->keys()) {
      ContextID contextID(key.first);
      auto name = identifiers[key.second];
      auto info = *Impl.ObjCPropertyTable->find(key);
      visitor.visitObjCProperty(contextID, name, info);
    }
  }

  // Visit global functions.
  if (Impl.GlobalFunctionTable) {
    for (auto key : Impl.GlobalFunctionTable->keys()) {
      auto name = identifiers[key];
      auto info = *Impl.GlobalFunctionTable->find(key);
      visitor.visitGlobalFunction(name, info);
    }
  }

  // Visit global variables.
  if (Impl.GlobalVariableTable) {
    for (auto key : Impl.GlobalVariableTable->keys()) {
      auto name = identifiers[key];
      auto info = *Impl.GlobalVariableTable->find(key);
      visitor.visitGlobalVariable(name, info);
    }
  }

  // Visit global variables.
  if (Impl.EnumConstantTable) {
    for (auto key : Impl.EnumConstantTable->keys()) {
      auto name = identifiers[key];
      auto info = *Impl.EnumConstantTable->find(key);
      visitor.visitEnumConstant(name, info);
    }
  }

  // Visit tags.
  if (Impl.TagTable) {
    for (auto key : Impl.TagTable->keys()) {
      auto name = identifiers[key];
      auto info = *Impl.TagTable->find(key);
      visitor.visitTag(name, info);
    }
  }

  // Visit typedefs.
  if (Impl.TypedefTable) {
    for (auto key : Impl.TypedefTable->keys()) {
      auto name = identifiers[key];
      auto info = *Impl.TypedefTable->find(key);
      visitor.visitTypedef(name, info);
    }
  }
}

