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
#include "llvm/Support/DJB.h"
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
  /// Deserialize a version tuple.
  VersionTuple readVersionTuple(const uint8_t *&data) {
    uint8_t numVersions = (*data++) & 0x03;

    unsigned major = endian::readNext<uint32_t, little, unaligned>(data);
    if (numVersions == 0)
      return VersionTuple(major);

    unsigned minor = endian::readNext<uint32_t, little, unaligned>(data);
    if (numVersions == 1)
      return VersionTuple(major, minor);

    unsigned subminor = endian::readNext<uint32_t, little, unaligned>(data);
    if (numVersions == 2)
      return VersionTuple(major, minor, subminor);

    unsigned build = endian::readNext<uint32_t, little, unaligned>(data);
    return VersionTuple(major, minor, subminor, build);
  }

  /// An on-disk hash table whose data is versioned based on the Swift version.
  template<typename Derived, typename KeyType, typename UnversionedDataType>
  class VersionedTableInfo {
  public:
    using internal_key_type = KeyType;
    using external_key_type = KeyType;
    using data_type = SmallVector<std::pair<VersionTuple, UnversionedDataType>, 1>;
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

    static data_type ReadData(internal_key_type key, const uint8_t *data,
                              unsigned length) {
      unsigned numElements = endian::readNext<uint16_t, little, unaligned>(data);
      data_type result;
      result.reserve(numElements);
      for (unsigned i = 0; i != numElements; ++i) {
        auto version = readVersionTuple(data);
        auto dataBefore = data; (void)dataBefore;
        auto unversionedData = Derived::readUnversioned(key, data);
        assert(data != dataBefore
               && "Unversioned data reader didn't move pointer");
        result.push_back({version, unversionedData});
      }
      return result;
    }
  };


  /// Read serialized CommonEntityInfo.
  void readCommonEntityInfo(const uint8_t *&data, CommonEntityInfo &info) {
    uint8_t unavailableBits = *data++;
    info.Unavailable = (unavailableBits >> 1) & 0x01;
    info.UnavailableInSwift = unavailableBits & 0x01;
    if ((unavailableBits >> 2) & 0x01)
      info.setSwiftPrivate(static_cast<bool>((unavailableBits >> 3) & 0x01));

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
    if (swiftBridgeLength > 0) {
      info.setSwiftBridge(
        std::string(reinterpret_cast<const char *>(data), swiftBridgeLength-1));
      data += swiftBridgeLength-1;
    }

    unsigned errorDomainLength =
      endian::readNext<uint16_t, little, unaligned>(data);
    if (errorDomainLength > 0) {
      info.setNSErrorDomain(
        std::string(reinterpret_cast<const char *>(data), errorDomainLength-1));
      data += errorDomainLength-1;
    }
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
      return llvm::djbHash(key);
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
  class ObjCContextIDTableInfo {
  public:
    // identifier ID, is-protocol
    using internal_key_type = std::pair<unsigned, char>;
    using external_key_type = internal_key_type;
    using data_type = unsigned;
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
      return endian::readNext<uint32_t, little, unaligned>(data);
    }
  };

  /// Used to deserialize the on-disk Objective-C property table.
  class ObjCContextInfoTableInfo
    : public VersionedTableInfo<ObjCContextInfoTableInfo,
                                unsigned,
                                ObjCContextInfo>
  {
  public:
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      return endian::readNext<uint32_t, little, unaligned>(data);
    }
    
    static ObjCContextInfo readUnversioned(internal_key_type key,
                                           const uint8_t *&data) {
      ObjCContextInfo info;
      readCommonTypeInfo(data, info);
      uint8_t payload = *data++;

      if (payload & 0x01)
        info.setHasDesignatedInits(true);
      payload = payload >> 1;

      if (payload & 0x4)
        info.setDefaultNullability(static_cast<NullabilityKind>(payload&0x03));
      payload >>= 3;

      if (payload & (1 << 1))
        info.setSwiftObjCMembers(payload & 1);
      payload >>= 2;

      if (payload & (1 << 1))
        info.setSwiftImportAsNonGeneric(payload & 1);

      return info;
    }
  };

  /// Read serialized VariableInfo.
  void readVariableInfo(const uint8_t *&data, VariableInfo &info) {
    readCommonEntityInfo(data, info);
    if (*data++) {
      info.setNullabilityAudited(static_cast<NullabilityKind>(*data));
    }
    ++data;

    auto typeLen
      = endian::readNext<uint16_t, little, unaligned>(data);
    info.setType(std::string(data, data + typeLen));
    data += typeLen;
  }

  /// Used to deserialize the on-disk Objective-C property table.
  class ObjCPropertyTableInfo
    : public VersionedTableInfo<ObjCPropertyTableInfo,
                                std::tuple<unsigned, unsigned, char>,
                                ObjCPropertyInfo>
  {
  public:
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto classID = endian::readNext<uint32_t, little, unaligned>(data);
      auto nameID = endian::readNext<uint32_t, little, unaligned>(data);
      char isInstance = endian::readNext<uint8_t, little, unaligned>(data);
      return std::make_tuple(classID, nameID, isInstance);
    }
    
    static ObjCPropertyInfo readUnversioned(internal_key_type key,
                                            const uint8_t *&data) {
      ObjCPropertyInfo info;
      readVariableInfo(data, info);
      uint8_t flags = *data++;
      if (flags & (1 << 0))
        info.setSwiftImportAsAccessors(flags & (1 << 1));
      return info;
    }
  };

  /// Read serialized ParamInfo.
  void readParamInfo(const uint8_t *&data, ParamInfo &info) {
    readVariableInfo(data, info);

    uint8_t payload = endian::readNext<uint8_t, little, unaligned>(data);
    if (auto rawConvention = payload & 0x7) {
      auto convention = static_cast<RetainCountConventionKind>(rawConvention-1);
      info.setRetainCountConvention(convention);
    }
    payload >>= 3;
    if (payload & 0x01) {
      info.setNoEscape(payload & 0x02);
    }
    payload >>= 2; assert(payload == 0 && "Bad API notes");
  }

  /// Read serialized FunctionInfo.
  void readFunctionInfo(const uint8_t *&data, FunctionInfo &info) {
    readCommonEntityInfo(data, info);

    uint8_t payload = endian::readNext<uint8_t, little, unaligned>(data);
    if (auto rawConvention = payload & 0x7) {
      auto convention = static_cast<RetainCountConventionKind>(rawConvention-1);
      info.setRetainCountConvention(convention);
    }
    payload >>= 3;
    info.NullabilityAudited = payload & 0x1;
    payload >>= 1; assert(payload == 0 && "Bad API notes");

    info.NumAdjustedNullable
      = endian::readNext<uint8_t, little, unaligned>(data);
    info.NullabilityPayload
      = endian::readNext<uint64_t, little, unaligned>(data);

    unsigned numParams = endian::readNext<uint16_t, little, unaligned>(data);
    while (numParams > 0) {
      ParamInfo pi;
      readParamInfo(data, pi);
      info.Params.push_back(pi);
      --numParams;
    }

    unsigned resultTypeLen
      = endian::readNext<uint16_t, little, unaligned>(data);
    info.ResultType = std::string(data, data + resultTypeLen);
    data += resultTypeLen;
  }

  /// Used to deserialize the on-disk Objective-C method table.
  class ObjCMethodTableInfo
    : public VersionedTableInfo<ObjCMethodTableInfo,
                                std::tuple<unsigned, unsigned, char>,
                                ObjCMethodInfo> {
  public:
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto classID = endian::readNext<uint32_t, little, unaligned>(data);
      auto selectorID = endian::readNext<uint32_t, little, unaligned>(data);
      auto isInstance = endian::readNext<uint8_t, little, unaligned>(data);
      return internal_key_type{ classID, selectorID, isInstance };
    }
    
    static ObjCMethodInfo readUnversioned(internal_key_type key,
                                          const uint8_t *&data) {
      ObjCMethodInfo info;
      uint8_t payload = *data++;
      info.Required = payload & 0x01;
      payload >>= 1;
      info.DesignatedInit = payload & 0x01;
      payload >>= 1;

      readFunctionInfo(data, info);
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
  class GlobalVariableTableInfo
    : public VersionedTableInfo<GlobalVariableTableInfo, unsigned,
                                GlobalVariableInfo> {
  public:
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto nameID = endian::readNext<uint32_t, little, unaligned>(data);
      return nameID;
    }

    static GlobalVariableInfo readUnversioned(internal_key_type key,
                                              const uint8_t *&data) {
      GlobalVariableInfo info;
      readVariableInfo(data, info);
      return info;
    }
  };

  /// Used to deserialize the on-disk global function table.
  class GlobalFunctionTableInfo
    : public VersionedTableInfo<GlobalFunctionTableInfo, unsigned,
                                GlobalFunctionInfo> {
  public:
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto nameID = endian::readNext<uint32_t, little, unaligned>(data);
      return nameID;
    }
    
    static GlobalFunctionInfo readUnversioned(internal_key_type key,
                                              const uint8_t *&data) {
      GlobalFunctionInfo info;
      readFunctionInfo(data, info);
      return info;
    }
  };

  /// Used to deserialize the on-disk enumerator table.
  class EnumConstantTableInfo
    : public VersionedTableInfo<EnumConstantTableInfo, unsigned,
                                EnumConstantInfo> {
  public:
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto nameID = endian::readNext<uint32_t, little, unaligned>(data);
      return nameID;
    }
    
    static EnumConstantInfo readUnversioned(internal_key_type key,
                                            const uint8_t *&data) {
      EnumConstantInfo info;
      readCommonEntityInfo(data, info);
      return info;
    }
  };

  /// Used to deserialize the on-disk tag table.
  class TagTableInfo
    : public VersionedTableInfo<TagTableInfo, unsigned, TagInfo> {
  public:
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto nameID = endian::readNext<IdentifierID, little, unaligned>(data);
      return nameID;
    }
    
    static TagInfo readUnversioned(internal_key_type key,
                                   const uint8_t *&data) {
      TagInfo info;

      uint8_t payload = *data++;
      if (payload & 1) {
        info.setFlagEnum(payload & 2);
      }
      payload >>= 2;
      if (payload > 0) {
        info.EnumExtensibility =
            static_cast<EnumExtensibilityKind>((payload & 0x3) - 1);
      }

      readCommonTypeInfo(data, info);
      return info;
    }
  };

  /// Used to deserialize the on-disk typedef table.
  class TypedefTableInfo
    : public VersionedTableInfo<TypedefTableInfo, unsigned, TypedefInfo> {
  public:
    static internal_key_type ReadKey(const uint8_t *data, unsigned length) {
      auto nameID = endian::readNext<IdentifierID, little, unaligned>(data);
      return nameID;
    }

    static TypedefInfo readUnversioned(internal_key_type key,
                                       const uint8_t *&data) {
      TypedefInfo info;

      uint8_t payload = *data++;
      if (payload > 0) {
        info.SwiftWrapper = static_cast<SwiftWrapperKind>((payload & 0x3) - 1);
      }

      readCommonTypeInfo(data, info);
      return info;
    }
  };
} // end anonymous namespace

class APINotesReader::Implementation {
public:
  /// The input buffer for the API notes data.
  llvm::MemoryBuffer *InputBuffer;

  /// Whether we own the input buffer.
  bool OwnsInputBuffer;

  /// The Swift version to use for filtering.
  VersionTuple SwiftVersion;

  /// The name of the module that we read from the control block.
  std::string ModuleName;

  // The size and modification time of the source file from
  // which this API notes file was created, if known.
  Optional<std::pair<off_t, time_t>> SourceFileSizeAndModTime;

  /// Various options and attributes for the module
  ModuleOptions ModuleOpts;

  using SerializedIdentifierTable =
      llvm::OnDiskIterableChainedHashTable<IdentifierTableInfo>;

  /// The identifier table.
  std::unique_ptr<SerializedIdentifierTable> IdentifierTable;

  using SerializedObjCContextIDTable =
      llvm::OnDiskIterableChainedHashTable<ObjCContextIDTableInfo>;

  /// The Objective-C context ID table.
  std::unique_ptr<SerializedObjCContextIDTable> ObjCContextIDTable;

  using SerializedObjCContextInfoTable =
    llvm::OnDiskIterableChainedHashTable<ObjCContextInfoTableInfo>;

  /// The Objective-C context info table.
  std::unique_ptr<SerializedObjCContextInfoTable> ObjCContextInfoTable;

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

    case control_block::SOURCE_FILE:
      SourceFileSizeAndModTime = { scratch[0], scratch[1] };
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
    case objc_context_block::OBJC_CONTEXT_ID_DATA: {
      // Already saw Objective-C context ID table.
      if (ObjCContextIDTable)
        return true;

      uint32_t tableOffset;
      objc_context_block::ObjCContextIDLayout::readRecord(scratch, tableOffset);
      auto base = reinterpret_cast<const uint8_t *>(blobData.data());

      ObjCContextIDTable.reset(
        SerializedObjCContextIDTable::Create(base + tableOffset,
                                             base + sizeof(uint32_t),
                                             base));
      break;
    }

    case objc_context_block::OBJC_CONTEXT_INFO_DATA: {
      // Already saw Objective-C context info table.
      if (ObjCContextInfoTable)
        return true;

      uint32_t tableOffset;
      objc_context_block::ObjCContextInfoLayout::readRecord(scratch,
                                                            tableOffset);
      auto base = reinterpret_cast<const uint8_t *>(blobData.data());

      ObjCContextInfoTable.reset(
        SerializedObjCContextInfoTable::Create(base + tableOffset,
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

APINotesReader::APINotesReader(llvm::MemoryBuffer *inputBuffer, 
                               bool ownsInputBuffer,
                               VersionTuple swiftVersion,
                               bool &failed) 
  : Impl(*new Implementation)
{
  failed = false;

  // Initialize the input buffer.
  Impl.InputBuffer = inputBuffer;
  Impl.OwnsInputBuffer = ownsInputBuffer;
  Impl.SwiftVersion = swiftVersion;
  llvm::BitstreamCursor cursor(*Impl.InputBuffer);

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
  while (!cursor.AtEndOfStream()) {
    auto topLevelEntry = cursor.advance();
    if (topLevelEntry.Kind != llvm::BitstreamEntry::SubBlock)
      break;

    switch (topLevelEntry.ID) {
    case llvm::bitc::BLOCKINFO_BLOCK_ID:
      if (!cursor.ReadBlockInfoBlock()) {
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
  }

  if (!cursor.AtEndOfStream()) {
    failed = true;
    return;
  }
}

APINotesReader::~APINotesReader() {
  if (Impl.OwnsInputBuffer)
    delete Impl.InputBuffer;

  delete &Impl;
}

std::unique_ptr<APINotesReader> 
APINotesReader::get(std::unique_ptr<llvm::MemoryBuffer> inputBuffer,
                    VersionTuple swiftVersion) {
  bool failed = false;
  std::unique_ptr<APINotesReader> 
    reader(new APINotesReader(inputBuffer.release(), /*ownsInputBuffer=*/true,
                              swiftVersion, failed));
  if (failed)
    return nullptr;

  return reader;
}

std::unique_ptr<APINotesReader> 
APINotesReader::getUnmanaged(llvm::MemoryBuffer *inputBuffer,
                             VersionTuple swiftVersion) {
  bool failed = false;
  std::unique_ptr<APINotesReader> 
    reader(new APINotesReader(inputBuffer, /*ownsInputBuffer=*/false,
                              swiftVersion, failed));
  if (failed)
    return nullptr;

  return reader;
}

StringRef APINotesReader::getModuleName() const {
  return Impl.ModuleName;
}

Optional<std::pair<off_t, time_t>>
APINotesReader::getSourceFileSizeAndModTime() const {
  return Impl.SourceFileSizeAndModTime;
}

ModuleOptions APINotesReader::getModuleOptions() const {
  return Impl.ModuleOpts;
}

template<typename T>
APINotesReader::VersionedInfo<T>::VersionedInfo(
    VersionTuple version,
    SmallVector<std::pair<VersionTuple, T>, 1> results)
  : Results(std::move(results)) {

  assert(!Results.empty());
  assert(std::is_sorted(Results.begin(), Results.end(),
                        [](const std::pair<VersionTuple, T> &left,
                           const std::pair<VersionTuple, T> &right) -> bool {
    assert(left.first != right.first && "two entries for the same version");
    return left.first < right.first;
  }));

  Selected = Results.size();
  for (unsigned i = 0, n = Results.size(); i != n; ++i) {
    if (version && Results[i].first >= version) {
      // If the current version is "4", then entries for 4 are better than
      // entries for 5, but both are valid. Because entries are sorted, we get
      // that behavior by picking the first match.
      Selected = i;
      break;
    }
  }

  // If we didn't find a match but we have an unversioned result, use the
  // unversioned result. This will always be the first entry because we encode
  // it as version 0.
  if (Selected == Results.size() && Results[0].first.empty())
    Selected = 0;
}

auto APINotesReader::lookupObjCClassID(StringRef name) -> Optional<ContextID> {
  if (!Impl.ObjCContextIDTable)
    return None;

  Optional<IdentifierID> classID = Impl.getIdentifier(name);
  if (!classID)
    return None;

  auto knownID = Impl.ObjCContextIDTable->find({*classID, '\0'});
  if (knownID == Impl.ObjCContextIDTable->end())
    return None;

  return ContextID(*knownID);
}

auto APINotesReader::lookupObjCClassInfo(StringRef name)
       -> VersionedInfo<ObjCContextInfo> {
  if (!Impl.ObjCContextInfoTable)
    return None;

  Optional<ContextID> contextID = lookupObjCClassID(name);
  if (!contextID)
    return None;

  auto knownInfo = Impl.ObjCContextInfoTable->find(contextID->Value);
  if (knownInfo == Impl.ObjCContextInfoTable->end())
    return None;

  return { Impl.SwiftVersion, *knownInfo };
}

auto APINotesReader::lookupObjCProtocolID(StringRef name)
       -> Optional<ContextID> {
   if (!Impl.ObjCContextIDTable)
     return None;

   Optional<IdentifierID> classID = Impl.getIdentifier(name);
   if (!classID)
     return None;

   auto knownID = Impl.ObjCContextIDTable->find({*classID, '\1'});
   if (knownID == Impl.ObjCContextIDTable->end())
     return None;

   return ContextID(*knownID);
}

auto APINotesReader::lookupObjCProtocolInfo(StringRef name)
       -> VersionedInfo<ObjCContextInfo> {
   if (!Impl.ObjCContextInfoTable)
     return None;

   Optional<ContextID> contextID = lookupObjCProtocolID(name);
   if (!contextID)
     return None;

   auto knownInfo = Impl.ObjCContextInfoTable->find(contextID->Value);
   if (knownInfo == Impl.ObjCContextInfoTable->end())
     return None;
   
   return { Impl.SwiftVersion, *knownInfo };
}


auto APINotesReader::lookupObjCProperty(ContextID contextID,
                                        StringRef name,
                                        bool isInstance)
    -> VersionedInfo<ObjCPropertyInfo> {
  if (!Impl.ObjCPropertyTable)
    return None;

  Optional<IdentifierID> propertyID = Impl.getIdentifier(name);
  if (!propertyID)
    return None;

  auto known = Impl.ObjCPropertyTable->find(std::make_tuple(contextID.Value,
                                                            *propertyID,
                                                            (char)isInstance));
  if (known == Impl.ObjCPropertyTable->end())
    return None;

  return { Impl.SwiftVersion, *known };
}

auto APINotesReader::lookupObjCMethod(
                                      ContextID contextID,
                                      ObjCSelectorRef selector,
                                      bool isInstanceMethod)
    -> VersionedInfo<ObjCMethodInfo> {
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

  return { Impl.SwiftVersion, *known };
}

auto APINotesReader::lookupGlobalVariable(
                                          StringRef name)
    -> VersionedInfo<GlobalVariableInfo> {
  if (!Impl.GlobalVariableTable)
    return None;

  Optional<IdentifierID> nameID = Impl.getIdentifier(name);
  if (!nameID)
    return None;

  auto known = Impl.GlobalVariableTable->find(*nameID);
  if (known == Impl.GlobalVariableTable->end())
    return None;

  return { Impl.SwiftVersion, *known };
}

auto APINotesReader::lookupGlobalFunction(StringRef name)
    -> VersionedInfo<GlobalFunctionInfo> {
  if (!Impl.GlobalFunctionTable)
    return None;

  Optional<IdentifierID> nameID = Impl.getIdentifier(name);
  if (!nameID)
    return None;

  auto known = Impl.GlobalFunctionTable->find(*nameID);
  if (known == Impl.GlobalFunctionTable->end())
    return None;

  return { Impl.SwiftVersion, *known };
}

auto APINotesReader::lookupEnumConstant(StringRef name)
    -> VersionedInfo<EnumConstantInfo> {
  if (!Impl.EnumConstantTable)
    return None;

  Optional<IdentifierID> nameID = Impl.getIdentifier(name);
  if (!nameID)
    return None;

  auto known = Impl.EnumConstantTable->find(*nameID);
  if (known == Impl.EnumConstantTable->end())
    return None;

  return { Impl.SwiftVersion, *known };
}

auto APINotesReader::lookupTag(StringRef name) -> VersionedInfo<TagInfo> {
  if (!Impl.TagTable)
    return None;

  Optional<IdentifierID> nameID = Impl.getIdentifier(name);
  if (!nameID)
    return None;

  auto known = Impl.TagTable->find(*nameID);
  if (known == Impl.TagTable->end())
    return None;

  return { Impl.SwiftVersion, *known };
}

auto APINotesReader::lookupTypedef(StringRef name)
    -> VersionedInfo<TypedefInfo> {
  if (!Impl.TypedefTable)
    return None;

  Optional<IdentifierID> nameID = Impl.getIdentifier(name);
  if (!nameID)
    return None;

  auto known = Impl.TypedefTable->find(*nameID);
  if (known == Impl.TypedefTable->end())
    return None;

  return { Impl.SwiftVersion, *known };
}
