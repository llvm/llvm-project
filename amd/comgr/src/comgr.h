/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#ifndef COMGR_DATA_H_
#define COMGR_DATA_H_

#include "amd_comgr.h"
#include "comgr-symbol.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ObjectFile.h"

namespace COMGR {
struct DataMeta;
struct DataSymbol;

/// Update @p Dest to point to a newly allocated C-style (null terminated)
/// string with the contents of @p Src, optionally updating @p Size with the
/// length of the string (not including the null terminator).
///
/// If @p Dest is non-null, it will first be freed.
///
/// @p Src may contain null bytes.
amd_comgr_status_t setCStr(char *&Dest, llvm::StringRef Src,
                           size_t *Size = nullptr);

/// Components of a "Code Object Target Identification" string.
///
/// See https://llvm.org/docs/AMDGPUUsage.html#code-object-target-identification
/// for details.
struct TargetIdentifier {
  llvm::StringRef Arch;
  llvm::StringRef Vendor;
  llvm::StringRef OS;
  llvm::StringRef Environ;
  llvm::StringRef Processor;
  llvm::SmallVector<llvm::StringRef, 2> Features;
};

/// Parse a "Code Object Target Identification" string into it's components.
///
/// See https://llvm.org/docs/AMDGPUUsage.html#code-object-target-identification
/// for details.
///
/// @param IdentStr [in] The string to parse.
/// @param Ident [out] The components of the identification string.
amd_comgr_status_t parseTargetIdentifier(llvm::StringRef IdentStr,
                                         TargetIdentifier &Ident);

/// Ensure all required LLVM initialization functions have been invoked at least
/// once in this process.
void ensureLLVMInitialized();

/// Reset all `llvm::cl` options to their default values.
void clearLLVMOptions();

/// Return `true` if the kind is valid, or false otherwise.
bool isDataKindValid(amd_comgr_data_kind_t DataKind);

struct DataObject {

  // Allocate a new DataObject and return a pointer to it.
  static DataObject *allocate(amd_comgr_data_kind_t DataKind);

  // Decrement the refcount of this DataObject, and free it when it reaches 0.
  void release();

  static amd_comgr_data_t convert(DataObject *Data) {
    amd_comgr_data_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Data))};
    return Handle;
  }

  static const amd_comgr_data_t convert(const DataObject *Data) {
    const amd_comgr_data_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Data))};
    return Handle;
  }

  static DataObject *convert(amd_comgr_data_t Data) {
    return reinterpret_cast<DataObject *>(Data.handle);
  }

  bool hasValidDataKind() { return isDataKindValid(DataKind); }

  amd_comgr_status_t setName(llvm::StringRef Name);
  amd_comgr_status_t setData(llvm::StringRef Data);
  amd_comgr_status_t setData(std::unique_ptr<llvm::MemoryBuffer> Buffer);

  void setMetadata(DataMeta *Metadata);

  amd_comgr_data_kind_t DataKind;
  char *Data;
  char *Name;
  size_t Size;
  int RefCount;
  DataSymbol *DataSym;

private:
  std::unique_ptr<llvm::MemoryBuffer> Buffer;

  void clearData();
  // We require this type be allocated via new, specifically through calling
  // allocate, because we want to be able to `delete this` in release. To make
  // sure the type is not constructed without new, or destructed without
  // checking the reference count, we mark the constructor and destructor
  // private.
  DataObject(amd_comgr_data_kind_t Kind);
  ~DataObject();
};

/// Should be used to ensure references to transient data objects are properly
/// released when they go out of scope.
class ScopedDataObjectReleaser {
  DataObject *Obj;

public:
  ScopedDataObjectReleaser(DataObject *Obj) : Obj(Obj) {}

  ScopedDataObjectReleaser(amd_comgr_data_t Obj)
      : Obj(DataObject::convert(Obj)) {}

  ~ScopedDataObjectReleaser() { Obj->release(); }
};

struct DataSet {

  DataSet();
  ~DataSet();

  static amd_comgr_data_set_t convert(DataSet *Set) {
    amd_comgr_data_set_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Set))};
    return Handle;
  }

  static const amd_comgr_data_set_t convert(const DataSet *Set) {
    const amd_comgr_data_set_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Set))};
    return Handle;
  }

  static DataSet *convert(amd_comgr_data_set_t Set) {
    return reinterpret_cast<DataSet *>(Set.handle);
  }

  llvm::SmallSetVector<DataObject *, 8> DataObjects;
};

struct DataAction {
  // Some actions involving llvm we want to do it only once for the entire
  // duration of the COMGR library. Once initialized, they should never be
  // reset.

  DataAction();
  ~DataAction();

  static amd_comgr_action_info_t convert(DataAction *Action) {
    amd_comgr_action_info_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Action))};
    return Handle;
  }

  static const amd_comgr_action_info_t convert(const DataAction *Action) {
    const amd_comgr_action_info_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Action))};
    return Handle;
  }

  static DataAction *convert(amd_comgr_action_info_t Action) {
    return reinterpret_cast<DataAction *>(Action.handle);
  }

  amd_comgr_status_t setIsaName(llvm::StringRef IsaName);
  amd_comgr_status_t setActionPath(llvm::StringRef ActionPath);

  // Set the options to be the legacy "flat" string.
  amd_comgr_status_t setOptionsFlat(llvm::StringRef Options);
  // If the options were set via setOptionsFlag, return a reference to the
  // string (including the null terminator).
  amd_comgr_status_t getOptionsFlat(llvm::StringRef &Options);

  // Set the options to be the new list.
  amd_comgr_status_t setOptionList(llvm::ArrayRef<const char *> Options);
  // If the options were set via setOptionList, return the length of the list.
  amd_comgr_status_t getOptionListCount(size_t &Size);
  // If the options were set via setOptionList, return a reference to the
  // string at Index in the list (including the null terminator).
  amd_comgr_status_t getOptionListItem(size_t Index, llvm::StringRef &Option);

  // Return a normalized array of options, possibly splitting a flat options
  // string. If splitting, ' ' is used as a delimiter if IsDeviceLibs is false,
  // otherwise ',' is used. The returned array reference is only valid as long
  // as no other option APIs are called.
  llvm::ArrayRef<std::string> getOptions(bool IsDeviceLibs = false);

  char *IsaName;
  char *Path;
  amd_comgr_language_t Language;
  bool Logging;

private:
  bool AreOptionsList;
  std::string FlatOptions;
  std::vector<std::string> ListOptions;
};

// Elements common to all DataMeta which refer to the same "document".
struct MetaDocument {
  // The MsgPack document, which owns all memory allocated during parsing.
  llvm::msgpack::Document Document;
  // The MsgPack parser is zero-copy, so we retain a copy of the input buffer.
  std::string RawDocument;
  std::vector<std::string> RawDocumentList;
  // The old YAML parser would produce the strings "true" and "false" for
  // booleans, whereas the old MsgPack parser produced "0" and "1". The new
  // universal parser produces "true" and "false", but we need to remain
  // backwards compatible, so we set a flag when parsing MsgPack.
  bool EmitIntegerBooleans = false;
};

struct DataMeta {
  static amd_comgr_metadata_node_t convert(DataMeta *Meta) {
    amd_comgr_metadata_node_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Meta))};
    return Handle;
  }

  static const amd_comgr_metadata_node_t convert(const DataMeta *Meta) {
    const amd_comgr_metadata_node_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Meta))};
    return Handle;
  }

  static DataMeta *convert(amd_comgr_metadata_node_t Meta) {
    return reinterpret_cast<DataMeta *>(Meta.handle);
  }

  amd_comgr_metadata_kind_t getMetadataKind();
  // Get the canonical string representation of @p DocNode, assuming
  // it is a scalar node.
  std::string convertDocNodeToString(llvm::msgpack::DocNode DocNode);

  // This DataMeta's "meta document", shared by all instances derived from the
  // same metadata.
  std::shared_ptr<MetaDocument> MetaDoc;
  // This DataMeta's "view" into the shared llvm::msgpack::Document.
  llvm::msgpack::DocNode DocNode;
};

struct DataSymbol {
  DataSymbol(SymbolContext *DataSym);
  ~DataSymbol();

  static amd_comgr_symbol_t convert(DataSymbol *Sym) {
    amd_comgr_symbol_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Sym))};
    return Handle;
  }

  static const amd_comgr_symbol_t convert(const DataSymbol *Sym) {
    const amd_comgr_symbol_t Handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(Sym))};
    return Handle;
  }

  static DataSymbol *convert(amd_comgr_symbol_t Sym) {
    return reinterpret_cast<DataSymbol *>(Sym.handle);
  }

  SymbolContext *DataSym;
};

} // namespace COMGR

#endif // header guard
