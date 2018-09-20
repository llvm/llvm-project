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
*******************************************************************************/

#ifndef COMGR_DATA_H_
#define COMGR_DATA_H_

#include "comgr-msgpack.h"
#include "comgr-symbol.h"
#include "comgr/amd_comgr.h"
#include "yaml-cpp/yaml.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
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
amd_comgr_status_t SetCStr(char *&Dest, llvm::StringRef Src,
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
amd_comgr_status_t ParseTargetIdentifier(llvm::StringRef IdentStr,
        TargetIdentifier &Ident);

struct DataObject {

  // Allocate a new DataObject and return a pointer to it.
  static DataObject *allocate(amd_comgr_data_kind_t data_kind);

  // Decrement the refcount of this DataObject, and free it when it reaches 0.
  void release();

  static amd_comgr_data_t Convert(DataObject* data) {
    amd_comgr_data_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data))};
    return handle;
  }

  static const amd_comgr_data_t Convert(const DataObject* data) {
    const amd_comgr_data_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data))};
    return handle;
  }

  static DataObject *Convert(amd_comgr_data_t data) {
    return reinterpret_cast<DataObject *>(data.handle);
  }

  bool kind_is_valid() {
    if (data_kind > AMD_COMGR_DATA_KIND_UNDEF &&
        data_kind <= AMD_COMGR_DATA_KIND_LAST)
      return true;
    return false;
  }

  amd_comgr_status_t SetName(llvm::StringRef Name);
  amd_comgr_status_t SetData(llvm::StringRef Data);
  void SetMetadata(DataMeta *Metadata);

  void dump();

  amd_comgr_data_kind_t data_kind;
  char *data;
  char *name;
  size_t size;
  int refcount;
  DataSymbol *data_sym;

private:
  // We require this type be allocated via new, specifically through calling
  // allocate, because we want to be able to `delete this` in release. To make
  // sure the type is not constructed without new, or destructed without
  // checking the reference count, we mark the constructor and destructor
  // private.
  DataObject(amd_comgr_data_kind_t kind);
  ~DataObject();
};

/// Should be used to ensure references to transient data objects are properly
/// released when they go out of scope.
class ScopedDataObjectReleaser {
  DataObject *Obj;

public:
  ScopedDataObjectReleaser(DataObject *Obj) : Obj(Obj) {}

  ScopedDataObjectReleaser(amd_comgr_data_t Obj)
      : Obj(DataObject::Convert(Obj)) {}

  ~ScopedDataObjectReleaser() { Obj->release(); }
};

struct DataSet {

  DataSet();
  ~DataSet();

  static amd_comgr_data_set_t Convert(DataSet* data_set ) {
    amd_comgr_data_set_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data_set))};
    return handle;
  }

  static const amd_comgr_data_set_t Convert(const DataSet* data_set) {
    const amd_comgr_data_set_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data_set))};
    return handle;
  }

  static DataSet *Convert(amd_comgr_data_set_t data_set) {
    return reinterpret_cast<DataSet *>(data_set.handle);
  }

  void dump();

  llvm::SmallSetVector<DataObject *, 8> data_objects;
};

struct DataAction {
  // Some actions involving llvm we want to do it only once for the entire
  // duration of the COMGR library. Once initialized, they should never be
  // reset.

  static bool llvm_initialized;   // must be statically initialized

  DataAction();
  ~DataAction();

  static amd_comgr_action_info_t Convert(DataAction* action) {
    amd_comgr_action_info_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(action))};
    return handle;
  }

  static const amd_comgr_action_info_t Convert(const DataAction* action) {
    const amd_comgr_action_info_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(action))};
    return handle;
  }

  static DataAction *Convert(amd_comgr_action_info_t action) {
    return reinterpret_cast<DataAction *>(action.handle);
  }

  amd_comgr_status_t SetIsaName(llvm::StringRef IsaName);
  amd_comgr_status_t SetActionOptions(llvm::StringRef ActionOptions);
  amd_comgr_status_t SetActionPath(llvm::StringRef ActionPath);

  char *isa_name;
  char *action_options;
  char *action_path;
  amd_comgr_language_t language;
  bool logging;
};

struct DataMeta {

  DataMeta();
  ~DataMeta();

  static amd_comgr_metadata_node_t Convert(DataMeta* meta) {
    amd_comgr_metadata_node_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(meta))};
    return handle;
  }

  static const amd_comgr_metadata_node_t Convert(const DataMeta* meta) {
    const amd_comgr_metadata_node_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(meta))};
    return handle;
  }

  static DataMeta *Convert(amd_comgr_metadata_node_t meta) {
    return reinterpret_cast<DataMeta *>(meta.handle);
  }

  void dump();

  YAML::Node node;
  std::shared_ptr<msgpack::Node> msgpack_node;
  amd_comgr_metadata_kind_t get_metadata_kind();
};

struct DataSymbol {
  DataSymbol(SymbolContext *data_sym);
  ~DataSymbol();

  static amd_comgr_symbol_t Convert(DataSymbol* sym) {
    amd_comgr_symbol_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sym))};
    return handle;
  }

  static const amd_comgr_symbol_t Convert(const DataSymbol* sym) {
    const amd_comgr_symbol_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(sym))};
    return handle;
  }

  static DataSymbol *Convert(amd_comgr_symbol_t sym) {
    return reinterpret_cast<DataSymbol *>(sym.handle);
  }

  SymbolContext *data_sym;
};

}  // namespace CO

#endif // header guard
