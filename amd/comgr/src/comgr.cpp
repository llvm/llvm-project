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

#include "comgr.h" // C++ class header
#include "comgr-compiler.h"
#ifdef DEVICE_LIBS
  #include "comgr-device-libs.h"
#endif
#include "comgr-metadata.h"
#include "comgr-objdump.h"
#include "comgr-symbol.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/TargetSelect.h"
#include <string>

#ifndef AMD_NOINLINE
#ifdef __GNUC__
#define AMD_NOINLINE __attribute__((noinline))
#else
#define AMD_NOINLINE __declspec(noinline)
#endif
#endif

using namespace llvm;
using namespace COMGR;

// static member initialization
bool DataAction::llvm_initialized = false;

// Module static functions

static amd_comgr_status_t print_entry(amd_comgr_metadata_node_t key,
                                      amd_comgr_metadata_node_t value,
                                      void *data) {
  amd_comgr_metadata_kind_t kind;
  amd_comgr_metadata_node_t son;
  amd_comgr_status_t status;
  size_t size;
  char keybuf[50];
  char buf[50];
  int *indent = (int *)data;

  (void) status;

  // assume key to be string in this test function
  status = amd_comgr_get_metadata_kind(key, &kind);
  assert(status == AMD_COMGR_STATUS_SUCCESS);
  assert(kind != AMD_COMGR_METADATA_KIND_STRING);
  status = amd_comgr_get_metadata_string(key, &size, NULL);
  assert(status == AMD_COMGR_STATUS_SUCCESS);
  status = amd_comgr_get_metadata_string(key, &size, keybuf);
  assert(status == AMD_COMGR_STATUS_SUCCESS);

  status = amd_comgr_get_metadata_kind(value, &kind);
  assert(status == AMD_COMGR_STATUS_SUCCESS);
  for (int i=0; i<*indent; i++)
    printf("  ");

  switch (kind) {
  case AMD_COMGR_METADATA_KIND_STRING: {
    printf("%s  :  ", size ? keybuf : "");
    status = amd_comgr_get_metadata_string(value, &size, NULL);
    assert(status == AMD_COMGR_STATUS_SUCCESS);
    status = amd_comgr_get_metadata_string(value, &size, buf);
    assert(status == AMD_COMGR_STATUS_SUCCESS);
    printf(" %s\n", buf);
    break;
  }
  case AMD_COMGR_METADATA_KIND_LIST: {
    *indent += 1;
    status = amd_comgr_get_metadata_list_size(value, &size);
    assert(status == AMD_COMGR_STATUS_SUCCESS);
    printf("LIST %s %ld entries = \n", keybuf, size);
    for (size_t i=0; i<size; i++) {
      status = amd_comgr_index_list_metadata(value, i, &son);
      assert(status == AMD_COMGR_STATUS_SUCCESS);
      print_entry(key, son, data);
      assert(status == AMD_COMGR_STATUS_SUCCESS);
    }
    *indent = *indent > 0 ? *indent-1 : 0;
    break;
  }
  case AMD_COMGR_METADATA_KIND_MAP: {
    *indent += 1;
    status = amd_comgr_get_metadata_map_size(value, &size);
    assert(status == AMD_COMGR_STATUS_SUCCESS);
    printf("MAP %ld entries = \n", size);
    status = amd_comgr_iterate_map_metadata(value, print_entry, data);
    assert(status == AMD_COMGR_STATUS_SUCCESS);
    *indent = *indent > 0 ? *indent-1 : 0;
    break;
  }
  default:
    assert(0);
  } // switch

  return AMD_COMGR_STATUS_SUCCESS;
}

static bool language_is_valid(amd_comgr_language_t language) {
  if (language >= AMD_COMGR_LANGUAGE_NONE &&
      language <= AMD_COMGR_LANGUAGE_LAST)
    return true;
  return false;
}

static bool
action_is_valid(
  amd_comgr_action_kind_t action_kind)
{
  if (action_kind <= AMD_COMGR_ACTION_LAST)
    return true;
  return false;
}

static amd_comgr_status_t disassemble_object(DisassemHelper &helper,
                                             StringRef cpu, DataObject *input,
                                             DataObject *resp) {
  amd_comgr_status_t status;

  status = (amd_comgr_status_t)helper.DisassembleAction(input->data,
                                                        input->size, cpu);
  if (status)
    return status;

  std::string &result = helper.get_result();

  resp->data_kind = AMD_COMGR_DATA_KIND_SOURCE;
  status = resp->SetData(result);
  result.clear();

  if (status)
    return status;

  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t
dispatch_disassemble_action(amd_comgr_action_kind_t action_kind,
                            DataAction *action_info, DataSet *input_set,
                            DataSet *result_set) {
  amd_comgr_status_t status;
  bool byte_disassem =
      action_kind == AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE;
  DisassemHelper helper(byte_disassem);

  TargetIdentifier Ident;
  if ((status = ParseTargetIdentifier(action_info->isa_name, Ident)))
    return status;

  // Handle the data object in set relevant to the action only
  auto objects =
      make_filter_range(input_set->data_objects, [&](const DataObject *DO) {
        if (action_kind == AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE &&
            DO->data_kind == AMD_COMGR_DATA_KIND_RELOCATABLE)
          return true;
        if (action_kind == AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE &&
            DO->data_kind == AMD_COMGR_DATA_KIND_EXECUTABLE)
          return true;
        if (action_kind == AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE &&
            DO->data_kind == AMD_COMGR_DATA_KIND_BYTES)
          return true;
        return false;
      });
  // Loop through the input data set, perform actions and add result
  // to output data set.
  for (auto input : objects) {
    amd_comgr_data_t result;
    status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_LOG, &result);
    if (status != AMD_COMGR_STATUS_SUCCESS)
      return status;
    ScopedDataObjectReleaser SDOR(result);

    DataObject *resp = DataObject::Convert(result);
    resp->SetName(std::string(input->name) + ".s");
    status = disassemble_object(helper, Ident.Processor, input, resp);
    if (status != AMD_COMGR_STATUS_SUCCESS)
      return status;

    amd_comgr_data_set_t amd_result_set = DataSet::Convert(result_set);
    status = amd_comgr_data_set_add(amd_result_set, result);
    if (status != AMD_COMGR_STATUS_SUCCESS)
      return status;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

static amd_comgr_status_t
dispatch_compiler_action(amd_comgr_action_kind_t action_kind,
                         DataAction *action_info, DataSet *input_set,
                         DataSet *result_set) {
  AMDGPUCompiler Compiler(action_info, input_set, result_set);
  amd_comgr_status_t CompilerStatus;
  switch (action_kind) {
  case AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR:
    CompilerStatus = Compiler.PreprocessToSource();
    break;
  case AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC:
    CompilerStatus = Compiler.CompileToBitcode();
    break;
  case AMD_COMGR_ACTION_LINK_BC_TO_BC:
    CompilerStatus = Compiler.LinkBitcodeToBitcode();
    break;
  case AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE:
    CompilerStatus = Compiler.CodeGenBitcodeToRelocatable();
    break;
  case AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY:
    CompilerStatus = Compiler.CodeGenBitcodeToAssembly();
    break;
  case AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE:
    CompilerStatus = Compiler.AssembleToRelocatable();
    break;
  case AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE:
    CompilerStatus = Compiler.LinkToRelocatable();
    break;
  case AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE:
    CompilerStatus = Compiler.LinkToExecutable();
    break;
  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  amd_comgr_status_t LogsStatus = Compiler.AddLogs();

  return CompilerStatus ? CompilerStatus : LogsStatus;
}

static amd_comgr_status_t
dispatch_add_action(amd_comgr_action_kind_t action_kind,
                    DataAction *action_info, DataSet *input_set,
                    DataSet *result_set) {
#ifdef DEVICE_LIBS
  for (DataObject *datap : input_set->data_objects) {
    datap->refcount++;
    result_set->data_objects.insert(datap);
  }
  switch (action_kind) {
  case AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS:
    return add_precompiled_headers(action_info, result_set);
  case AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES:
    return add_device_libraries(action_info, result_set);
  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
#else
  return AMD_COMGR_STATUS_ERROR;
#endif
}

static bool symbol_info_is_valid(amd_comgr_symbol_info_t attr) {
  if (attr >= AMD_COMGR_SYMBOL_INFO_NAME_LENGTH ||
      attr <= AMD_COMGR_SYMBOL_INFO_LAST)
    return true;
  return false;
}

amd_comgr_status_t COMGR::SetCStr(char *&Dest, StringRef Src, size_t *Size) {
  free(Dest);
  Dest = reinterpret_cast<char *>(malloc(Src.size() + 1));
  if (!Dest)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  memcpy(Dest, Src.data(), Src.size());
  Dest[Src.size()] = '\0';
  if (Size)
    *Size = Src.size();
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t COMGR::ParseTargetIdentifier(StringRef IdentStr,
                                                TargetIdentifier &Ident) {
  SmallVector<StringRef, 5> IsaNameComponents;
  IdentStr.split(IsaNameComponents, '-', 4);
  if (IsaNameComponents.size() != 5)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  Ident.Arch = IsaNameComponents[0];
  Ident.Vendor = IsaNameComponents[1];
  Ident.OS = IsaNameComponents[2];
  Ident.Environ = IsaNameComponents[3];

  Ident.Features.clear();
  IsaNameComponents[4].split(Ident.Features, '+');

  Ident.Processor = Ident.Features[0];
  Ident.Features.erase(Ident.Features.begin());

  return AMD_COMGR_STATUS_SUCCESS;
}

DataObject::DataObject(amd_comgr_data_kind_t kind)
    : data_kind(kind), data(nullptr), name(nullptr), size(0), refcount(1),
      data_sym(nullptr) {}

DataObject::~DataObject() {
  data_kind = AMD_COMGR_DATA_KIND_UNDEF;
  free(data);
  free(name);
  size = 0;
  delete data_sym;
}

DataObject *DataObject::allocate(amd_comgr_data_kind_t data_kind) {
  return new (std::nothrow) DataObject(data_kind);
}

void DataObject::release() {
  if (--refcount == 0)
    delete this;
}

amd_comgr_status_t DataObject::SetName(llvm::StringRef Name) {
  return SetCStr(name, Name);
}

amd_comgr_status_t DataObject::SetData(llvm::StringRef Data) {
  return SetCStr(data, Data, &size);
}

DataSet::DataSet() : data_objects() {}
DataSet::~DataSet() {
  for (DataObject *datap : data_objects)
    datap->release();
}

DataAction::DataAction()
    : isa_name(nullptr),
      action_options(nullptr),
      action_path(nullptr),
      language(AMD_COMGR_LANGUAGE_NONE),
      logging(false) {}

DataAction::~DataAction() {
  free(isa_name);
  free(action_options);
  free(action_path);
}

amd_comgr_status_t DataAction::SetIsaName(llvm::StringRef IsaName) {
  return SetCStr(isa_name, IsaName);
}

amd_comgr_status_t DataAction::SetActionOptions(llvm::StringRef ActionOptions) {
  return SetCStr(action_options, ActionOptions);
}

amd_comgr_status_t DataAction::SetActionPath(llvm::StringRef ActionPath) {
  return SetCStr(action_path, ActionPath);
}

DataMeta::DataMeta() : node(0) {}

DataMeta::~DataMeta() {}

amd_comgr_metadata_kind_t DataMeta::get_metadata_kind() {
  if (msgpack_node)
    return msgpack_node->getKind();
  else {
    if (node.IsScalar())
      return AMD_COMGR_METADATA_KIND_STRING;
    else if (node.IsSequence())
      return AMD_COMGR_METADATA_KIND_LIST;
    else if (node.IsMap())
      return AMD_COMGR_METADATA_KIND_MAP;
    else
      // treat as NULL
      return AMD_COMGR_METADATA_KIND_NULL;
  }
}

DataSymbol::DataSymbol(SymbolContext *data_sym) : data_sym(data_sym) {}
DataSymbol::~DataSymbol() {
  delete data_sym;
}

// Class specific dump functions

void AMD_NOINLINE
DataObject::dump() {
  printf("Data Kind: %d\n", data_kind);
  printf("Name: %s\n", name);
  printf("Size: %ld\n", size);
  printf("Refcount: %d\n", refcount);
}

void AMD_NOINLINE
DataSet::dump() {
  printf("Total data objects: %ld\n", data_objects.size());
  int i = 0;
  for (DataObject *datap: data_objects) {
    printf("--- Data %d ---\n", i++);
    datap->dump();
  }
}

void AMD_NOINLINE
DataMeta::dump() {
  int indent = 0;
  amd_comgr_metadata_node_t key, value;

  DataMeta *keyp = new (std::nothrow) DataMeta();
  keyp->node = NULL;
  key = DataMeta::Convert(keyp);
  DataMeta *valuep = new (std::nothrow) DataMeta();
  valuep->node = node;
  value = DataMeta::Convert(valuep);

  (void)print_entry(key, value, (void*)&indent);
  // debugging use, no error check

  delete(keyp);
  delete(valuep);
}

// Miscellaneous

amd_comgr_status_t AMD_API amd_comgr_status_string(amd_comgr_status_t status,
                                                   const char **status_string) {
  if (status_string == NULL ||
      ((int)status < AMD_COMGR_STATUS_SUCCESS &&
       (int)status > AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  switch(status) {
    case AMD_COMGR_STATUS_SUCCESS:
      *status_string = "SUCCESS";
      break;
    case AMD_COMGR_STATUS_ERROR:
      *status_string = "ERROR";
      break;
    case AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT:
      *status_string = "INVALID_ARGUMENT";
      break;
    case AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES:
      *status_string = "OUT_OF_RESOURCES";
      break;
  }

  // *status_string is a const char *, no need to free

  return AMD_COMGR_STATUS_SUCCESS;
}

void AMD_API amd_comgr_get_version(size_t *major, size_t *minor) {
  *major = AMD_COMGR_INTERFACE_VERSION_MAJOR;
  *minor = AMD_COMGR_INTERFACE_VERSION_MINOR;
}

// API functions on ISA

amd_comgr_status_t AMD_API
amd_comgr_get_isa_count(
  size_t *count)
{
  if (count == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *count = metadata::getIsaCount();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API amd_comgr_get_isa_name(size_t index,
                                                  const char **isa_name) {
  if (isa_name == NULL || index >= metadata::getIsaCount())
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *isa_name = metadata::getIsaName(index);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_get_isa_metadata(
  const char *isa_name,
  amd_comgr_metadata_node_t *metadata)
{
  amd_comgr_status_t status;

  if (isa_name == NULL || metadata == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  DataMeta *metap = new (std::nothrow) DataMeta();
  if (metap == NULL)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  status = metadata::getIsaMetadata(isa_name, metap);
  if (status)
    return status;

  *metadata = DataMeta::Convert(metap);

  return AMD_COMGR_STATUS_SUCCESS;
}

// API functions on Data Object

amd_comgr_status_t AMD_API
amd_comgr_create_data(
  amd_comgr_data_kind_t kind,   // IN
  amd_comgr_data_t *data)       // OUT
{
  if (data == NULL || kind <= AMD_COMGR_DATA_KIND_UNDEF ||
      kind > AMD_COMGR_DATA_KIND_LAST)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  DataObject *datap = DataObject::allocate(kind);
  if (datap == NULL)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  *data = DataObject::Convert(datap);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_release_data(
  amd_comgr_data_t data)        // IN
{
  DataObject *datap = DataObject::Convert(data);

  if (datap == NULL ||
      !datap->kind_is_valid())
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  datap->release();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_get_data_kind(
  amd_comgr_data_t data,        // IN
  amd_comgr_data_kind_t *kind)  // OUT
{
  DataObject *datap = DataObject::Convert(data);

  if (datap == NULL ||
      !datap->kind_is_valid() ||
      kind == NULL) {
    *kind = AMD_COMGR_DATA_KIND_UNDEF;
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *kind = datap->data_kind;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_set_data(
  amd_comgr_data_t data,
  size_t size,
  const char* bytes)
{
  DataObject *datap = DataObject::Convert(data);

  if (bytes == NULL || size <= 0 || datap == NULL || !datap->kind_is_valid() ||
      datap->data_kind == AMD_COMGR_DATA_KIND_UNDEF)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  return datap->SetData(StringRef(bytes, size));
}

amd_comgr_status_t AMD_API
amd_comgr_get_data(
  amd_comgr_data_t data,
  size_t *size,
  char *bytes)
{
  DataObject *datap = DataObject::Convert(data);

  if (datap == NULL || datap->data == NULL ||
      datap->data_kind == AMD_COMGR_DATA_KIND_UNDEF || size == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (bytes != NULL)
    if (*size == datap->size)
      memcpy(bytes, datap->data, *size);
    else
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  else
    *size = datap->size;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_set_data_name(
  amd_comgr_data_t data,
  const char* name)
{
  DataObject *datap = DataObject::Convert(data);

  if (datap == NULL ||
      !datap->kind_is_valid())
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  return datap->SetName(name);
}

amd_comgr_status_t AMD_API
amd_comgr_get_data_name(
  amd_comgr_data_t data,
  size_t *size,
  char *name)
{
  DataObject *datap = DataObject::Convert(data);

  if (datap == NULL ||
      !datap->kind_is_valid() ||
      datap->data_kind == AMD_COMGR_DATA_KIND_UNDEF ||
      size == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (name == NULL)
    *size = strlen(datap->name) + 1;  // include terminating null
  else
    memcpy(name, datap->name, *size);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_get_data_isa_name(
  amd_comgr_data_t data,
  size_t *size,
  char *isa_name)
{
  DataObject *datap = DataObject::Convert(data);

  if (datap == NULL || size == NULL ||
      (datap->data_kind != AMD_COMGR_DATA_KIND_RELOCATABLE &&
       datap->data_kind != AMD_COMGR_DATA_KIND_EXECUTABLE))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  return metadata::getElfIsaName(datap, size, isa_name);
}

// API functions on Data Set

amd_comgr_status_t AMD_API
amd_comgr_create_data_set(
  amd_comgr_data_set_t *data_set)
{
  if (data_set == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  DataSet *datap = new (std::nothrow) DataSet();
  if (datap == NULL)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  *data_set = DataSet::Convert(datap);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_destroy_data_set(
  amd_comgr_data_set_t data_set)
{
  DataSet *setp = DataSet::Convert(data_set);

  if (setp == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  delete setp;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_data_set_add(
  amd_comgr_data_set_t data_set,
  amd_comgr_data_t data)
{
  DataSet *setp = DataSet::Convert(data_set);
  DataObject *datap = DataObject::Convert(data);

  if (setp == NULL ||
      datap == NULL ||
      !datap->kind_is_valid() ||
      datap->data_kind == AMD_COMGR_DATA_KIND_UNDEF ||
      datap->name == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // SmallSetVector: will not add if data was already added
  if (setp->data_objects.insert(datap))
    datap->refcount++;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_data_set_remove(
  amd_comgr_data_set_t data_set,
  amd_comgr_data_kind_t data_kind)
{
  DataSet *setp = DataSet::Convert(data_set);

  if (setp == NULL || data_kind == AMD_COMGR_DATA_KIND_UNDEF ||
      data_kind > AMD_COMGR_DATA_KIND_LAST)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Deleting entries by iterating a setvector is fishy.
  // Use an alternate way.
  SmallVector<DataObject *,8> tmp;
  tmp = setp->data_objects.takeVector();   // take and delete

  for (DataObject *datap: tmp) {
    if (data_kind != AMD_COMGR_DATA_KIND_UNDEF &&
        datap->data_kind != data_kind)
      setp->data_objects.insert(datap);
    else
      datap->release();
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

// API functions on data action

amd_comgr_status_t AMD_API
amd_comgr_action_data_count(
  amd_comgr_data_set_t data_set,
  amd_comgr_data_kind_t data_kind,
  size_t *count)
{
  DataSet *setp = DataSet::Convert(data_set);

  if (setp == NULL || data_kind == AMD_COMGR_DATA_KIND_UNDEF ||
      data_kind > AMD_COMGR_DATA_KIND_LAST || count == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *count = 0;
  for (DataObject *datap: setp->data_objects) {
    if (datap->data_kind == data_kind)
      *count += 1;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_action_data_get_data(
  amd_comgr_data_set_t data_set,
  amd_comgr_data_kind_t data_kind,
  size_t index,
  amd_comgr_data_t *data)
{
  DataSet *setp = DataSet::Convert(data_set);
  amd_comgr_status_t status;

  if (setp == NULL || data_kind == AMD_COMGR_DATA_KIND_UNDEF ||
      data_kind > AMD_COMGR_DATA_KIND_LAST || data == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  size_t n;
  status = amd_comgr_action_data_count(data_set, data_kind, &n);
  if (status != AMD_COMGR_STATUS_SUCCESS)
    return status;
  if (index > n)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  n = 0;
  for (SmallSetVector<DataObject *, 8>::iterator I = setp->data_objects.begin(),
                                                 E = setp->data_objects.end();
       I != E; ++I) {
    if ((*I)->data_kind == data_kind) {
      if (n++ == index) {
        (*I)->refcount++;
        *data = DataObject::Convert(*I);
        return AMD_COMGR_STATUS_SUCCESS;
      }
    }
  }

  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}

amd_comgr_status_t AMD_API
amd_comgr_create_action_info(
  amd_comgr_action_info_t *action_info)
{
  if (action_info == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  DataAction *actionp = new (std::nothrow) DataAction();
  if (actionp == NULL)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  *action_info = DataAction::Convert(actionp);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_destroy_action_info(
  amd_comgr_action_info_t action_info)
{
  DataAction *actionp = DataAction::Convert(action_info);

  if (actionp == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  delete actionp;

  return AMD_COMGR_STATUS_SUCCESS;
}


amd_comgr_status_t AMD_API
amd_comgr_action_info_set_isa_name(
  amd_comgr_action_info_t action_info,
  const char *isa_name)
{
  DataAction *actionp = DataAction::Convert(action_info);

  if (!actionp)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (!isa_name || StringRef(isa_name) == "") {
    free(actionp->isa_name);
    actionp->isa_name = nullptr;
    return AMD_COMGR_STATUS_SUCCESS;
  }

  if (!metadata::isValidIsaName(isa_name))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  return actionp->SetIsaName(isa_name);
}

amd_comgr_status_t AMD_API
amd_comgr_action_info_get_isa_name(
  amd_comgr_action_info_t action_info,
  size_t *size,
  char *isa_name)
{
  DataAction *actionp = DataAction::Convert(action_info);

  if (actionp == NULL ||
      size == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (isa_name == NULL)
    *size = strlen(actionp->isa_name) + 1; // include terminating null
  else
    memcpy(isa_name, actionp->isa_name, *size);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_action_info_set_language(
  amd_comgr_action_info_t action_info,
  amd_comgr_language_t language)
{
  DataAction *actionp = DataAction::Convert(action_info);

  if (actionp == NULL ||
      !language_is_valid(language))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  actionp->language = language;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_action_info_get_language(
  amd_comgr_action_info_t action_info,
  amd_comgr_language_t *language)
{
  DataAction *actionp = DataAction::Convert(action_info);

  if (actionp == NULL ||
      language == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *language = actionp->language;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_action_info_set_options(
  amd_comgr_action_info_t action_info,
  const char *options)
{
  DataAction *actionp = DataAction::Convert(action_info);

  if (actionp == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  return actionp->SetActionOptions(options);
}

amd_comgr_status_t AMD_API
amd_comgr_action_info_get_options(
  amd_comgr_action_info_t action_info,
  size_t *size,
  char *options)
{
  DataAction *actionp = DataAction::Convert(action_info);

  if (actionp == NULL ||
      size == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (options == NULL)
    *size = strlen(actionp->action_options) + 1;  // include terminating 0
  else {
    memcpy(options, actionp->action_options, *size);
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_action_info_set_working_directory_path(
  amd_comgr_action_info_t action_info,
  const char *path)
{
  DataAction *actionp = DataAction::Convert(action_info);

  if (actionp == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  actionp->SetActionPath(path);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_action_info_get_working_directory_path(
  amd_comgr_action_info_t action_info,
  size_t *size,
  char *path)
{
  DataAction *actionp = DataAction::Convert(action_info);

  if (actionp == NULL ||
      size == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (path == NULL)
    *size = strlen(actionp->action_path) + 1;  // include terminating 0
  else {
    memcpy(path, actionp->action_path, *size);
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_action_info_set_logging(
  amd_comgr_action_info_t action_info,
  bool logging)
{
  DataAction *actionp = DataAction::Convert(action_info);

  if (actionp == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  actionp->logging = logging;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_action_info_get_logging(
  amd_comgr_action_info_t action_info,
  bool *logging)
{
  DataAction *actionp = DataAction::Convert(action_info);

  if (actionp == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *logging = actionp->logging;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_do_action(
  amd_comgr_action_kind_t action_kind,
  amd_comgr_action_info_t action_info,
  amd_comgr_data_set_t input_set,
  amd_comgr_data_set_t result_set)
{
  DataAction *actioninfop = DataAction::Convert(action_info);
  DataSet *insetp = DataSet::Convert(input_set);
  DataSet *outsetp = DataSet::Convert(result_set);

  if (!action_is_valid(action_kind) ||
      insetp == NULL ||
      outsetp == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Initialize LLVM once
  if (!DataAction::llvm_initialized) {
    // initialized only once for the entire duration of using the API
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    DataAction::llvm_initialized = true;
  }

  switch (action_kind) {
  case AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE:
  case AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE:
  case AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE:
    return dispatch_disassemble_action(action_kind, actioninfop, insetp,
                                       outsetp);
  case AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR:
  case AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC:
  case AMD_COMGR_ACTION_LINK_BC_TO_BC:
  case AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE:
  case AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY:
  case AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE:
  case AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE:
  case AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE:
    return dispatch_compiler_action(action_kind, actioninfop, insetp, outsetp);
  case AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS:
  case AMD_COMGR_ACTION_ADD_DEVICE_LIBRARIES:
    return dispatch_add_action(action_kind, actioninfop, insetp, outsetp);
  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

// API functions on metadata

amd_comgr_status_t AMD_API amd_comgr_get_data_metadata(
    amd_comgr_data_t data, amd_comgr_metadata_node_t *metadata) {
  DataObject *datap = DataObject::Convert(data);
  amd_comgr_status_t status;

  if (datap == NULL ||
      !datap->kind_is_valid() ||
      datap->data_kind == AMD_COMGR_DATA_KIND_UNDEF ||
      metadata == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  DataMeta *metap = new (std::nothrow) DataMeta();
  if (metap == NULL)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  status = metadata::getMetadataRoot(datap, metap);
  if (status)
    return status;

  // if no metadata found in this data object, still return SUCCESS but
  // with default NULL kind

  // set return metadata
  *metadata = DataMeta::Convert(metap);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_destroy_metadata(amd_comgr_metadata_node_t metadata) {
  DataMeta *metap = DataMeta::Convert(metadata);
  delete metap;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API amd_comgr_get_metadata_kind(
    amd_comgr_metadata_node_t metadata, amd_comgr_metadata_kind_t *kind) {
  DataMeta *metap = DataMeta::Convert(metadata);

  if (kind == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *kind = metap->get_metadata_kind();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t amd_comgr_get_metadata_string_yaml(DataMeta *metap,
                                                      size_t *size,
                                                      char *string) {
  if (size == NULL || !metap->node.IsDefined() ||
      metap->get_metadata_kind() != AMD_COMGR_METADATA_KIND_STRING)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // get length
  if (string == NULL) {
    std::string str = metap->node.as<std::string>();
    *size = str.size() + 1; // ensure null teminator
    return AMD_COMGR_STATUS_SUCCESS;
  }

  // get string
  std::string str = metap->node.as<std::string>();
  memcpy(string, str.c_str(), *size);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API amd_comgr_get_metadata_string(
    amd_comgr_metadata_node_t metadata, size_t *size, char *string) {
  DataMeta *metap = DataMeta::Convert(metadata);

  if (!metap->msgpack_node)
    return amd_comgr_get_metadata_string_yaml(metap, size, string);

  auto String = dyn_cast_or_null<COMGR::msgpack::String>(metap->msgpack_node.get());
  if (!String)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (string)
    memcpy(string, String->Value.c_str(), *size);
  else
    *size = String->Value.size() + 1;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API amd_comgr_get_metadata_map_size_yaml(DataMeta *metap,
                                                                size_t *size) {
  if (size == NULL || metap->get_metadata_kind() != AMD_COMGR_METADATA_KIND_MAP)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *size = metap->node.size();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API amd_comgr_get_metadata_map_size(
    amd_comgr_metadata_node_t metadata, size_t *size) {
  DataMeta *metap = DataMeta::Convert(metadata);

  if (!metap->msgpack_node)
    return amd_comgr_get_metadata_map_size_yaml(metap, size);

  auto Map = dyn_cast_or_null<COMGR::msgpack::Map>(metap->msgpack_node.get());
  if (!Map)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *size = Map->Elements.size();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t amd_comgr_iterate_map_metadata_yaml(
    DataMeta *metap,
    amd_comgr_status_t (*callback)(amd_comgr_metadata_node_t key,
                                   amd_comgr_metadata_node_t value, void *data),
    void *data) {
  if (callback == NULL || !metap->node.IsDefined() ||
      metap->get_metadata_kind() != AMD_COMGR_METADATA_KIND_MAP)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  for (YAML::const_iterator it=metap->node.begin();
       it != metap->node.end();
       ++it) {
    DataMeta *keyp;
    DataMeta *valuep;
    amd_comgr_metadata_node_t key_meta;
    amd_comgr_metadata_node_t value_meta;

    // create new metadata node for key (usually string, but can be anything)
    if (it->first) {
      keyp = new (std::nothrow) DataMeta();
      if (keyp != NULL) {
        keyp->node = it->first;
        key_meta = DataMeta::Convert(keyp);
      } else
        return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    } else
      // something wrong, no key for this map entry
      return AMD_COMGR_STATUS_ERROR;

    // create new metadata node from entry value
    if (it->second) {
      valuep = new (std::nothrow) DataMeta();
      if (valuep != NULL) {
        valuep->node = it->second;
        value_meta = DataMeta::Convert(valuep);
      } else
        return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    } else {
      // something wrong, no value for this map entry
      delete keyp;
      return AMD_COMGR_STATUS_ERROR;
    }

    // call user callback function
    (*callback)(key_meta, value_meta, data);

    delete keyp;
    delete valuep;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API amd_comgr_iterate_map_metadata(
    amd_comgr_metadata_node_t metadata,
    amd_comgr_status_t (*callback)(amd_comgr_metadata_node_t key,
                                   amd_comgr_metadata_node_t value, void *data),
    void *data) {
  DataMeta *metap = DataMeta::Convert(metadata);

  if (!metap->msgpack_node)
    return amd_comgr_iterate_map_metadata_yaml(metap, callback, data);

  auto Map = dyn_cast_or_null<COMGR::msgpack::Map>(metap->msgpack_node.get());
  if (!Map)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  for (auto &KV : Map->Elements) {
    if (!KV.first || !KV.second)
      return AMD_COMGR_STATUS_ERROR;
    auto keyp = std::unique_ptr<DataMeta>(new (std::nothrow) DataMeta());
    auto valuep = std::unique_ptr<DataMeta>(new (std::nothrow) DataMeta());
    if (!keyp || !valuep)
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    keyp->msgpack_node = KV.first;
    valuep->msgpack_node = KV.second;
    (*callback)(DataMeta::Convert(keyp.get()), DataMeta::Convert(valuep.get()),
                data);
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t
amd_comgr_metadata_lookup_yaml(DataMeta *metap, const char *key,
                               amd_comgr_metadata_node_t *value) {
  if (key == NULL || value == NULL || !metap->node.IsDefined() ||
      metap->get_metadata_kind() != AMD_COMGR_METADATA_KIND_MAP)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  DataMeta *new_mp = new (std::nothrow) DataMeta();
  if (new_mp == NULL)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  if (metap->node[key])
    new_mp->node = metap->node[key];
  else
    // not found
    return AMD_COMGR_STATUS_ERROR;

  *value = DataMeta::Convert(new_mp);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_metadata_lookup(amd_comgr_metadata_node_t metadata, const char *key,
                          amd_comgr_metadata_node_t *value) {
  DataMeta *metap = DataMeta::Convert(metadata);

  if (!metap->msgpack_node)
    return amd_comgr_metadata_lookup_yaml(metap, key, value);

  auto Map = dyn_cast_or_null<COMGR::msgpack::Map>(metap->msgpack_node.get());
  if (!Map)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  for (auto &KV : Map->Elements) {
    if (!KV.first || !KV.second)
      return AMD_COMGR_STATUS_ERROR;
    auto String = dyn_cast_or_null<COMGR::msgpack::String>(KV.first.get());
    if (!String)
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    if (String->Value == key) {
      DataMeta *new_mp = new (std::nothrow) DataMeta();
      if (!new_mp)
        return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
      new_mp->msgpack_node = KV.second;
      *value = DataMeta::Convert(new_mp);
      return AMD_COMGR_STATUS_SUCCESS;
    }
  }

  return AMD_COMGR_STATUS_ERROR;
}

amd_comgr_status_t AMD_API
amd_comgr_get_metadata_list_size_yaml(DataMeta *metap, size_t *size) {
  if (size == NULL || !metap->node.IsDefined() ||
      metap->get_metadata_kind() != AMD_COMGR_METADATA_KIND_LIST)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *size = metap->node.size();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API amd_comgr_get_metadata_list_size(
    amd_comgr_metadata_node_t metadata, size_t *size) {
  DataMeta *metap = DataMeta::Convert(metadata);

  if (!metap->msgpack_node)
    return amd_comgr_get_metadata_list_size_yaml(metap, size);

  auto List = dyn_cast_or_null<COMGR::msgpack::List>(metap->msgpack_node.get());
  if (!List)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  *size = List->Elements.size();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t
amd_comgr_index_list_metadata_yaml(DataMeta *metap, size_t index,
                                   amd_comgr_metadata_node_t *value) {
  if (value == NULL || !metap->node.IsDefined() ||
      metap->get_metadata_kind() != AMD_COMGR_METADATA_KIND_LIST ||
      index >= metap->node.size())
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  DataMeta *new_mp = new (std::nothrow) DataMeta();
  if (new_mp == NULL)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  if (metap->node[index])
    new_mp->node = metap->node[index];
  else
    // not found, not possible
    return AMD_COMGR_STATUS_ERROR;

  *value = DataMeta::Convert(new_mp);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_index_list_metadata(amd_comgr_metadata_node_t metadata, size_t index,
                              amd_comgr_metadata_node_t *value) {
  DataMeta *metap = DataMeta::Convert(metadata);

  if (!metap->msgpack_node)
    return amd_comgr_index_list_metadata_yaml(metap, index, value);

  auto List = dyn_cast_or_null<COMGR::msgpack::List>(metap->msgpack_node.get());
  if (!List)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  DataMeta *new_mp = new (std::nothrow) DataMeta();
  if (!new_mp)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  if (index < List->Elements.size())
    new_mp->msgpack_node = List->Elements[index];
  else
    return AMD_COMGR_STATUS_ERROR;

  *value = DataMeta::Convert(new_mp);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API amd_comgr_iterate_symbols(
    amd_comgr_data_t data,
    amd_comgr_status_t (*callback)(amd_comgr_symbol_t symbol, void *user_data),
    void *user_data) {
  amd_comgr_status_t status;
  SymbolHelper helper;
  DataObject *datap = DataObject::Convert(data);

  if (datap == NULL ||
      !datap->kind_is_valid() ||
      !(datap->data_kind == AMD_COMGR_DATA_KIND_RELOCATABLE ||
        datap->data_kind == AMD_COMGR_DATA_KIND_EXECUTABLE) ||
      callback == NULL)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (!DataAction::llvm_initialized)
    LLVMInitializeAMDGPUTargetInfo();  // just need this one only

  StringRef ins(datap->data, datap->size);
  status = helper.iterate_table(ins, datap->data_kind, callback, user_data);

  return status;
}

amd_comgr_status_t AMD_API
amd_comgr_symbol_lookup(
  amd_comgr_data_t data,
  const char *name,
  amd_comgr_symbol_t *symbol)
{
  DataObject *datap = DataObject::Convert(data);
  SymbolHelper helper;

  if (datap == NULL ||
      !datap->kind_is_valid() ||
      !(datap->data_kind == AMD_COMGR_DATA_KIND_RELOCATABLE ||
        datap->data_kind == AMD_COMGR_DATA_KIND_EXECUTABLE))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (!DataAction::llvm_initialized)
    LLVMInitializeAMDGPUTargetInfo();  // just need this one only

  // look through the symbol table for a symbol name based
  // on the data object.

  StringRef ins(datap->data, datap->size);
  SymbolContext *sym = helper.search_symbol(ins, name, datap->data_kind);
  if (!sym)
    return AMD_COMGR_STATUS_ERROR;

  DataSymbol *symp = new (std::nothrow) DataSymbol(sym);
  if (symp == NULL)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  *symbol = DataSymbol::Convert(symp);

  // Update the symbol field in the data object
  delete datap->data_sym;
  datap->data_sym = symp;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_API
amd_comgr_symbol_get_info(
  amd_comgr_symbol_t symbol,
  amd_comgr_symbol_info_t attribute,
  void *value)
{
  DataSymbol *symp = DataSymbol::Convert(symbol);

  if (value == NULL ||
      !symbol_info_is_valid(attribute) ||
      !symp->data_sym)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  SymbolContext *sym = symp->data_sym;

  // retrieve specified symbol info
  switch (attribute) {
    case AMD_COMGR_SYMBOL_INFO_NAME_LENGTH:
      *(size_t *)value = strlen(sym->name);
      break;
    case AMD_COMGR_SYMBOL_INFO_NAME:
      // if user did not allocate AMD_COMGR_SYMBOL_INFO_NAME_LENGTH+1 characters
      // space and passed its address, this may corrupt space.
      // symp->name always have a null terminator.
      strcpy((char *)value, sym->name);
      break;
    case AMD_COMGR_SYMBOL_INFO_TYPE:
      *(amd_comgr_symbol_type_t *)value = sym->type;
      break;
    case AMD_COMGR_SYMBOL_INFO_SIZE:
      *(uint64_t*)value = sym->size;
      break;
    case AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED:
      *(bool*)value = sym->undefined;
      break;
    case AMD_COMGR_SYMBOL_INFO_VALUE:
      *(uint64_t*)value = sym->value;
      break;

    default: return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}
