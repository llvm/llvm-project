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

#include "comgr.h"
#include "comgr-compiler.h"
#include "comgr-device-libs.h"
#include "comgr-disassembly.h"
#include "comgr-env.h"
#include "comgr-metadata.h"
#include "comgr-objdump.h"
#include "comgr-signal.h"
#include "comgr-symbol.h"
#include "comgr-symbolizer.h"

#include "clang/Basic/Version.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/TargetSelect.h"
#include <fstream>
#include <mutex>
#include <string>

#include "time-stat/ts-interface.h"

#ifndef AMD_NOINLINE
#ifdef __GNUC__
#define AMD_NOINLINE __attribute__((noinline))
#else
#define AMD_NOINLINE __declspec(noinline)
#endif
#endif

// Needed for stringification of macro expansions for git branch/commit macros
#define xstringify(x) stringify(x)
#define stringify(x) #x

using namespace llvm;
using namespace COMGR;
using namespace COMGR::TimeStatistics;

namespace {
bool isLanguageValid(amd_comgr_language_t Language) {
  return Language >= AMD_COMGR_LANGUAGE_NONE &&
         Language <= AMD_COMGR_LANGUAGE_LAST;
}

bool isActionValid(amd_comgr_action_kind_t ActionKind) {
  return ActionKind <= AMD_COMGR_ACTION_LAST;
}

bool isSymbolInfoValid(amd_comgr_symbol_info_t SymbolInfo) {
  return SymbolInfo >= AMD_COMGR_SYMBOL_INFO_NAME_LENGTH &&
         SymbolInfo <= AMD_COMGR_SYMBOL_INFO_LAST;
}

amd_comgr_status_t dispatchDisassembleAction(amd_comgr_action_kind_t ActionKind,
                                             DataAction *ActionInfo,
                                             DataSet *InputSet,
                                             DataSet *ResultSet,
                                             raw_ostream &LogS) {
  amd_comgr_data_set_t ResultSetT = DataSet::convert(ResultSet);

  std::string Out;
  raw_string_ostream OutS(Out);
  DisassemHelper Helper(OutS, LogS);

  TargetIdentifier Ident;
  if (auto Status = parseTargetIdentifier(ActionInfo->IsaName, Ident)) {
    return Status;
  }

  // Handle the data object in set relevant to the action only
  auto Objects =
      make_filter_range(InputSet->DataObjects, [&](const DataObject *DO) {
        if (ActionKind == AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE &&
            DO->DataKind == AMD_COMGR_DATA_KIND_RELOCATABLE) {
          return true;
        }
        if (ActionKind == AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE &&
            DO->DataKind == AMD_COMGR_DATA_KIND_EXECUTABLE) {
          return true;
        }
        if (ActionKind == AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE &&
            DO->DataKind == AMD_COMGR_DATA_KIND_BYTES) {
          return true;
        }
        return false;
      });
  std::vector<std::string> Options;
  Options.emplace_back("-disassemble");
  Options.push_back((Twine("-mcpu=") + Ident.Processor).str());
  auto ActionOptions = ActionInfo->getOptions();
  Options.insert(Options.end(), ActionOptions.begin(), ActionOptions.end());
  // Loop through the input data set, perform actions and add result
  // to output data set.
  for (auto *Input : Objects) {
    if (auto Status = Helper.disassembleAction(
            StringRef(Input->Data, Input->Size), Options)) {
      return Status;
    }

    amd_comgr_data_t ResultT;
    if (auto Status =
            amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &ResultT)) {
      return Status;
    }
    ScopedDataObjectReleaser ResultSDOR(ResultT);
    DataObject *Result = DataObject::convert(ResultT);
    if (auto Status = Result->setName(std::string(Input->Name) + ".s")) {
      return Status;
    }
    if (auto Status = Result->setData(OutS.str())) {
      return Status;
    }
    Out.clear();
    if (auto Status = amd_comgr_data_set_add(ResultSetT, ResultT)) {
      return Status;
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t dispatchCompilerAction(amd_comgr_action_kind_t ActionKind,
                                          DataAction *ActionInfo,
                                          DataSet *InputSet, DataSet *ResultSet,
                                          raw_ostream &LogS) {
  AMDGPUCompiler Compiler(ActionInfo, InputSet, ResultSet, LogS);
  switch (ActionKind) {
  case AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR:
    return Compiler.preprocessToSource();
  case AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC:
    return Compiler.compileToBitcode();
  case AMD_COMGR_ACTION_UNBUNDLE:
    return Compiler.unbundle();
  case AMD_COMGR_ACTION_LINK_BC_TO_BC:
    return Compiler.linkBitcodeToBitcode();
  case AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE:
    return Compiler.codeGenBitcodeToRelocatable();
  case AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY:
    return Compiler.codeGenBitcodeToAssembly();
  case AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE:
    return Compiler.assembleToRelocatable();
  case AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE:
    return Compiler.linkToRelocatable();
  case AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE:
    return Compiler.linkToExecutable();
  case AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE:
    return Compiler.compileToRelocatable();
  case AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC:
    return Compiler.compileToBitcode(true);
  case AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE:
    return Compiler.compileToExecutable();
  case AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC:
    return Compiler.translateSpirvToBitcode();

  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

amd_comgr_status_t dispatchAddAction(amd_comgr_action_kind_t ActionKind,
                                     DataAction *ActionInfo, DataSet *InputSet,
                                     DataSet *ResultSet) {
  for (DataObject *Data : InputSet->DataObjects) {
    Data->RefCount++;
    ResultSet->DataObjects.insert(Data);
  }
  switch (ActionKind) {
  case AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS:
    return addPrecompiledHeaders(ActionInfo, ResultSet);
  default:
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

StringRef getLanguageName(amd_comgr_language_t Language) {
  switch (Language) {
  case AMD_COMGR_LANGUAGE_NONE:
    return "AMD_COMGR_LANGUAGE_NONE";
  case AMD_COMGR_LANGUAGE_OPENCL_1_2:
    return "AMD_COMGR_LANGUAGE_OPENCL_1_2";
  case AMD_COMGR_LANGUAGE_OPENCL_2_0:
    return "AMD_COMGR_LANGUAGE_OPENCL_2_0";
  case AMD_COMGR_LANGUAGE_HIP:
    return "AMD_COMGR_LANGUAGE_HIP";
  case AMD_COMGR_LANGUAGE_LLVM_IR:
    return "AMD_COMGR_LANGUAGE_LLVM_IR";
  }

  llvm_unreachable("invalid language");
}

StringRef getStatusName(amd_comgr_status_t Status) {
  switch (Status) {
  case AMD_COMGR_STATUS_SUCCESS:
    return "AMD_COMGR_STATUS_SUCCESS";
  case AMD_COMGR_STATUS_ERROR:
    return "AMD_COMGR_STATUS_ERROR";
  case AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT:
    return "AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT";
  case AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES:
    return "AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES";
  }

  llvm_unreachable("invalid status");
}

/// Perform a simple quoting of an option to allow separating options with
/// space in debug output. The option is surrounded by double quotes, and
/// any embedded double quotes or backslashes are preceeded by a backslash.
void printQuotedOption(raw_ostream &OS, StringRef Option) {
  OS << '"';
  for (const char C : Option) {
    if (C == '"' || C == '\\') {
      OS << '\\';
    }
    OS << C;
  }
  OS << '"';
}
} // namespace

StringRef getActionKindName(amd_comgr_action_kind_t ActionKind) {
  switch (ActionKind) {
  case AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR:
    return "AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR";
  case AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS:
    return "AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS";
  case AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC:
    return "AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC";
  case AMD_COMGR_ACTION_LINK_BC_TO_BC:
    return "AMD_COMGR_ACTION_LINK_BC_TO_BC";
  case AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE:
    return "AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE";
  case AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY:
    return "AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY";
  case AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE:
    return "AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE";
  case AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE:
    return "AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE";
  case AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE:
    return "AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE";
  case AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE:
    return "AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE";
  case AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE:
    return "AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE";
  case AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE:
    return "AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE";
  case AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE:
    return "AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE";
  case AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC:
    return "AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC";
  case AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE:
    return "AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE";
  case AMD_COMGR_ACTION_UNBUNDLE:
    return "AMD_COMGR_ACTION_UNBUNDLE";
  case AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC:
    return "AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC";
  }

  llvm_unreachable("invalid action");
}

bool COMGR::isDataKindValid(amd_comgr_data_kind_t DataKind) {
  return DataKind > AMD_COMGR_DATA_KIND_UNDEF &&
         DataKind <= AMD_COMGR_DATA_KIND_LAST;
}

amd_comgr_status_t COMGR::setCStr(char *&Dest, StringRef Src, size_t *Size) {
  free(Dest);
  Dest = reinterpret_cast<char *>(malloc(Src.size() + 1));
  if (!Dest) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  memcpy(Dest, Src.data(), Src.size());
  Dest[Src.size()] = '\0';
  if (Size) {
    *Size = Src.size();
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

StringRef COMGR::getComgrHashIdentifier() {
  return xstringify(AMD_COMGR_VERSION_ID);
}

amd_comgr_status_t COMGR::parseTargetIdentifier(StringRef IdentStr,
                                                TargetIdentifier &Ident) {
  SmallVector<StringRef, 5> IsaNameComponents;
  IdentStr.split(IsaNameComponents, '-', 4);
  if (IsaNameComponents.size() != 5) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  Ident.Arch = IsaNameComponents[0];
  Ident.Vendor = IsaNameComponents[1];
  Ident.OS = IsaNameComponents[2];
  Ident.Environ = IsaNameComponents[3];

  Ident.Features.clear();
  IsaNameComponents[4].split(Ident.Features, ':');

  Ident.Processor = Ident.Features[0];
  Ident.Features.erase(Ident.Features.begin());

  size_t IsaIndex;

  amd_comgr_status_t Status = metadata::getIsaIndex(IdentStr, IsaIndex);
  if (Status != AMD_COMGR_STATUS_SUCCESS) {
    return Status;
  }

  for (auto Feature : Ident.Features) {
    if (!metadata::isSupportedFeature(IsaIndex, Feature)) {
      return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

void COMGR::ensureLLVMInitialized() {

  // LLVMInitializeAMDGPUTargetInfo calls TargetRegistry.cpp:RegisterTarget()
  // This function is not thread safe. There may be thread safety issues
  // with the other LLVMInitialize functions as well. For completeness, we
  // include all of these initialization functions in mutual exclusion region
  // TODO: remove mutex once LLVM multi-threading issues are resolved
  static std::mutex LlvmInitMutex;
  {
    std::scoped_lock<std::mutex> LlvmInitLock(LlvmInitMutex);

    static bool LLVMInitialized = false;
    if (LLVMInitialized) {
      return;
    }
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitialized = true;
  }
}

void COMGR::clearLLVMOptions() {
  cl::ResetAllOptionOccurrences();
  for (auto *SC : cl::getRegisteredSubcommands()) {
    for (auto &OM : SC->OptionsMap) {
      cl::Option *O = OM.second;
      O->setDefault();
    }
  }
}

DataObject::DataObject(amd_comgr_data_kind_t DataKind)
    : DataKind(DataKind), Data(nullptr), Name(nullptr), Size(0), RefCount(1),
      DataSym(nullptr) {}

DataObject::~DataObject() {
  DataKind = AMD_COMGR_DATA_KIND_UNDEF;
  clearData();
  free(Name);
  delete DataSym;
}

DataObject *DataObject::allocate(amd_comgr_data_kind_t DataKind) {
  return new (std::nothrow) DataObject(DataKind);
}

void DataObject::release() {
  if (--RefCount == 0) {
    delete this;
  }
}

amd_comgr_status_t DataObject::setName(llvm::StringRef Name) {
  return setCStr(this->Name, Name);
}

amd_comgr_status_t DataObject::setData(llvm::StringRef Data) {
  clearData();
  return setCStr(this->Data, Data, &Size);
}

amd_comgr_status_t DataObject::setData(std::unique_ptr<llvm::MemoryBuffer> MB) {
  Buffer = std::move(MB);
  Data = const_cast<char *>(Buffer->getBufferStart());
  Size = Buffer->getBufferSize();
  MangledNames.clear();
  return AMD_COMGR_STATUS_SUCCESS;
}

void DataObject::clearData() {
  if (Buffer) {
    Buffer.reset();
  } else {
    free(Data);
  }

  Data = nullptr;
  Size = 0;
  MangledNames.clear();
}

DataSet::DataSet() : DataObjects() {}
DataSet::~DataSet() {
  for (DataObject *Data : DataObjects) {
    Data->release();
  }
}

DataAction::DataAction()
    : IsaName(nullptr), Path(nullptr), Language(AMD_COMGR_LANGUAGE_NONE),
      Logging(false) {}

DataAction::~DataAction() {
  free(IsaName);
  free(Path);
}

amd_comgr_status_t DataAction::setIsaName(llvm::StringRef IsaName) {
  return setCStr(this->IsaName, IsaName);
}

amd_comgr_status_t DataAction::setActionPath(llvm::StringRef ActionPath) {
  return setCStr(this->Path, ActionPath);
}

amd_comgr_status_t DataAction::setOptionList(ArrayRef<const char *> Options) {
  ListOptions.clear();
  for (auto &Option : Options) {
    ListOptions.push_back(Option);
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t DataAction::getOptionListCount(size_t &Size) {
  Size = ListOptions.size();
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t DataAction::getOptionListItem(size_t Index,
                                                 StringRef &Option) {
  if (Index >= ListOptions.size()) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }
  auto &Str = ListOptions[Index];
  Option = StringRef(Str.c_str(), Str.size() + 1);
  return AMD_COMGR_STATUS_SUCCESS;
}

ArrayRef<std::string> DataAction::getOptions() { return ListOptions; }

amd_comgr_status_t
DataAction::setBundleEntryIDs(ArrayRef<const char *> EntryIDs) {
  BundleEntryIDs.clear();
  for (auto &ID : EntryIDs) {
    BundleEntryIDs.push_back(ID);
  }
  return AMD_COMGR_STATUS_SUCCESS;
}

ArrayRef<std::string> DataAction::getBundleEntryIDs() { return BundleEntryIDs; }

amd_comgr_metadata_kind_t DataMeta::getMetadataKind() {
  if (DocNode.isScalar()) {
    return AMD_COMGR_METADATA_KIND_STRING;
  }
  if (DocNode.isArray()) {
    return AMD_COMGR_METADATA_KIND_LIST;
  }
  if (DocNode.isMap()) {
    return AMD_COMGR_METADATA_KIND_MAP;
  }
  // treat as NULL
  return AMD_COMGR_METADATA_KIND_NULL;
}

std::string DataMeta::convertDocNodeToString(msgpack::DocNode DocNode) {
  assert(DocNode.isScalar() && "cannot convert non-scalar DocNode to string");
  if (MetaDoc->EmitIntegerBooleans &&
      DocNode.getKind() == msgpack::Type::Boolean) {
    return DocNode.getBool() ? "1" : "0";
  }
  return DocNode.toString();
}

DataSymbol::DataSymbol(SymbolContext *DataSym) : DataSym(DataSym) {}
DataSymbol::~DataSymbol() { delete DataSym; }

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_status_string
    //
    (amd_comgr_status_t Status, const char **StatusString) {
  if (!StatusString || Status < AMD_COMGR_STATUS_SUCCESS ||
      Status > AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  switch (Status) {
  case AMD_COMGR_STATUS_SUCCESS:
    *StatusString = "SUCCESS";
    break;
  case AMD_COMGR_STATUS_ERROR:
    *StatusString = "ERROR";
    break;
  case AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT:
    *StatusString = "INVALID_ARGUMENT";
    break;
  case AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES:
    *StatusString = "OUT_OF_RESOURCES";
    break;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

void AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_version
    //
    (size_t *Major, size_t *Minor) {
  *Major = AMD_COMGR_INTERFACE_VERSION_MAJOR;
  *Minor = AMD_COMGR_INTERFACE_VERSION_MINOR;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_isa_count
    //
    (size_t *Count) {
  if (!Count) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *Count = metadata::getIsaCount();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_isa_name
    //
    (size_t Index, const char **IsaName) {
  if (!IsaName || Index >= metadata::getIsaCount()) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *IsaName = metadata::getIsaName(Index);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_isa_metadata
    //
    (const char *IsaName, amd_comgr_metadata_node_t *MetadataNode) {
  if (!IsaName || !MetadataNode) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  std::unique_ptr<DataMeta> MetaP(new (std::nothrow) DataMeta());
  if (!MetaP) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  std::unique_ptr<MetaDocument> MetaDoc(new (std::nothrow) MetaDocument());
  if (!MetaDoc) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  if (auto Status = metadata::getIsaMetadata(IsaName, MetaDoc->Document)) {
    return Status;
  }

  MetaP->MetaDoc = std::move(MetaDoc);
  MetaP->MetaDoc->EmitIntegerBooleans = true;
  MetaP->DocNode = MetaP->MetaDoc->Document.getRoot();

  *MetadataNode = DataMeta::convert(MetaP.release());

  return AMD_COMGR_STATUS_SUCCESS;
}

// API functions on Data Object

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_create_data
    //
    (amd_comgr_data_kind_t DataKind, amd_comgr_data_t *Data) {
  if (!Data || DataKind <= AMD_COMGR_DATA_KIND_UNDEF ||
      DataKind > AMD_COMGR_DATA_KIND_LAST) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  DataObject *DataP = DataObject::allocate(DataKind);
  if (!DataP) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  *Data = DataObject::convert(DataP);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_release_data
    //
    (amd_comgr_data_t Data) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind()) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  DataP->release();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_data_kind
    //
    (amd_comgr_data_t Data, amd_comgr_data_kind_t *DataKind) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind() || !DataKind) {
    *DataKind = AMD_COMGR_DATA_KIND_UNDEF;
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *DataKind = DataP->DataKind;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_set_data
    //
    (amd_comgr_data_t Data, size_t Size, const char *Bytes) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind() || !Size || !Bytes) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return DataP->setData(StringRef(Bytes, Size));
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_data
    //
    (amd_comgr_data_t Data, size_t *Size, char *Bytes) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->Data || !DataP->hasValidDataKind() || !Size) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (Bytes) {
    memcpy(Bytes, DataP->Data, *Size);
  } else {
    *Size = DataP->Size;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_set_data_name
    //
    (amd_comgr_data_t Data, const char *Name) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind()) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return DataP->setName(Name);
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_data_name
    //
    (amd_comgr_data_t Data, size_t *Size, char *Name) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind() || !Size) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (Name) {
    memcpy(Name, DataP->Name, *Size);
  } else {
    *Size = strlen(DataP->Name) + 1; // include terminating null
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_create_symbolizer_info
    //
    (amd_comgr_data_t CodeObject,
     void (*PrintSymbolCallback)(const char *, void *),
     amd_comgr_symbolizer_info_t *SymbolizerInfo) {

  DataObject *CodeObjectP = DataObject::convert(CodeObject);
  if (!CodeObjectP || !PrintSymbolCallback ||
      !(CodeObjectP->DataKind == AMD_COMGR_DATA_KIND_RELOCATABLE ||
        CodeObjectP->DataKind == AMD_COMGR_DATA_KIND_EXECUTABLE ||
        CodeObjectP->DataKind == AMD_COMGR_DATA_KIND_BYTES))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  ensureLLVMInitialized();

  return Symbolizer::create(CodeObjectP, PrintSymbolCallback, SymbolizerInfo);
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_destroy_symbolizer_info
    //
    (amd_comgr_symbolizer_info_t SymbolizerInfo) {

  Symbolizer *SI = Symbolizer::convert(SymbolizerInfo);
  if (!SI) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  delete SI;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_symbolize
    //
    (amd_comgr_symbolizer_info_t SymbolizeInfo, uint64_t Address, bool IsCode,
     void *UserData) {

  Symbolizer *SI = Symbolizer::convert(SymbolizeInfo);
  if (!SI || !UserData) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return SI->symbolize(Address, IsCode, UserData);
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_data_isa_name
    //
    (amd_comgr_data_t Data, size_t *Size, char *IsaName) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !Size ||
      (DataP->DataKind != AMD_COMGR_DATA_KIND_RELOCATABLE &&
       DataP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  std::string ElfIsaName;
  amd_comgr_status_t Status = metadata::getElfIsaName(DataP, ElfIsaName);

  if (Status == AMD_COMGR_STATUS_SUCCESS) {
    if (IsaName) {
      memcpy(IsaName, ElfIsaName.c_str(),
             std::min(*Size, ElfIsaName.size() + 1));
    }

    *Size = ElfIsaName.size() + 1;
  }

  return Status;
}

// API functions on Data Set

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_create_data_set
    //
    (amd_comgr_data_set_t *Set) {
  if (!Set) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  DataSet *SetP = new (std::nothrow) DataSet();
  if (!SetP) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  *Set = DataSet::convert(SetP);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_destroy_data_set
    //
    (amd_comgr_data_set_t Set) {
  DataSet *SetP = DataSet::convert(Set);

  if (!SetP) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  delete SetP;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_data_set_add
    //
    (amd_comgr_data_set_t Set, amd_comgr_data_t Data) {
  DataSet *SetP = DataSet::convert(Set);
  DataObject *DataP = DataObject::convert(Data);

  if (!SetP || !DataP || !DataP->hasValidDataKind() || !DataP->Name) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // SmallSetVector: will not add if data was already added
  if (SetP->DataObjects.insert(DataP)) {
    DataP->RefCount++;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_data_set_remove
    //
    (amd_comgr_data_set_t Set, amd_comgr_data_kind_t DataKind) {
  DataSet *SetP = DataSet::convert(Set);

  if (!SetP || !isDataKindValid(DataKind)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  SmallVector<DataObject *, 8> Tmp = SetP->DataObjects.takeVector();

  for (DataObject *Data : Tmp) {
    if (Data->DataKind == DataKind) {
      Data->release();
    } else {
      SetP->DataObjects.insert(Data);
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_data_count
    //
    (amd_comgr_data_set_t Set, amd_comgr_data_kind_t DataKind, size_t *Count) {
  DataSet *SetP = DataSet::convert(Set);

  if (!SetP || !isDataKindValid(DataKind) || !Count) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *Count = 0;
  for (DataObject *Data : SetP->DataObjects) {
    if (Data->DataKind == DataKind) {
      *Count += 1;
    }
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_data_get_data
    //
    (amd_comgr_data_set_t Set, amd_comgr_data_kind_t DataKind, size_t Index,
     amd_comgr_data_t *Data) {
  DataSet *SetP = DataSet::convert(Set);

  if (!SetP || !isDataKindValid(DataKind) || !Data) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  size_t N;
  if (auto Status = amd_comgr_action_data_count(Set, DataKind, &N)) {
    return Status;
  }
  if (Index > N) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  N = 0;
  for (auto &I : SetP->DataObjects) {
    if (I->DataKind == DataKind) {
      if (N++ == Index) {
        I->RefCount++;
        *Data = DataObject::convert(I);
        return AMD_COMGR_STATUS_SUCCESS;
      }
    }
  }

  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_create_action_info
    //
    (amd_comgr_action_info_t *ActionInfo) {
  if (!ActionInfo) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  DataAction *ActionP = new (std::nothrow) DataAction();
  if (!ActionP) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  *ActionInfo = DataAction::convert(ActionP);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_destroy_action_info
    //
    (amd_comgr_action_info_t ActionInfo) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  delete ActionP;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_set_isa_name
    //
    (amd_comgr_action_info_t ActionInfo, const char *IsaName) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (!IsaName || StringRef(IsaName) == "") {
    free(ActionP->IsaName);
    ActionP->IsaName = nullptr;
    return AMD_COMGR_STATUS_SUCCESS;
  }

  if (!metadata::isValidIsaName(IsaName)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ActionP->setIsaName(IsaName);
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_get_isa_name
    //
    (amd_comgr_action_info_t ActionInfo, size_t *Size, char *IsaName) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP || !Size) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (IsaName) {
    memcpy(IsaName, ActionP->IsaName, *Size);
  } else {
    *Size = strlen(ActionP->IsaName) + 1; // include terminating null
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_set_language
    //
    (amd_comgr_action_info_t ActionInfo, amd_comgr_language_t Language) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP || !isLanguageValid(Language)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ActionP->Language = Language;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_get_language
    //
    (amd_comgr_action_info_t ActionInfo, amd_comgr_language_t *Language) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP || !Language) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *Language = ActionP->Language;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_set_option_list
    //
    (amd_comgr_action_info_t ActionInfo, const char *Options[], size_t Count) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP || (!Options && Count)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ActionP->setOptionList(ArrayRef<const char *>(Options, Count));
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_get_option_list_count
    //
    (amd_comgr_action_info_t ActionInfo, size_t *Count) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP || !Count) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ActionP->getOptionListCount(*Count);
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_get_option_list_item
    //
    (amd_comgr_action_info_t ActionInfo, size_t Index, size_t *Size,
     char *Option) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP || !Size) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  StringRef ActionOption;
  if (auto Status = ActionP->getOptionListItem(Index, ActionOption)) {
    return Status;
  }

  if (Option) {
    memcpy(Option, ActionOption.data(), *Size);
  } else {
    *Size = ActionOption.size();
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_get_bundle_entry_id_count
    //
    (amd_comgr_action_info_t ActionInfo, size_t *Count) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *Count = ActionP->getBundleEntryIDs().size();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_get_bundle_entry_id
    //
    (amd_comgr_action_info_t ActionInfo, size_t Index, size_t *Size,
     char *BundleEntryID) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP || !Size) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ArrayRef<std::string> ActionBundleEntryIDs = ActionP->getBundleEntryIDs();

  if (Index >= ActionBundleEntryIDs.size()) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // First return the size of the BundleEntryID
  if (BundleEntryID == NULL)
    *Size = ActionBundleEntryIDs[Index].size() + 1;

  // Now that the calling API has had a chance to allocate memory, copy the
  // bundle entry ID at Index to BundleEntryID
  else
    memcpy(BundleEntryID, ActionBundleEntryIDs[Index].c_str(), *Size);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_set_bundle_entry_ids
    //
    (amd_comgr_action_info_t ActionInfo, const char *EntryIDs[], size_t Count) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP || (!EntryIDs && Count)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return ActionP->setBundleEntryIDs(ArrayRef<const char *>(EntryIDs, Count));
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_set_device_lib_linking
    //
    (amd_comgr_action_info_t ActionInfo, bool ShouldLinkDeviceLibs) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ActionP->ShouldLinkDeviceLibs = ShouldLinkDeviceLibs;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_set_working_directory_path
    //
    (amd_comgr_action_info_t ActionInfo, const char *Path) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ActionP->setActionPath(Path);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_get_working_directory_path
    //
    (amd_comgr_action_info_t ActionInfo, size_t *Size, char *Path) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP || !Size) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (Path) {
    memcpy(Path, ActionP->Path, *Size);
  } else {
    *Size = strlen(ActionP->Path) + 1; // include terminating 0
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_set_logging
    //
    (amd_comgr_action_info_t ActionInfo, bool Logging) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ActionP->Logging = Logging;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_action_info_get_logging
    //
    (amd_comgr_action_info_t ActionInfo, bool *Logging) {
  DataAction *ActionP = DataAction::convert(ActionInfo);

  if (!ActionP) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *Logging = ActionP->Logging;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_do_action
    //
    (amd_comgr_action_kind_t ActionKind, amd_comgr_action_info_t ActionInfo,
     amd_comgr_data_set_t InputSet, amd_comgr_data_set_t ResultSet) {
  DataAction *ActionInfoP = DataAction::convert(ActionInfo);
  DataSet *InputSetP = DataSet::convert(InputSet);
  DataSet *ResultSetP = DataSet::convert(ResultSet);

  if (!isActionValid(ActionKind) || !InputSetP || !ResultSetP) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  amd_comgr_status_t ActionStatus;

  // Enclose core Comgr actions in a mutally excusive region to avoid
  // multithreading issues stemming from concurrently maintaing multiple
  // LLVM instances.
  // TODO: Remove the scoped lock once updates to LLVM enable thread saftey
  static std::mutex ComgrMutex;
  {
    std::scoped_lock<std::mutex> ComgrLock(ComgrMutex);

    ensureLLVMInitialized();

    // Save signal handlers so that they can be restored after the action has
    // completed.
    if (auto Status = signal::saveHandlers()) {
      return Status;
    }

    // The normal log stream, used to return via a AMD_COMGR_DATA_KIND_LOG
    // object.
    std::string LogStr;
    std::string PerfLog = "PerfStatsLog.txt";
    raw_string_ostream LogS(LogStr);

    // The log stream when redirecting to a file.
    std::unique_ptr<raw_fd_ostream> LogF;

    // Pointer to the currently selected log stream.
    raw_ostream *LogP = &LogS;

    if (std::optional<StringRef> RedirectLogs = env::getRedirectLogs()) {
      StringRef RedirectLog = *RedirectLogs;
      if (RedirectLog == "stdout") {
        LogP = &outs();
      } else if (RedirectLog == "stderr") {
        LogP = &errs();
      } else {
        std::error_code EC;
        LogF.reset(new (std::nothrow) raw_fd_ostream(
            RedirectLog, EC, sys::fs::OF_Text | sys::fs::OF_Append));
        if (EC) {
          LogF.reset();
          *LogP << "Comgr unable to redirect log to file '" << RedirectLog
                << "': " << EC.message() << "\n";
        } else {
          LogP = LogF.get();
          PerfLog = RedirectLog.str();
        }
      }
    }

    InitTimeStatistics(PerfLog);

    if (env::shouldEmitVerboseLogs()) {
      *LogP << "amd_comgr_do_action:\n"
            << "\t  ActionKind: " << getActionKindName(ActionKind) << '\n'
            << "\t     IsaName: " << ActionInfoP->IsaName << '\n'
            << "\t     Options:";
      for (auto &Option : ActionInfoP->getOptions()) {
        *LogP << ' ';
        printQuotedOption(*LogP, Option);
      }
      *LogP << '\n'
            << "\t        Path: " << ActionInfoP->Path << '\n'
            << "\t    Language: " << getLanguageName(ActionInfoP->Language)
            << '\n'
            << " Comgr Branch-Commit: " << xstringify(AMD_COMGR_GIT_BRANCH)
            << '-' << xstringify(AMD_COMGR_GIT_COMMIT) << '\n'
            << "\t LLVM Commit: " << clang::getLLVMRevision() << '\n';
      (*LogP).flush();
    }

    ProfilePoint ProfileAction(getActionKindName(ActionKind));
    switch (ActionKind) {
    case AMD_COMGR_ACTION_DISASSEMBLE_RELOCATABLE_TO_SOURCE:
    case AMD_COMGR_ACTION_DISASSEMBLE_EXECUTABLE_TO_SOURCE:
    case AMD_COMGR_ACTION_DISASSEMBLE_BYTES_TO_SOURCE:
      ActionStatus = dispatchDisassembleAction(ActionKind, ActionInfoP,
                                               InputSetP, ResultSetP, *LogP);
      break;
    case AMD_COMGR_ACTION_SOURCE_TO_PREPROCESSOR:
    case AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC:
    case AMD_COMGR_ACTION_UNBUNDLE:
    case AMD_COMGR_ACTION_LINK_BC_TO_BC:
    case AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE:
    case AMD_COMGR_ACTION_CODEGEN_BC_TO_ASSEMBLY:
    case AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE:
    case AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_RELOCATABLE:
    case AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE:
    case AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE:
    case AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC:
    case AMD_COMGR_ACTION_COMPILE_SOURCE_TO_EXECUTABLE:
    case AMD_COMGR_ACTION_TRANSLATE_SPIRV_TO_BC:
      ActionStatus = dispatchCompilerAction(ActionKind, ActionInfoP, InputSetP,
                                            ResultSetP, *LogP);
      break;
    case AMD_COMGR_ACTION_ADD_PRECOMPILED_HEADERS:
      ActionStatus =
          dispatchAddAction(ActionKind, ActionInfoP, InputSetP, ResultSetP);
      break;
    default:
      ActionStatus = AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
    }
    ProfileAction.finish();

    // Restore signal handlers.
    if (auto Status = signal::restoreHandlers()) {
      return Status;
    }

    if (env::shouldEmitVerboseLogs()) {
      *LogP << "\tReturnStatus: " << getStatusName(ActionStatus) << "\n\n";
    }

    if (ActionInfoP->Logging) {
      amd_comgr_data_t LogT;
      if (auto Status = amd_comgr_create_data(AMD_COMGR_DATA_KIND_LOG, &LogT)) {
        return Status;
      }
      ScopedDataObjectReleaser LogSDOR(LogT);
      DataObject *Log = DataObject::convert(LogT);
      if (auto Status = Log->setName("comgr.log")) {
        return Status;
      }
      if (auto Status = Log->setData(LogS.str())) {
        return Status;
      }
      if (auto Status = amd_comgr_data_set_add(ResultSet, LogT)) {
        return Status;
      }
    }
  } // exit scoped_lock region

  return ActionStatus;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_data_metadata
    //
    (amd_comgr_data_t Data, amd_comgr_metadata_node_t *MetadataNode) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind() ||
      DataP->DataKind == AMD_COMGR_DATA_KIND_UNDEF || !MetadataNode) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  std::unique_ptr<DataMeta> MetaP(new (std::nothrow) DataMeta());
  if (!MetaP) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  MetaDocument *MetaDoc = new (std::nothrow) MetaDocument();
  if (!MetaDoc) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  MetaP->MetaDoc.reset(MetaDoc);
  MetaP->DocNode = MetaP->MetaDoc->Document.getRoot();

  if (auto Status = metadata::getMetadataRoot(DataP, MetaP.get())) {
    return Status;
  }

  // if no metadata found in this data object, still return SUCCESS but
  // with default NULL kind

  *MetadataNode = DataMeta::convert(MetaP.release());

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_destroy_metadata
    //
    (amd_comgr_metadata_node_t MetadataNode) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);
  delete MetaP;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_metadata_kind
    //
    (amd_comgr_metadata_node_t MetadataNode,
     amd_comgr_metadata_kind_t *MetadataKind) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (!MetadataKind) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *MetadataKind = MetaP->getMetadataKind();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_metadata_string
    //
    (amd_comgr_metadata_node_t MetadataNode, size_t *Size, char *String) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (MetaP->getMetadataKind() != AMD_COMGR_METADATA_KIND_STRING || !Size) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  std::string Str = MetaP->convertDocNodeToString(MetaP->DocNode);

  if (String) {
    memcpy(String, Str.c_str(), *Size);
  } else {
    *Size = Str.size() + 1;
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_metadata_map_size
    //
    (amd_comgr_metadata_node_t MetadataNode, size_t *Size) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (MetaP->getMetadataKind() != AMD_COMGR_METADATA_KIND_MAP || !Size) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *Size = MetaP->DocNode.getMap().size();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_iterate_map_metadata
    //
    (amd_comgr_metadata_node_t MetadataNode,
     amd_comgr_status_t (*Callback)(amd_comgr_metadata_node_t,
                                    amd_comgr_metadata_node_t, void *),
     void *UserData) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (MetaP->getMetadataKind() != AMD_COMGR_METADATA_KIND_MAP || !Callback) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto Map = MetaP->DocNode.getMap();

  for (auto &KV : Map) {
    if (KV.first.isEmpty() || KV.second.isEmpty()) {
      return AMD_COMGR_STATUS_ERROR;
    }
    std::unique_ptr<DataMeta> KeyP(new (std::nothrow) DataMeta());
    std::unique_ptr<DataMeta> ValueP(new (std::nothrow) DataMeta());
    if (!KeyP || !ValueP) {
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    }
    KeyP->MetaDoc = MetaP->MetaDoc;
    KeyP->DocNode = KV.first;
    ValueP->MetaDoc = MetaP->MetaDoc;
    ValueP->DocNode = KV.second;
    (*Callback)(DataMeta::convert(KeyP.get()), DataMeta::convert(ValueP.get()),
                UserData);
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_metadata_lookup
    //
    (amd_comgr_metadata_node_t MetadataNode, const char *Key,
     amd_comgr_metadata_node_t *Value) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (MetaP->getMetadataKind() != AMD_COMGR_METADATA_KIND_MAP || !Key ||
      !Value) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  for (auto Iter : MetaP->DocNode.getMap()) {
    if (!Iter.first.isScalar() ||
        StringRef(Key) != MetaP->convertDocNodeToString(Iter.first)) {
      continue;
    }

    DataMeta *NewMetaP = new (std::nothrow) DataMeta();
    if (!NewMetaP) {
      return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    NewMetaP->MetaDoc = MetaP->MetaDoc;
    NewMetaP->DocNode = Iter.second;
    *Value = DataMeta::convert(NewMetaP);

    return AMD_COMGR_STATUS_SUCCESS;
  }

  return AMD_COMGR_STATUS_ERROR;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_get_metadata_list_size
    //
    (amd_comgr_metadata_node_t MetadataNode, size_t *Size) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (MetaP->getMetadataKind() != AMD_COMGR_METADATA_KIND_LIST || !Size) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *Size = MetaP->DocNode.getArray().size();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_index_list_metadata
    //
    (amd_comgr_metadata_node_t MetadataNode, size_t Index,
     amd_comgr_metadata_node_t *Value) {
  DataMeta *MetaP = DataMeta::convert(MetadataNode);

  if (MetaP->getMetadataKind() != AMD_COMGR_METADATA_KIND_LIST || !Value) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto List = MetaP->DocNode.getArray();

  if (Index >= List.size()) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  DataMeta *NewMetaP = new (std::nothrow) DataMeta();
  if (!NewMetaP) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  NewMetaP->MetaDoc = MetaP->MetaDoc;
  NewMetaP->DocNode = List[Index];
  *Value = DataMeta::convert(NewMetaP);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_iterate_symbols
    //
    (amd_comgr_data_t Data,
     amd_comgr_status_t (*Callback)(amd_comgr_symbol_t, void *),
     void *UserData) {
  SymbolHelper Helper;
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind() ||
      !(DataP->DataKind == AMD_COMGR_DATA_KIND_RELOCATABLE ||
        DataP->DataKind == AMD_COMGR_DATA_KIND_EXECUTABLE) ||
      !Callback) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ensureLLVMInitialized();

  StringRef Ins(DataP->Data, DataP->Size);
  return Helper.iterateTable(Ins, DataP->DataKind, Callback, UserData);
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_symbol_lookup
    //
    (amd_comgr_data_t Data, const char *Name, amd_comgr_symbol_t *Symbol) {
  DataObject *DataP = DataObject::convert(Data);
  SymbolHelper Helper;

  if (!DataP || !DataP->hasValidDataKind() ||
      !(DataP->DataKind == AMD_COMGR_DATA_KIND_RELOCATABLE ||
        DataP->DataKind == AMD_COMGR_DATA_KIND_EXECUTABLE)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  ensureLLVMInitialized();

  // look through the symbol table for a symbol name based
  // on the data object.

  StringRef Ins(DataP->Data, DataP->Size);
  SymbolContext *Sym = Helper.createBinary(Ins, Name, DataP->DataKind);
  if (!Sym) {
    return AMD_COMGR_STATUS_ERROR;
  }

  DataSymbol *SymP = new (std::nothrow) DataSymbol(Sym);
  if (!SymP) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  *Symbol = DataSymbol::convert(SymP);

  // Update the symbol field in the data object
  delete DataP->DataSym;
  DataP->DataSym = SymP;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_symbol_get_info
    //
    (amd_comgr_symbol_t Symbol, amd_comgr_symbol_info_t SymbolInfo,
     void *Value) {
  DataSymbol *SymP = DataSymbol::convert(Symbol);

  if (!Value || !isSymbolInfoValid(SymbolInfo) || !SymP->DataSym) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  SymbolContext *Sym = SymP->DataSym;

  switch (SymbolInfo) {
  case AMD_COMGR_SYMBOL_INFO_NAME_LENGTH:
    *(size_t *)Value = strlen(Sym->Name);
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_SYMBOL_INFO_NAME:
    strcpy((char *)Value, Sym->Name);
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_SYMBOL_INFO_TYPE:
    *(amd_comgr_symbol_type_t *)Value = Sym->Type;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_SYMBOL_INFO_SIZE:
    *(uint64_t *)Value = Sym->Size;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED:
    *(bool *)Value = Sym->Undefined;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_SYMBOL_INFO_VALUE:
    *(uint64_t *)Value = Sym->Value;
    return AMD_COMGR_STATUS_SUCCESS;
  }

  llvm_unreachable("invalid symbol info");
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_create_disassembly_info
    //
    (const char *IsaName,
     uint64_t (*ReadMemoryCallback)(uint64_t, char *, uint64_t, void *),
     void (*PrintInstructionCallback)(const char *, void *),
     void (*PrintAddressAnnotationCallback)(uint64_t, void *),
     amd_comgr_disassembly_info_t *DisasmInfo) {

  if (!IsaName || !metadata::isValidIsaName(IsaName) || !ReadMemoryCallback ||
      !PrintInstructionCallback || !PrintAddressAnnotationCallback ||
      !DisasmInfo) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  TargetIdentifier Ident;
  if (auto Status = parseTargetIdentifier(IsaName, Ident)) {
    return Status;
  }

  ensureLLVMInitialized();

  return DisassemblyInfo::create(Ident, ReadMemoryCallback,
                                 PrintInstructionCallback,
                                 PrintAddressAnnotationCallback, DisasmInfo);
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_destroy_disassembly_info
    //
    (amd_comgr_disassembly_info_t DisasmInfo) {

  DisassemblyInfo *DI = DisassemblyInfo::convert(DisasmInfo);

  if (!DI) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  delete DI;

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_disassemble_instruction
    //
    (amd_comgr_disassembly_info_t DisasmInfo, uint64_t Address, void *UserData,
     uint64_t *Size) {

  DisassemblyInfo *DI = DisassemblyInfo::convert(DisasmInfo);
  if (!DI || !Size) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return DI->disassembleInstruction(Address, UserData, *Size);
}

amd_comgr_status_t AMD_COMGR_API
// NOLINTNEXTLINE(readability-identifier-naming)
amd_comgr_demangle_symbol_name(amd_comgr_data_t MangledSymbolName,
                               amd_comgr_data_t *DemangledSymbolName) {
  DataObject *DataP = DataObject::convert(MangledSymbolName);
  if (!DataP || !DataP->Data || DataP->DataKind != AMD_COMGR_DATA_KIND_BYTES ||
      !DemangledSymbolName) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  DataObject *DemangledDataP = DataObject::allocate(AMD_COMGR_DATA_KIND_BYTES);
  if (!DemangledDataP) {
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  DemangledDataP->setData(
      llvm::demangle(std::string(DataP->Data, DataP->Size)));
  *DemangledSymbolName = DataObject::convert(DemangledDataP);
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
// NOLINTNEXTLINE(readability-identifier-naming)
amd_comgr_populate_mangled_names(amd_comgr_data_t Data, size_t *Count) {
  DataObject *DataP = DataObject::convert(Data);
  if (!DataP || !DataP->Data ||
      (DataP->DataKind != AMD_COMGR_DATA_KIND_BC &&
       DataP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  DataP->MangledNames.clear();

  LLVMContext Context;

  if (DataP->DataKind == AMD_COMGR_DATA_KIND_BC) {

    MemoryBufferRef BcMemBufRef = MemoryBufferRef(
        StringRef(DataP->Data, DataP->Size), StringRef(DataP->Name));

    auto BcModVecOrErr = getBitcodeModuleList(BcMemBufRef);
    if (!BcModVecOrErr) {
      llvm::logAllUnhandledErrors(BcModVecOrErr.takeError(), llvm::errs(),
                                  "Bitcode Contents error: ");
      return AMD_COMGR_STATUS_ERROR;
    }

    std::vector<BitcodeModule> BcModVec = BcModVecOrErr.get();
    for (BitcodeModule BcMod : BcModVec) {

      Expected<std::unique_ptr<Module>> ModOrError =
          BcMod.getLazyModule(Context, true, true);
      if (!ModOrError) {
        llvm::logAllUnhandledErrors(ModOrError.takeError(), llvm::errs(),
                                    "Bitcode Contents error: ");
        return AMD_COMGR_STATUS_ERROR;
      }

      std::unique_ptr<Module> M = std::move(ModOrError.get());
      for (llvm::GlobalVariable &GlobalVar : M->globals())
        DataP->MangledNames.push_back(GlobalVar.getName().str());
      for (llvm::Function &Function : M->getFunctionList())
        DataP->MangledNames.push_back(Function.getName().str());
    }
  }

  if (DataP->DataKind == AMD_COMGR_DATA_KIND_EXECUTABLE) {
    // Callback to iterate_symbols that error checks and appends lowered names
    // to "data"
    auto Callback = [](amd_comgr_symbol_t Symbol, void *Data) {
      size_t Len = 0;
      if (auto Res = amd_comgr_symbol_get_info(
              Symbol, AMD_COMGR_SYMBOL_INFO_NAME_LENGTH, &Len);
          Res != AMD_COMGR_STATUS_SUCCESS)
        return Res;
      std::string Name(Len, 0);
      if (auto Res = amd_comgr_symbol_get_info(
              Symbol, AMD_COMGR_SYMBOL_INFO_NAME, &Name[0]);
          Res != AMD_COMGR_STATUS_SUCCESS)
        return Res;
      auto *Rv = reinterpret_cast<std::vector<std::string> *>(Data);
      Rv->push_back(Name);
      return AMD_COMGR_STATUS_SUCCESS;
    };

    if (auto Res = amd_comgr_iterate_symbols(
            Data, Callback, reinterpret_cast<void *>(&(DataP->MangledNames)));
        Res != AMD_COMGR_STATUS_SUCCESS) {
      return AMD_COMGR_STATUS_ERROR;
    }
  }

  *Count = DataP->MangledNames.size();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
// NOLINTNEXTLINE(readability-identifier-naming)
amd_comgr_get_mangled_name(amd_comgr_data_t Data, size_t Index, size_t *Size,
                           char *MangledName) {
  DataObject *DataP = DataObject::convert(Data);
  if (!DataP || !DataP->Data ||
      (DataP->DataKind != AMD_COMGR_DATA_KIND_BC &&
       DataP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (Index >= DataP->MangledNames.size())
    return AMD_COMGR_STATUS_ERROR;

  if (MangledName == NULL)
    *Size = DataP->MangledNames[Index].size() + 1;
  else
    memcpy(MangledName, DataP->MangledNames[Index].c_str(), *Size);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
// NOLINTNEXTLINE(readability-identifier-naming)
amd_comgr_populate_name_expression_map(amd_comgr_data_t Data, size_t *Count) {

  DataObject *DataP = DataObject::convert(Data);
  if (!DataP || !DataP->Data ||
      (DataP->DataKind != AMD_COMGR_DATA_KIND_BC &&
       DataP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  DataP->NameExpressionMap.clear();
  LLVMContext Context;

  // For bitcodes, the name expression and function pointer can be found by
  // creating a bitcode module data structure, and searching through the
  // initalizers of global variables
  if (DataP->DataKind == AMD_COMGR_DATA_KIND_BC) {

    MemoryBufferRef BcMemBufRef = MemoryBufferRef(
        StringRef(DataP->Data, DataP->Size), StringRef(DataP->Name));

    auto BcModVecOrErr = getBitcodeModuleList(BcMemBufRef);
    if (!BcModVecOrErr) {
      llvm::logAllUnhandledErrors(BcModVecOrErr.takeError(), llvm::errs(),
                                  "Bitcode Contents error: ");
      return AMD_COMGR_STATUS_ERROR;
    }

    std::vector<BitcodeModule> BcModVec = BcModVecOrErr.get();
    for (BitcodeModule BcMod : BcModVec) {

      Expected<std::unique_ptr<Module>> ModOrError =
          BcMod.getLazyModule(Context, true, true);
      if (!ModOrError) {
        llvm::logAllUnhandledErrors(ModOrError.takeError(), llvm::errs(),
                                    "Bitcode Contents error: ");
        return AMD_COMGR_STATUS_ERROR;
      }

      // Collect initial values of all global variables starting with
      // `__amdgcn_name_expr_`.
      std::unique_ptr<Module> M = std::move(ModOrError.get());
      for (llvm::GlobalVariable &GlobalVar : M->globals()) {
        if (GlobalVar.getName().contains("__amdgcn_name_expr_")) {

          std::string MapKey, MapVal;

          // 1. use getInitalizer() to get a pointer to [2xi8*]
          auto *Initalizer = GlobalVar.getInitializer();

          // 2. Get NameExpression map value from second operand name
          MapVal = Initalizer->getOperand(1)->getName().str();

          // 3 Get NameExpression map key from first operand
          llvm::Value *V = Initalizer->getOperand(0);

          // Cast initalizer operand 0 to ConstantExpr
          if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
            // Cast ConstantExpr operand 0 to GlobalVaribale
            if (llvm::GlobalVariable *GV =
                    dyn_cast<llvm::GlobalVariable>(CE->getOperand(0))) {
              // Cast GlobalVariable initializer to ConstantDataSequential
              if (ConstantDataSequential *CDS =
                      dyn_cast<ConstantDataSequential>(GV->getInitializer())) {

                MapKey = CDS->getAsString().str();
              }
            }
          }

          MapKey.erase(std::find(MapKey.begin(), MapKey.end(), '\0'),
                       MapKey.end());
          MapVal.erase(std::find(MapVal.begin(), MapVal.end(), '\0'),
                       MapVal.end());
          if (env::shouldEmitVerboseLogs()) {
            llvm::errs() << "   Comgr NameExpressionMap[" << MapKey
                         << "] = " << MapVal << "\n";
          }
          DataP->NameExpressionMap[MapKey] = MapVal;
        }
      } // end M->globals() loop
    } // end BcModVec loop
  } // end AMD_COMGR_DATA_KIND_BC conditional

  // For code objects, we can get the needed information by creating an ELF
  // object and traversing the .dynsym, .rela.dyn, and .rodata sections.
  if (DataP->DataKind == AMD_COMGR_DATA_KIND_EXECUTABLE) {
    auto ELFFileOrError =
        llvm::object::ELF64LEFile::create(StringRef(DataP->Data, DataP->Size));
    if (!ELFFileOrError) {
      llvm::logAllUnhandledErrors(ELFFileOrError.takeError(), llvm::errs(),
                                  "ELFObj creation error: ");
      return AMD_COMGR_STATUS_ERROR;
    }
    auto ELFFile = std::move(ELFFileOrError.get());

    std::vector<struct NameExpressionData *> NameExpDataVec;
    std::map<int, StringRef> DynsymMap;

    // Collect references for .dynsym, .rela.dyn, and .rodata sections
    auto SectionsOrError = ELFFile.sections();
    if (!SectionsOrError) {
      llvm::logAllUnhandledErrors(SectionsOrError.takeError(), llvm::errs(),
                                  "Sections creation error: ");
      return AMD_COMGR_STATUS_ERROR;
    }
    auto Sections = std::move(SectionsOrError.get());

    Elf_Shdr_Impl<ELF64LE> DynsymShdr, RelaShdr, RodataShdr;
    for (auto Shdr : Sections) {

      if (Shdr.sh_type == ELF::SHT_DYNSYM)
        DynsymShdr = Shdr;

      // Check sh_info to differentiate .rela.dyn and not .rela
      if (Shdr.sh_type == ELF::SHT_RELA && Shdr.sh_info == 0)
        RelaShdr = Shdr;

      // We can't uniquely identify the .rodata section using the type and flag
      // because other sections may use the exact same flags and type (i.e.
      // .interp).  For correctness, we can check the name instead
      if (Shdr.sh_type == ELF::SHT_PROGBITS &&
          (Shdr.sh_flags & ELF::SHF_ALLOC)) {

        Expected<StringRef> SecNameOrError = ELFFile.getSectionName(Shdr);
        if (!SecNameOrError) {
          llvm::logAllUnhandledErrors(SecNameOrError.takeError(), llvm::errs(),
                                      "ELFObj creation error: ");
          return AMD_COMGR_STATUS_ERROR;
        }
        StringRef SecName = std::move(SecNameOrError.get());

        if (SecName == StringRef(".rodata"))
          RodataShdr = Shdr;
      }
    }

    // .dynsym - Find name expressions with amdgcn_name_expr and store their
    // Value fields
    Expected<StringRef> StrTabOrError =
        ELFFile.getStringTableForSymtab(DynsymShdr);
    if (!StrTabOrError) {
      llvm::logAllUnhandledErrors(StrTabOrError.takeError(), llvm::errs(),
                                  "StrTab creation error: ");
      return AMD_COMGR_STATUS_ERROR;
    }
    StringRef StrTab = std::move(StrTabOrError.get());

    // Check each .dynsym entry
    for (unsigned int I = 0; I < DynsymShdr.getEntityCount(); ++I) {

      // Get symbol from entry
      auto SymbolOrError = ELFFile.getSymbol(&DynsymShdr, I);
      if (!SymbolOrError) {
        llvm::logAllUnhandledErrors(SymbolOrError.takeError(), llvm::errs(),
                                    "Symbol creation error: ");
        return AMD_COMGR_STATUS_ERROR;
      }
      const auto *Symbol = std::move(SymbolOrError.get());

      // Get symbol name from symbol
      Expected<StringRef> SymbolNameOrError = Symbol->getName(StrTab);
      if (!SymbolNameOrError) {
        llvm::logAllUnhandledErrors(SymbolNameOrError.takeError(), llvm::errs(),
                                    "SymbolName creation error: ");
        return AMD_COMGR_STATUS_ERROR;
      }
      StringRef SymbolName = std::move(SymbolNameOrError.get());

      // Process symbol names containing amdgcn_name_expr
      if (SymbolName.contains(StringRef("__amdgcn_name_expr_"))) {
        struct NameExpressionData *ExpData = new NameExpressionData();
        ExpData->StubName = SymbolName;
        ExpData->StubValue = Symbol->getValue();
        NameExpDataVec.push_back(ExpData);
      }

      // Store all symbols to later quickly find mangled name
      DynsymMap[Symbol->getValue()] = SymbolName;
    } // end entry loop

    // .rela.dyn - Use Values collected from .dynsym
    //   Offset == Value: Store 'Symbol's Name + Addend'
    //      - needed to get unmangled name from .rodata
    //   Offset == Value + 8: Store 'Symbol's Name + Addend'
    //      - needed to get mangled name from .dynsym
    auto RelaRangeOrError = ELFFile.relas(RelaShdr);
    if (!RelaRangeOrError) {
      llvm::logAllUnhandledErrors(RelaRangeOrError.takeError(), llvm::errs(),
                                  "RelaRange creation error: ");
      return AMD_COMGR_STATUS_ERROR;
    }
    auto RelaRange = std::move(RelaRangeOrError.get());

    for (auto Rela : RelaRange) {
      for (auto *ExpData : NameExpDataVec) {
        if (Rela.r_offset == ExpData->StubValue)
          ExpData->RodataOffset = Rela.r_addend;

        if (Rela.r_offset == ExpData->StubValue + 8)
          ExpData->MangledValue = Rela.r_addend;
      }
    }

    // rodata - Use the difference between the .rela.dyn Names and .rodata
    // offset to collect unmangled strings
    auto RodataOrError = ELFFile.getSectionContents(RodataShdr);
    if (!RodataOrError) {
      llvm::logAllUnhandledErrors(RodataOrError.takeError(), llvm::errs(),
                                  "Rodata creation error: ");
      return AMD_COMGR_STATUS_ERROR;
    }
    auto Rodata = std::move(RodataOrError.get());

    // Collect an unmangled name for each name expression
    for (auto *ExpData : NameExpDataVec) {
      // TODO: If/when an accessor API becomes available to get the starting
      // address for the section, switch to that
      size_t Offset = ExpData->RodataOffset - RodataShdr.sh_offset;

      // Store from the offset up until the first '\0'
      const char *Unmangled = reinterpret_cast<const char *>(&Rodata[Offset]);
      ExpData->UnmangledName = StringRef(Unmangled);
    }

    // Populate mangled names now that mangled values are set
    for (auto *ExpData : NameExpDataVec)
      ExpData->MangledName = DynsymMap[ExpData->MangledValue];

    // Populate map
    for (auto *ExpData : NameExpDataVec) {
      DataP->NameExpressionMap[ExpData->UnmangledName.str()] =
          ExpData->MangledName.str();

      if (env::shouldEmitVerboseLogs()) {
        llvm::errs() << "   Comgr NameExpressionMap[" << ExpData->UnmangledName
                     << "] = " << ExpData->MangledName << "\n";
      }
    }

  } // end AMD_COMGR_DATA_KIND_EXECUTABLE conditional

  *Count = DataP->NameExpressionMap.size();

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
// NOLINTNEXTLINE(readability-identifier-naming)
amd_comgr_map_name_expression_to_symbol_name(amd_comgr_data_t Data,
                                             size_t *Size, char *NameExpression,
                                             char *SymbolName) {
  DataObject *DataP = DataObject::convert(Data);
  if (!DataP || !DataP->Data ||
      (DataP->DataKind != AMD_COMGR_DATA_KIND_BC &&
       DataP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Check if the provided NameExpression is in the map
  std::string NameExpStr(NameExpression);
  if (DataP->NameExpressionMap.find(NameExpStr) ==
      DataP->NameExpressionMap.end()) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // First return the size of the SymbolName
  if (SymbolName == NULL)
    *Size = DataP->NameExpressionMap[NameExpression].size() + 1;

  // Now that the calling API has had a chance to allocate memory, copy the
  // symbol name associated with the provided name expression to the provided
  // buffer.
  else
    memcpy(SymbolName, DataP->NameExpressionMap[NameExpression].c_str(), *Size);

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_set_data_from_file_slice
    //
    (amd_comgr_data_t Data, int FD, uint64_t Offset, uint64_t Size) {
  DataObject *DataP = DataObject::convert(Data);
  if (!DataP || !DataP->hasValidDataKind())
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  auto FileHandle = sys::fs::convertFDToNativeFile(FD);
  auto BufferOrErr = MemoryBuffer::getOpenFileSlice(
      FileHandle, "" /* Name not set */, Size, Offset);
  if (BufferOrErr.getError()) {
    return AMD_COMGR_STATUS_ERROR;
  }

  DataP->setData(std::move(*BufferOrErr));

  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API
    // NOLINTNEXTLINE(readability-identifier-naming)
    amd_comgr_lookup_code_object
    //
    (amd_comgr_data_t Data, amd_comgr_code_object_info_t *QueryList,
     size_t QueryListSize) {
  DataObject *DataP = DataObject::convert(Data);

  if (!DataP || !DataP->hasValidDataKind() ||
      !(DataP->DataKind == AMD_COMGR_DATA_KIND_FATBIN ||
        DataP->DataKind == AMD_COMGR_DATA_KIND_BYTES ||
        DataP->DataKind == AMD_COMGR_DATA_KIND_EXECUTABLE))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (!QueryList)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  return metadata::lookUpCodeObject(DataP, QueryList, QueryListSize);
}

amd_comgr_status_t AMD_COMGR_API
// NOLINTNEXTLINE(readability-identifier-naming)
amd_comgr_map_elf_virtual_address_to_code_object_offset(
    amd_comgr_data_t Data, uint64_t ElfVirtualAddress,
    uint64_t *CodeObjectOffset, uint64_t *SliceSize, bool *Nobits) {

  DataObject *DataP = DataObject::convert(Data);
  if (!DataP || !DataP->Data ||
      (DataP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE)) {
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Create ELF Object file
  auto ELFFileOrError =
      llvm::object::ELF64LEFile::create(StringRef(DataP->Data, DataP->Size));
  if (!ELFFileOrError) {
    llvm::logAllUnhandledErrors(ELFFileOrError.takeError(), llvm::errs(),
                                "ELFObj creation error: ");
    return AMD_COMGR_STATUS_ERROR;
  }
  auto ELFFile = std::move(ELFFileOrError.get());

  // Error check the ELF file
  auto ELFHeader = ELFFile.getHeader();
  if (!ELFHeader.checkMagic())
    return AMD_COMGR_STATUS_ERROR;

  if (ELFHeader.e_ident[llvm::ELF::EI_CLASS] != llvm::ELF::ELFCLASS64 ||
      ELFHeader.e_ident[llvm::ELF::EI_DATA] != llvm::ELF::ELFDATA2LSB ||
      ELFHeader.e_ident[llvm::ELF::EI_VERSION] != llvm::ELF::EV_CURRENT ||
      ELFHeader.e_ident[llvm::ELF::EI_OSABI] != llvm::ELF::ELFOSABI_AMDGPU_HSA)
    return AMD_COMGR_STATUS_ERROR;

  unsigned EIdent = ELFHeader.e_ident[llvm::ELF::EI_ABIVERSION];
  if (EIdent != llvm::ELF::ELFABIVERSION_AMDGPU_HSA_V4 &&
      EIdent != llvm::ELF::ELFABIVERSION_AMDGPU_HSA_V5 &&
      EIdent != llvm::ELF::ELFABIVERSION_AMDGPU_HSA_V6)
    return AMD_COMGR_STATUS_ERROR;

  if (ELFHeader.e_type != llvm::ELF::ET_DYN ||
      ELFHeader.e_machine != llvm::ELF::EM_AMDGPU || ELFHeader.e_phoff == 0)
    return AMD_COMGR_STATUS_ERROR;

  // Access program headers
  auto ProgHeadersOrError = ELFFile.program_headers();
  if (!ProgHeadersOrError) {
    llvm::logAllUnhandledErrors(ProgHeadersOrError.takeError(), llvm::errs(),
                                "ProgHeaders creation error: ");
    return AMD_COMGR_STATUS_ERROR;
  }
  auto ProgHeaders = std::move(ProgHeadersOrError.get());

  for (auto Phdr : ProgHeaders) {

    // Check if ELF virtual address defined in this header
    if (Phdr.p_type == llvm::ELF::PT_LOAD &&
        ElfVirtualAddress >= Phdr.p_vaddr &&
        ElfVirtualAddress < Phdr.p_vaddr + Phdr.p_memsz) {

      *CodeObjectOffset = ElfVirtualAddress - Phdr.p_vaddr + Phdr.p_offset;
      *Nobits = ElfVirtualAddress - Phdr.p_vaddr >= Phdr.p_filesz;

      if (*Nobits) // end of segment to relative address difference
        *SliceSize = Phdr.p_filesz - (ElfVirtualAddress - Phdr.p_vaddr);
      else // end of valid memory to relative address difference
        *SliceSize = Phdr.p_memsz - (ElfVirtualAddress - Phdr.p_vaddr);

      return AMD_COMGR_STATUS_SUCCESS;
    }
  }

  // If the provided ELF virtual address is not mapped to an offset
  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}
