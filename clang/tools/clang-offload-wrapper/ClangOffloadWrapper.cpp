//===-- clang-offload-wrapper/ClangOffloadWrapper.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the offload wrapper tool. It takes offload target binaries
/// as input and creates wrapper bitcode from them which, after linking with the
/// offload application, provides access to the binaries.
/// TODO Add Windows support.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace llvm;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory
    ClangOffloadWrapperCategory("clang-offload-wrapper options");

static cl::opt<std::string> Output("o", cl::Required,
                                   cl::desc("Output filename"),
                                   cl::value_desc("filename"),
                                   cl::cat(ClangOffloadWrapperCategory));

static cl::list<std::string> Inputs(cl::Positional, cl::OneOrMore,
                                    cl::desc("<input  files>"),
                                    cl::cat(ClangOffloadWrapperCategory));

static cl::opt<std::string>
    Target("target", cl::Required,
           cl::desc("Offload kind + target triple of the wrapper object: "
                    "<offload kind>-<target triple>"),
           cl::value_desc("kind-triple"), cl::cat(ClangOffloadWrapperCategory));

static cl::opt<bool> EmitEntryTable("emit-entry-table", cl::NotHidden,
                                    cl::init(true), cl::Optional,
                                    cl::desc("Emit offload entry table"),
                                    cl::cat(ClangOffloadWrapperCategory));

static cl::opt<bool> EmitRegFuncs("emit-reg-funcs", cl::NotHidden,
                                  cl::init(true), cl::Optional,
                                  cl::desc("Emit [un-]registration functions"),
                                  cl::cat(ClangOffloadWrapperCategory));

static cl::opt<std::string>
    RegFuncName("reg-func-name", cl::Optional, cl::init("__tgt_register_lib"),
                cl::desc("Offload descriptor registration function name"),
                cl::value_desc("name"), cl::cat(ClangOffloadWrapperCategory));

static cl::opt<std::string>
    UnregFuncName("unreg-func-name", cl::Optional,
                  cl::init("__tgt_unregister_lib"),
                  cl::desc("Offload descriptor un-registration function name"),
                  cl::value_desc("name"), cl::cat(ClangOffloadWrapperCategory));

static cl::opt<std::string> DescriptorName(
    "desc-name", cl::Optional, cl::init("descriptor"),
    cl::desc(
        "Specifies offload descriptor symbol name: '.<offload kind>.<name>'"
        ", and makes it globally visible"),
    cl::value_desc("name"), cl::cat(ClangOffloadWrapperCategory));

namespace {
// TODO offload bundler and wrapper should share this.
// Offload kinds this tool supports
enum class OffloadKind {
  Unknown = 0,
  Host,
  OpenMP,
  HIP,
  SYCL,
  First = Host,
  Last = SYCL
};

OffloadKind parseOffloadKind(StringRef KindStr) {
  OffloadKind Kind = StringSwitch<OffloadKind>(KindStr)
                         .Case("host", OffloadKind::Host)
                         .Case("openmp", OffloadKind::OpenMP)
                         .Case("hip", OffloadKind::HIP)
                         .Case("sycl", OffloadKind::SYCL)
                         .Default(OffloadKind::Unknown);
  return Kind;
}

StringRef offloadKindToString(OffloadKind Kind) {
  switch (Kind) {
  case OffloadKind::Unknown:
    return "unknown";
  case OffloadKind::Host:
    return "host";
  case OffloadKind::OpenMP:
    return "openmp";
  case OffloadKind::HIP:
    return "hip";
  case OffloadKind::SYCL:
    return "sycl";
  default:
    llvm_unreachable("bad offload kind");
  }
  return "<ERROR>";
}

void dumpOffloadKinds(raw_ostream &OS) {
  OffloadKind Kinds[] = {OffloadKind::Host, OffloadKind::OpenMP,
                         OffloadKind::HIP, OffloadKind::SYCL};
  for (auto K : Kinds) {
    if (K != OffloadKind::Host)
      OS << " ";
    OS << offloadKindToString(K);
  }
}

class BinaryWrapper {
  LLVMContext C;
  Module M;
  std::string OffloadKindTag;

  StructType *EntryTy = nullptr;
  StructType *ImageTy = nullptr;
  StructType *DescTy = nullptr;

  using MemoryBuffersVector = SmallVectorImpl<std::unique_ptr<MemoryBuffer>>;

private:
  IntegerType *getSizeTTy() {
    auto PtrSize = M.getDataLayout().getPointerTypeSize(Type::getInt8PtrTy(C));
    return PtrSize == 8 ? Type::getInt64Ty(C) : Type::getInt32Ty(C);
  }

  // struct __tgt_offload_entry {
  //   void *addr;
  //   char *name;
  //   size_t size;
  //   int32_t flags;
  //   int32_t reserved;
  // };
  StructType *getEntryTy() {
    if (!EntryTy)
      EntryTy = StructType::create("__tgt_offload_entry", Type::getInt8PtrTy(C),
                                   Type::getInt8PtrTy(C), getSizeTTy(),
                                   Type::getInt32Ty(C), Type::getInt32Ty(C));
    return EntryTy;
  }

  PointerType *getEntryPtrTy() { return PointerType::getUnqual(getEntryTy()); }

  // struct __tgt_device_image {
  //   void *ImageStart;
  //   void *ImageEnd;
  //   __tgt_offload_entry *EntriesBegin; [optional]
  //   __tgt_offload_entry *EntriesEnd;   [optional]
  // };
  StructType *getDeviceImageTy() {
    if (!ImageTy) {
      SmallVector<Type *, 4> FieldTypes(
          {Type::getInt8PtrTy(C), Type::getInt8PtrTy(C)});
      if (EmitEntryTable)
        FieldTypes.append({getEntryPtrTy(), getEntryPtrTy()});
      ImageTy = StructType::create(FieldTypes, "__tgt_device_image");
    }
    return ImageTy;
  }

  PointerType *getDeviceImagePtrTy() {
    return PointerType::getUnqual(getDeviceImageTy());
  }

  // struct __tgt_bin_desc {
  //   int32_t NumDeviceImages;
  //   __tgt_device_image *DeviceImages;
  //   __tgt_offload_entry *HostEntriesBegin; [optional]
  //   __tgt_offload_entry *HostEntriesEnd;   [optional]
  // };
  StructType *getBinDescTy() {
    if (!DescTy) {
      SmallVector<Type *, 4> FieldTypes(
          {Type::getInt32Ty(C), getDeviceImagePtrTy()});
      if (EmitEntryTable)
        FieldTypes.append({getEntryPtrTy(), getEntryPtrTy()});
      DescTy = StructType::create(FieldTypes, "__tgt_bin_desc");
    }
    return DescTy;
  }

  PointerType *getBinDescPtrTy() {
    return PointerType::getUnqual(getBinDescTy());
  }

  GlobalVariable *createBinDesc(const MemoryBuffersVector &Bufs) {
    GlobalVariable *EntriesB = nullptr, *EntriesE = nullptr;

    if (EmitEntryTable) {
      EntriesB = new GlobalVariable(M, getEntryTy(), true,
                                    GlobalValue::ExternalLinkage, nullptr,
                                    OffloadKindTag + "entries_begin");
      EntriesE = new GlobalVariable(M, getEntryTy(), true,
                                    GlobalValue::ExternalLinkage, nullptr,
                                    OffloadKindTag + "entries_end");
    }
    auto *Zero = ConstantInt::get(getSizeTTy(), 0u);
    Constant *ZeroZero[] = {Zero, Zero};

    SmallVector<Constant *, 4> ImagesInits;
    for (const auto &Buf : Bufs) {
      auto *Data = ConstantDataArray::get(
          C, makeArrayRef(Buf->getBufferStart(), Buf->getBufferSize()));

      auto *Image = new GlobalVariable(M, Data->getType(), true,
                                       GlobalVariable::InternalLinkage, Data,
                                       OffloadKindTag + "device_image");
      Image->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

      auto *Size = ConstantInt::get(getSizeTTy(), Buf->getBufferSize());
      Constant *ZeroSize[] = {Zero, Size};

      auto *ImageB = ConstantExpr::getGetElementPtr(Image->getValueType(),
                                                    Image, ZeroZero);
      auto *ImageE = ConstantExpr::getGetElementPtr(Image->getValueType(),
                                                    Image, ZeroSize);

      SmallVector<Constant *, 4> Inits({ImageB, ImageE});
      if (EmitEntryTable)
        Inits.append({EntriesB, EntriesE});
      ImagesInits.push_back(ConstantStruct::get(getDeviceImageTy(), Inits));
    }

    auto *ImagesData = ConstantArray::get(
        ArrayType::get(getDeviceImageTy(), ImagesInits.size()), ImagesInits);

    auto *Images = new GlobalVariable(M, ImagesData->getType(), true,
                                      GlobalValue::InternalLinkage, ImagesData,
                                      OffloadKindTag + "device_images");
    Images->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    auto *ImagesB = ConstantExpr::getGetElementPtr(Images->getValueType(),
                                                   Images, ZeroZero);

    SmallVector<Constant *, 4> Inits(
        {ConstantInt::get(Type::getInt32Ty(C), ImagesInits.size()), ImagesB});
    if (EmitEntryTable)
      Inits.append({EntriesB, EntriesE});

    auto *DescInit = ConstantStruct::get(getBinDescTy(), Inits);

    GlobalValue::LinkageTypes Lnk = DescriptorName.getNumOccurrences() > 0
                                        ? GlobalValue::ExternalLinkage
                                        : GlobalValue::InternalLinkage;
    Twine DescName = Twine(OffloadKindTag) + Twine(DescriptorName);

    return new GlobalVariable(M, DescInit->getType(), true, Lnk, DescInit,
                              DescName);
  }

  void createRegisterFunction(GlobalVariable *BinDesc) {
    auto *FuncTy = FunctionType::get(Type::getVoidTy(C), {}, false);
    auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                  OffloadKindTag + "descriptor_reg", &M);
    Func->setSection(".text.startup");

    // Get RegFuncName function declaration.
    auto *RegFuncTy =
        FunctionType::get(Type::getVoidTy(C), {getBinDescPtrTy()}, false);
    FunctionCallee RegFunc = M.getOrInsertFunction(RegFuncName, RegFuncTy);

    // Construct function body
    IRBuilder<> Builder(BasicBlock::Create(C, "entry", Func));
    Builder.CreateCall(RegFunc, {BinDesc});
    Builder.CreateRetVoid();

    // Add this function to constructors.
    appendToGlobalCtors(M, Func, 0);
  }

  void createUnregisterFunction(GlobalVariable *BinDesc) {
    auto *FuncTy = FunctionType::get(Type::getVoidTy(C), {}, false);
    auto *Func = Function::Create(FuncTy, GlobalValue::InternalLinkage,
                                  OffloadKindTag + "descriptor_unreg", &M);
    Func->setSection(".text.startup");

    // Get UnregFuncName function declaration.
    auto *UnRegFuncTy =
        FunctionType::get(Type::getVoidTy(C), {getBinDescPtrTy()}, false);
    FunctionCallee UnRegFunc = M.getOrInsertFunction(UnregFuncName, UnRegFuncTy);

    // Construct function body
    IRBuilder<> Builder(BasicBlock::Create(C, "entry", Func));
    Builder.CreateCall(UnRegFunc, {BinDesc});
    Builder.CreateRetVoid();

    // Add this function to global destructors.
    appendToGlobalDtors(M, Func, 0);
  }

public:
  BinaryWrapper(const StringRef &KindStr, const StringRef &Target)
      : M("offload.wrapper.object", C) {

    OffloadKindTag =
        (Twine(".") + Twine(KindStr) + Twine("_offloading.")).str();
    M.setTargetTriple(Target);
  }

  const Module &wrapBinaries(const MemoryBuffersVector &Binaries) {
    auto *Desc = createBinDesc(Binaries);
    assert(Desc && "no binary descriptor");

    if (EmitRegFuncs) {
      createRegisterFunction(Desc);
      createUnregisterFunction(Desc);
    }
    return M;
  }
};

} // anonymous namespace

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  cl::HideUnrelatedOptions(ClangOffloadWrapperCategory);
  cl::SetVersionPrinter([](raw_ostream &OS) {
    OS << clang::getClangToolFullVersion("clang-offload-wrapper") << '\n';
  });
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to create a wrapper bitcode for offload target binaries.\n"
      "Takes offload target binaries as input and produces bitcode file "
      "containing\ntarget binaries packaged as data and initialization code "
      "which registers target\nbinaries in offload runtime.");

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }

  std::pair<StringRef, StringRef> KindTriplePair = StringRef(Target).split('-');
  auto OffloadKindStr = KindTriplePair.first;
  auto TargetStr = KindTriplePair.second;

  if (OffloadKindStr.empty()) {
    errs() << "error: no offload kind specified\n";
    return 1;
  }
  OffloadKind Kind = parseOffloadKind(OffloadKindStr);

  if (Kind == OffloadKind::Unknown) {
    errs() << "error: unknown offload kind: " << OffloadKindStr << "\n";
    errs() << "valid offload kinds: ";
    dumpOffloadKinds(errs());
    errs() << "\n";
    return 1;
  }
  if (TargetStr.empty()) {
    errs() << "error: no target specified\n";
    return 1;
  }
  // Create the bitcode file to write the resulting code to.
  {
    std::error_code EC;
    raw_fd_ostream OutF(Output, EC, sys::fs::F_None);
    if (EC) {
      errs() << "error: unable to open output file: " << EC.message() << ".\n";
      return 1;
    }

    // Read device binaries.
    SmallVector<std::unique_ptr<MemoryBuffer>, 4> DeviceBinaries;
    for (const auto &File : Inputs) {
      auto InputOrErr = MemoryBuffer::getFileOrSTDIN(File);
      if (auto EC = InputOrErr.getError()) {
        errs() << "error: can't open file " << File << ": " << EC.message()
               << "\n";
        return 1;
      }
      DeviceBinaries.emplace_back(std::move(*InputOrErr));
    }

    // Create a wrapper for device binaries and write its bitcode to the file.
    WriteBitcodeToFile(
        BinaryWrapper(OffloadKindStr, TargetStr).wrapBinaries(DeviceBinaries),
        OutF);
  }
  return 0;
}
