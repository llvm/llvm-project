//===-- clang-offload-wrapper/ClangOffloadWrapper.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
#include <iterator>
#include <memory>
#include <tuple>

using namespace llvm;

// Offload models supported by this tool. The support basically means mapping
// a string representation given at the command line to a value from this
// enum.
enum OffloadKind {
  Unknown = 0,
  Host,
  OpenMP,
  HIP,
  SYCL,
  First = Host,
  Last = SYCL
};

namespace llvm {
template <> struct DenseMapInfo<OffloadKind> {
  static inline OffloadKind getEmptyKey() {
    return static_cast<OffloadKind>(DenseMapInfo<unsigned>::getEmptyKey());
  }

  static inline OffloadKind getTombstoneKey() {
    return static_cast<OffloadKind>(DenseMapInfo<unsigned>::getTombstoneKey());
  }

  static unsigned getHashValue(const OffloadKind &Val) {
    return DenseMapInfo<unsigned>::getHashValue(static_cast<unsigned>(Val));
  }

  static bool isEqual(const OffloadKind &LHS, const OffloadKind &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

namespace {

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category, everything else (except for -version
// and -help) will be hidden.
static cl::OptionCategory
    ClangOffloadWrapperCategory("clang-offload-wrapper options");

static cl::opt<std::string> Output("o", cl::Required,
                                   cl::desc("Output filename"),
                                   cl::value_desc("filename"),
                                   cl::cat(ClangOffloadWrapperCategory));

static cl::opt<bool> Verbose("v", cl::desc("verbose output"),
                             cl::cat(ClangOffloadWrapperCategory));

static cl::list<std::string> Inputs(cl::Positional, cl::OneOrMore,
                                    cl::desc("<input  files>"),
                                    cl::cat(ClangOffloadWrapperCategory));

// Binary image formats supported by this tool. The support basically means
// mapping string representation given at the command line to a value from this
// enum. No format checking is performed.
enum BinaryImageFormat {
  none,   // image kind is not determined
  native, // image kind is native
  // portable image kinds go next
  spirv, // SPIR-V
  llvmbc // LLVM bitcode
};

/// Sets offload kind.
static cl::list<OffloadKind> Kinds(
    "kind", cl::desc("offload kind:"), cl::OneOrMore,
    cl::values(clEnumValN(Unknown, "unknown", "unknown"),
               clEnumValN(Host, "host", "host"),
               clEnumValN(OpenMP, "openmp", "OpenMP"),
               clEnumValN(HIP, "hip", "HIP"), clEnumValN(SYCL, "sycl", "SYCL")),
    cl::cat(ClangOffloadWrapperCategory));

/// Sets binary image format.
static cl::list<BinaryImageFormat>
    Formats("format", cl::desc("device binary image formats:"), cl::ZeroOrMore,
            cl::values(clEnumVal(none, "not set"),
                       clEnumVal(native, "unknown or native"),
                       clEnumVal(spirv, "SPIRV binary"),
                       clEnumVal(llvmbc, "LLVMIR bitcode")),
            cl::cat(ClangOffloadWrapperCategory));

/// Sets offload target.
static cl::list<std::string> Targets("target", cl::ZeroOrMore,
                                     cl::desc("offload target triple"),
                                     cl::cat(ClangOffloadWrapperCategory),
                                     cl::cat(ClangOffloadWrapperCategory));

/// Sets build options for device binary image.
static cl::list<std::string>
    Options("build-opts", cl::ZeroOrMore,
            cl::desc("build options passed to the offload runtime"),
            cl::cat(ClangOffloadWrapperCategory),
            cl::cat(ClangOffloadWrapperCategory));

/// Specifies the target triple of the host wrapper.
static cl::opt<std::string> Target("host", cl::Optional,
                                   cl::desc("wrapper object target triple"),
                                   cl::value_desc("triple"),
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

static StringRef offloadKindToString(OffloadKind Kind) {
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

static StringRef formatToString(BinaryImageFormat Fmt) {
  switch (Fmt) {
  case BinaryImageFormat::none:
    return "none";
  case BinaryImageFormat::spirv:
    return "spirv";
  case BinaryImageFormat::llvmbc:
    return "llvmbc";
  case BinaryImageFormat::native:
    return "native";
  default:
    llvm_unreachable("bad format");
  }
  return "<ERROR>";
}

struct OffloadKindToUint {
  using argument_type = OffloadKind;
  unsigned operator()(argument_type Kind) const {
    return static_cast<unsigned>(Kind);
  }
};

/// Implements binary image information collecting and wrapping it in a host
/// bitcode file.
class BinaryWrapper {
public:
  /// Represents a single image to wrap.
  class Image {
  public:
    Image(const llvm::StringRef File_, const llvm::StringRef Manif_,
          const llvm::StringRef Tgt_, BinaryImageFormat Fmt_,
          const llvm::StringRef Opts_)
        : File(File_), Manif(Manif_), Tgt(Tgt_), Fmt(Fmt_), Opts(Opts_) {}

    /// Name of the file with actual contents
    const llvm::StringRef File;
    /// Name of the manifest file
    const llvm::StringRef Manif;
    /// Offload target architecture
    const llvm::StringRef Tgt;
    /// Format
    const BinaryImageFormat Fmt;
    /// Build options
    const llvm::StringRef Opts;

    friend raw_ostream &operator<<(raw_ostream &Out, const Image &Img);
  };

private:
  using SameKindPack = llvm::SmallVector<std::unique_ptr<Image>, 4>;

  LLVMContext C;
  Module M;

  StructType *EntryTy = nullptr;
  StructType *ImageTy = nullptr;
  StructType *DescTy = nullptr;

  /// Records all added device binary images per offload kind.
  llvm::DenseMap<OffloadKind, std::unique_ptr<SameKindPack>> Packs;
  /// Records all created memory buffers for safe auto-gc
  llvm::SmallVector<std::unique_ptr<MemoryBuffer>, 4> AutoGcBufs;

public:
  void addImage(const OffloadKind Kind, const llvm::StringRef File,
                const llvm::StringRef Manif, const llvm::StringRef Tgt,
                const BinaryImageFormat Fmt, const llvm::StringRef Opts) {
    std::unique_ptr<SameKindPack> &Pack = Packs[Kind];
    if (!Pack)
      Pack.reset(new SameKindPack());
    Pack->emplace_back(llvm::make_unique<Image>(File, Manif, Tgt, Fmt, Opts));
  }

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
      EntryTy = StructType::create(
          {
              Type::getInt8PtrTy(C), // addr
              Type::getInt8PtrTy(C), // name
              getSizeTTy(),          // size
              Type::getInt32Ty(C),   // flags
              Type::getInt32Ty(C)    // reserved
          },
          "__tgt_offload_entry");
    return EntryTy;
  }

  PointerType *getEntryPtrTy() { return PointerType::getUnqual(getEntryTy()); }

  const uint16_t DeviceImageStructVersion = 1;

  //  struct __tgt_device_image {
  //    /// version of this structure - for backward compatibility;
  //    /// all modifications which change order/type/offsets of existing fields
  //    /// should increment the version.
  //    uint16_t Version;
  //    /// the kind of offload model the image employs.
  //    uint8_t OffloadKind;
  //    /// format of the image data - SPIRV, LLVMIR bitcode,...
  //    uint8_t Format;
  //    /// null-terminated string representation of the device's target
  //    /// architecture
  //    const char *DeviceTargetSpec;
  //    /// a null-terminated string; target- and compiler-specific options
  //    /// which are suggested to use to "build" program at runtime
  //    const char *BuildOptions;
  //    /// Pointer to the manifest data start
  //    const unsigned char *ManifestStart;
  //    /// Pointer to the manifest data end
  //    const unsigned char *ManifestEnd;
  //    /// Pointer to the device binary image start
  //    void *ImageStart;
  //    /// Pointer to the device binary image end
  //    void *ImageEnd;
  //    /// the entry table
  //    __tgt_offload_entry *EntriesBegin;
  //    __tgt_offload_entry *EntriesEnd;
  //  };
  //
  StructType *getDeviceImageTy() {
    if (!ImageTy) {
      ImageTy = StructType::create(
          {
              Type::getInt16Ty(C),   // Version
              Type::getInt8Ty(C),    // OffloadKind
              Type::getInt8Ty(C),    // Format
              Type::getInt8PtrTy(C), // DeviceTargetSpec
              Type::getInt8PtrTy(C), // BuildOptions
              Type::getInt8PtrTy(C), // ManifestStart
              Type::getInt8PtrTy(C), // ManifestEnd
              Type::getInt8PtrTy(C), // ImageStart
              Type::getInt8PtrTy(C), // ImageEnd
              getEntryPtrTy(),       // EntriesBegin
              getEntryPtrTy()        // EntriesEnd
          },
          "__tgt_device_image");
    }
    return ImageTy;
  }

  PointerType *getDeviceImagePtrTy() {
    return PointerType::getUnqual(getDeviceImageTy());
  }

  const uint16_t BinDescStructVersion = 1;

  // struct __tgt_bin_desc {
  //   /// version of this structure - for backward compatibility;
  //   /// all modifications which change order/type/offsets of existing fields
  //   /// should increment the version.
  //   uint16_t Version;
  //   uint16_t NumDeviceImages;
  //   __tgt_device_image *DeviceImages;
  //   /// the offload entry table
  //   __tgt_offload_entry *HostEntriesBegin;
  //   __tgt_offload_entry *HostEntriesEnd;
  // };
  StructType *getBinDescTy() {
    if (!DescTy) {
      DescTy = StructType::create(
          {
              Type::getInt16Ty(C),   // Version
              Type::getInt16Ty(C),   // NumDeviceImages
              getDeviceImagePtrTy(), // DeviceImages
              getEntryPtrTy(),       // HostEntriesBegin
              getEntryPtrTy()        // HostEntriesEnd
          },
          "__tgt_bin_desc");
    }
    return DescTy;
  }

  PointerType *getBinDescPtrTy() {
    return PointerType::getUnqual(getBinDescTy());
  }

  MemoryBuffer *loadFile(llvm::StringRef Name) {
    auto InputOrErr = MemoryBuffer::getFileOrSTDIN(Name);

    if (auto EC = InputOrErr.getError()) {
      errs() << "error: can't read file " << Name << ": " << EC.message()
             << "\n";
      exit(1);
    }
    AutoGcBufs.emplace_back(std::move(InputOrErr.get()));
    return AutoGcBufs.back().get();
  }

  // Adds given buffer as a global variable into the module and, depending on
  // the StartEnd flag, returns either a pair of pointers to the beginning
  // and end of the variable or a <pointer to the beginning, size> pair. The
  // input memory buffer must outlive 'this' object.
  std::pair<Constant *, Constant *>
  addMemBufToModule(Module &M, MemoryBuffer *Buf, const Twine &Name) {
    auto *Buf1 = ConstantDataArray::get(
        C, makeArrayRef(Buf->getBufferStart(), Buf->getBufferSize()));
    auto *Var = new GlobalVariable(M, Buf1->getType(), true,
                                   GlobalVariable::InternalLinkage, Buf1, Name);
    if (Verbose)
      errs() << "  global added: " << Var->getName() << "\n";
    Var->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    auto *Zero = ConstantInt::get(getSizeTTy(), 0u);
    Constant *ZeroZero[] = {Zero, Zero};
    auto *ImageB =
        ConstantExpr::getGetElementPtr(Var->getValueType(), Var, ZeroZero);
    auto *Size = ConstantInt::get(getSizeTTy(), Buf->getBufferSize());

    Constant *ZeroSize[] = {Zero, Size};
    auto *ImageE =
        ConstantExpr::getGetElementPtr(Var->getValueType(), Var, ZeroSize);
    return std::make_pair(ImageB, ImageE);
  }

  // Creates a global variable of const char* type and creates an
  // initializer that initializes it with given null-terminated string.
  // Returns a link-time constant pointer (constant expr) to that variable.
  Constant *addStringToModule(Module &M, const std::string &Str,
                              const Twine &Name) {
    Constant *Arr =
        ConstantDataArray::get(C, makeArrayRef(Str.c_str(), Str.size() + 1));
    auto *Var = new GlobalVariable(M, Arr->getType(), true,
                                   GlobalVariable::InternalLinkage, Arr, Name);
    if (Verbose)
      errs() << "  global added: " << Var->getName() << "\n";
    Var->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    auto *Zero = ConstantInt::get(getSizeTTy(), 0u);
    Constant *ZeroZero[] = {Zero, Zero};
    return ConstantExpr::getGetElementPtr(Var->getValueType(), Var, ZeroZero);
  }

  GlobalVariable *createBinDesc(OffloadKind Kind, SameKindPack &Pack) {
    const std::string OffloadKindTag =
        (Twine(".") + offloadKindToString(Kind) + Twine("_offloading.")).str();

    Constant *EntriesB = nullptr, *EntriesE = nullptr;

    if (Kind != OffloadKind::SYCL) {
      EntriesB = new GlobalVariable(
          M, getEntryTy(), true, GlobalValue::ExternalLinkage, nullptr,
          Twine(OffloadKindTag) + Twine("entries_begin"));
      EntriesE = new GlobalVariable(
          M, getEntryTy(), true, GlobalValue::ExternalLinkage, nullptr,
          Twine(OffloadKindTag) + Twine("entries_end"));

      if (Verbose) {
        errs() << "  global added: " << EntriesB->getName() << "\n";
        errs() << "  global added: " << EntriesE->getName() << "\n";
      }
    } else {
      EntriesB = Constant::getNullValue(getEntryPtrTy());
      EntriesE = Constant::getNullValue(getEntryPtrTy());
    }
    auto *Zero = ConstantInt::get(getSizeTTy(), 0u);
    auto *NullPtr = Constant::getNullValue(Type::getInt8PtrTy(C));
    Constant *ZeroZero[] = {Zero, Zero};

    SmallVector<Constant *, 4> ImagesInits;
    unsigned ImgId = 0;

    for (const auto &ImgPtr : Pack) {
      const BinaryWrapper::Image &Img = *(ImgPtr.get());
      if (Verbose)
        errs() << "adding image: offload kind=" << offloadKindToString(Kind)
               << Img << "\n";
      auto *Fver =
          ConstantInt::get(Type::getInt16Ty(C), DeviceImageStructVersion);
      auto *Fknd = ConstantInt::get(Type::getInt8Ty(C), Kind);
      auto *Ffmt = ConstantInt::get(Type::getInt8Ty(C), Img.Fmt);
      auto *Ftgt = addStringToModule(
          M, Img.Tgt, Twine(OffloadKindTag) + Twine("target.") + Twine(ImgId));
      auto *Fopt = addStringToModule(
          M, Img.Opts, Twine(OffloadKindTag) + Twine("opts.") + Twine(ImgId));
      std::pair<Constant *, Constant *> FMnf;

      if (Img.Manif.empty()) {
        // no manifest - zero out the fields
        FMnf = std::make_pair(NullPtr, NullPtr);
      } else {
        MemoryBuffer *Mnf = loadFile(Img.Manif);
        FMnf = addMemBufToModule(
            M, Mnf, Twine(OffloadKindTag) + Twine(ImgId) + Twine(".manifest"));
      }
      if (Img.File.empty()) {
        errs() << "error: image file name missing\n";
        exit(1);
      }
      MemoryBuffer *Bin = loadFile(Img.File);
      std::pair<Constant *, Constant *> Fbin = addMemBufToModule(
          M, Bin, Twine(OffloadKindTag) + Twine(ImgId) + Twine(".data"));

      ImagesInits.push_back(ConstantStruct::get(
          getDeviceImageTy(),
          {Fver, Fknd, Ffmt, Ftgt, Fopt, FMnf.first, FMnf.second, Fbin.first,
           Fbin.second, EntriesB, EntriesE}));
      ImgId++;
    }
    auto *ImagesData = ConstantArray::get(
        ArrayType::get(getDeviceImageTy(), ImagesInits.size()), ImagesInits);

    auto *Images = new GlobalVariable(
        M, ImagesData->getType(), true, GlobalValue::InternalLinkage,
        ImagesData, Twine(OffloadKindTag) + Twine("device_images"));
    if (Verbose)
      errs() << "  global added: " << Images->getName() << "\n";
    Images->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

    auto *ImagesB = ConstantExpr::getGetElementPtr(Images->getValueType(),
                                                   Images, ZeroZero);
    Constant *Version =
        ConstantInt::get(Type::getInt16Ty(C), BinDescStructVersion);
    Constant *NumImages =
        ConstantInt::get(Type::getInt16Ty(C), ImagesInits.size());
    auto *DescInit = ConstantStruct::get(
        getBinDescTy(), {Version, NumImages, ImagesB, EntriesB, EntriesE});

    GlobalValue::LinkageTypes Lnk = DescriptorName.getNumOccurrences() > 0
                                        ? GlobalValue::ExternalLinkage
                                        : GlobalValue::InternalLinkage;
    auto *Res =
        new GlobalVariable(M, DescInit->getType(), true, Lnk, DescInit,
                           Twine(OffloadKindTag) + Twine(DescriptorName));
    if (Verbose)
      errs() << "  global added: " << Res->getName() << "\n";
    return Res;
  }

  void createRegisterFunction(OffloadKind Kind, GlobalVariable *BinDesc) {
    const std::string OffloadKindTag =
        (Twine(".") + offloadKindToString(Kind) + Twine("_offloading.")).str();
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

  void createUnregisterFunction(OffloadKind Kind, GlobalVariable *BinDesc) {
    const std::string OffloadKindTag =
        (Twine(".") + offloadKindToString(Kind) + Twine("_offloading.")).str();
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
  BinaryWrapper(const StringRef &Target) : M("offload.wrapper.object", C) {
    M.setTargetTriple(Target);
  }

  const Module &wrap() {
    for (auto &X : Packs) {
      OffloadKind Kind = X.first;
      SameKindPack *Pack = X.second.get();
      auto *Desc = createBinDesc(Kind, *Pack);
      assert(Desc && "no binary descriptor");

      if (EmitRegFuncs) {
        createRegisterFunction(Kind, Desc);
        createUnregisterFunction(Kind, Desc);
      }
    }
    return M;
  }
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &Out,
                              const BinaryWrapper::Image &Img) {
  Out << "\n{\n";
  Out << "  file     = " << Img.File << "\n";
  Out << "  manifest = " << (Img.Manif.empty() ? "-" : Img.Manif) << "\n";
  Out << "  format   = " << formatToString(Img.Fmt) << "\n";
  Out << "  target   = " << (Img.Tgt.empty() ? "-" : Img.Tgt) << "\n";
  Out << "  options  = " << (Img.Opts.empty() ? "-" : Img.Opts) << "\n";
  Out << "}\n";
  return Out;
}

// enable_if_t is available only starting with C++14
template <bool Cond, typename T = void>
using my_enable_if_t = typename std::enable_if<Cond, T>::type;

// Helper class to order elements of multiple cl::list option lists according to
// the sequence they occurred on the command line. Each cl::list defines a
// separate options "class" to identify which class current options belongs to.
// The ID of a class is simply the ordinal of its corresponding cl::list object
// as passed to the constructor. Typical usage:
//  do {
//    ID = ArgSeq.next();
//
//    switch (ID) {
//    case -1: // Done
//      break;
//    case 0: // An option from the cl::list which came first in the constructor
//      (*(ArgSeq.template get<0>())); // get the option value
//      break;
//    case 1: // An option from the cl::list which came second in the
//    constructor
//      (*(ArgSeq.template get<1>())); // get the option value
//      break;
//    ...
//    default:
//      llvm_unreachable("bad option class ID");
//    }
//  } while (ID != -1);
//
template <typename... Tys> class ListArgsSequencer {
private:
  /// The class ID of current option
  int Cur = -1;

  /// Class IDs of all options from all lists. Filled in the constructor.
  std::unique_ptr<std::vector<int>> OptListIDs;

  using tuple_of_iters_t = std::tuple<typename Tys::iterator...>;

  template <size_t I>
  using iter_t = typename std::tuple_element<I, tuple_of_iters_t>::type;

  /// Tuple of all lists' iterators pointing to "previous" option value -
  /// before latest next() was called
  tuple_of_iters_t Prevs;

  /// Holds "current" iterators - after next()
  tuple_of_iters_t Iters;

public:
  /// The only constructor.
  /// Sz   - total number of options on the command line
  /// Args - the cl::list objects to sequence elements of
  ListArgsSequencer(size_t Sz, Tys &... Args)
      : Prevs(Args.end()...), Iters(Args.begin()...) {
    assert(Sz >= sizeof...(Tys));
    OptListIDs.reset(new std::vector<int>(Sz, -1));
    addLists<sizeof...(Tys) - 1, 0>(Args...);
  }

  ListArgsSequencer() = delete;

  /// Advances to the next option in the sequence. Returns the option class ID
  /// or -1 when all lists' elements have been iterated over.
  int next() {
    size_t Sz = OptListIDs->size();

    if ((Cur > 0) && (((size_t)Cur) >= Sz))
      return -1;
    while ((((size_t)++Cur) < Sz) && (cur() == -1))
      ;

    if (((size_t)Cur) < Sz)
      inc<sizeof...(Tys) - 1>();
    return ((size_t)Cur) >= Sz ? -1 : cur();
  }

  /// Retrieves the value of current option. ID must match is the option class
  /// returned by next(), otherwise compile error can happen or incorrect option
  /// value will be retrieved.
  template <int ID> decltype(std::get<ID>(Prevs)) get() {
    return std::get<ID>(Prevs);
  }

private:
  int cur() {
    assert(Cur >= 0 && ((size_t)Cur) < OptListIDs->size());
    return (*OptListIDs)[Cur];
  }

  template <int MAX, int ID, typename XTy, typename... XTys>
      my_enable_if_t < ID<MAX> addLists(XTy &Arg, XTys &... Args) {
    addListImpl<ID>(Arg);
    addLists<MAX, ID + 1>(Args...);
  }

  template <int MAX, int ID, typename XTy>
  my_enable_if_t<ID == MAX> addLists(XTy &Arg) {
    addListImpl<ID>(Arg);
  }

  /// Does the actual sequencing of options found in given list.
  template <int ID, typename T> void addListImpl(T &L) {
    for (auto It = L.begin(); It != L.end(); It++) {
      unsigned Pos = L.getPosition(It - L.begin());
      assert((*OptListIDs)[Pos] == -1);
      (*OptListIDs)[Pos] = ID;
    }
  }

  template <int N> void incImpl() {
    if (cur() == -1)
      return;
    if (N == cur()) {
      std::get<N>(Prevs) = std::get<N>(Iters);
      std::get<N>(Iters)++;
    }
  }

  template <int N> my_enable_if_t<N != 0> inc() {
    incImpl<N>();
    inc<N - 1>();
  }

  template <int N> my_enable_if_t<N == 0> inc() { incImpl<N>(); }
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
      "Takes offload target binaries and optional manifest files as input\n"
      "and produces bitcode file containing target binaries packaged as data\n"
      "and initialization code which registers target binaries in the offload\n"
      "runtime. Manifest files format and contents are not restricted and are\n"
      "a subject of agreement between the device compiler and the native\n"
      "runtime for that device. When present, manifest file name should\n"
      "immediately follow the corresponding device image filename on the\n"
      "command line. Options annotating a device binary have effect on all\n"
      "subsequent input, until redefined. For example:\n"
      "$clang-offload-wrapper -host x86_64-pc-linux-gnu \\\n"
      "  -kind=sycl -target=spir64 -format=spirv -build-opts=-g \\\n"
      "  a.spv a_mf.txt \\\n"
      "             -target=xxx -format=native -build-opts=\"\"  \\\n"
      "  b.bin b_mf.txt \\\n"
      "  -kind=openmp \\\n"
      "  c.bin\n"
      "will generate an x86 wrapper object (.bc) enclosing the following\n"
      "tuples describing a single device binary each ('-' means 'none')\n\n"
      "offload kind | target | data format | data | manifest | build options:\n"
      "----------------------------------------------------------------------\n"
      "    sycl     | spir64 | spirv       | a.spv| a_mf.txt | -g\n"
      "    sycl     | xxx    | native      | b.bin| b_mf.txt | -\n"
      "    openmp   | xxx    | native      | c.bin| -        | -\n");

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }
  if (Target.empty()) {
    errs() << "error: no target specified\n";
    return 1;
  }

  // Construct BinaryWrapper::Image instances based on command line args and
  // add them to the wrapper

  BinaryWrapper Wr(Target);
  OffloadKind Knd = OffloadKind::Unknown;
  llvm::StringRef Tgt = "";
  BinaryImageFormat Fmt = BinaryImageFormat::none;
  llvm::StringRef Opts = "";
  llvm::SmallVector<llvm::StringRef, 2> CurInputPair;

  ListArgsSequencer<decltype(Inputs), decltype(Kinds), decltype(Formats),
                    decltype(Targets), decltype(Options)>
      ArgSeq((size_t)argc, Inputs, Kinds, Formats, Targets, Options);
  int ID = -1;

  do {
    ID = ArgSeq.next();

    if (ID != 0) {
      // cur option is not an input - create and image instance using current
      // state
      if (CurInputPair.size() > 2) {
        errs() << "too many inputs for a single binary image, <binary file> "
                  "<manifest file>{opt}expected\n";
        return 1;
      }
      if (CurInputPair.size() != 0) {
        if (Knd == OffloadKind::Unknown) {
          errs() << "error: offload model not set\n";
          return 1;
        }
        StringRef File = CurInputPair[0];
        StringRef Manif = CurInputPair.size() > 1 ? CurInputPair[1] : "";
        Wr.addImage(Knd, File, Manif, Tgt, Fmt, Opts);
        CurInputPair.clear();
      }
    }
    switch (ID) {
    case -1: // Done
      break;
    case 0: // Inputs
      CurInputPair.push_back(*(ArgSeq.template get<0>()));
      break;
    case 1: // Kinds
      Knd = *(ArgSeq.template get<1>());
      break;
    case 2: // Formats
      Fmt = *(ArgSeq.template get<2>());
      break;
    case 3: // Targets
      Tgt = *(ArgSeq.template get<3>());
      break;
    case 4: // Options
      Opts = *(ArgSeq.template get<4>());
      break;
    default:
      llvm_unreachable("bad option class ID");
    }
  } while (ID != -1);

  // Create the bitcode file to write the resulting code to.
  std::error_code EC;
  raw_fd_ostream OutF(Output, EC, sys::fs::F_None);
  if (EC) {
    errs() << "error: unable to open output file: " << EC.message() << ".\n";
    return 1;
  }
  // Create a wrapper for device binaries and write its bitcode to the file.
  WriteBitcodeToFile(Wr.wrap(), OutF);

  return 0;
}
