//===-- clang-offload-packager/ClangOffloadPackager.cpp - file bundler ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This tool takes several device object files and bundles them into a single
// binary image using a custom binary format. This is intended to be used to
// embed many device files into an application to create a fat binary.
//
//===---------------------------------------------------------------------===//

#include "clang/Basic/Version.h"

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace llvm::object;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

static cl::OptionCategory
    ClangOffloadPackagerCategory("clang-offload-packager options");

static cl::opt<std::string> OutputFile("o", cl::desc("Write output to <file>."),
                                       cl::value_desc("file"),
                                       cl::cat(ClangOffloadPackagerCategory));

static cl::opt<std::string> InputFile(cl::Positional,
                                      cl::desc("Extract from <file>."),
                                      cl::value_desc("file"),
                                      cl::cat(ClangOffloadPackagerCategory));

static cl::list<std::string>
    DeviceImages("image",
                 cl::desc("List of key and value arguments. Required keywords "
                          "are 'file' and 'triple'."),
                 cl::value_desc("<key>=<value>,..."),
                 cl::cat(ClangOffloadPackagerCategory));

static void PrintVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("clang-offload-packager") << '\n';
}

// Get a map containing all the arguments for the image. Repeated arguments will
// be placed in a comma separated list.
static DenseMap<StringRef, StringRef> getImageArguments(StringRef Image,
                                                        StringSaver &Saver) {
  DenseMap<StringRef, StringRef> Args;
  for (StringRef Arg : llvm::split(Image, ",")) {
    auto [Key, Value] = Arg.split("=");
    if (Args.count(Key))
      Args[Key] = Saver.save(Args[Key] + "," + Value);
    else
      Args[Key] = Value;
  }

  return Args;
}

static Error bundleImages() {
  SmallVector<char, 1024> BinaryData;
  raw_svector_ostream OS(BinaryData);
  for (StringRef Image : DeviceImages) {
    BumpPtrAllocator Alloc;
    StringSaver Saver(Alloc);
    DenseMap<StringRef, StringRef> Args = getImageArguments(Image, Saver);

    if (!Args.count("triple") || !Args.count("file"))
      return createStringError(
          inconvertibleErrorCode(),
          "'file' and 'triple' are required image arguments");

    OffloadBinary::OffloadingImage ImageBinary{};
    std::unique_ptr<llvm::MemoryBuffer> DeviceImage;
    for (const auto &[Key, Value] : Args) {
      if (Key == "file") {
        llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ObjectOrErr =
            llvm::MemoryBuffer::getFileOrSTDIN(Value);
        if (std::error_code EC = ObjectOrErr.getError())
          return errorCodeToError(EC);

        // Clang uses the '.o' suffix for LTO bitcode.
        if (identify_magic((*ObjectOrErr)->getBuffer()) == file_magic::bitcode)
          ImageBinary.TheImageKind = object::IMG_Bitcode;
        else
          ImageBinary.TheImageKind =
              getImageKind(sys::path::extension(Value).drop_front());
        ImageBinary.Image = std::move(*ObjectOrErr);
      } else if (Key == "kind") {
        ImageBinary.TheOffloadKind = getOffloadKind(Value);
      } else {
        ImageBinary.StringData[Key] = Value;
      }
    }
    std::unique_ptr<MemoryBuffer> Buffer = OffloadBinary::write(ImageBinary);
    if (Buffer->getBufferSize() % OffloadBinary::getAlignment() != 0)
      return createStringError(inconvertibleErrorCode(),
                               "Offload binary has invalid size alignment");
    OS << Buffer->getBuffer();
  }

  Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
      FileOutputBuffer::create(OutputFile, BinaryData.size());
  if (!OutputOrErr)
    return OutputOrErr.takeError();
  std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
  std::copy(BinaryData.begin(), BinaryData.end(), Output->getBufferStart());
  if (Error E = Output->commit())
    return E;
  return Error::success();
}

static Error unbundleImages() {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFileOrSTDIN(InputFile);
  if (std::error_code EC = BufferOrErr.getError())
    return createFileError(InputFile, EC);
  std::unique_ptr<MemoryBuffer> Buffer = std::move(*BufferOrErr);

  // This data can be misaligned if extracted from an archive.
  if (!isAddrAligned(Align(OffloadBinary::getAlignment()),
                     Buffer->getBufferStart()))
    Buffer = MemoryBuffer::getMemBufferCopy(Buffer->getBuffer(),
                                            Buffer->getBufferIdentifier());

  SmallVector<OffloadFile> Binaries;
  if (Error Err = extractOffloadBinaries(*Buffer, Binaries))
    return Err;

  // Try to extract each device image specified by the user from the input file.
  for (StringRef Image : DeviceImages) {
    BumpPtrAllocator Alloc;
    StringSaver Saver(Alloc);
    auto Args = getImageArguments(Image, Saver);

    for (uint64_t I = 0, E = Binaries.size(); I != E; ++I) {
      const auto *Binary = Binaries[I].getBinary();
      // We handle the 'file' and 'kind' identifiers differently.
      bool Match = llvm::all_of(Args, [&](auto &Arg) {
        const auto [Key, Value] = Arg;
        if (Key == "file")
          return true;
        if (Key == "kind")
          return Binary->getOffloadKind() == getOffloadKind(Value);
        return Binary->getString(Key) == Value;
      });
      if (!Match)
        continue;

      // If the user did not provide a filename derive one from the input and
      // image.
      StringRef Filename =
          !Args.count("file")
              ? Saver.save(sys::path::stem(InputFile) + "-" +
                           Binary->getTriple() + "-" + Binary->getArch() + "." +
                           std::to_string(I) + "." +
                           getImageKindName(Binary->getImageKind()))
              : Args["file"];

      Expected<std::unique_ptr<FileOutputBuffer>> OutputOrErr =
          FileOutputBuffer::create(Filename, Binary->getImage().size());
      if (!OutputOrErr)
        return OutputOrErr.takeError();
      std::unique_ptr<FileOutputBuffer> Output = std::move(*OutputOrErr);
      llvm::copy(Binary->getImage(), Output->getBufferStart());
      if (Error E = Output->commit())
        return E;
    }
  }

  return Error::success();
}

int main(int argc, const char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  cl::HideUnrelatedOptions(ClangOffloadPackagerCategory);
  cl::SetVersionPrinter(PrintVersion);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A utility for bundling several object files into a single binary.\n"
      "The output binary can then be embedded into the host section table\n"
      "to create a fatbinary containing offloading code.\n");

  if (Help) {
    cl::PrintHelpMessage();
    return EXIT_SUCCESS;
  }

  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
    return EXIT_FAILURE;
  };

  if (!InputFile.empty() && !OutputFile.empty())
    return reportError(
        createStringError(inconvertibleErrorCode(),
                          "Packaging to an output file and extracting from an "
                          "input file are mutually exclusive."));

  if (!OutputFile.empty()) {
    if (Error Err = bundleImages())
      return reportError(std::move(Err));
  } else if (!InputFile.empty()) {
    if (Error Err = unbundleImages())
      return reportError(std::move(Err));
  }

  return EXIT_SUCCESS;
}
