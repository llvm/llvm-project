//===- comgr-unbundle-command.cpp - UnbundleCommand implementation --------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the CacheCommandAdaptor interface for
/// llvm::OffloadBundler::Unbundle() routines that are stored in the cache.
///
//===----------------------------------------------------------------------===//

#include <comgr-unbundle-command.h>

#include <clang/Driver/OffloadBundler.h>
#include <llvm/BinaryFormat/Magic.h>

namespace COMGR {
using namespace llvm;
using namespace clang;

using SizeFieldType = uint32_t;

bool UnbundleCommand::canCache() const {
  // The header format for AR files is not the same as object files
  if (Kind == AMD_COMGR_DATA_KIND_AR_BUNDLE)
    return false;

  StringRef InputFilename = Config.InputFileNames.front();
  file_magic Magic;
  if (identify_magic(InputFilename, Magic))
    return false;

  // Check the input file magic. Handle only compressed bundles
  // It's not worth to cache other types of bundles
  return Magic == file_magic::offload_bundle_compressed;
}

Error UnbundleCommand::writeExecuteOutput(StringRef CachedBuffer) {
  for (StringRef OutputFilename : Config.OutputFileNames) {
    SizeFieldType OutputFileSize;
    if (CachedBuffer.size() < sizeof(OutputFileSize))
      return createStringError(std::errc::invalid_argument,
                               "Not enough bytes to read output file size");
    memcpy(&OutputFileSize, CachedBuffer.data(), sizeof(OutputFileSize));
    CachedBuffer = CachedBuffer.drop_front(sizeof(OutputFileSize));

    if (CachedBuffer.size() < OutputFileSize)
      return createStringError(std::errc::invalid_argument,
                               "Not enough bytes to read output file contents");

    StringRef OutputFileContents = CachedBuffer.substr(0, OutputFileSize);
    CachedBuffer = CachedBuffer.drop_front(OutputFileSize);

    if (Error Err = CachedCommandAdaptor::writeSingleOutputFile(
            OutputFilename, OutputFileContents))
      return Err;
  }

  if (!CachedBuffer.empty())
    return createStringError(std::errc::invalid_argument,
                             "Bytes in cache entry not used for the output");
  return Error::success();
}

Expected<StringRef> UnbundleCommand::readExecuteOutput() {
  size_t OutputSize = 0;
  for (StringRef OutputFilename : Config.OutputFileNames) {
    auto MaybeOneOutput =
        CachedCommandAdaptor::readSingleOutputFile(OutputFilename);
    if (!MaybeOneOutput)
      return MaybeOneOutput.takeError();

    const MemoryBuffer &OneOutputBuffer = **MaybeOneOutput;
    SizeFieldType OneOutputFileSize = OneOutputBuffer.getBufferSize();

    OutputBuffer.resize_for_overwrite(OutputSize + sizeof(OneOutputFileSize) +
                                      OneOutputFileSize);

    memcpy(OutputBuffer.data() + OutputSize, &OneOutputFileSize,
           sizeof(OneOutputFileSize));
    OutputSize += sizeof(OneOutputFileSize);
    memcpy(OutputBuffer.data() + OutputSize, OneOutputBuffer.getBufferStart(),
           OneOutputFileSize);
    OutputSize += OneOutputFileSize;
  }
  return OutputBuffer;
}

amd_comgr_status_t UnbundleCommand::execute(raw_ostream &LogS) {
  assert(Config.InputFileNames.size() == 1);

  OffloadBundler Bundler(Config);

  switch (Kind) {
  case AMD_COMGR_DATA_KIND_BC_BUNDLE:
  case AMD_COMGR_DATA_KIND_OBJ_BUNDLE: {
    if (Error Err = Bundler.UnbundleFiles()) {
      logAllUnhandledErrors(std::move(Err), LogS, "Unbundle Error: ");
      return AMD_COMGR_STATUS_ERROR;
    }
    break;
  }
  case AMD_COMGR_DATA_KIND_AR_BUNDLE: {
    if (Error Err = Bundler.UnbundleArchive()) {
      logAllUnhandledErrors(std::move(Err), LogS, "Unbundle Archives Error: ");
      return AMD_COMGR_STATUS_ERROR;
    }
    break;
  }
  default:
    llvm_unreachable("invalid bundle type");
  }

  return AMD_COMGR_STATUS_SUCCESS;
}

CachedCommandAdaptor::ActionClass UnbundleCommand::getClass() const {
  return clang::driver::Action::OffloadUnbundlingJobClass;
}

void UnbundleCommand::addOptionsIdentifier(HashAlgorithm &H) const {
  addUInt(H, Config.TargetNames.size());
  for (StringRef Target : Config.TargetNames) {
    CachedCommandAdaptor::addString(H, Target);
  }
}

Error UnbundleCommand::addInputIdentifier(HashAlgorithm &H) const {
  StringRef InputFilename = Config.InputFileNames.front();

  ErrorOr<std::unique_ptr<MemoryBuffer>> MaybeInputBuffer =
      MemoryBuffer::getFile(InputFilename);
  if (!MaybeInputBuffer) {
    std::error_code EC = MaybeInputBuffer.getError();
    return createStringError(EC, Twine("Failed to open ") + InputFilename +
                                     " : " + EC.message() + "\n");
  }

  MemoryBuffer &InputBuffer = **MaybeInputBuffer;

  using Header = CompressedOffloadBundle::CompressedBundleHeader;
  Expected<Header> MaybeHeader = Header::tryParse(InputBuffer.getBuffer());
  if (!MaybeHeader)
    return MaybeHeader.takeError();

  // The hash represents the contents of the bundle. Extracting the same
  // contents should give the same result, regardless of the compression
  // algorithm or header version. Since the hash used by the offload bundler is
  // not a cryptographic hash, we also add the uncompressed file size.
  addUInt(H, MaybeHeader->Hash);
  addUInt(H, MaybeHeader->UncompressedFileSize);
  return Error::success();
}

} // namespace COMGR
