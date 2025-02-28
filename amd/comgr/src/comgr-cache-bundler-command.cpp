/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2003-2017 University of Illinois at Urbana-Champaign.
 * Modifications (c) 2018 Advanced Micro Devices, Inc.
 * All rights reserved.
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
 *     * Neither the names of the LLVM Team, University of Illinois at
 *       Urbana-Champaign, nor the names of its contributors may be used to
 *       endorse or promote products derived from this Software without specific
 *       prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include <comgr-cache-bundler-command.h>

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

    if (Error Err = CachedCommandAdaptor::writeUniqueExecuteOutput(
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
        CachedCommandAdaptor::readUniqueExecuteOutput(OutputFilename);
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
  H.update(Config.TargetNames.size());
  for (StringRef Target : Config.TargetNames) {
    CachedCommandAdaptor::addString(H, Target);
  }
}

Error UnbundleCommand::addInputIdentifier(HashAlgorithm &H) const {
  StringRef InputFilename = Config.InputFileNames.front();

  constexpr size_t LargestHeaderSize = CompressedOffloadBundle::V3HeaderSize;

  ErrorOr<std::unique_ptr<MemoryBuffer>> MaybeInputBuffer =
      MemoryBuffer::getFileSlice(InputFilename, LargestHeaderSize, 0);
  if (!MaybeInputBuffer) {
    std::error_code EC = MaybeInputBuffer.getError();
    return createStringError(EC, Twine("Failed to open ") + InputFilename +
                                     " : " + EC.message() + "\n");
  }

  MemoryBuffer &InputBuffer = **MaybeInputBuffer;

  uint8_t Header[LargestHeaderSize];
  memset(Header, 0, sizeof(Header));
  memcpy(Header, InputBuffer.getBufferStart(),
         std::min(LargestHeaderSize, InputBuffer.getBufferSize()));

  // only hash the input file, not the whole header. Colissions are unlikely
  // since the header includes a hash (weak) of the contents
  H.update(Header);
  return Error::success();
}

} // namespace COMGR
