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

#include "comgr-cache.h"
#include "comgr-cache-command.h"
#include "comgr-env.h"
#include "comgr.h"

#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Caching.h>
#include <llvm/Support/MemoryBuffer.h>

namespace COMGR {
using namespace llvm;
using namespace clang::driver;

namespace {

const unsigned CacheTask = 1;

void serializeCacheEntry(llvm::raw_ostream &FS, StringRef FileContents,
                         StringRef Log) {
  auto WriteStringRef = [&FS](StringRef Buf) {
    uint64_t Size = Buf.size();
    constexpr size_t NSize = sizeof(Size);
    char SizeBuf[NSize];
    memcpy(SizeBuf, &Size, NSize);
    FS.write(SizeBuf, NSize);
    FS.write(Buf.data(), Size);
  };

  for (StringRef *Buf : {&FileContents, &Log}) {
    WriteStringRef(*Buf);
  }
}

Error deserializeCacheEntry(const llvm::MemoryBuffer &Buffer,
                            StringRef &FileContents, StringRef &Log) {
  auto ConsumeStringRef = [&](StringRef Buffer,
                              StringRef &Buf) -> Expected<StringRef> {
    uint64_t Size;
    constexpr size_t NSize = sizeof(Size);
    if (NSize > Buffer.size())
      return createStringError(
          "Cache entry file too small: couldn't read buffer size");
    memcpy(&Size, Buffer.data(), NSize);
    Buffer = Buffer.substr(NSize);
    if (Size > Buffer.size())
      return createStringError(
          "Cache entry file too small: couldn't read buffer");
    Buf = Buffer.substr(0, Size);
    return Buffer.substr(Size);
  };

  StringRef UnreadBuffer = Buffer.getBuffer();
  for (StringRef *Buf : {&FileContents, &Log}) {
    auto ErrOrUnread = ConsumeStringRef(UnreadBuffer, *Buf);
    if (!ErrOrUnread)
      return ErrOrUnread.takeError();
    UnreadBuffer = *ErrOrUnread;
  }

  if (!UnreadBuffer.empty())
    return createStringError(
        "Cache entry file too big: extra bytes after the end");

  return Error::success();
}

std::function<void(Error, const char *)>
getComgrCacheErrorHandler(llvm::raw_ostream &LogS) {
  if (!env::shouldEmitVerboseLogs()) {
    return [](Error E, const char *) { consumeError(std::move(E)); };
  }

  return [&LogS](Error E, const char *When) {
    logAllUnhandledErrors(std::move(E), LogS,
                          Twine("Comgr cache, ") + When + ": ");
  };
}

void saveCommandOutput(CachedCommandAdaptor &C, AddStreamFn &AddStream,
                       StringRef CapturedLogS, raw_ostream &LogS) {
  auto ErrorHandler = getComgrCacheErrorHandler(LogS);

  Expected<std::unique_ptr<CachedFileStream>> FileOrErr =
      AddStream(CacheTask, "");
  if (!FileOrErr) {
    ErrorHandler(FileOrErr.takeError(), "when getting the cached file stream");
    return;
  }

  Expected<StringRef> Buffer = C.readExecuteOutput();
  if (!Buffer) {
    ErrorHandler(Buffer.takeError(), "when reading command's output");
    return;
  }

  CachedFileStream *CFS = FileOrErr->get();
  serializeCacheEntry(*CFS->OS, *Buffer, CapturedLogS);
  ErrorHandler(CFS->commit(), "when commiting file stream");
}

bool readEntryFromCache(CachedCommandAdaptor &C, MemoryBuffer &CachedBuffer,
                        raw_ostream &LogS) {
  auto ErrorHandler = getComgrCacheErrorHandler(LogS);

  StringRef CachedOutputFile;
  StringRef CachedLogS;
  if (Error E =
          deserializeCacheEntry(CachedBuffer, CachedOutputFile, CachedLogS)) {
    ErrorHandler(std::move(E), "when reading the cache entry");
    return false;
  }

  if (Error E = C.writeExecuteOutput(CachedOutputFile)) {
    ErrorHandler(std::move(E), "when writing the command output");
    return false;
  }

  LogS << CachedLogS;
  return true;
}
} // namespace

std::optional<CachePruningPolicy>
CommandCache::getPolicyFromEnv(llvm::raw_ostream &LogS) {
  StringRef PolicyString = COMGR::env::getCachePolicy();
  if (PolicyString.empty()) {
    // Default policy: scan at most once per hour, take up at most 75% of
    // available disk space or 5GB (whichever is smaller), no limit on number
    // or age of files.

    CachePruningPolicy DefaultPolicy;
    DefaultPolicy.Interval = std::chrono::hours(1);
    DefaultPolicy.Expiration = std::chrono::hours(0);
    DefaultPolicy.MaxSizePercentageOfAvailableSpace = 75;
    DefaultPolicy.MaxSizeBytes = 5ul << 30; // Gb to byte;
    DefaultPolicy.MaxSizeFiles = 0;
    return DefaultPolicy;
  }

  Expected<CachePruningPolicy> PolicyOrErr =
      parseCachePruningPolicy(PolicyString);
  if (!PolicyOrErr) {
    auto ErrorHandler = getComgrCacheErrorHandler(LogS);
    ErrorHandler(PolicyOrErr.takeError(), "when parsing the cache policy");
    return std::nullopt;
  }
  return *PolicyOrErr;
}

void CommandCache::prune() { pruneCache(CacheDir, Policy); }

std::unique_ptr<CommandCache> CommandCache::get(raw_ostream &LogS) {
  StringRef CacheDir = env::getCacheDirectory();
  if (CacheDir.empty())
    return nullptr;

  std::optional<CachePruningPolicy> Policy =
      CommandCache::getPolicyFromEnv(LogS);
  if (!Policy)
    return nullptr;

  return std::unique_ptr<CommandCache>(new CommandCache(CacheDir, *Policy));
}

CommandCache::CommandCache(StringRef CacheDir, const CachePruningPolicy &Policy)
    : CacheDir(CacheDir.str()), Policy(Policy) {
  assert(!CacheDir.empty());
}

CommandCache::~CommandCache() { prune(); }

amd_comgr_status_t CommandCache::execute(CachedCommandAdaptor &C,
                                         raw_ostream &LogS) {

  if (!C.canCache()) {
    // Do not cache preprocessor commands.
    // Handling include directories and constants is hard and this simplifies
    // our implementation. Preprocessing is fast.
    return C.execute(LogS);
  }

  // This lambda will get called when the data is gotten from the cache and
  // also after the data was set for a given key.
  std::unique_ptr<MemoryBuffer> CachedBuffer;
  auto AddBuffer = [&CachedBuffer](unsigned Task, const Twine &ModuleName,
                                   std::unique_ptr<MemoryBuffer> M) {
    CachedBuffer = std::move(M);
  };

  auto ErrorHandler = getComgrCacheErrorHandler(LogS);

  Expected<FileCache> CacheOrErr =
      localCache("AMDGPUCompilerCache", "amdgpu-compiler", CacheDir, AddBuffer);
  if (!CacheOrErr) {
    ErrorHandler(CacheOrErr.takeError(), "when creating cache directory");
    return C.execute(LogS);
  }

  auto MaybeId = C.getIdentifier();
  if (!MaybeId) {
    ErrorHandler(MaybeId.takeError(),
                 "when computing the identifier for the command");
    return C.execute(LogS);
  }

  FileCache &Cache = *CacheOrErr;

  // If we call the "Cache" function and the data is cached, it will call the
  // "AddBuffer" lambda function from the constructor which will in turn take
  // ownership of the member buffer that is passed to the callback and put it
  // into the CachedBuffer member variable.
  Expected<AddStreamFn> AddStreamOrErr = Cache(CacheTask, *MaybeId, "");
  if (!AddStreamOrErr) {
    ErrorHandler(AddStreamOrErr.takeError(),
                 "when building the add stream callback");
    return C.execute(LogS);
  }

  // If the "AddStream" is nullptr, then the data was cached and we already
  // called the "AddBuffer" lambda.
  AddStreamFn &AddStream = *AddStreamOrErr;
  if (!AddStream && readEntryFromCache(C, *CachedBuffer, LogS)) {
    if (env::shouldEmitVerboseLogs())
      LogS << "Comgr cache: entry " << *MaybeId << " found in cache.\n";
    return AMD_COMGR_STATUS_SUCCESS;
  }

  std::string CapturedLogS;
  llvm::raw_string_ostream CaptureLogS(CapturedLogS);
  amd_comgr_status_t Result = C.execute(CaptureLogS);
  CaptureLogS.flush();
  LogS << CapturedLogS;

  if (Result == AMD_COMGR_STATUS_SUCCESS && AddStream) {
    saveCommandOutput(C, AddStream, CapturedLogS, LogS);
  }

  return Result;
}
} // namespace COMGR
