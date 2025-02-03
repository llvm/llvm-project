#include "comgr-cache-command.h"
#include "comgr-cache.h"
#include "comgr-device-libs.h"
#include "comgr-env.h"
#include "comgr.h"

#include <clang/Basic/Version.h>
#include <llvm/ADT/StringExtras.h>

#include <optional>

namespace COMGR {
using namespace llvm;
using namespace clang;

namespace {
// std::isalnum is locale dependent and can have issues
// depending on the stdlib version and application. We prefer to avoid it
bool isalnum(char c) {
  char low[] = {'0', 'a', 'A'};
  char hi[] = {'9', 'z', 'Z'};
  for (unsigned i = 0; i != 3; ++i) {
    if (low[i] <= c && c <= hi[i])
      return true;
  }
  return false;
}

} // namespace

std::optional<size_t> CachedCommandAdaptor::searchComgrTmpModel(StringRef S) {
  // Ideally, we would use std::regex_search with the regex
  // "comgr-[[:alnum:]]{6}". However, due to a bug in stdlibc++
  // (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85824) we have to roll our
  // own search of this regular expression. This bug resulted in a crash in
  // luxmarkv3, during the std::regex constructor.
  const StringRef Prefix = "comgr-";
  const size_t AlnumCount = 6;

  size_t N = S.size();
  size_t Pos = S.find(Prefix);

  size_t AlnumStart = Pos + Prefix.size();
  size_t AlnumEnd = AlnumStart + AlnumCount;
  if (Pos == StringRef::npos || N < AlnumEnd)
    return std::nullopt;

  for (size_t i = AlnumStart; i < AlnumEnd; ++i) {
    if (!isalnum(S[i]))
      return std::nullopt;
  }

  return Pos;
}

void CachedCommandAdaptor::addString(CachedCommandAdaptor::HashAlgorithm &H,
                                     StringRef S) {
  // hash size + contents to avoid collisions
  // for example, we have to ensure that the result of hashing "AA" "BB" is
  // different from "A" "ABB"
  H.update(S.size());
  H.update(S);
}

void CachedCommandAdaptor::addFileContents(
    CachedCommandAdaptor::HashAlgorithm &H, StringRef Buf) {
  // this is a workaround temporary paths getting in the output files of the
  // different commands in #line directives in preprocessed files, and the
  // ModuleID or source_filename in the bitcode.
  while (!Buf.empty()) {
    std::optional<size_t> ComgrTmpPos = searchComgrTmpModel(Buf);
    if (!ComgrTmpPos) {
      addString(H, Buf);
      break;
    }

    StringRef ToHash = Buf.substr(0, *ComgrTmpPos);
    addString(H, ToHash);
    Buf = Buf.substr(ToHash.size() + StringRef("comgr-xxxxxx").size());
  }
}

Expected<CachedCommandAdaptor::Identifier>
CachedCommandAdaptor::getIdentifier() const {
  CachedCommandAdaptor::HashAlgorithm H;
  H.update(getClass());
  H.update(env::shouldEmitVerboseLogs());
  addString(H, getClangFullVersion());
  addString(H, getComgrHashIdentifier());
  H.update(getDeviceLibrariesIdentifier());

  if (Error E = addInputIdentifier(H))
    return E;

  addOptionsIdentifier(H);

  CachedCommandAdaptor::Identifier Id;
  toHex(H.final(), true, Id);
  return Id;
}

llvm::Error
CachedCommandAdaptor::writeUniqueExecuteOutput(StringRef OutputFilename,
                                               StringRef CachedBuffer) {
  std::error_code EC;
  raw_fd_ostream Out(OutputFilename, EC);
  if (EC) {
    Error E = createStringError(EC, Twine("Failed to open ") + OutputFilename +
                                        " : " + EC.message() + "\n");
    return E;
  }

  Out.write(CachedBuffer.data(), CachedBuffer.size());
  Out.close();
  if (Out.has_error()) {
    Error E = createStringError(EC, Twine("Failed to write ") + OutputFilename +
                                        " : " + EC.message() + "\n");
    return E;
  }

  return Error::success();
}

Expected<std::unique_ptr<MemoryBuffer>>
CachedCommandAdaptor::readUniqueExecuteOutput(StringRef OutputFilename) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
      MemoryBuffer::getFile(OutputFilename);
  if (!MBOrErr) {
    std::error_code EC = MBOrErr.getError();
    return createStringError(EC, Twine("Failed to open ") + OutputFilename +
                                     " : " + EC.message() + "\n");
  }

  return std::move(*MBOrErr);
}
} // namespace COMGR
