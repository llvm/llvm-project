//===- Utils.cpp - Shared utilities for SSAF tools -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Tool/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <memory>
#include <string>

using namespace clang;
using namespace ssaf;

namespace path = llvm::sys::path;

namespace {

llvm::StringRef ToolName;
llvm::StringRef ToolVersion;

void printVersion(llvm::raw_ostream &OS) {
  OS << ToolName << " " << ToolVersion << "\n";
}

// Returns the SerializationFormat registered for \p Extension, or nullptr if
// none is registered. Results are cached for the lifetime of the process.
// FIXME: This will be revisited after we add support for registering formats
// with extensions.
SerializationFormat *getFormatForExtension(llvm::StringRef Extension) {
  // This cache is not thread-safe. SSAF tools are single-threaded CLIs, so
  // concurrent calls to this function are not expected.

  // Realistically, we don't expect to encounter more than four registered
  // formats.
  static llvm::SmallVector<
      std::pair<std::string, std::unique_ptr<SerializationFormat>>, 4>
      ExtensionFormatList;

  // Most recently used format is most likely to be reused again.
  auto ReversedList = llvm::reverse(ExtensionFormatList);
  auto It = llvm::find_if(ReversedList, [&](const auto &Entry) {
    return Entry.first == Extension;
  });
  if (It != ReversedList.end()) {
    return It->second.get();
  }

  if (!isFormatRegistered(Extension)) {
    return nullptr;
  }

  auto Format = makeFormat(Extension);
  SerializationFormat *Result = Format.get();
  assert(Result &&
         "makeFormat must return non-null for a registered extension");

  ExtensionFormatList.emplace_back(Extension, std::move(Format));

  return Result;
}

} // namespace

llvm::StringRef ssaf::getToolName() { return ToolName; }

[[noreturn]] void ssaf::fail(const char *Msg) {
  llvm::WithColor::error(llvm::errs(), ToolName) << Msg << "\n";
  llvm::sys::Process::Exit(1);
}

[[noreturn]] void ssaf::fail(llvm::Error Err) {
  std::string Message = llvm::toString(std::move(Err));
  ssaf::fail(Message.data());
}

void ssaf::loadPlugins(llvm::ArrayRef<std::string> Paths) {
  for (const std::string &PluginPath : Paths) {
    std::string ErrMsg;
    if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(PluginPath.c_str(),
                                                          &ErrMsg)) {
      fail(ErrorMessages::FailedToLoadPlugin, PluginPath, ErrMsg);
    }
  }
}

void ssaf::initTool(int argc, const char **argv, llvm::StringRef Version,
                    llvm::cl::OptionCategory &Category,
                    llvm::StringRef ToolHeading) {
  // path::stem strips the .exe extension on Windows so ToolName is consistent.
  ToolName = path::stem(argv[0]);

  // Set tool version for the version printer.
  ToolVersion = Version;

  // Hide options unrelated to the tool from --help output.
  llvm::cl::HideUnrelatedOptions(Category);

  // Register a custom version printer for the --version flag.
  llvm::cl::SetVersionPrinter(printVersion);

  // Parse command-line arguments and exit with an error if they are invalid.
  std::string Overview = (ToolHeading + "\n").str();
  llvm::cl::ParseCommandLineOptions(argc, argv, Overview);
}

SummaryFile SummaryFile::fromPath(llvm::StringRef Path) {
  llvm::StringRef Extension = path::extension(Path);
  if (Extension.empty()) {
    fail(ErrorMessages::CannotValidateSummary, Path,
         ErrorMessages::ExtensionNotSupplied);
  }

  Extension = Extension.drop_front();
  SerializationFormat *Format = getFormatForExtension(Extension);
  if (!Format) {
    std::string BadExtension =
        llvm::formatv(ErrorMessages::NoFormatForExtension, Extension);
    fail(ErrorMessages::CannotValidateSummary, Path, BadExtension);
  }

  return {Path.str(), Format};
}
