//===- lib/Driver/Driver.cpp - Linker Driver Emulator -----------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/Core/ArchiveLibraryFile.h"
#include "lld/Core/File.h"
#include "lld/Core/Instrumentation.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Parallel.h"
#include "lld/Core/PassManager.h"
#include "lld/Core/Reader.h"
#include "lld/Core/Resolver.h"
#include "lld/Core/Writer.h"
#include "lld/Driver/Driver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <mutex>

namespace lld {

FileVector makeErrorFile(StringRef path, std::error_code ec) {
  std::vector<std::unique_ptr<File>> result;
  result.push_back(llvm::make_unique<ErrorFile>(path, ec));
  return result;
}

FileVector parseMemberFiles(std::unique_ptr<File> file) {
  std::vector<std::unique_ptr<File>> members;
  if (auto *archive = dyn_cast<ArchiveLibraryFile>(file.get())) {
    if (std::error_code ec = archive->parseAllMembers(members))
      return makeErrorFile(file->path(), ec);
  } else {
    members.push_back(std::move(file));
  }
  return members;
}

FileVector loadFile(LinkingContext &ctx, StringRef path, bool wholeArchive) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> mb
      = MemoryBuffer::getFileOrSTDIN(path);
  if (std::error_code ec = mb.getError())
    return makeErrorFile(path, ec);
  ErrorOr<std::unique_ptr<File>> fileOrErr =
      ctx.registry().loadFile(std::move(mb.get()));
  if (std::error_code ec = fileOrErr.getError())
    return makeErrorFile(path, ec);
  std::unique_ptr<File> &file = fileOrErr.get();
  if (wholeArchive)
    return parseMemberFiles(std::move(file));
  std::vector<std::unique_ptr<File>> files;
  files.push_back(std::move(file));
  return files;
}

/// This is where the link is actually performed.
bool Driver::link(LinkingContext &ctx, raw_ostream &diagnostics) {
  // Honor -mllvm
  if (!ctx.llvmOptions().empty()) {
    unsigned numArgs = ctx.llvmOptions().size();
    auto **args = new const char *[numArgs + 2];
    args[0] = "lld (LLVM option parsing)";
    for (unsigned i = 0; i != numArgs; ++i)
      args[i + 1] = ctx.llvmOptions()[i];
    args[numArgs + 1] = nullptr;
    llvm::cl::ParseCommandLineOptions(numArgs + 1, args);
  }
  if (ctx.getNodes().empty())
    return false;

  for (std::unique_ptr<Node> &ie : ctx.getNodes())
    if (FileNode *node = dyn_cast<FileNode>(ie.get()))
      ctx.getTaskGroup().spawn([node] { node->getFile()->parse(); });

  std::vector<std::unique_ptr<File>> internalFiles;
  ctx.createInternalFiles(internalFiles);
  for (auto i = internalFiles.rbegin(), e = internalFiles.rend(); i != e; ++i) {
    auto &members = ctx.getNodes();
    members.insert(members.begin(), llvm::make_unique<FileNode>(std::move(*i)));
  }

  // Give target a chance to add files.
  std::vector<std::unique_ptr<File>> implicitFiles;
  ctx.createImplicitFiles(implicitFiles);
  for (auto i = implicitFiles.rbegin(), e = implicitFiles.rend(); i != e; ++i) {
    auto &members = ctx.getNodes();
    members.insert(members.begin(), llvm::make_unique<FileNode>(std::move(*i)));
  }

  // Give target a chance to postprocess input files.
  // Mach-O uses this chance to move all object files before library files.
  // ELF adds specific undefined symbols resolver.
  ctx.finalizeInputFiles();

  // Do core linking.
  ScopedTask resolveTask(getDefaultDomain(), "Resolve");
  Resolver resolver(ctx);
  if (!resolver.resolve()) {
    ctx.getTaskGroup().sync();
    return false;
  }
  std::unique_ptr<SimpleFile> merged = resolver.resultFile();
  resolveTask.end();

  // Run passes on linked atoms.
  ScopedTask passTask(getDefaultDomain(), "Passes");
  PassManager pm;
  ctx.addPasses(pm);
  if (std::error_code ec = pm.runOnFile(*merged)) {
    diagnostics << "Failed to write file '" << ctx.outputPath()
                << "': " << ec.message() << "\n";
    return false;
  }

  passTask.end();

  // Give linked atoms to Writer to generate output file.
  ScopedTask writeTask(getDefaultDomain(), "Write");
  if (std::error_code ec = ctx.writeFile(*merged)) {
    diagnostics << "Failed to write file '" << ctx.outputPath()
                << "': " << ec.message() << "\n";
    return false;
  }

  return true;
}

} // namespace lld
