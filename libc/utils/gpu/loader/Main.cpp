//===-- Main entry into the loader interface ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file opens a device image passed on the command line and passes it to
// one of the loader implementations for launch.
//
//===----------------------------------------------------------------------===//

#include "Loader.h"

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/WithColor.h"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace llvm;

static cl::OptionCategory loader_category("loader options");

static cl::opt<bool> help("h", cl::desc("Alias for -help"), cl::Hidden,
                          cl::cat(loader_category));

static cl::opt<unsigned>
    threads_x("threads-x", cl::desc("Number of threads in the 'x' dimension"),
              cl::init(1), cl::cat(loader_category));
static cl::opt<unsigned>
    threads_y("threads-y", cl::desc("Number of threads in the 'y' dimension"),
              cl::init(1), cl::cat(loader_category));
static cl::opt<unsigned>
    threads_z("threads-z", cl::desc("Number of threads in the 'z' dimension"),
              cl::init(1), cl::cat(loader_category));
static cl::alias threads("threads", cl::aliasopt(threads_x),
                         cl::desc("Alias for --threads-x"),
                         cl::cat(loader_category));

static cl::opt<unsigned>
    blocks_x("blocks-x", cl::desc("Number of blocks in the 'x' dimension"),
             cl::init(1), cl::cat(loader_category));
static cl::opt<unsigned>
    blocks_y("blocks-y", cl::desc("Number of blocks in the 'y' dimension"),
             cl::init(1), cl::cat(loader_category));
static cl::opt<unsigned>
    blocks_z("blocks-z", cl::desc("Number of blocks in the 'z' dimension"),
             cl::init(1), cl::cat(loader_category));
static cl::alias blocks("blocks", cl::aliasopt(blocks_x),
                        cl::desc("Alias for --blocks-x"),
                        cl::cat(loader_category));

static cl::opt<bool>
    print_resource_usage("print-resource-usage",
                         cl::desc("Output resource usage of launched kernels"),
                         cl::init(false), cl::cat(loader_category));

static cl::opt<std::string> file(cl::Positional, cl::Required,
                                 cl::desc("<gpu executable>"),
                                 cl::cat(loader_category));
static cl::list<std::string> args(cl::ConsumeAfter,
                                  cl::desc("<program arguments>..."),
                                  cl::cat(loader_category));

[[noreturn]] void report_error(Error E) {
  outs().flush();
  logAllUnhandledErrors(std::move(E), WithColor::error(errs(), "loader"));
  exit(EXIT_FAILURE);
}

int main(int argc, const char **argv, const char **envp) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  cl::HideUnrelatedOptions(loader_category);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A utility used to launch unit tests built for a GPU target. This is\n"
      "intended to provide an intrface simular to cross-compiling emulators\n");

  if (help) {
    cl::PrintHelpMessage();
    return EXIT_SUCCESS;
  }

  ErrorOr<std::unique_ptr<MemoryBuffer>> image_or_err =
      MemoryBuffer::getFileOrSTDIN(file);
  if (std::error_code ec = image_or_err.getError())
    report_error(errorCodeToError(ec));
  MemoryBufferRef image = **image_or_err;

  SmallVector<const char *> new_argv = {file.c_str()};
  llvm::transform(args, std::back_inserter(new_argv),
                  [](const std::string &arg) { return arg.c_str(); });

  // Drop the loader from the program arguments.
  LaunchParameters params{threads_x, threads_y, threads_z,
                          blocks_x,  blocks_y,  blocks_z};
  int ret = load(new_argv.size(), new_argv.data(), envp,
                 const_cast<char *>(image.getBufferStart()),
                 image.getBufferSize(), params, print_resource_usage);

  return ret;
}
