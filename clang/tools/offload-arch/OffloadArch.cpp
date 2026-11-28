//===- OffloadArch.cpp - list available GPUs ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lib/OffloadArchOptions.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"
#include <optional>

using namespace llvm;

namespace {
using namespace llvm::opt;

enum ID {
  OPT_INVALID = 0,
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define OPTTABLE_CODE
#include "Opts.inc"

class OffloadArchOptTable : public OptTable {
public:
  OffloadArchOptTable() : OptTable(optionTables()) {}
};
} // namespace

enum VendorName {
  all,
  amdgpu,
  nvptx,
  intel,
};

[[noreturn]] static void reportError(const Twine &Msg) {
  WithColor::error(errs(), "offload-arch") << Msg << '\n';
  exit(1);
}

int printGPUsByKFD();
int printGPUsByHIP();
int printGPUsByCUDA();
int printGPUsByLevelZero();

static int printAMD() {
#ifndef _WIN32
  if (!printGPUsByKFD())
    return 0;
#endif

  return printGPUsByHIP();
}

static int printNVIDIA() { return printGPUsByCUDA(); }
static int printIntel() { return printGPUsByLevelZero(); }

const std::array<std::pair<VendorName, function_ref<int()>>, 3> VendorTable{
    {{VendorName::amdgpu, printAMD},
     {VendorName::nvptx, printNVIDIA},
     {VendorName::intel, printIntel}}};

int main(int argc, char *argv[]) {
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  OffloadArchOptTable Tbl;
  opt::InputArgList Args = Tbl.parseArgs(
      argc, argv, OPT_UNKNOWN, Saver, [](StringRef Msg) { reportError(Msg); });

  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(
        outs(), "offload-arch [options]",
        "A tool to detect the presence of offloading devices on the system");
    return 0;
  }
  if (Args.hasArg(OPT_version)) {
    outs() << clang::getClangToolFullVersion("offload-arch") << '\n';
    return 0;
  }
  if (const opt::Arg *A = Args.getLastArg(OPT_INPUT))
    reportError("unexpected argument '" + Twine(A->getValue()) + "'");

  Verbose = Args.hasArg(OPT_verbose);

  StringRef HipArg = Args.getLastArgValue(OPT_hip_api_version_EQ, "auto");
  std::optional<HipApiVersion> Hip =
      StringSwitch<std::optional<HipApiVersion>>(HipArg)
          .Case("auto", HipApiVersion::Auto)
          .Case("r0600", HipApiVersion::R0600)
          .Case("r0000", HipApiVersion::R0000)
          .Case("unversioned", HipApiVersion::Unversioned)
          .Default(std::nullopt);
  if (!Hip)
    reportError("invalid value for --hip-api-version: '" + HipArg + "'");
  HipApi = *Hip;

  StringRef OnlyArg = Args.getLastArgValue(OPT_only_EQ, "all");
  std::optional<VendorName> Only =
      StringSwitch<std::optional<VendorName>>(OnlyArg)
          .Case("all", all)
          .Case("amdgpu", amdgpu)
          .Case("nvptx", nvptx)
          .Case("intel", intel)
          .Default(std::nullopt);
  if (!Only)
    reportError("invalid value for --only: '" + OnlyArg + "'");

  // Support legacy binaries.
  if (sys::path::stem(argv[0]).starts_with("amdgpu-arch"))
    Only = VendorName::amdgpu;
  if (sys::path::stem(argv[0]).starts_with("nvptx-arch"))
    Only = VendorName::nvptx;

  int Result = 1;
  for (auto [Name, Func] : VendorTable) {
    if (*Only == VendorName::all || *Only == Name)
      Result &= Func();
  }

  return Result;
}
