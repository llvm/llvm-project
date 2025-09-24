//===- OffloadArch.cpp - list available GPUs ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"

using namespace llvm;

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

// Mark all our options with this category.
static cl::OptionCategory OffloadArchCategory("offload-arch options");

enum VendorName {
  all,
  amdgpu,
  nvptx,
  intel,
};

static cl::opt<VendorName>
    Only("only", cl::desc("Restrict to vendor:"), cl::cat(OffloadArchCategory),
         cl::init(all),
         cl::values(clEnumVal(all, "Print all GPUs (default)"),
                    clEnumVal(amdgpu, "Only print AMD GPUs"),
                    clEnumVal(nvptx, "Only print NVIDIA GPUs"),
                    clEnumVal(intel, "Only print Intel GPUs")));

cl::opt<bool> Verbose("verbose", cl::desc("Enable verbose output"),
                      cl::init(false), cl::cat(OffloadArchCategory));

static void PrintVersion(raw_ostream &OS) {
  OS << clang::getClangToolFullVersion("offload-arch") << '\n';
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

struct vendor_entry_t {
  VendorName name;
  function_ref<int()> printFunc;
};
std::array<vendor_entry_t, 3> VendorTable{{{VendorName::amdgpu, printAMD},
                                           {VendorName::nvptx, printNVIDIA},
                                           {VendorName::intel, printIntel}}};

int main(int argc, char *argv[]) {
  cl::HideUnrelatedOptions(OffloadArchCategory);

  cl::SetVersionPrinter(PrintVersion);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to detect the presence of offloading devices on the system. \n\n"
      "The tool will output each detected GPU architecture separated by a\n"
      "newline character. If multiple GPUs of the same architecture are found\n"
      "a string will be printed for each\n");

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }

  // Support legacy binaries
  if (sys::path::stem(argv[0]).starts_with("amdgpu-arch"))
    Only = VendorName::amdgpu;
  if (sys::path::stem(argv[0]).starts_with("nvptx-arch"))
    Only = VendorName::nvptx;

  llvm::SmallVector<int> results(VendorTable.size());
  llvm::transform(VendorTable, results.begin(), [&](const auto &entry) {
    if (Only == VendorName::all || Only == entry.name)
      return entry.printFunc();
    return 0;
  });

  return llvm::all_of(results, [](int r) { return r == 1; });
}
