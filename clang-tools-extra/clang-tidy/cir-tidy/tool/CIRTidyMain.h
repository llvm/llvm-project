//===--- tools/extra/clang-tidy/cir/CIRTidyMain.h - cir tidy tool ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
///  \file This file declares the main function for the cir-tidy tool.
///
///  This tool uses the Clang Tooling infrastructure, see
///    http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
///  for details on setting it up with LLVM source tree.
///
//===----------------------------------------------------------------------===//

namespace cir {
namespace tidy {

int CIRTidyMain(int argc, const char **argv);

} // namespace tidy
} // namespace cir
