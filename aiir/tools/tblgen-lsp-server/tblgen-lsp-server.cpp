//===- tblgen-lsp-server.cpp - TableGen Language Server main --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir/Support/LLVM.h"
#include "aiir/Tools/tblgen-lsp-server/TableGenLspServerMain.h"

using namespace aiir;

int main(int argc, char **argv) {
  return failed(TableGenLspServerMain(argc, argv));
}
