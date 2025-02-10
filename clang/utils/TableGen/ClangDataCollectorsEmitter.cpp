//===-- ClangDataCollectorsEmitter.cpp - Generate Clang data collector ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emit Clang data collector tables.
//
//===----------------------------------------------------------------------===//

#include "TableGenBackends.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

void clang::EmitClangDataCollectors(const RecordKeeper &RK, raw_ostream &OS) {
  const auto &Defs = RK.getClasses();
  for (const auto &Entry : Defs) {
    Record &R = *Entry.second;
    OS << "DEF_ADD_DATA(" << R.getName() << ", {\n";
    auto Code = R.getValue("Code")->getValue();
    OS << Code->getAsUnquotedString() << "}\n)";
    OS << "\n";
  }
  OS << "#undef DEF_ADD_DATA\n";
}
