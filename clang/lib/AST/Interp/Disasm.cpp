//===--- Disasm.cpp - Disassembler for bytecode functions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dump method for Function which disassembles the bytecode.
//
//===----------------------------------------------------------------------===//

#include "Function.h"
#include "Opcode.h"
#include "PrimType.h"
#include "Program.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/Support/Compiler.h"

using namespace clang;
using namespace clang::interp;

LLVM_DUMP_METHOD void Function::dump() const { dump(llvm::errs()); }

LLVM_DUMP_METHOD void Function::dump(llvm::raw_ostream &OS) const {
  if (F) {
    if (auto *Cons = dyn_cast<CXXConstructorDecl>(F)) {
      const std::string &Name = Cons->getParent()->getNameAsString();
      OS << Name << "::" << Name << ":\n";
    } else {
      OS << F->getNameAsString() << ":\n";
    }
  } else {
    OS << "<<expr>>\n";
  }

  OS << "frame size: " << getFrameSize() << "\n";
  OS << "arg size:   " << getArgSize() << "\n";
  OS << "rvo:        " << hasRVO() << "\n";

  auto PrintName = [&OS](const char *Name) {
    OS << Name;
    for (long I = 0, N = strlen(Name); I < 30 - N; ++I) {
      OS << ' ';
    }
  };

  for (CodePtr Start = getCodeBegin(), PC = Start; PC != getCodeEnd();) {
    size_t Addr = PC - Start;
    auto Op = PC.read<Opcode>();
    OS << llvm::format("%8d", Addr) << " ";
    switch (Op) {
#define GET_DISASM
#include "Opcodes.inc"
#undef GET_DISASM
    }
  }
}

LLVM_DUMP_METHOD void Program::dump() const { dump(llvm::errs()); }

LLVM_DUMP_METHOD void Program::dump(llvm::raw_ostream &OS) const {
  for (auto &Func : Funcs) {
    Func.second->dump();
  }
  for (auto &Anon : AnonFuncs) {
    Anon->dump();
  }
}
