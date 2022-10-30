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

#include "Floating.h"
#include "Function.h"
#include "Opcode.h"
#include "PrimType.h"
#include "Program.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Format.h"

using namespace clang;
using namespace clang::interp;

template <typename T> inline T ReadArg(Program &P, CodePtr &OpPC) {
  if constexpr (std::is_pointer_v<T>) {
    uint32_t ID = OpPC.read<uint32_t>();
    return reinterpret_cast<T>(P.getNativePointer(ID));
  } else {
    return OpPC.read<T>();
  }
}

LLVM_DUMP_METHOD void Function::dump() const { dump(llvm::errs()); }

LLVM_DUMP_METHOD void Function::dump(llvm::raw_ostream &OS) const {
  if (F) {
    if (auto *Cons = dyn_cast<CXXConstructorDecl>(F)) {
      DeclarationName Name = Cons->getParent()->getDeclName();
      OS << Name << "::" << Name;
    } else {
      OS << F->getDeclName();
    }
    OS << " " << (const void*)this << ":\n";
  } else {
    OS << "<<expr>>\n";
  }

  OS << "frame size: " << getFrameSize() << "\n";
  OS << "arg size:   " << getArgSize() << "\n";
  OS << "rvo:        " << hasRVO() << "\n";
  OS << "this arg:   " << hasThisPointer() << "\n";

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
  OS << ":: Program\n";
  OS << "Global Variables: " << Globals.size() << "\n";
  OS << "Functions: " << Funcs.size() << "\n";
  OS << "\n";
  for (auto &Func : Funcs) {
    Func.second->dump();
  }
  for (auto &Anon : AnonFuncs) {
    Anon->dump();
  }
}
