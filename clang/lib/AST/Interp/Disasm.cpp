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
#include "IntegralAP.h"
#include "Opcode.h"
#include "PrimType.h"
#include "Program.h"
#include "clang/AST/ASTDumperUtils.h"
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

template <> inline Floating ReadArg<Floating>(Program &P, CodePtr &OpPC) {
  Floating F = Floating::deserialize(*OpPC);
  OpPC += align(F.bytesToSerialize());
  return F;
}

template <>
inline IntegralAP<false> ReadArg<IntegralAP<false>>(Program &P, CodePtr &OpPC) {
  IntegralAP<false> I = IntegralAP<false>::deserialize(*OpPC);
  OpPC += align(I.bytesToSerialize());
  return I;
}

template <>
inline IntegralAP<true> ReadArg<IntegralAP<true>>(Program &P, CodePtr &OpPC) {
  IntegralAP<true> I = IntegralAP<true>::deserialize(*OpPC);
  OpPC += align(I.bytesToSerialize());
  return I;
}

LLVM_DUMP_METHOD void Function::dump() const { dump(llvm::errs()); }

LLVM_DUMP_METHOD void Function::dump(llvm::raw_ostream &OS) const {
  {
    ColorScope SC(OS, true, {llvm::raw_ostream::BRIGHT_GREEN, true});
    OS << getName() << " " << (const void *)this << "\n";
  }
  OS << "frame size: " << getFrameSize() << "\n";
  OS << "arg size:   " << getArgSize() << "\n";
  OS << "rvo:        " << hasRVO() << "\n";
  OS << "this arg:   " << hasThisPointer() << "\n";

  auto PrintName = [&OS](const char *Name) {
    OS << Name;
    long N = 30 - strlen(Name);
    if (N > 0)
      OS.indent(N);
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
  {
    ColorScope SC(OS, true, {llvm::raw_ostream::BRIGHT_RED, true});
    OS << "\n:: Program\n";
  }

  {
    ColorScope SC(OS, true, {llvm::raw_ostream::WHITE, true});
    OS << "Total memory : " << Allocator.getTotalMemory() << " bytes\n";
    OS << "Global Variables: " << Globals.size() << "\n";
  }
  unsigned GI = 0;
  for (const Global *G : Globals) {
    const Descriptor *Desc = G->block()->getDescriptor();
    OS << GI << ": " << (void *)G->block() << " ";
    {
      Pointer GP = getPtrGlobal(GI);
      ColorScope SC(OS, true,
                    GP.isInitialized()
                        ? TerminalColor{llvm::raw_ostream::GREEN, false}
                        : TerminalColor{llvm::raw_ostream::RED, false});
      OS << (GP.isInitialized() ? "initialized " : "uninitialized ");
    }
    Desc->dump(OS);
    OS << "\n";
    ++GI;
  }

  {
    ColorScope SC(OS, true, {llvm::raw_ostream::WHITE, true});
    OS << "Functions: " << Funcs.size() << "\n";
  }
  for (const auto &Func : Funcs) {
    Func.second->dump();
  }
  for (const auto &Anon : AnonFuncs) {
    Anon->dump();
  }
}

LLVM_DUMP_METHOD void Descriptor::dump() const {
  dump(llvm::errs());
  llvm::errs() << '\n';
}

LLVM_DUMP_METHOD void Descriptor::dump(llvm::raw_ostream &OS) const {
  // Source
  {
    ColorScope SC(OS, true, {llvm::raw_ostream::BLUE, true});
    if (const auto *ND = dyn_cast_if_present<NamedDecl>(asDecl()))
      OS << ND->getName();
    else if (asExpr())
      OS << "expr (TODO)";
  }

  // Print a few interesting bits about the descriptor.
  if (isPrimitiveArray())
    OS << " primitive-array";
  else if (isCompositeArray())
    OS << " composite-array";
  else if (isRecord())
    OS << " record";
  else if (isPrimitive())
    OS << " primitive";

  if (isZeroSizeArray())
    OS << " zero-size-arrary";
  else if (isUnknownSizeArray())
    OS << " unknown-size-array";

  if (isDummy())
    OS << " dummy";
}
