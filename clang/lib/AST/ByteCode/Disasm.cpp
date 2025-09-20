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

#include "Boolean.h"
#include "Context.h"
#include "EvaluationResult.h"
#include "FixedPoint.h"
#include "Floating.h"
#include "Function.h"
#include "FunctionPointer.h"
#include "Integral.h"
#include "IntegralAP.h"
#include "InterpFrame.h"
#include "MemberPointer.h"
#include "Opcode.h"
#include "PrimType.h"
#include "Program.h"
#include "clang/AST/ASTDumperUtils.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/Support/Compiler.h"

using namespace clang;
using namespace clang::interp;

template <typename T>
inline static std::string printArg(Program &P, CodePtr &OpPC) {
  if constexpr (std::is_pointer_v<T>) {
    uint32_t ID = OpPC.read<uint32_t>();
    std::string Result;
    llvm::raw_string_ostream SS(Result);
    SS << reinterpret_cast<T>(P.getNativePointer(ID));
    return Result;
  } else {
    std::string Result;
    llvm::raw_string_ostream SS(Result);
    auto Arg = OpPC.read<T>();
    // Make sure we print the integral value of chars.
    if constexpr (std::is_integral_v<T>) {
      if constexpr (sizeof(T) == 1) {
        if constexpr (std::is_signed_v<T>)
          SS << static_cast<int32_t>(Arg);
        else
          SS << static_cast<uint32_t>(Arg);
      } else {
        SS << Arg;
      }
    } else {
      SS << Arg;
    }

    return Result;
  }
}

template <> inline std::string printArg<Floating>(Program &P, CodePtr &OpPC) {
  auto Sem = Floating::deserializeSemantics(*OpPC);

  unsigned BitWidth = llvm::APFloatBase::semanticsSizeInBits(
      llvm::APFloatBase::EnumToSemantics(Sem));
  auto Memory =
      std::make_unique<uint64_t[]>(llvm::APInt::getNumWords(BitWidth));
  Floating Result(Memory.get(), Sem);
  Floating::deserialize(*OpPC, &Result);

  OpPC += align(Result.bytesToSerialize());

  std::string S;
  llvm::raw_string_ostream SS(S);
  SS << std::move(Result);
  return S;
}

template <>
inline std::string printArg<IntegralAP<false>>(Program &P, CodePtr &OpPC) {
  using T = IntegralAP<false>;
  uint32_t BitWidth = T::deserializeSize(*OpPC);
  auto Memory =
      std::make_unique<uint64_t[]>(llvm::APInt::getNumWords(BitWidth));

  T Result(Memory.get(), BitWidth);
  T::deserialize(*OpPC, &Result);

  OpPC += align(Result.bytesToSerialize());

  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << std::move(Result);
  return Str;
}

template <>
inline std::string printArg<IntegralAP<true>>(Program &P, CodePtr &OpPC) {
  using T = IntegralAP<true>;
  uint32_t BitWidth = T::deserializeSize(*OpPC);
  auto Memory =
      std::make_unique<uint64_t[]>(llvm::APInt::getNumWords(BitWidth));

  T Result(Memory.get(), BitWidth);
  T::deserialize(*OpPC, &Result);

  OpPC += align(Result.bytesToSerialize());

  std::string Str;
  llvm::raw_string_ostream SS(Str);
  SS << std::move(Result);
  return Str;
}

template <> inline std::string printArg<FixedPoint>(Program &P, CodePtr &OpPC) {
  auto F = FixedPoint::deserialize(*OpPC);
  OpPC += align(F.bytesToSerialize());

  std::string Result;
  llvm::raw_string_ostream SS(Result);
  SS << std::move(F);
  return Result;
}

static bool isJumpOpcode(Opcode Op) {
  return Op == OP_Jmp || Op == OP_Jf || Op == OP_Jt;
}

static size_t getNumDisplayWidth(size_t N) {
  unsigned L = 1u, M = 10u;
  while (M <= N && ++L != std::numeric_limits<size_t>::digits10 + 1)
    M *= 10u;

  return L;
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

  struct OpText {
    size_t Addr;
    std::string Op;
    bool IsJump;
    llvm::SmallVector<std::string> Args;
  };

  auto PrintName = [](const char *Name) -> std::string {
    return std::string(Name);
  };

  llvm::SmallVector<OpText> Code;
  size_t LongestAddr = 0;
  size_t LongestOp = 0;

  for (CodePtr Start = getCodeBegin(), PC = Start; PC != getCodeEnd();) {
    size_t Addr = PC - Start;
    OpText Text;
    auto Op = PC.read<Opcode>();
    Text.Addr = Addr;
    Text.IsJump = isJumpOpcode(Op);
    switch (Op) {
#define GET_DISASM
#include "Opcodes.inc"
#undef GET_DISASM
    }
    Code.push_back(Text);
    LongestOp = std::max(Text.Op.size(), LongestOp);
    LongestAddr = std::max(getNumDisplayWidth(Addr), LongestAddr);
  }

  // Record jumps and their targets.
  struct JmpData {
    size_t From;
    size_t To;
  };
  llvm::SmallVector<JmpData> Jumps;
  for (auto &Text : Code) {
    if (Text.IsJump)
      Jumps.push_back({Text.Addr, Text.Addr + std::stoi(Text.Args[0]) +
                                      align(sizeof(Opcode)) +
                                      align(sizeof(int32_t))});
  }

  llvm::SmallVector<std::string> Text;
  Text.reserve(Code.size());
  size_t LongestLine = 0;
  // Print code to a string, one at a time.
  for (auto C : Code) {
    std::string Line;
    llvm::raw_string_ostream LS(Line);
    LS << C.Addr;
    LS.indent(LongestAddr - getNumDisplayWidth(C.Addr) + 4);
    LS << C.Op;
    LS.indent(LongestOp - C.Op.size() + 4);
    for (auto &Arg : C.Args) {
      LS << Arg << ' ';
    }
    Text.push_back(Line);
    LongestLine = std::max(Line.size(), LongestLine);
  }

  assert(Code.size() == Text.size());

  auto spaces = [](unsigned N) -> std::string {
    std::string S;
    for (unsigned I = 0; I != N; ++I)
      S += ' ';
    return S;
  };

  // Now, draw the jump lines.
  for (auto &J : Jumps) {
    if (J.To > J.From) {
      bool FoundStart = false;
      for (size_t LineIndex = 0; LineIndex != Text.size(); ++LineIndex) {
        Text[LineIndex] += spaces(LongestLine - Text[LineIndex].size());

        if (Code[LineIndex].Addr == J.From) {
          Text[LineIndex] += "  --+";
          FoundStart = true;
        } else if (Code[LineIndex].Addr == J.To) {
          Text[LineIndex] += "  <-+";
          break;
        } else if (FoundStart) {
          Text[LineIndex] += "    |";
        }
      }
      LongestLine += 5;
    } else {
      bool FoundStart = false;
      for (ssize_t LineIndex = Text.size() - 1; LineIndex >= 0; --LineIndex) {
        Text[LineIndex] += spaces(LongestLine - Text[LineIndex].size());
        if (Code[LineIndex].Addr == J.From) {
          Text[LineIndex] += "  --+";
          FoundStart = true;
        } else if (Code[LineIndex].Addr == J.To) {
          Text[LineIndex] += "  <-+";
          break;
        } else if (FoundStart) {
          Text[LineIndex] += "    |";
        }
      }
      LongestLine += 5;
    }
  }

  for (auto &Line : Text)
    OS << Line << '\n';
}

LLVM_DUMP_METHOD void Program::dump() const { dump(llvm::errs()); }

static const char *primTypeToString(PrimType T) {
  switch (T) {
  case PT_Sint8:
    return "Sint8";
  case PT_Uint8:
    return "Uint8";
  case PT_Sint16:
    return "Sint16";
  case PT_Uint16:
    return "Uint16";
  case PT_Sint32:
    return "Sint32";
  case PT_Uint32:
    return "Uint32";
  case PT_Sint64:
    return "Sint64";
  case PT_Uint64:
    return "Uint64";
  case PT_IntAP:
    return "IntAP";
  case PT_IntAPS:
    return "IntAPS";
  case PT_Bool:
    return "Bool";
  case PT_Float:
    return "Float";
  case PT_Ptr:
    return "Ptr";
  case PT_MemberPtr:
    return "MemberPtr";
  case PT_FixedPoint:
    return "FixedPoint";
  }
  llvm_unreachable("Unhandled PrimType");
}

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
    Pointer GP = getPtrGlobal(GI);

    OS << GI << ": " << (const void *)G->block() << " ";
    {
      ColorScope SC(OS, true,
                    GP.isInitialized()
                        ? TerminalColor{llvm::raw_ostream::GREEN, false}
                        : TerminalColor{llvm::raw_ostream::RED, false});
      OS << (GP.isInitialized() ? "initialized " : "uninitialized ");
    }
    Desc->dump(OS);

    if (GP.isInitialized() && Desc->IsTemporary) {
      if (const auto *MTE =
              dyn_cast_if_present<MaterializeTemporaryExpr>(Desc->asExpr());
          MTE && MTE->getLifetimeExtendedTemporaryDecl()) {
        if (const APValue *V =
                MTE->getLifetimeExtendedTemporaryDecl()->getValue()) {
          OS << " (global temporary value: ";
          {
            ColorScope SC(OS, true, {llvm::raw_ostream::BRIGHT_MAGENTA, true});
            std::string VStr;
            llvm::raw_string_ostream SS(VStr);
            V->dump(SS, Ctx.getASTContext());

            for (unsigned I = 0; I != VStr.size(); ++I) {
              if (VStr[I] == '\n')
                VStr[I] = ' ';
            }
            VStr.pop_back(); // Remove the newline (or now space) at the end.
            OS << VStr;
          }
          OS << ')';
        }
      }
    }

    OS << "\n";
    if (GP.isInitialized() && Desc->isPrimitive() && !G->block()->isDummy()) {
      OS << "   ";
      {
        ColorScope SC(OS, true, {llvm::raw_ostream::BRIGHT_CYAN, false});
        OS << primTypeToString(Desc->getPrimType()) << " ";
      }
      TYPE_SWITCH(Desc->getPrimType(), { GP.deref<T>().print(OS); });
      OS << "\n";
    }
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
      ND->printQualifiedName(OS);
    else if (asExpr())
      OS << "Expr " << (const void *)asExpr();
  }

  // Print a few interesting bits about the descriptor.
  if (isPrimitiveArray())
    OS << " primitive-array";
  else if (isCompositeArray())
    OS << " composite-array";
  else if (isUnion())
    OS << " union";
  else if (isRecord())
    OS << " record";
  else if (isPrimitive())
    OS << " primitive " << primTypeToString(getPrimType());

  if (isZeroSizeArray())
    OS << " zero-size-array";
  else if (isUnknownSizeArray())
    OS << " unknown-size-array";

  if (IsConstexprUnknown)
    OS << " constexpr-unknown";
}

/// Dump descriptor, including all valid offsets.
LLVM_DUMP_METHOD void Descriptor::dumpFull(unsigned Offset,
                                           unsigned Indent) const {
  unsigned Spaces = Indent * 2;
  llvm::raw_ostream &OS = llvm::errs();
  OS.indent(Spaces);
  dump(OS);
  OS << '\n';
  OS.indent(Spaces) << "Metadata: " << getMetadataSize() << " bytes\n";
  OS.indent(Spaces) << "Size: " << getSize() << " bytes\n";
  OS.indent(Spaces) << "AllocSize: " << getAllocSize() << " bytes\n";
  Offset += getMetadataSize();
  if (isCompositeArray()) {
    OS.indent(Spaces) << "Elements: " << getNumElems() << '\n';
    unsigned FO = Offset;
    for (unsigned I = 0; I != getNumElems(); ++I) {
      FO += sizeof(InlineDescriptor);
      assert(ElemDesc->getMetadataSize() == 0);
      OS.indent(Spaces) << "Element " << I << " offset: " << FO << '\n';
      ElemDesc->dumpFull(FO, Indent + 1);

      FO += ElemDesc->getAllocSize();
    }
  } else if (isRecord()) {
    ElemRecord->dump(OS, Indent + 1, Offset);
  } else if (isPrimitive()) {
  } else {
  }

  OS << '\n';
}

LLVM_DUMP_METHOD void InlineDescriptor::dump(llvm::raw_ostream &OS) const {
  {
    ColorScope SC(OS, true, {llvm::raw_ostream::BLUE, true});
    OS << "InlineDescriptor " << (const void *)this << "\n";
  }
  OS << "Offset: " << Offset << "\n";
  OS << "IsConst: " << IsConst << "\n";
  OS << "IsInitialized: " << IsInitialized << "\n";
  OS << "IsBase: " << IsBase << "\n";
  OS << "IsActive: " << IsActive << "\n";
  OS << "InUnion: " << InUnion << "\n";
  OS << "IsFieldMutable: " << IsFieldMutable << "\n";
  OS << "IsArrayElement: " << IsArrayElement << "\n";
  OS << "IsConstInMutable: " << IsConstInMutable << '\n';
  OS << "Desc: ";
  if (Desc)
    Desc->dump(OS);
  else
    OS << "nullptr";
  OS << "\n";
}

LLVM_DUMP_METHOD void InterpFrame::dump(llvm::raw_ostream &OS,
                                        unsigned Indent) const {
  unsigned Spaces = Indent * 2;
  {
    ColorScope SC(OS, true, {llvm::raw_ostream::BLUE, true});
    OS.indent(Spaces);
    if (getCallee())
      describe(OS);
    else
      OS << "Frame (Depth: " << getDepth() << ")";
    OS << "\n";
  }
  OS.indent(Spaces) << "Function: " << getFunction();
  if (const Function *F = getFunction()) {
    OS << " (" << F->getName() << ")";
  }
  OS << "\n";
  OS.indent(Spaces) << "This: " << getThis() << "\n";
  OS.indent(Spaces) << "RVO: " << getRVOPtr() << "\n";
  OS.indent(Spaces) << "Depth: " << Depth << "\n";
  OS.indent(Spaces) << "ArgSize: " << ArgSize << "\n";
  OS.indent(Spaces) << "Args: " << (void *)Args << "\n";
  OS.indent(Spaces) << "FrameOffset: " << FrameOffset << "\n";
  OS.indent(Spaces) << "FrameSize: " << (Func ? Func->getFrameSize() : 0)
                    << "\n";

  for (const InterpFrame *F = this->Caller; F; F = F->Caller) {
    F->dump(OS, Indent + 1);
  }
}

LLVM_DUMP_METHOD void Record::dump(llvm::raw_ostream &OS, unsigned Indentation,
                                   unsigned Offset) const {
  unsigned Indent = Indentation * 2;
  OS.indent(Indent);
  {
    ColorScope SC(OS, true, {llvm::raw_ostream::BLUE, true});
    OS << getName() << "\n";
  }

  unsigned I = 0;
  for (const Record::Base &B : bases()) {
    OS.indent(Indent) << "- Base " << I << ". Offset " << (Offset + B.Offset)
                      << "\n";
    B.R->dump(OS, Indentation + 1, Offset + B.Offset);
    ++I;
  }

  I = 0;
  for (const Record::Field &F : fields()) {
    OS.indent(Indent) << "- Field " << I << ": ";
    {
      ColorScope SC(OS, true, {llvm::raw_ostream::BRIGHT_RED, true});
      OS << F.Decl->getName();
    }
    OS << ". Offset " << (Offset + F.Offset) << "\n";
    ++I;
  }

  I = 0;
  for (const Record::Base &B : virtual_bases()) {
    OS.indent(Indent) << "- Virtual Base " << I << ". Offset "
                      << (Offset + B.Offset) << "\n";
    B.R->dump(OS, Indentation + 1, Offset + B.Offset);
    ++I;
  }
}

LLVM_DUMP_METHOD void Block::dump(llvm::raw_ostream &OS) const {
  {
    ColorScope SC(OS, true, {llvm::raw_ostream::BRIGHT_BLUE, true});
    OS << "Block " << (const void *)this;
  }
  OS << " (";
  Desc->dump(OS);
  OS << ")\n";
  unsigned NPointers = 0;
  for (const Pointer *P = Pointers; P; P = P->asBlockPointer().Next) {
    ++NPointers;
  }
  OS << "  EvalID: " << EvalID << '\n';
  OS << "  DeclID: ";
  if (DeclID)
    OS << *DeclID << '\n';
  else
    OS << "-\n";
  OS << "  Pointers: " << NPointers << "\n";
  OS << "  Dead: " << isDead() << "\n";
  OS << "  Static: " << IsStatic << "\n";
  OS << "  Extern: " << isExtern() << "\n";
  OS << "  Initialized: " << IsInitialized << "\n";
  OS << "  Weak: " << isWeak() << "\n";
  OS << "  Dummy: " << isDummy() << '\n';
  OS << "  Dynamic: " << isDynamic() << "\n";
}

LLVM_DUMP_METHOD void EvaluationResult::dump() const {
  auto &OS = llvm::errs();

  if (empty()) {
    OS << "Empty\n";
  } else if (isInvalid()) {
    OS << "Invalid\n";
  } else {
    OS << "Value: ";
#ifndef NDEBUG
    assert(Ctx);
    Value.dump(OS, Ctx->getASTContext());
#endif
  }
}
