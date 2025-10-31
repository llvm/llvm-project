//===--- CIRGenAsm.cpp - Inline Assembly Support for CIR CodeGen ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains code to emit inline assembly.
//
//===----------------------------------------------------------------------===//

#include "CIRGenFunction.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;
using namespace cir;

static AsmFlavor inferFlavor(const CIRGenModule &cgm, const AsmStmt &s) {
  AsmFlavor gnuAsmFlavor =
      cgm.getCodeGenOpts().getInlineAsmDialect() == CodeGenOptions::IAD_ATT
          ? AsmFlavor::x86_att
          : AsmFlavor::x86_intel;

  return isa<MSAsmStmt>(&s) ? AsmFlavor::x86_intel : gnuAsmFlavor;
}

static void collectClobbers(const CIRGenFunction &cgf, const AsmStmt &s,
                            std::string &constraints, bool &hasUnwindClobber,
                            bool &readOnly, bool readNone) {

  hasUnwindClobber = false;
  const CIRGenModule &cgm = cgf.getCIRGenModule();

  // Clobbers
  for (unsigned i = 0, e = s.getNumClobbers(); i != e; i++) {
    std::string clobber = s.getClobber(i);
    if (clobber == "memory") {
      readOnly = readNone = false;
    } else if (clobber == "unwind") {
      hasUnwindClobber = true;
      continue;
    } else if (clobber != "cc") {
      clobber = cgf.getTarget().getNormalizedGCCRegisterName(clobber);
      if (cgm.getCodeGenOpts().StackClashProtector &&
          cgf.getTarget().isSPRegName(clobber))
        cgm.getDiags().Report(s.getAsmLoc(),
                              diag::warn_stack_clash_protection_inline_asm);
    }

    if (isa<MSAsmStmt>(&s)) {
      if (clobber == "eax" || clobber == "edx") {
        if (constraints.find("=&A") != std::string::npos)
          continue;
        std::string::size_type position1 =
            constraints.find("={" + clobber + "}");
        if (position1 != std::string::npos) {
          constraints.insert(position1 + 1, "&");
          continue;
        }
        std::string::size_type position2 = constraints.find("=A");
        if (position2 != std::string::npos) {
          constraints.insert(position2 + 1, "&");
          continue;
        }
      }
    }
    if (!constraints.empty())
      constraints += ',';

    constraints += "~{";
    constraints += clobber;
    constraints += '}';
  }

  // Add machine specific clobbers
  std::string_view machineClobbers = cgf.getTarget().getClobbers();
  if (!machineClobbers.empty()) {
    if (!constraints.empty())
      constraints += ',';
    constraints += machineClobbers;
  }
}

mlir::LogicalResult CIRGenFunction::emitAsmStmt(const AsmStmt &s) {
  // Assemble the final asm string.
  std::string asmString = s.generateAsmString(getContext());

  bool isGCCAsmGoto = false;

  std::string constraints;
  std::vector<mlir::Value> outArgs;
  std::vector<mlir::Value> inArgs;
  std::vector<mlir::Value> inOutArgs;

  // An inline asm can be marked readonly if it meets the following conditions:
  //  - it doesn't have any sideeffects
  //  - it doesn't clobber memory
  //  - it doesn't return a value by-reference
  // It can be marked readnone if it doesn't have any input memory constraints
  // in addition to meeting the conditions listed above.
  bool readOnly = true, readNone = true;

  if (s.getNumInputs() != 0 || s.getNumOutputs() != 0) {
    assert(!cir::MissingFeatures::asmInputOperands());
    assert(!cir::MissingFeatures::asmOutputOperands());
    cgm.errorNYI(s.getAsmLoc(), "asm with operands");
  }

  bool hasUnwindClobber = false;
  collectClobbers(*this, s, constraints, hasUnwindClobber, readOnly, readNone);

  std::array<mlir::ValueRange, 3> operands = {outArgs, inArgs, inOutArgs};

  mlir::Type resultType;

  bool hasSideEffect = s.isVolatile() || s.getNumOutputs() == 0;

  cir::InlineAsmOp ia = builder.create<cir::InlineAsmOp>(
      getLoc(s.getAsmLoc()), resultType, operands, asmString, constraints,
      hasSideEffect, inferFlavor(cgm, s), mlir::ArrayAttr());

  if (isGCCAsmGoto) {
    assert(!cir::MissingFeatures::asmGoto());
  } else if (hasUnwindClobber) {
    assert(!cir::MissingFeatures::asmUnwindClobber());
  } else {
    assert(!cir::MissingFeatures::asmMemoryEffects());
  }

  llvm::SmallVector<mlir::Attribute> operandAttrs;
  ia.setOperandAttrsAttr(builder.getArrayAttr(operandAttrs));

  return mlir::success();
}
