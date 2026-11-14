//==---- RISCVLpadLabelGen.h - generates func-sig labels for landing pads --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Computes RISC-V zicfilp landing-pad labels for the psABI "func-sig"
/// scheme, deriving each label from a function's mangled signature.
/// Shared by function definitions and indirect call sites so their
/// labels agree.
///
//===----------------------------------------------------------------------===//
#ifndef CLANG_LIB_CODEGEN_RISCV_LPADLABELGEN_H
#define CLANG_LIB_CODEGEN_RISCV_LPADLABELGEN_H

#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MD5.h"
#include <cstdint>
#include <string>

namespace clang {
class ASTContext;
class ItaniumMangleContext;
class FunctionDecl;

namespace CodeGen {

std::string getRISCVLpadSignatureString(ASTContext &Ctx,
                                        ItaniumMangleContext &MC, QualType Ty,
                                        SourceLocation Loc = SourceLocation());
std::string getRISCVLpadSignatureString(ASTContext &Ctx,
                                        ItaniumMangleContext &MC,
                                        const FunctionDecl *FD);

uint32_t computeRISCVLpadLabel(const llvm::MD5::MD5Result &Result);

uint32_t computeRISCVLpadLabel(llvm::StringRef Str);

QualType buildBlockCopyHelperSigTy(ASTContext &Ctx);

QualType buildDtorSigTy(ASTContext &Ctx);

QualType buildVoidPtrToVoidPtrSigTy(ASTContext &Ctx);

/// Generate a 20-bit type identifier for riscv landing pad labels.
/// Entry point when a Decl is known, to compute the function entry label.
uint32_t createRISCVEntryLpadTypeId(ASTContext &Ctx, ItaniumMangleContext &MC,
                                    const FunctionDecl *FD);

/// Generate a 20-bit type identifier for riscv landing pad labels.
/// Entry point when a Decl is known, to compute the callsite label.
uint32_t createRISCVCallSiteLpadTypeId(ASTContext &Ctx,
                                       ItaniumMangleContext &MC,
                                       const FunctionDecl *FD);

/// Generate a 20-bit type identifier for riscv landing pad labels.
/// Canonical/structural entry point. Used directly for indirect calls with
/// no resolvable Decl, and internally by the FunctionDecl overload after
/// Decl-only special cases are resolved. This is the single source of
/// truth for the actual mangling + hashing.
uint32_t createRISCVLpadTypeId(ASTContext &Ctx, ItaniumMangleContext &MC,
                               QualType Ty,
                               SourceLocation Loc = SourceLocation());

} // namespace CodeGen
} // namespace clang

#endif
