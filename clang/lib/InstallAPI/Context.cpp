//===--- InstallAPI/Context.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/InstallAPI/Context.h"
#include "clang/AST/ASTContext.h"
#include "llvm/TextAPI/TextAPIWriter.h"

using namespace clang;
using namespace clang::installapi;
using namespace llvm::MachO;

void InstallAPIConsumer::HandleTranslationUnit(ASTContext &Context) {
  if (Context.getDiagnostics().hasErrorOccurred())
    return;
  InterfaceFile IF;
  // Set library attributes captured through cc1 args.
  Target T(Ctx.TargetTriple);
  IF.addTarget(T);
  IF.setFromBinaryAttrs(Ctx.BA, T);
  if (auto Err = TextAPIWriter::writeToStream(*Ctx.OS, IF, Ctx.FT))
    Ctx.Diags->Report(diag::err_cannot_open_file) << Ctx.OutputLoc;
}
