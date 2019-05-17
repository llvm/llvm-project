//===--- ASTDumper.h - Dumping implementation for ASTs --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_ASTDUMPER_H
#define LLVM_CLANG_AST_ASTDUMPER_H

#include "clang/AST/ASTNodeTraverser.h"
#include "clang/AST/TextNodeDumper.h"

namespace clang {

class ASTDumper : public ASTNodeTraverser<ASTDumper, TextNodeDumper> {

  TextNodeDumper NodeDumper;

  raw_ostream &OS;

  const bool ShowColors;

public:
  ASTDumper(raw_ostream &OS, const comments::CommandTraits *Traits,
            const SourceManager *SM)
      : ASTDumper(OS, Traits, SM, SM && SM->getDiagnostics().getShowColors()) {}

  ASTDumper(raw_ostream &OS, const comments::CommandTraits *Traits,
            const SourceManager *SM, bool ShowColors)
      : ASTDumper(OS, Traits, SM, ShowColors, LangOptions()) {}
  ASTDumper(raw_ostream &OS, const comments::CommandTraits *Traits,
            const SourceManager *SM, bool ShowColors,
            const PrintingPolicy &PrintPolicy)
      : NodeDumper(OS, ShowColors, SM, PrintPolicy, Traits), OS(OS),
        ShowColors(ShowColors) {}

  TextNodeDumper &doGetNodeDelegate() { return NodeDumper; }

  void dumpLookups(const DeclContext *DC, bool DumpDecls);

  template <typename SpecializationDecl>
  void dumpTemplateDeclSpecialization(const SpecializationDecl *D,
                                      bool DumpExplicitInst, bool DumpRefOnly);
  template <typename TemplateDecl>
  void dumpTemplateDecl(const TemplateDecl *D, bool DumpExplicitInst);

  void VisitFunctionTemplateDecl(const FunctionTemplateDecl *D);
  void VisitClassTemplateDecl(const ClassTemplateDecl *D);
  void VisitVarTemplateDecl(const VarTemplateDecl *D);
};

} // namespace clang

#endif
