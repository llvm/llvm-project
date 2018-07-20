//===- unittest/Tooling/RecursiveASTVisitorTests/TemplateArgumentLocTraverser.cpp -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"

using namespace clang;

namespace {

class TemplateArgumentLocTraverser
  : public ExpectedLocationVisitor<TemplateArgumentLocTraverser> {
public:
  bool TraverseTemplateArgumentLoc(const TemplateArgumentLoc &ArgLoc) {
    std::string ArgStr;
    llvm::raw_string_ostream Stream(ArgStr);
    const TemplateArgument &Arg = ArgLoc.getArgument();

    Arg.print(Context->getPrintingPolicy(), Stream);
    Match(Stream.str(), ArgLoc.getLocation());
    return ExpectedLocationVisitor<TemplateArgumentLocTraverser>::
      TraverseTemplateArgumentLoc(ArgLoc);
  }
};

TEST(RecursiveASTVisitor, VisitsClassTemplateTemplateParmDefaultArgument) {
  TemplateArgumentLocTraverser Visitor;
  Visitor.ExpectMatch("X", 2, 40);
  EXPECT_TRUE(Visitor.runOver(
    "template<typename T> class X;\n"
    "template<template <typename> class T = X> class Y;\n"
    "template<template <typename> class T> class Y {};\n"));
}

} // end anonymous namespace
