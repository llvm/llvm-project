//===--- AttrImpl.cpp - Classes for representing attributes -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file contains out-of-line virtual methods for Attr classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/Attr.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/AST/Expr.h"
using namespace clang;

Attr::~Attr() { }

void InheritableAttr::anchor() { }

void InheritableParamAttr::anchor() { }

#include "clang/AST/AttrImpl.inc"
