//===--- Mangle.cpp - Mangle C++ Names --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements generic name mangling support for blocks and Objective-C.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/Mangle.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/ABI.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"

#define MANGLE_CHECKER 0

#if MANGLE_CHECKER
#include <cxxabi.h>
#endif

using namespace clang;

// FIXME: For blocks we currently mimic GCC's mangling scheme, which leaves
// much to be desired. Come up with a better mangling scheme.

namespace {

static void mangleFunctionBlock(MangleContext &Context,
                                StringRef Outer,
                                const BlockDecl *BD,
                                raw_ostream &Out) {
  unsigned discriminator = Context.getBlockId(BD, true);
  if (discriminator == 0)
    Out << "__" << Outer << "_block_invoke";
  else
    Out << "__" << Outer << "_block_invoke_" << discriminator+1; 
}

static void checkMangleDC(const DeclContext *DC, const BlockDecl *BD) {
#ifndef NDEBUG
  const DeclContext *ExpectedDC = BD->getDeclContext();
  while (isa<BlockDecl>(ExpectedDC) || isa<EnumDecl>(ExpectedDC))
    ExpectedDC = ExpectedDC->getParent();
  // In-class initializers for non-static data members are lexically defined
  // within the class, but are mangled as if they were specified as constructor
  // member initializers.
  if (isa<CXXRecordDecl>(ExpectedDC) && DC != ExpectedDC)
    DC = DC->getParent();
  assert(DC == ExpectedDC && "Given decl context did not match expected!");
#endif
}

}

void MangleContext::anchor() { }

void MangleContext::mangleGlobalBlock(const BlockDecl *BD,
                                      const NamedDecl *ID,
                                      raw_ostream &Out) {
  unsigned discriminator = getBlockId(BD, false);
  if (ID) {
    if (shouldMangleDeclName(ID))
      mangleName(ID, Out);
    else {
      Out << ID->getIdentifier()->getName();
    }
  }
  if (discriminator == 0)
    Out << "_block_invoke";
  else
    Out << "_block_invoke_" << discriminator+1;
}

void MangleContext::mangleCtorBlock(const CXXConstructorDecl *CD,
                                    CXXCtorType CT, const BlockDecl *BD,
                                    raw_ostream &ResStream) {
  checkMangleDC(CD, BD);
  SmallString<64> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  mangleCXXCtor(CD, CT, Out);
  Out.flush();
  mangleFunctionBlock(*this, Buffer, BD, ResStream);
}

void MangleContext::mangleDtorBlock(const CXXDestructorDecl *DD,
                                    CXXDtorType DT, const BlockDecl *BD,
                                    raw_ostream &ResStream) {
  checkMangleDC(DD, BD);
  SmallString<64> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  mangleCXXDtor(DD, DT, Out);
  Out.flush();
  mangleFunctionBlock(*this, Buffer, BD, ResStream);
}

void MangleContext::mangleBlock(const DeclContext *DC, const BlockDecl *BD,
                                raw_ostream &Out) {
  assert(!isa<CXXConstructorDecl>(DC) && !isa<CXXDestructorDecl>(DC));
  checkMangleDC(DC, BD);

  SmallString<64> Buffer;
  llvm::raw_svector_ostream Stream(Buffer);
  if (const ObjCMethodDecl *Method = dyn_cast<ObjCMethodDecl>(DC)) {
    mangleObjCMethodName(Method, Stream);
  } else {
    const NamedDecl *ND = cast<NamedDecl>(DC);
    if (!shouldMangleDeclName(ND) && ND->getIdentifier())
      Stream << ND->getIdentifier()->getName();
    else {
      // FIXME: We were doing a mangleUnqualifiedName() before, but that's
      // a private member of a class that will soon itself be private to the
      // Itanium C++ ABI object. What should we do now? Right now, I'm just
      // calling the mangleName() method on the MangleContext; is there a
      // better way?
      mangleName(ND, Stream);
    }
  }
  Stream.flush();
  mangleFunctionBlock(*this, Buffer, BD, Out);
}

void MangleContext::mangleObjCMethodName(const ObjCMethodDecl *MD,
                                         raw_ostream &Out) {
  SmallString<64> Name;
  llvm::raw_svector_ostream OS(Name);
  
  const ObjCContainerDecl *CD =
  dyn_cast<ObjCContainerDecl>(MD->getDeclContext());
  assert (CD && "Missing container decl in GetNameForMethod");
  OS << (MD->isInstanceMethod() ? '-' : '+') << '[' << CD->getName();
  if (const ObjCCategoryImplDecl *CID = dyn_cast<ObjCCategoryImplDecl>(CD))
    OS << '(' << *CID << ')';
  OS << ' ' << MD->getSelector().getAsString() << ']';
  
  Out << OS.str().size() << OS.str();
}

void MangleContext::mangleBlock(const BlockDecl *BD,
                                raw_ostream &Out,
                                const NamedDecl *ID) {
  const DeclContext *DC = BD->getDeclContext();
  while (isa<BlockDecl>(DC) || isa<EnumDecl>(DC))
    DC = DC->getParent();
  if (DC->isFunctionOrMethod())
    mangleBlock(DC, BD, Out);
  else
    mangleGlobalBlock(BD, ID, Out);
}
