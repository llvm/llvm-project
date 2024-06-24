//===--- SemaOpenCL.cpp --- Semantic Analysis for OpenCL constructs -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements semantic analysis for OpenCL.
///
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaOpenCL.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclBase.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"

namespace clang {
SemaOpenCL::SemaOpenCL(Sema &S) : SemaBase(S) {}

void SemaOpenCL::handleNoSVMAttr(Decl *D, const ParsedAttr &AL) {
  if (getLangOpts().getOpenCLCompatibleVersion() < 200)
    Diag(AL.getLoc(), diag::err_attribute_requires_opencl_version)
        << AL << "2.0" << 1;
  else
    Diag(AL.getLoc(), diag::warn_opencl_attr_deprecated_ignored)
        << AL << getLangOpts().getOpenCLVersionString();
}

void SemaOpenCL::handleAccessAttr(Decl *D, const ParsedAttr &AL) {
  if (D->isInvalidDecl())
    return;

  // Check if there is only one access qualifier.
  if (D->hasAttr<OpenCLAccessAttr>()) {
    if (D->getAttr<OpenCLAccessAttr>()->getSemanticSpelling() ==
        AL.getSemanticSpelling()) {
      Diag(AL.getLoc(), diag::warn_duplicate_declspec)
          << AL.getAttrName()->getName() << AL.getRange();
    } else {
      Diag(AL.getLoc(), diag::err_opencl_multiple_access_qualifiers)
          << D->getSourceRange();
      D->setInvalidDecl(true);
      return;
    }
  }

  // OpenCL v2.0 s6.6 - read_write can be used for image types to specify that
  // an image object can be read and written. OpenCL v2.0 s6.13.6 - A kernel
  // cannot read from and write to the same pipe object. Using the read_write
  // (or __read_write) qualifier with the pipe qualifier is a compilation error.
  // OpenCL v3.0 s6.8 - For OpenCL C 2.0, or with the
  // __opencl_c_read_write_images feature, image objects specified as arguments
  // to a kernel can additionally be declared to be read-write.
  // C++ for OpenCL 1.0 inherits rule from OpenCL C v2.0.
  // C++ for OpenCL 2021 inherits rule from OpenCL C v3.0.
  if (const auto *PDecl = dyn_cast<ParmVarDecl>(D)) {
    const Type *DeclTy = PDecl->getType().getCanonicalType().getTypePtr();
    if (AL.getAttrName()->getName().contains("read_write")) {
      bool ReadWriteImagesUnsupported =
          (getLangOpts().getOpenCLCompatibleVersion() < 200) ||
          (getLangOpts().getOpenCLCompatibleVersion() == 300 &&
           !SemaRef.getOpenCLOptions().isSupported(
               "__opencl_c_read_write_images", getLangOpts()));
      if (ReadWriteImagesUnsupported || DeclTy->isPipeType()) {
        Diag(AL.getLoc(), diag::err_opencl_invalid_read_write)
            << AL << PDecl->getType() << DeclTy->isImageType();
        D->setInvalidDecl(true);
        return;
      }
    }
  }

  D->addAttr(::new (getASTContext()) OpenCLAccessAttr(getASTContext(), AL));
}

void SemaOpenCL::handleSubGroupSize(Decl *D, const ParsedAttr &AL) {
  uint32_t SGSize;
  const Expr *E = AL.getArgAsExpr(0);
  if (!SemaRef.checkUInt32Argument(AL, E, SGSize))
    return;
  if (SGSize == 0) {
    Diag(AL.getLoc(), diag::err_attribute_argument_is_zero)
        << AL << E->getSourceRange();
    return;
  }

  OpenCLIntelReqdSubGroupSizeAttr *Existing =
      D->getAttr<OpenCLIntelReqdSubGroupSizeAttr>();
  if (Existing && Existing->getSubGroupSize() != SGSize)
    Diag(AL.getLoc(), diag::warn_duplicate_attribute) << AL;

  D->addAttr(::new (getASTContext())
                 OpenCLIntelReqdSubGroupSizeAttr(getASTContext(), AL, SGSize));
}

} // namespace clang
