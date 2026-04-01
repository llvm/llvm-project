//===- CIRCXXABI.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file partially mimics clang/lib/CodeGen/CGCXXABI.cpp. The queries are
// adapted to operate on the CIR dialect, however.
//
//===----------------------------------------------------------------------===//

#include "CIRCXXABI.h"
#include "LowerModule.h"

namespace cir {

CIRCXXABI::~CIRCXXABI() {}

unsigned CIRCXXABI::getPtrSizeInBits() const {
  return lm.getTarget().getPointerWidth(clang::LangAS::Default);
}

void CIRCXXABI::readArrayCookie(aiir::Location loc, aiir::Value elementPtr,
                                const aiir::DataLayout &dataLayout,
                                CIRBaseBuilderTy &builder,
                                aiir::Value &numElements, aiir::Value &allocPtr,
                                clang::CharUnits &cookieSize) const {
  auto u8PtrTy = builder.getPointerTo(builder.getUIntNTy(8));
  auto ptrDiffTy = builder.getSIntNTy(getPtrSizeInBits());
  auto voidPtrTy = builder.getVoidPtrTy();

  auto ptrTy = aiir::cast<cir::PointerType>(elementPtr.getType());
  cookieSize = getArrayCookieSizeImpl(ptrTy.getPointee(), dataLayout);

  aiir::Value bytePtr = cir::CastOp::create(builder, loc, u8PtrTy,
                                            cir::CastKind::bitcast, elementPtr);

  aiir::Value negCookieSize = cir::ConstantOp::create(
      builder, loc, cir::IntAttr::get(ptrDiffTy, -cookieSize.getQuantity()));
  aiir::Value allocBytePtr =
      cir::PtrStrideOp::create(builder, loc, u8PtrTy, bytePtr, negCookieSize);

  allocPtr = cir::CastOp::create(builder, loc, voidPtrTy,
                                 cir::CastKind::bitcast, allocBytePtr);

  // cookieSize is always a multiple of the element ABI alignment (both are
  // powers of 2 and cookieSize >= elementAlign), so subtracting it preserves
  // alignment. The cookie alignment therefore equals the element alignment.
  clang::CharUnits cookieAlignment = clang::CharUnits::fromQuantity(
      dataLayout.getTypePreferredAlignment(ptrTy.getPointee()));
  numElements = readArrayCookieImpl(loc, allocBytePtr, cookieSize,
                                    cookieAlignment, dataLayout, builder);
}

} // namespace cir
