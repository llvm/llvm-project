//===-- ConvertExpr.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Common/idioms.h"
#include "flang/Lower/IntrinsicCall.h"
#include "flang/Lower/Support/BoxValue.h"

mlir::Value fir::getBase(const fir::ExtendedValue &ex) {
  return std::visit(Fortran::common::visitors{
                        [](const fir::UnboxedValue &x) { return x; },
                        [](const auto &x) { return x.getAddr(); },
                    },
                    ex.box);
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::CharBoxValue &box) {
  os << "boxchar { addr: " << box.getAddr() << ", len: " << box.getLen()
     << " }";
  return os;
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::ArrayBoxValue &box) {
  os << "boxarray { addr: " << box.getAddr();
  if (box.getLBounds().size()) {
    os << ", lbounds: [";
    llvm::interleaveComma(box.getLBounds(), os);
    os << "]";
  } else {
    os << ", lbounds: all-ones";
  }
  os << ", shape: [";
  llvm::interleaveComma(box.getExtents(), os);
  os << "]}";
  return os;
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::CharArrayBoxValue &box) {
  os << "boxchararray { addr: " << box.getAddr() << ", len : " << box.getLen();
  if (box.getLBounds().size()) {
    os << ", lbounds: [";
    llvm::interleaveComma(box.getLBounds(), os);
    os << "]";
  } else {
    os << " lbounds: all-ones";
  }
  os << ", shape: [";
  llvm::interleaveComma(box.getExtents(), os);
  os << "]}";
  return os;
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::BoxValue &box) {
  os << "box { addr: " << box.getAddr();
  if (box.getLen())
    os << ", size: " << box.getLen();
  if (box.params.size()) {
    os << ", type params: [";
    llvm::interleaveComma(box.params, os);
    os << "]";
  }
  if (box.getLBounds().size()) {
    os << ", lbounds: [";
    llvm::interleaveComma(box.getLBounds(), os);
    os << "]";
  }
  if (box.getExtents().size()) {
    os << ", shape: [";
    llvm::interleaveComma(box.getExtents(), os);
    os << "]";
  }
  os << "}";
  return os;
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::ProcBoxValue &box) {
  os << "boxproc: { addr: " << box.getAddr() << ", context: " << box.hostContext
     << "}";
  return os;
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::ExtendedValue &ex) {
  std::visit([&](const auto &value) { os << value; }, ex.box);
  return os;
}
