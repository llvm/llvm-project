//===-- BoxValue.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pretty printers for box values, etc.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Lower/Todo.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "flang-box-value"

mlir::Value fir::getBase(const fir::ExtendedValue &exv) {
  return exv.match([](const fir::UnboxedValue &x) { return x; },
                   [](const auto &x) { return x.getAddr(); });
}

mlir::Value fir::getLen(const fir::ExtendedValue &exv) {
  return exv.match(
      [](const fir::CharBoxValue &x) { return x.getLen(); },
      [](const fir::CharArrayBoxValue &x) { return x.getLen(); },
      [](const fir::BoxValue &) -> mlir::Value {
        llvm::report_fatal_error("Need to read len from BoxValue Exv");
      },
      [](const fir::MutableBoxValue &) -> mlir::Value {
        llvm::report_fatal_error("Need to read len from MutableBoxValue Exv");
      },
      [](const auto &) { return mlir::Value{}; });
}

fir::ExtendedValue fir::substBase(const fir::ExtendedValue &exv,
                                  mlir::Value base) {
  return exv.match(
      [=](const fir::UnboxedValue &x) { return fir::ExtendedValue(base); },
      [=](const auto &x) { return fir::ExtendedValue(x.clone(base)); });
}

llvm::SmallVector<mlir::Value>
fir::getTypeParams(const fir::ExtendedValue &exv) {
  using RT = llvm::SmallVector<mlir::Value>;
  auto baseTy = fir::getBase(exv).getType();
  if (auto t = fir::dyn_cast_ptrEleTy(baseTy))
    baseTy = t;
  baseTy = fir::unwrapSequenceType(baseTy);
  if (!fir::hasDynamicSize(baseTy))
    return {}; // type has constant size, no type parameters needed
  [[maybe_unused]] auto loc = fir::getBase(exv).getLoc();
  return exv.match(
      [](const fir::CharBoxValue &x) -> RT { return {x.getLen()}; },
      [](const fir::CharArrayBoxValue &x) -> RT { return {x.getLen()}; },
      [&](const fir::BoxValue &) -> RT {
        TODO(loc, "box value is missing type parameters");
        return {};
      },
      [&](const fir::MutableBoxValue &) -> RT {
        // In this case, the type params may be bound to the variable in an
        // ALLOCATE statement as part of a type-spec.
        TODO(loc, "mutable box value is missing type parameters");
        return {};
      },
      [](const auto &) -> RT { return {}; });
}

bool fir::isArray(const fir::ExtendedValue &exv) {
  return exv.match(
      [](const fir::ArrayBoxValue &) { return true; },
      [](const fir::CharArrayBoxValue &) { return true; },
      [](const fir::BoxValue &box) { return box.hasRank(); },
      [](const fir::MutableBoxValue &box) { return box.hasRank(); },
      [](auto) { return false; });
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::CharBoxValue &box) {
  return os << "boxchar { addr: " << box.getAddr() << ", len: " << box.getLen()
            << " }";
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
  return os << "]}";
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
  return os << "]}";
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::ProcBoxValue &box) {
  return os << "boxproc: { procedure: " << box.getAddr()
            << ", context: " << box.hostContext << "}";
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::BoxValue &box) {
  os << "box: { value: " << box.getAddr();
  if (box.lbounds.size()) {
    os << ", lbounds: [";
    llvm::interleaveComma(box.lbounds, os);
    os << "]";
  }
  if (!box.explicitParams.empty()) {
    os << ", explicit type params: [";
    llvm::interleaveComma(box.explicitParams, os);
    os << "]";
  }
  if (!box.extents.empty()) {
    os << ", explicit extents: [";
    llvm::interleaveComma(box.extents, os);
    os << "]";
  }
  return os << "}";
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::MutableBoxValue &box) {
  os << "mutablebox: { addr: " << box.getAddr();
  if (!box.lenParams.empty()) {
    os << ", non deferred type params: [";
    llvm::interleaveComma(box.lenParams, os);
    os << "]";
  }
  const auto &properties = box.mutableProperties;
  if (!properties.isEmpty()) {
    os << ", mutableProperties: { addr: " << properties.addr;
    if (!properties.lbounds.empty()) {
      os << ", lbounds: [";
      llvm::interleaveComma(properties.lbounds, os);
      os << "]";
    }
    if (!properties.extents.empty()) {
      os << ", shape: [";
      llvm::interleaveComma(properties.extents, os);
      os << "]";
    }
    if (!properties.deferredParams.empty()) {
      os << ", deferred type params: [";
      llvm::interleaveComma(properties.deferredParams, os);
      os << "]";
    }
    os << "}";
  }
  return os << "}";
}

llvm::raw_ostream &fir::operator<<(llvm::raw_ostream &os,
                                   const fir::ExtendedValue &exv) {
  exv.match([&](const auto &value) { os << value; });
  return os;
}

/// Debug verifier for MutableBox ctor. There is no guarantee that this will
/// always be called, so it should not have any functional side effects,
/// the const is here to enforce that.
bool fir::MutableBoxValue::verify() const {
  mlir::Type type = fir::dyn_cast_ptrEleTy(getAddr().getType());
  if (!type)
    return false;
  auto box = type.dyn_cast<fir::BoxType>();
  if (!box)
    return false;
  // A boxed value always takes a memory reference,

  auto nParams = lenParams.size();
  if (isCharacter()) {
    if (nParams > 1)
      return false;
  } else if (!isDerived()) {
    if (nParams != 0)
      return false;
  }
  return true;
}

/// Debug verifier for BoxValue ctor. There is no guarantee this will
/// always be called.
bool fir::BoxValue::verify() const {
  if (!addr.getType().isa<fir::BoxType>())
    return false;
  if (!lbounds.empty() && lbounds.size() != rank())
    return false;
  // Explicit extents are here to cover cases where an explicit-shape dummy
  // argument comes as a fir.box. This can only happen with derived types and
  // unlimited polymorphic.
  if (!extents.empty() && !(isDerived() || isUnlimitedPolymorphic()))
    return false;
  if (!extents.empty() && extents.size() != rank())
    return false;
  if (isCharacter() && explicitParams.size() > 1)
    return false;
  return true;
}

/// Get exactly one extent for any array-like extended value, \p exv. If \p exv
/// is not an array or has rank less then \p dim, the result will be a nullptr.
mlir::Value fir::factory::getExtentAtDimension(mlir::Location loc,
                                               fir::FirOpBuilder &builder,
                                               const fir::ExtendedValue &exv,
                                               unsigned dim) {
  auto extents = fir::factory::getExtents(loc, builder, exv);
  if (dim < extents.size())
    return extents[dim];
  return {};
}

static inline bool isUndefOp(mlir::Value v) {
  return mlir::isa_and_nonnull<fir::UndefOp>(v.getDefiningOp());
}

bool fir::ExtendedValue::isAssumedSize() const {
  return match(
      [](const fir::ArrayBoxValue &box) -> bool {
        return !box.getExtents().empty() && isUndefOp(box.getExtents().back());
        ;
      },
      [](const fir::CharArrayBoxValue &box) -> bool {
        return !box.getExtents().empty() && isUndefOp(box.getExtents().back());
      },
      [](const auto &box) -> bool { return false; });
}
