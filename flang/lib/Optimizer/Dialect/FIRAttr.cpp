//===-- FIRAttr.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Dialect/FIRAttr.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Support/KindMapping.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallString.h"

using namespace fir;

namespace fir::detail {

struct RealAttributeStorage : public mlir::AttributeStorage {
  using KeyTy = std::pair<int, llvm::APFloat>;

  RealAttributeStorage(int kind, const llvm::APFloat &value)
      : kind(kind), value(value) {}
  RealAttributeStorage(const KeyTy &key)
      : RealAttributeStorage(key.first, key.second) {}

  static unsigned hashKey(const KeyTy &key) { return llvm::hash_value(key); }

  bool operator==(const KeyTy &key) const {
    return key.first == kind &&
           key.second.compare(value) == llvm::APFloatBase::cmpEqual;
  }

  static RealAttributeStorage *
  construct(mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<RealAttributeStorage>())
        RealAttributeStorage(key);
  }

  KindTy getFKind() const { return kind; }
  llvm::APFloat getValue() const { return value; }

private:
  int kind;
  llvm::APFloat value;
};

/// An attribute representing a reference to a type.
struct TypeAttributeStorage : public mlir::AttributeStorage {
  using KeyTy = mlir::Type;

  TypeAttributeStorage(mlir::Type value) : value(value) {
    assert(value && "must not be of Type null");
  }

  /// Key equality function.
  bool operator==(const KeyTy &key) const { return key == value; }

  /// Construct a new storage instance.
  static TypeAttributeStorage *
  construct(mlir::AttributeStorageAllocator &allocator, KeyTy key) {
    return new (allocator.allocate<TypeAttributeStorage>())
        TypeAttributeStorage(key);
  }

  mlir::Type getType() const { return value; }

private:
  mlir::Type value;
};
} // namespace fir::detail

//===----------------------------------------------------------------------===//
// Attributes for SELECT TYPE
//===----------------------------------------------------------------------===//

ExactTypeAttr fir::ExactTypeAttr::get(mlir::Type value) {
  return Base::get(value.getContext(), value);
}

mlir::Type fir::ExactTypeAttr::getType() const { return getImpl()->getType(); }

SubclassAttr fir::SubclassAttr::get(mlir::Type value) {
  return Base::get(value.getContext(), value);
}

mlir::Type fir::SubclassAttr::getType() const { return getImpl()->getType(); }

//===----------------------------------------------------------------------===//
// Attributes for SELECT CASE
//===----------------------------------------------------------------------===//

using AttributeUniquer = mlir::detail::AttributeUniquer;

ClosedIntervalAttr fir::ClosedIntervalAttr::get(mlir::MLIRContext *ctxt) {
  return AttributeUniquer::get<ClosedIntervalAttr>(ctxt);
}

UpperBoundAttr fir::UpperBoundAttr::get(mlir::MLIRContext *ctxt) {
  return AttributeUniquer::get<UpperBoundAttr>(ctxt);
}

LowerBoundAttr fir::LowerBoundAttr::get(mlir::MLIRContext *ctxt) {
  return AttributeUniquer::get<LowerBoundAttr>(ctxt);
}

PointIntervalAttr fir::PointIntervalAttr::get(mlir::MLIRContext *ctxt) {
  return AttributeUniquer::get<PointIntervalAttr>(ctxt);
}

//===----------------------------------------------------------------------===//
// RealAttr
//===----------------------------------------------------------------------===//

RealAttr fir::RealAttr::get(mlir::MLIRContext *ctxt,
                            const RealAttr::ValueType &key) {
  return Base::get(ctxt, key);
}

KindTy fir::RealAttr::getFKind() const { return getImpl()->getFKind(); }

llvm::APFloat fir::RealAttr::getValue() const { return getImpl()->getValue(); }

//===----------------------------------------------------------------------===//
// FIR attribute parsing
//===----------------------------------------------------------------------===//

static mlir::Attribute parseFirRealAttr(FIROpsDialect *dialect,
                                        mlir::DialectAsmParser &parser,
                                        mlir::Type type) {
  int kind = 0;
  if (parser.parseLess() || parser.parseInteger(kind) || parser.parseComma()) {
    parser.emitError(parser.getNameLoc(), "expected '<' kind ','");
    return {};
  }
  KindMapping kindMap(dialect->getContext());
  llvm::APFloat value(0.);
  if (parser.parseOptionalKeyword("i")) {
    // `i` not present, so literal float must be present
    double dontCare;
    if (parser.parseFloat(dontCare) || parser.parseGreater()) {
      parser.emitError(parser.getNameLoc(), "expected real constant '>'");
      return {};
    }
    auto fltStr = parser.getFullSymbolSpec()
                      .drop_until([](char c) { return c == ','; })
                      .drop_front()
                      .drop_while([](char c) { return c == ' ' || c == '\t'; })
                      .take_until([](char c) {
                        return c == '>' || c == ' ' || c == '\t';
                      });
    value = llvm::APFloat(kindMap.getFloatSemantics(kind), fltStr);
  } else {
    // `i` is present, so literal bitstring (hex) must be present
    llvm::StringRef hex;
    if (parser.parseKeyword(&hex) || parser.parseGreater()) {
      parser.emitError(parser.getNameLoc(), "expected real constant '>'");
      return {};
    }
    auto bits = llvm::APInt(kind * 8, hex.drop_front(), 16);
    value = llvm::APFloat(kindMap.getFloatSemantics(kind), bits);
  }
  return RealAttr::get(dialect->getContext(), {kind, value});
}

mlir::Attribute fir::parseFirAttribute(FIROpsDialect *dialect,
                                       mlir::DialectAsmParser &parser,
                                       mlir::Type type) {
  auto loc = parser.getNameLoc();
  llvm::StringRef attrName;
  if (parser.parseKeyword(&attrName)) {
    parser.emitError(loc, "expected an attribute name");
    return {};
  }

  if (attrName == ExactTypeAttr::getAttrName()) {
    mlir::Type type;
    if (parser.parseLess() || parser.parseType(type) || parser.parseGreater()) {
      parser.emitError(loc, "expected a type");
      return {};
    }
    return ExactTypeAttr::get(type);
  }
  if (attrName == SubclassAttr::getAttrName()) {
    mlir::Type type;
    if (parser.parseLess() || parser.parseType(type) || parser.parseGreater()) {
      parser.emitError(loc, "expected a subtype");
      return {};
    }
    return SubclassAttr::get(type);
  }
  if (attrName == PointIntervalAttr::getAttrName())
    return PointIntervalAttr::get(dialect->getContext());
  if (attrName == LowerBoundAttr::getAttrName())
    return LowerBoundAttr::get(dialect->getContext());
  if (attrName == UpperBoundAttr::getAttrName())
    return UpperBoundAttr::get(dialect->getContext());
  if (attrName == ClosedIntervalAttr::getAttrName())
    return ClosedIntervalAttr::get(dialect->getContext());
  if (attrName == RealAttr::getAttrName())
    return parseFirRealAttr(dialect, parser, type);

  parser.emitError(loc, "unknown FIR attribute: ") << attrName;
  return {};
}

//===----------------------------------------------------------------------===//
// FIR attribute pretty printer
//===----------------------------------------------------------------------===//

void fir::printFirAttribute(FIROpsDialect *dialect, mlir::Attribute attr,
                            mlir::DialectAsmPrinter &p) {
  auto &os = p.getStream();
  if (auto exact = attr.dyn_cast<fir::ExactTypeAttr>()) {
    os << fir::ExactTypeAttr::getAttrName() << '<';
    p.printType(exact.getType());
    os << '>';
  } else if (auto sub = attr.dyn_cast<fir::SubclassAttr>()) {
    os << fir::SubclassAttr::getAttrName() << '<';
    p.printType(sub.getType());
    os << '>';
  } else if (attr.dyn_cast_or_null<fir::PointIntervalAttr>()) {
    os << fir::PointIntervalAttr::getAttrName();
  } else if (attr.dyn_cast_or_null<fir::ClosedIntervalAttr>()) {
    os << fir::ClosedIntervalAttr::getAttrName();
  } else if (attr.dyn_cast_or_null<fir::LowerBoundAttr>()) {
    os << fir::LowerBoundAttr::getAttrName();
  } else if (attr.dyn_cast_or_null<fir::UpperBoundAttr>()) {
    os << fir::UpperBoundAttr::getAttrName();
  } else if (auto a = attr.dyn_cast_or_null<fir::RealAttr>()) {
    os << fir::RealAttr::getAttrName() << '<' << a.getFKind() << ", i x";
    llvm::SmallString<40> ss;
    a.getValue().bitcastToAPInt().toStringUnsigned(ss, 16);
    os << ss << '>';
  } else {
    // don't know how to print the attribute, so use a default
    os << "<(unknown attribute)>";
  }
}

//===----------------------------------------------------------------------===//
// FIROpsDialect
//===----------------------------------------------------------------------===//

void FIROpsDialect::registerAttributes() {
  addAttributes<ClosedIntervalAttr, ExactTypeAttr, LowerBoundAttr,
                PointIntervalAttr, RealAttr, SubclassAttr, UpperBoundAttr>();
}
