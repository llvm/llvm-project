//===-- BoxValue.h -- internal box values -----------------------*- C++ -*-===//
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

#ifndef FORTRAN_OPTIMIZER_BUILDER_BOXVALUE_H
#define FORTRAN_OPTIMIZER_BUILDER_BOXVALUE_H

#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/FatalError.h"
#include "flang/Optimizer/Support/Matcher.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

namespace fir {
class FirOpBuilder;
class ArrayLoadOp;

class ArrayBoxValue;
class BoxValue;
class CharBoxValue;
class CharArrayBoxValue;
class MutableBoxValue;
class PolymorphicValue;
class ProcBoxValue;

llvm::raw_ostream &operator<<(llvm::raw_ostream &, const CharBoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ArrayBoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const CharArrayBoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ProcBoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const MutableBoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const BoxValue &);
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const PolymorphicValue &);

//===----------------------------------------------------------------------===//
//
// Boxed values
//
// Define a set of containers used internally by the lowering bridge to keep
// track of extended values associated with a Fortran subexpression. These
// associations are maintained during the construction of FIR.
//
//===----------------------------------------------------------------------===//

/// Most expressions of intrinsic type can be passed unboxed. Their properties
/// are known statically.
using UnboxedValue = mlir::Value;

/// Abstract base class.
class AbstractBox {
public:
  AbstractBox() = delete;
  AbstractBox(mlir::Value addr) : addr{addr} {}

  /// An abstract box most often contains a memory reference to a value. Despite
  /// the name here, it is possible that `addr` is a scalar value that is not a
  /// memory reference.
  mlir::Value getAddr() const { return addr; }

protected:
  mlir::Value addr;
};

/// Expressions of CHARACTER type have an associated, possibly dynamic LEN
/// value.
class CharBoxValue : public AbstractBox {
public:
  CharBoxValue(mlir::Value addr, mlir::Value len)
      : AbstractBox{addr}, len{len} {
    if (addr && addr.getType().template isa<fir::BoxCharType>())
      fir::emitFatalError(addr.getLoc(),
                          "BoxChar should not be in CharBoxValue");
  }

  CharBoxValue clone(mlir::Value newBase) const { return {newBase, len}; }

  /// Convenience alias to get the memory reference to the buffer.
  mlir::Value getBuffer() const { return getAddr(); }

  mlir::Value getLen() const { return len; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const CharBoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }

protected:
  mlir::Value len;
};

/// Polymorphic value associated with a dynamic type descriptor.
class PolymorphicValue : public AbstractBox {
public:
  PolymorphicValue(mlir::Value addr, mlir::Value tdesc)
      : AbstractBox{addr}, tdesc{tdesc} {}

  PolymorphicValue clone(mlir::Value newBase) const { return {newBase, tdesc}; }

  mlir::Value getTdesc() const { return tdesc; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const PolymorphicValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }

protected:
  mlir::Value tdesc;
};

/// Abstract base class.
/// Expressions of type array have at minimum a shape. These expressions may
/// have lbound attributes (dynamic values) that affect the interpretation of
/// indexing expressions.
class AbstractArrayBox {
public:
  AbstractArrayBox() = default;
  AbstractArrayBox(llvm::ArrayRef<mlir::Value> extents,
                   llvm::ArrayRef<mlir::Value> lbounds)
      : extents{extents.begin(), extents.end()}, lbounds{lbounds.begin(),
                                                         lbounds.end()} {}

  // Every array has extents that describe its shape.
  const llvm::SmallVectorImpl<mlir::Value> &getExtents() const {
    return extents;
  }

  // An array expression may have user-defined lower bound values.
  // If this vector is empty, the default in all dimensions in `1`.
  const llvm::SmallVectorImpl<mlir::Value> &getLBounds() const {
    return lbounds;
  }

  bool lboundsAllOne() const { return lbounds.empty(); }
  std::size_t rank() const { return extents.size(); }

protected:
  llvm::SmallVector<mlir::Value, 4> extents;
  llvm::SmallVector<mlir::Value, 4> lbounds;
};

/// Expressions with rank > 0 have extents. They may also have lbounds that are
/// not 1.
class ArrayBoxValue : public PolymorphicValue, public AbstractArrayBox {
public:
  ArrayBoxValue(mlir::Value addr, llvm::ArrayRef<mlir::Value> extents,
                llvm::ArrayRef<mlir::Value> lbounds = {},
                mlir::Value tdesc = {})
      : PolymorphicValue{addr, tdesc}, AbstractArrayBox{extents, lbounds} {}

  ArrayBoxValue clone(mlir::Value newBase) const {
    return {newBase, extents, lbounds};
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ArrayBoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }
};

/// Expressions of type CHARACTER and with rank > 0.
class CharArrayBoxValue : public CharBoxValue, public AbstractArrayBox {
public:
  CharArrayBoxValue(mlir::Value addr, mlir::Value len,
                    llvm::ArrayRef<mlir::Value> extents,
                    llvm::ArrayRef<mlir::Value> lbounds = {})
      : CharBoxValue{addr, len}, AbstractArrayBox{extents, lbounds} {}

  CharArrayBoxValue clone(mlir::Value newBase) const {
    return {newBase, len, extents, lbounds};
  }

  CharBoxValue cloneElement(mlir::Value newBase) const {
    return {newBase, len};
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const CharArrayBoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }
};

/// Expressions that are procedure POINTERs may need a set of references to
/// variables in the host scope.
class ProcBoxValue : public AbstractBox {
public:
  ProcBoxValue(mlir::Value addr, mlir::Value context)
      : AbstractBox{addr}, hostContext{context} {}

  ProcBoxValue clone(mlir::Value newBase) const {
    return {newBase, hostContext};
  }

  mlir::Value getHostContext() const { return hostContext; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ProcBoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }

protected:
  mlir::Value hostContext;
};

/// Base class for values associated to a fir.box or fir.ref<fir.box>.
class AbstractIrBox : public AbstractBox, public AbstractArrayBox {
public:
  AbstractIrBox(mlir::Value addr) : AbstractBox{addr} {}
  AbstractIrBox(mlir::Value addr, llvm::ArrayRef<mlir::Value> lbounds,
                llvm::ArrayRef<mlir::Value> extents)
      : AbstractBox{addr}, AbstractArrayBox(extents, lbounds) {}
  /// Get the fir.box<type> part of the address type.
  fir::BaseBoxType getBoxTy() const {
    auto type = getAddr().getType();
    if (auto pointedTy = fir::dyn_cast_ptrEleTy(type))
      type = pointedTy;
    return type.cast<fir::BaseBoxType>();
  }
  /// Return the part of the address type after memory and box types. That is
  /// the element type, maybe wrapped in a fir.array type.
  mlir::Type getBaseTy() const {
    return fir::dyn_cast_ptrOrBoxEleTy(getBoxTy());
  }

  /// Return the memory type of the data address inside the box:
  /// - for fir.box<fir.ptr<T>>, return fir.ptr<T>
  /// - for fir.box<fir.heap<T>>, return fir.heap<T>
  /// - for fir.box<T>, return fir.ref<T>
  mlir::Type getMemTy() const {
    auto ty = getBoxTy().getEleTy();
    if (fir::isa_ref_type(ty))
      return ty;
    return fir::ReferenceType::get(ty);
  }

  /// Get the scalar type related to the described entity
  mlir::Type getEleTy() const {
    auto type = getBaseTy();
    if (auto seqTy = type.dyn_cast<fir::SequenceType>())
      return seqTy.getEleTy();
    return type;
  }

  /// Is the entity an array or an assumed rank ?
  bool hasRank() const { return getBaseTy().isa<fir::SequenceType>(); }
  /// Is this an assumed rank ?
  bool hasAssumedRank() const {
    auto seqTy = getBaseTy().dyn_cast<fir::SequenceType>();
    return seqTy && seqTy.hasUnknownShape();
  }
  /// Returns the rank of the entity. Beware that zero will be returned for
  /// both scalars and assumed rank.
  unsigned rank() const {
    if (auto seqTy = getBaseTy().dyn_cast<fir::SequenceType>())
      return seqTy.getDimension();
    return 0;
  }

  /// Is this a character entity ?
  bool isCharacter() const { return fir::isa_char(getEleTy()); }

  /// Is this a derived type entity ?
  bool isDerived() const { return getEleTy().isa<fir::RecordType>(); }

  bool isDerivedWithLenParameters() const {
    return fir::isRecordWithTypeParameters(getEleTy());
  }

  /// Is this a polymorphic entity?
  bool isPolymorphic() const { return fir::isPolymorphicType(getBoxTy()); }

  /// Is this a CLASS(*)/TYPE(*)?
  bool isUnlimitedPolymorphic() const {
    return fir::isUnlimitedPolymorphicType(getBoxTy());
  }
};

/// An entity described by a fir.box value that cannot be read into
/// another ExtendedValue category, either because the fir.box may be an
/// absent optional and we need to wait until the user is referencing it
/// to read it, or because it contains important information that cannot
/// be exposed in FIR (e.g. non contiguous byte stride).
/// It may also store explicit bounds or LEN parameters that were specified
/// for the entity.
class BoxValue : public AbstractIrBox {
public:
  BoxValue(mlir::Value addr) : AbstractIrBox{addr} { assert(verify()); }
  BoxValue(mlir::Value addr, llvm::ArrayRef<mlir::Value> lbounds,
           llvm::ArrayRef<mlir::Value> explicitParams,
           llvm::ArrayRef<mlir::Value> explicitExtents = {})
      : AbstractIrBox{addr, lbounds, explicitExtents},
        explicitParams{explicitParams.begin(), explicitParams.end()} {
    assert(verify());
  }
  // TODO: check contiguous attribute of addr
  bool isContiguous() const { return false; }

  // Replace the fir.box, keeping any non-deferred parameters.
  BoxValue clone(mlir::Value newBox) const {
    return {newBox, lbounds, explicitParams, extents};
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const BoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }

  llvm::ArrayRef<mlir::Value> getLBounds() const { return lbounds; }

  // The extents member is not guaranteed to be field for arrays. It is only
  // guaranteed to be field for explicit shape arrays. In general,
  // explicit-shape will not come as descriptors, so this field will be empty in
  // most cases. The exception are derived types with LEN parameters and
  // polymorphic dummy argument arrays. It may be possible for the explicit
  // extents to conflict with the shape information that is in the box according
  // to 15.5.2.11 sequence association rules.
  llvm::ArrayRef<mlir::Value> getExplicitExtents() const { return extents; }

  llvm::ArrayRef<mlir::Value> getExplicitParameters() const {
    return explicitParams;
  }

protected:
  // Verify constructor invariants.
  bool verify() const;

  // Only field when the BoxValue has explicit LEN parameters.
  // Otherwise, the LEN parameters are in the fir.box.
  llvm::SmallVector<mlir::Value, 2> explicitParams;
};

/// Set of variables (addresses) holding the allocatable properties. These may
/// be empty in case it is not deemed safe to duplicate the descriptor
/// information locally (For instance, a volatile allocatable will always be
/// lowered to a descriptor to preserve the integrity of the entity and its
/// associated properties. As such, all references to the entity and its
/// property will go through the descriptor explicitly.).
class MutableProperties {
public:
  bool isEmpty() const { return !addr; }
  mlir::Value addr;
  llvm::SmallVector<mlir::Value, 2> extents;
  llvm::SmallVector<mlir::Value, 2> lbounds;
  /// Only keep track of the deferred LEN parameters through variables, since
  /// they are the only ones that can change as per the deferred type parameters
  /// definition in F2018 standard section 3.147.12.2.
  /// Non-deferred values are returned by
  /// MutableBoxValue.nonDeferredLenParams().
  llvm::SmallVector<mlir::Value, 2> deferredParams;
};

/// MutableBoxValue is used for entities that are represented by the address of
/// a box. This is intended to be used for entities whose base address, shape
/// and type are not constant in the entity lifetime (e.g Allocatables and
/// Pointers).
class MutableBoxValue : public AbstractIrBox {
public:
  /// Create MutableBoxValue given the address \p addr of the box and the non
  /// deferred LEN parameters \p lenParameters. The non deferred LEN parameters
  /// must always be provided, even if they are constant and already reflected
  /// in the address type.
  MutableBoxValue(mlir::Value addr, mlir::ValueRange lenParameters,
                  MutableProperties mutableProperties)
      : AbstractIrBox(addr), lenParams{lenParameters.begin(),
                                       lenParameters.end()},
        mutableProperties{mutableProperties} {
    // Currently only accepts fir.(ref/ptr/heap)<fir.box<type>> mlir::Value for
    // the address. This may change if we accept
    // fir.(ref/ptr/heap)<fir.heap<type>> for scalar without LEN parameters.
    assert(verify() &&
           "MutableBoxValue requires mem ref to fir.box<fir.[heap|ptr]<type>>");
  }
  /// Is this a Fortran pointer ?
  bool isPointer() const {
    return getBoxTy().getEleTy().isa<fir::PointerType>();
  }
  /// Is this an allocatable ?
  bool isAllocatable() const {
    return getBoxTy().getEleTy().isa<fir::HeapType>();
  }
  // Replace the fir.ref<fir.box>, keeping any non-deferred parameters.
  MutableBoxValue clone(mlir::Value newBox) const {
    return {newBox, lenParams, mutableProperties};
  }
  /// Does this entity has any non deferred LEN parameters?
  bool hasNonDeferredLenParams() const { return !lenParams.empty(); }
  /// Return the non deferred LEN parameters.
  llvm::ArrayRef<mlir::Value> nonDeferredLenParams() const { return lenParams; }
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const MutableBoxValue &);
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this; }

  /// Set of variable is used instead of a descriptor to hold the entity
  /// properties instead of a fir.ref<fir.box<>>.
  bool isDescribedByVariables() const { return !mutableProperties.isEmpty(); }

  const MutableProperties &getMutableProperties() const {
    return mutableProperties;
  }

protected:
  /// Validate the address type form in the constructor.
  bool verify() const;
  /// Hold the non-deferred LEN parameter values  (both for characters and
  /// derived). Non-deferred LEN parameters cannot change dynamically, as
  /// opposed to deferred type parameters (3.147.12.2).
  llvm::SmallVector<mlir::Value, 2> lenParams;
  /// Set of variables holding the extents, lower bounds and
  /// base address when it is deemed safe to work with these variables rather
  /// than directly with a descriptor.
  MutableProperties mutableProperties;
};

class ExtendedValue;

/// Get the base value of an extended value. Every type of extended value has a
/// base value or is null.
mlir::Value getBase(const ExtendedValue &exv);

/// Get the LEN property value of an extended value. CHARACTER values have a LEN
/// property.
mlir::Value getLen(const ExtendedValue &exv);

/// Pretty-print an extended value.
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ExtendedValue &);

/// Return a clone of the extended value `exv` with the base value `base`
/// substituted.
ExtendedValue substBase(const ExtendedValue &exv, mlir::Value base);

/// Is the extended value `exv` an array?
bool isArray(const ExtendedValue &exv);

/// Get the type parameters for `exv`.
llvm::SmallVector<mlir::Value> getTypeParams(const ExtendedValue &exv);

//===----------------------------------------------------------------------===//
// Functions that may generate IR to recover properties from extended values.
//===----------------------------------------------------------------------===//
namespace factory {

/// Generalized function to recover dependent type parameters. This does away
/// with the distinction between deferred and non-deferred LEN type parameters
/// (Fortran definition), since that categorization is irrelevant when getting
/// all type parameters for a value of dependent type.
llvm::SmallVector<mlir::Value> getTypeParams(mlir::Location loc,
                                             FirOpBuilder &builder,
                                             const ExtendedValue &exv);

/// Specialization of get type parameters for an ArrayLoadOp. An array load must
/// either have all type parameters given as arguments or be a boxed value.
llvm::SmallVector<mlir::Value>
getTypeParams(mlir::Location loc, FirOpBuilder &builder, ArrayLoadOp load);

// The generalized function to get a vector of extents is
/// Get extents from \p box. For fir::BoxValue and
/// fir::MutableBoxValue, this will generate code to read the extents.
llvm::SmallVector<mlir::Value>
getExtents(mlir::Location loc, FirOpBuilder &builder, const ExtendedValue &box);

/// Get exactly one extent for any array-like extended value, \p exv. If \p exv
/// is not an array or has rank less then \p dim, the result will be a nullptr.
mlir::Value getExtentAtDimension(mlir::Location loc, FirOpBuilder &builder,
                                 const ExtendedValue &exv, unsigned dim);

} // namespace factory

/// An extended value is a box of values pertaining to a discrete entity. It is
/// used in lowering to track all the runtime values related to an entity. For
/// example, an entity may have an address in memory that contains its value(s)
/// as well as various attribute values that describe the shape and starting
/// indices if it is an array entity.
class ExtendedValue : public details::matcher<ExtendedValue> {
public:
  using VT =
      std::variant<UnboxedValue, CharBoxValue, ArrayBoxValue, CharArrayBoxValue,
                   ProcBoxValue, BoxValue, MutableBoxValue, PolymorphicValue>;

  ExtendedValue() : box{UnboxedValue{}} {}
  template <typename A, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<A>, ExtendedValue>>>
  constexpr ExtendedValue(A &&a) : box{std::forward<A>(a)} {
    if (const auto *b = getUnboxed()) {
      if (*b) {
        auto type = b->getType();
        if (type.template isa<fir::BoxCharType>())
          fir::emitFatalError(b->getLoc(), "BoxChar should be unboxed");
        type = fir::unwrapSequenceType(fir::unwrapRefType(type));
        if (fir::isa_char(type))
          fir::emitFatalError(b->getLoc(),
                              "character buffer should be in CharBoxValue");
      }
    }
  }

  template <typename A>
  constexpr const A *getBoxOf() const {
    return std::get_if<A>(&box);
  }

  constexpr const CharBoxValue *getCharBox() const {
    return getBoxOf<CharBoxValue>();
  }

  constexpr const UnboxedValue *getUnboxed() const {
    return getBoxOf<UnboxedValue>();
  }

  unsigned rank() const {
    return match([](const fir::UnboxedValue &box) -> unsigned { return 0; },
                 [](const fir::CharBoxValue &box) -> unsigned { return 0; },
                 [](const fir::ProcBoxValue &box) -> unsigned { return 0; },
                 [](const fir::PolymorphicValue &box) -> unsigned { return 0; },
                 [](const auto &box) -> unsigned { return box.rank(); });
  }

  bool isPolymorphic() const {
    return match([](const fir::PolymorphicValue &box) -> bool { return true; },
                 [](const fir::ArrayBoxValue &box) -> bool {
                   return box.getTdesc() ? true : false;
                 },
                 [](const auto &box) -> bool { return false; });
  }

  /// Is this an assumed size array ?
  bool isAssumedSize() const;

  /// LLVM style debugging of extended values
  LLVM_DUMP_METHOD void dump() const { llvm::errs() << *this << '\n'; }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &,
                                       const ExtendedValue &);

  const VT &matchee() const { return box; }

private:
  VT box;
};

/// Is the extended value `exv` unboxed and non-null?
inline bool isUnboxedValue(const ExtendedValue &exv) {
  return exv.match(
      [](const fir::UnboxedValue &box) { return box ? true : false; },
      [](const auto &) { return false; });
}

/// Returns the base type of \p exv. This is the type of \p exv
/// without any memory or box type. The sequence type, if any, is kept.
inline mlir::Type getBaseTypeOf(const ExtendedValue &exv) {
  return exv.match(
      [](const fir::MutableBoxValue &box) { return box.getBaseTy(); },
      [](const fir::BoxValue &box) { return box.getBaseTy(); },
      [&](const auto &) {
        return fir::unwrapRefType(fir::getBase(exv).getType());
      });
}

/// Return the scalar type of \p exv type. This removes all
/// reference, box, or sequence type from \p exv base.
inline mlir::Type getElementTypeOf(const ExtendedValue &exv) {
  return fir::unwrapSequenceType(getBaseTypeOf(exv));
}

/// Is the extended value `exv` a derived type with LEN parameters?
inline bool isDerivedWithLenParameters(const ExtendedValue &exv) {
  return fir::isRecordWithTypeParameters(getElementTypeOf(exv));
}

} // namespace fir

#endif // FORTRAN_OPTIMIZER_BUILDER_BOXVALUE_H
