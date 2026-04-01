//===- Location.h - AIIR Location Classes -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// These classes provide the ability to relate AIIR objects back to source
// location position information.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_IR_LOCATION_H
#define AIIR_IR_LOCATION_H

#include "aiir/IR/Attributes.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace aiir {

class Location;
class WalkResult;
class UnknownLoc;

//===----------------------------------------------------------------------===//
// LocationAttr
//===----------------------------------------------------------------------===//

/// Location objects represent source locations information in AIIR.
/// LocationAttr acts as the anchor for all Location based attributes.
class LocationAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Walk all of the locations nested directly under, and including, the
  /// current. This means that if a location is nested under a non-location
  /// attribute, it will *not* be walked by this method. This walk is performed
  /// in pre-order to get this behavior.
  WalkResult walk(function_ref<WalkResult(Location)> walkFn);

  /// Return an instance of the given location type if one is nested under the
  /// current location. Returns nullptr if one could not be found.
  template <typename T>
  T findInstanceOf() {
    T result = {};
    walk([&](auto loc) {
      if (auto typedLoc = llvm::dyn_cast<T>(loc)) {
        result = typedLoc;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

  /// Return an instance of the given location type if one is nested under the
  /// current location else return unknown location.
  template <typename T, typename UnknownT = UnknownLoc>
  LocationAttr findInstanceOfOrUnknown() {
    if (T result = findInstanceOf<T>())
      return result;
    return UnknownT::get(getContext());
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// Location
//===----------------------------------------------------------------------===//

/// This class defines the main interface for locations in AIIR and acts as a
/// non-nullable wrapper around a LocationAttr.
class Location {
public:
  Location(LocationAttr loc) : impl(loc) {
    assert(loc && "location should never be null.");
  }
  Location(const LocationAttr::ImplType *impl) : impl(impl) {
    assert(impl && "location should never be null.");
  }

  /// Return the context this location is uniqued in.
  AIIRContext *getContext() const { return impl.getContext(); }

  /// Access the impl location attribute.
  operator LocationAttr() const { return impl; }
  LocationAttr *operator->() const { return const_cast<LocationAttr *>(&impl); }

  /// Comparison operators.
  bool operator==(Location rhs) const { return impl == rhs.impl; }
  bool operator!=(Location rhs) const { return !(*this == rhs); }

  /// Print the location.
  void print(raw_ostream &os) const { impl.print(os); }
  void dump() const { impl.dump(); }

  friend ::llvm::hash_code hash_value(Location arg);

  /// Methods for supporting PointerLikeTypeTraits.
  const void *getAsOpaquePointer() const { return impl.getAsOpaquePointer(); }
  static Location getFromOpaquePointer(const void *pointer) {
    return LocationAttr(reinterpret_cast<const AttributeStorage *>(pointer));
  }

  /// Support llvm style casting.
  static bool classof(Attribute attr) { return llvm::isa<LocationAttr>(attr); }

protected:
  /// The internal backing location attribute.
  LocationAttr impl;
};

inline raw_ostream &operator<<(raw_ostream &os, const Location &loc) {
  loc.print(os);
  return os;
}

// Make Location hashable.
inline ::llvm::hash_code hash_value(Location arg) {
  return hash_value(arg.impl);
}

} // namespace aiir

//===----------------------------------------------------------------------===//
// Tablegen Attribute Declarations
//===----------------------------------------------------------------------===//

// Forward declaration for class created later.
namespace aiir::detail {
struct FileLineColRangeAttrStorage;
} // namespace aiir::detail

#define GET_ATTRDEF_CLASSES
#include "aiir/IR/BuiltinLocationAttributes.h.inc"

namespace aiir {

//===----------------------------------------------------------------------===//
// FusedLoc
//===----------------------------------------------------------------------===//

/// This class represents a fused location whose metadata is known to be an
/// instance of the given type.
template <typename MetadataT>
class FusedLocWith : public FusedLoc {
public:
  using FusedLoc::FusedLoc;

  /// Return the metadata associated with this fused location.
  MetadataT getMetadata() const {
    return llvm::cast<MetadataT>(FusedLoc::getMetadata());
  }

  /// Support llvm style casting.
  static bool classof(Attribute attr) {
    auto fusedLoc = llvm::dyn_cast<FusedLoc>(attr);
    return fusedLoc && aiir::isa_and_nonnull<MetadataT>(fusedLoc.getMetadata());
  }
};

//===----------------------------------------------------------------------===//
// FileLineColLoc
//===----------------------------------------------------------------------===//

/// An instance of this location represents a tuple of file, line number, and
/// column number. This is similar to the type of location that you get from
/// most source languages.
///
/// FileLineColLoc is a view to FileLineColRange with one line and column.
class FileLineColLoc : public FileLineColRange {
public:
  using FileLineColRange::FileLineColRange;

  static FileLineColLoc get(StringAttr filename, unsigned line,
                            unsigned column);
  static FileLineColLoc get(AIIRContext *context, StringRef fileName,
                            unsigned line, unsigned column);

  StringAttr getFilename() const;
  unsigned getLine() const;
  unsigned getColumn() const;
};

/// Returns true iff the given location is a FileLineColRange with exactly one
/// line and column.
bool isStrictFileLineColLoc(Location loc);

//===----------------------------------------------------------------------===//
// OpaqueLoc
//===----------------------------------------------------------------------===//

/// Returns an instance of opaque location which contains a given pointer to
/// an object. The corresponding AIIR location is set to UnknownLoc.
template <typename T>
inline OpaqueLoc OpaqueLoc::get(T underlyingLocation, AIIRContext *context) {
  return get(reinterpret_cast<uintptr_t>(underlyingLocation), TypeID::get<T>(),
             UnknownLoc::get(context));
}

//===----------------------------------------------------------------------===//
// SubElements
//===----------------------------------------------------------------------===//

/// Enable locations to be introspected as sub-elements.
template <>
struct AttrTypeSubElementHandler<Location> {
  static void walk(Location param, AttrTypeImmediateSubElementWalker &walker) {
    walker.walk(param);
  }
  static Location replace(Location param, AttrSubElementReplacements &attrRepls,
                          TypeSubElementReplacements &typeRepls) {
    return cast<LocationAttr>(attrRepls.take_front(1)[0]);
  }
};

} // namespace aiir

//===----------------------------------------------------------------------===//
// LLVM Utilities
//===----------------------------------------------------------------------===//

namespace llvm {

// Type hash just like pointers.
template <>
struct DenseMapInfo<aiir::Location> {
  static aiir::Location getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return aiir::Location::getFromOpaquePointer(pointer);
  }
  static aiir::Location getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return aiir::Location::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(aiir::Location val) {
    return aiir::hash_value(val);
  }
  static bool isEqual(aiir::Location LHS, aiir::Location RHS) {
    return LHS == RHS;
  }
};

/// We align LocationStorage by 8, so allow LLVM to steal the low bits.
template <>
struct PointerLikeTypeTraits<aiir::Location> {
public:
  static inline void *getAsVoidPointer(aiir::Location I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline aiir::Location getFromVoidPointer(void *P) {
    return aiir::Location::getFromOpaquePointer(P);
  }
  static constexpr int NumLowBitsAvailable =
      PointerLikeTypeTraits<aiir::Attribute>::NumLowBitsAvailable;
};

/// The constructors in aiir::Location ensure that the class is a non-nullable
/// wrapper around aiir::LocationAttr. Override default behavior and always
/// return true for isPresent().
template <>
struct ValueIsPresent<aiir::Location> {
  using UnwrappedType = aiir::Location;
  static inline bool isPresent(const aiir::Location &location) { return true; }
};

/// Add support for llvm style casts. We provide a cast between To and From if
/// From is aiir::Location or derives from it.
template <typename To, typename From>
struct CastInfo<To, From,
                std::enable_if_t<
                    std::is_same_v<aiir::Location, std::remove_const_t<From>> ||
                    std::is_base_of_v<aiir::Location, From>>>
    : DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {

  static inline bool isPossible(aiir::Location location) {
    /// Return a constant true instead of a dynamic true when casting to self or
    /// up the hierarchy. Additionally, all casting info is deferred to the
    /// wrapped aiir::LocationAttr instance stored in aiir::Location.
    return std::is_same_v<To, std::remove_const_t<From>> ||
           isa<To>(static_cast<aiir::LocationAttr>(location));
  }

  static inline To castFailed() { return To(); }

  static inline To doCast(aiir::Location location) {
    return To(location->getImpl());
  }
};

} // namespace llvm

#endif
