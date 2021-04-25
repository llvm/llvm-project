//===- NodeIntrospection.h ------------------------------------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file contains the implementation of the NodeIntrospection.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_NODEINTROSPECTION_H
#define LLVM_CLANG_TOOLING_NODEINTROSPECTION_H

#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/DeclarationName.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include <set>

namespace clang {

class Stmt;
class Decl;
class CXXCtorInitializer;
class NestedNameSpecifierLoc;
class TemplateArgumentLoc;
class CXXBaseSpecifier;
struct DeclarationNameInfo;

namespace tooling {

class LocationCall;
using SharedLocationCall = llvm::IntrusiveRefCntPtr<LocationCall>;

class LocationCall : public llvm::ThreadSafeRefCountedBase<LocationCall> {
public:
  enum LocationCallFlags { NoFlags, ReturnsPointer, IsCast };
  LocationCall(SharedLocationCall on, std::string name,
               LocationCallFlags flags = NoFlags)
      : m_flags(flags), m_on(std::move(on)), m_name(std::move(name)) {}

  LocationCall *on() const { return m_on.get(); }
  StringRef name() const { return m_name; }
  bool returnsPointer() const { return m_flags & ReturnsPointer; }
  bool isCast() const { return m_flags & IsCast; }

private:
  LocationCallFlags m_flags;
  SharedLocationCall m_on;
  std::string m_name;
};

class LocationCallFormatterCpp {
public:
  static void print(const LocationCall &Call, llvm::raw_ostream &OS);
  static std::string format(const LocationCall &Call);
};

namespace internal {
struct RangeLessThan {
  bool operator()(std::pair<SourceRange, SharedLocationCall> const &LHS,
                  std::pair<SourceRange, SharedLocationCall> const &RHS) const;
  bool
  operator()(std::pair<SourceLocation, SharedLocationCall> const &LHS,
             std::pair<SourceLocation, SharedLocationCall> const &RHS) const;
};

} // namespace internal

// Note that this container stores unique results in a deterministic, but
// the location calls are in an unspecified order.  Clients which desire
// a particular order for the location calls, such as alphabetical,
// should sort results after retrieval, because the order is dependent
// on how the LocationCalls are formatted.
template <typename T, typename U>
using UniqueMultiMap = std::set<std::pair<T, U>, internal::RangeLessThan>;

using SourceLocationMap = UniqueMultiMap<SourceLocation, SharedLocationCall>;
using SourceRangeMap = UniqueMultiMap<SourceRange, SharedLocationCall>;

struct NodeLocationAccessors {
  SourceLocationMap LocationAccessors;
  SourceRangeMap RangeAccessors;
};

namespace NodeIntrospection {
bool hasIntrospectionSupport();
NodeLocationAccessors GetLocations(clang::Stmt const *Object);
NodeLocationAccessors GetLocations(clang::Decl const *Object);
NodeLocationAccessors GetLocations(clang::CXXCtorInitializer const *Object);
NodeLocationAccessors GetLocations(clang::NestedNameSpecifierLoc const &);
NodeLocationAccessors GetLocations(clang::TemplateArgumentLoc const &);
NodeLocationAccessors GetLocations(clang::CXXBaseSpecifier const *);
NodeLocationAccessors GetLocations(clang::TypeLoc const &);
NodeLocationAccessors GetLocations(clang::DeclarationNameInfo const &);
NodeLocationAccessors GetLocations(clang::DynTypedNode const &Node);
} // namespace NodeIntrospection
} // namespace tooling
} // namespace clang
#endif
