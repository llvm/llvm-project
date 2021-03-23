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

#include <memory>
#include <set>

namespace clang {

class Stmt;
class Decl;

namespace tooling {

class LocationCall {
public:
  enum LocationCallFlags { NoFlags, ReturnsPointer, IsCast };
  LocationCall(std::shared_ptr<LocationCall> on, std::string name,
               LocationCallFlags flags = NoFlags)
      : m_on(on), m_name(name), m_flags(flags) {}
  LocationCall(std::shared_ptr<LocationCall> on, std::string name,
               std::vector<std::string> const &args,
               LocationCallFlags flags = NoFlags)
      : m_on(on), m_name(name), m_flags(flags) {}

  LocationCall *on() const { return m_on.get(); }
  StringRef name() const { return m_name; }
  std::vector<std::string> const &args() const { return m_args; }
  bool returnsPointer() const { return m_flags & ReturnsPointer; }
  bool isCast() const { return m_flags & IsCast; }

private:
  std::shared_ptr<LocationCall> m_on;
  std::string m_name;
  std::vector<std::string> m_args;
  LocationCallFlags m_flags;
};

class LocationCallFormatterCpp {
public:
  static std::string format(LocationCall *Call);
};

namespace internal {
struct RangeLessThan {
  bool operator()(
      std::pair<SourceRange, std::shared_ptr<LocationCall>> const &LHS,
      std::pair<SourceRange, std::shared_ptr<LocationCall>> const &RHS) const;
};
} // namespace internal

template <typename T, typename U, typename Comp = std::less<std::pair<T, U>>>
using UniqueMultiMap = std::set<std::pair<T, U>, Comp>;

using SourceLocationMap =
    UniqueMultiMap<SourceLocation, std::shared_ptr<LocationCall>>;
using SourceRangeMap =
    UniqueMultiMap<SourceRange, std::shared_ptr<LocationCall>,
                   internal::RangeLessThan>;

struct NodeLocationAccessors {
  SourceLocationMap LocationAccessors;
  SourceRangeMap RangeAccessors;
};

namespace NodeIntrospection {
NodeLocationAccessors GetLocations(clang::Stmt const *Object);
NodeLocationAccessors GetLocations(clang::Decl const *Object);
NodeLocationAccessors GetLocations(clang::DynTypedNode const &Node);
} // namespace NodeIntrospection
} // namespace tooling
} // namespace clang
#endif
