//===- SerializationFormat.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Serialization/SerializationFormat.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/TUSummary/TUSummary.h"

using namespace clang::ssaf;

EntityIdTable &SerializationFormat::getIdTableForDeserialization(TUSummary &S) {
  return S.IdTable;
}

BuildNamespace &
SerializationFormat::getTUNamespaceForDeserialization(TUSummary &S) {
  return S.TUNamespace;
}

const EntityIdTable &SerializationFormat::getIdTable(const TUSummary &S) {
  return S.IdTable;
}

const BuildNamespace &SerializationFormat::getTUNamespace(const TUSummary &S) {
  return S.TUNamespace;
}

BuildNamespaceKind
SerializationFormat::getBuildNamespaceKind(const BuildNamespace &BN) {
  return BN.Kind;
}

llvm::StringRef
SerializationFormat::getBuildNamespaceName(const BuildNamespace &BN) {
  return BN.Name;
}

const std::vector<BuildNamespace> &
SerializationFormat::getNestedBuildNamespaces(const NestedBuildNamespace &NBN) {
  return NBN.Namespaces;
}

llvm::StringRef SerializationFormat::getEntityNameUSR(const EntityName &EN) {
  return EN.USR;
}

const llvm::SmallString<16> &
SerializationFormat::getEntityNameSuffix(const EntityName &EN) {
  return EN.Suffix;
}

const NestedBuildNamespace &
SerializationFormat::getEntityNameNamespace(const EntityName &EN) {
  return EN.Namespace;
}
