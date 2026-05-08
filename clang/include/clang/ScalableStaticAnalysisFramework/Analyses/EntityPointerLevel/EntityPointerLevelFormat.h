//===- EntityPointerLevelFormat.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_ENTITYPOINTERLEVEL_ENTITYPOINTERLEVELFORMAT_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_ENTITYPOINTERLEVEL_ENTITYPOINTERLEVELFORMAT_H

#include "clang/ScalableStaticAnalysisFramework/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysisFramework/Core/Serialization/JSONFormat.h"
#include "llvm/ADT/iterator_range.h"

namespace clang::ssaf {
llvm::json::Value
entityPointerLevelToJSON(const EntityPointerLevel &EPL,
                         JSONFormat::EntityIdToJSONFn EntityId2JSON);

Expected<EntityPointerLevel>
entityPointerLevelFromJSON(const llvm::json::Value &EPLData,
                           JSONFormat::EntityIdFromJSONFn EntityIdFromJSON);

llvm::json::Array entityPointerLevelSetToJSON(
    llvm::iterator_range<EntityPointerLevelSet::const_iterator> EPLs,
    JSONFormat::EntityIdToJSONFn EntityId2JSON);

Expected<EntityPointerLevelSet>
entityPointerLevelSetFromJSON(const llvm::json::Array &EPLsData,
                              JSONFormat::EntityIdFromJSONFn EntityIdFromJSON);
} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_ENTITYPOINTERLEVEL_ENTITYPOINTERLEVELFORMAT_H
