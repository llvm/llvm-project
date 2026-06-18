//===- EntityPointerLevelFormat.h -------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_ENTITYPOINTERLEVEL_ENTITYPOINTERLEVELFORMAT_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_ENTITYPOINTERLEVEL_ENTITYPOINTERLEVELFORMAT_H

#include "clang/ScalableStaticAnalysis/Analyses/EntityPointerLevel/EntityPointerLevel.h"
#include "clang/ScalableStaticAnalysis/Core/Serialization/JSONFormat.h"

namespace clang::ssaf {
llvm::json::Value
entityPointerLevelToJSON(const EntityPointerLevel &EPL,
                         JSONFormat::EntityIdToJSONFn EntityId2JSON);

Expected<EntityPointerLevel>
entityPointerLevelFromJSON(const llvm::json::Value &EPLData,
                           JSONFormat::EntityIdFromJSONFn EntityIdFromJSON);
} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_ANALYSES_ENTITYPOINTERLEVEL_ENTITYPOINTERLEVELFORMAT_H
