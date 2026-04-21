//===- EntityPointerLevelFormat.h -------------------------------*- C++-*-===//
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

template <typename... Ts>
llvm::Error makeSawButExpectedError(const llvm::json::Value &Saw,
                                    llvm::StringRef Expected,
                                    const Ts &...ExpectedArgs) {
  std::string Fmt = ("saw %s but expected " + Expected).str();
  std::string SawStr = llvm::formatv("{0:2}", Saw).str();

  return llvm::createStringError(Fmt.c_str(), SawStr.c_str(), ExpectedArgs...);
}

namespace clang::ssaf {
llvm::json::Value
entityPointerLevelToJSON(const EntityPointerLevel &EPL,
                         JSONFormat::EntityIdToJSONFn EntityId2JSON);

Expected<EntityPointerLevel>
entityPointerLevelFromJSON(const llvm::json::Value &EPLData,
                           JSONFormat::EntityIdFromJSONFn EntityIdFromJSON);
} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_ANALYSES_ENTITYPOINTERLEVEL_ENTITYPOINTERLEVELFORMAT_H
