//===-- cpp11-migrate/Transforms.cpp - class Transforms Impl ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the implementation for class Transforms.
///
//===----------------------------------------------------------------------===//

#include "Core/Transforms.h"
#include "Core/Transform.h"

namespace cl = llvm::cl;

Transforms::~Transforms() {
  for (std::vector<Transform*>::iterator I = ChosenTransforms.begin(),
       E = ChosenTransforms.end(); I != E; ++I) {
    delete *I;
  }
  for (OptionVec::iterator I = Options.begin(),
       E = Options.end(); I != E; ++I) {
    delete I->first;
  }
}

void Transforms::registerTransform(llvm::StringRef OptName,
                                   llvm::StringRef Description,
                                   TransformCreator Creator) {
  Options.push_back(OptionVec::value_type(
      new cl::opt<bool>(OptName.data(), cl::desc(Description.data())), Creator));
}

void Transforms::createSelectedTransforms() {
  for (OptionVec::iterator I = Options.begin(),
       E = Options.end(); I != E; ++I) {
    if (*I->first) {
      ChosenTransforms.push_back(I->second());
    }
  }
}
