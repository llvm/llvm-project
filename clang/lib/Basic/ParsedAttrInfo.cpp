//===- ParsedAttrInfo.cpp - Registry for attribute plugins ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Registry of attributes added by plugins which
// derive the ParsedAttrInfo class.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/ParsedAttrInfo.h"

using namespace clang;

LLVM_INSTANTIATE_REGISTRY(ParsedAttrInfoRegistry)
