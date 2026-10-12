//===- OffloadArchOptions.cpp - options shared by offload-arch -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the command line options used by both the offload-arch
// tool and the vendor-specific GPU detection code.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"

using namespace llvm;

// Mark all our options with this category.
cl::OptionCategory OffloadArchCategory("offload-arch options");

cl::opt<bool> Verbose("verbose", cl::desc("Enable verbose output"),
                      cl::init(false), cl::cat(OffloadArchCategory));
