//===- FuzzerMutate.h - Internal header for the Fuzzer ----------*- C++ -* ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// fuzzer::MutationDispatcher
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_MUTATE_H
#define LLVM_FUZZER_MUTATE_H

#include "FuzzerOptions.h"
#include "mutagen/Mutagen.h"
#include "mutagen/MutagenDispatcher.h"

namespace fuzzer {
namespace {

using mutagen::MutationDispatcher;

} // namespace

void ConfigureMutagen(unsigned int Seed, const FuzzingOptions &Options,
                      LLVMMutagenConfiguration *OutConfig);

void PrintRecommendedDictionary(MutationDispatcher &MD);

void PrintMutationSequence(MutationDispatcher &MD, bool Verbose = true);

}  // namespace fuzzer

#endif  // LLVM_FUZZER_MUTATE_H
