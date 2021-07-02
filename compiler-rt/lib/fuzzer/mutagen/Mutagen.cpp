//===- Mutagen.cpp - Interface header for the mutagen -----------*- C++ -* ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Define the interface between libMutagen and its consumers.
//===----------------------------------------------------------------------===//

#include "Mutagen.h"
#include "FuzzerDefs.h"
#include "MutagenDispatcher.h"
#include <algorithm>
#include <cstdio>
#include <string>

namespace mutagen {
namespace {

MutationDispatcher *MD = nullptr;

} // namespace

MutationDispatcher *GetMutationDispatcherForTest() { return MD; }

} // namespace mutagen

using fuzzer::Unit;
using mutagen::MD;
using mutagen::MutationDispatcher;
using mutagen::Word;

extern "C" {

ATTRIBUTE_INTERFACE void
LLVMMutagenConfigure(const LLVMMutagenConfiguration *Config) {
  if (MD)
    delete MD;
  MD = new MutationDispatcher(Config);
}

ATTRIBUTE_INTERFACE void LLVMMutagenResetSequence() {
  MD->StartMutationSequence();
}

ATTRIBUTE_INTERFACE void LLVMMutagenSetCrossOverWith(const uint8_t *Data,
                                                     size_t Size) {
  static Unit CrossOverWith;
  Unit U(Data, Data + Size);
  CrossOverWith = std::move(U);
  MD->SetCrossOverWith(&CrossOverWith);
}

ATTRIBUTE_INTERFACE size_t LLVMMutagenMutate(uint8_t *Data, size_t Size,
                                             size_t Max) {
  return MD->Mutate(Data, Size, Max);
}

ATTRIBUTE_INTERFACE size_t LLVMMutagenDefaultMutate(uint8_t *Data, size_t Size,
                                                    size_t Max) {
  return MD->DefaultMutate(Data, Size, Max);
}

ATTRIBUTE_INTERFACE void LLVMMutagenRecordSequence() {
  MD->RecordSuccessfulMutationSequence();
}

ATTRIBUTE_INTERFACE size_t LLVMMutagenGetMutationSequence(int Verbose,
                                                          char *Out, size_t Max,
                                                          size_t *OutNumItems) {
  const auto &Seq = MD->MutationSequence();
  if (OutNumItems)
    *OutNumItems = Seq.size();
  return snprintf(Out, Max, "%s", Seq.GetString(Verbose).c_str());
}

ATTRIBUTE_INTERFACE void LLVMMutagenAddWordToDictionary(const uint8_t *Data,
                                                        size_t Size) {
  MD->AddWordToManualDictionary(Word(Data, std::min(Size, Word::GetMaxSize())));
}

ATTRIBUTE_INTERFACE size_t LLVMMutagenGetDictionaryEntrySequence(
    int Verbose, char *Out, size_t Max, size_t *OutNumItems) {
  const auto &Seq = MD->DictionaryEntrySequence();
  if (OutNumItems)
    *OutNumItems = Seq.size();
  return snprintf(Out, Max, "%s", Seq.GetString(Verbose).c_str());
}

ATTRIBUTE_INTERFACE size_t LLVMMutagenRecommendDictionary() {
  return MD->RecommendDictionary().size();
}

ATTRIBUTE_INTERFACE const char *
LLVMMutagenRecommendDictionaryEntry(size_t *OutUseCount) {
  return MD->RecommendDictionaryEntry(OutUseCount);
}

} // extern "C"
