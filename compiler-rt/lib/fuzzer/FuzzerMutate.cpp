//===- FuzzerMutate.cpp - Mutation utilities -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Mutate utilities.
//===----------------------------------------------------------------------===//

#include "FuzzerMutate.h"
#include "FuzzerExtFunctions.h"
#include "FuzzerIO.h"
#include "FuzzerTracePC.h"
#include "FuzzerUtil.h"

namespace fuzzer {
namespace {

void FromTORC4(size_t Idx, uint32_t *A, uint32_t *B) {
  const auto &X = TPC.TORC4.Get(Idx);
  *A = X.A;
  *B = X.B;
}

void FromTORC8(size_t Idx, uint64_t *A, uint64_t *B) {
  const auto &X = TPC.TORC8.Get(Idx);
  *A = X.A;
  *B = X.B;
}

void FromTORCW(size_t Idx, const uint8_t **DataA, size_t *SizeA,
               const uint8_t **DataB, size_t *SizeB) {
  const auto &X = TPC.TORCW.Get(Idx);
  *DataA = X.A.data();
  *SizeA = X.A.size();
  *DataB = X.B.data();
  *SizeB = X.B.size();
}

void FromMMT(size_t Idx, const uint8_t **Data, size_t *Size) {
  const auto &W = TPC.MMT.Get(Idx);
  *Data = W.data();
  *Size = W.size();
}

void PrintASCII(const Word &W, const char *PrintAfter) {
  fuzzer::PrintASCII(W.data(), W.size(), PrintAfter);
}

} // namespace

void ConfigureMutagen(unsigned int Seed, const FuzzingOptions &Options,
                      LLVMMutagenConfiguration *OutConfig) {
  memset(OutConfig, 0, sizeof(*OutConfig));
  OutConfig->Seed = Seed;
  OutConfig->UseCmp = Options.UseCmp;
  OutConfig->FromTORC4 = FromTORC4;
  OutConfig->FromTORC8 = FromTORC8;
  OutConfig->FromTORCW = FromTORCW;
  OutConfig->UseMemmem = Options.UseMemmem;
  OutConfig->FromMMT = FromMMT;
  OutConfig->CustomMutator = EF->LLVMFuzzerCustomMutator;
  OutConfig->CustomCrossOver = EF->LLVMFuzzerCustomCrossOver;
  OutConfig->MSanUnpoison = EF->__msan_unpoison;
  OutConfig->MSanUnpoisonParam = EF->__msan_unpoison_param;
}

void PrintRecommendedDictionary(MutationDispatcher &MD) {
  auto RecommendedDictionary = MD.RecommendDictionary();
  if (RecommendedDictionary.empty())
    return;
  Printf("###### Recommended dictionary. ######\n");
  for (auto &DE : RecommendedDictionary) {
    assert(DE.GetW().size());
    Printf("\"");
    PrintASCII(DE.GetW(), "\"");
    Printf(" # Uses: %zd\n", DE.GetUseCount());
  }
  Printf("###### End of recommended dictionary. ######\n");
}

void PrintMutationSequence(MutationDispatcher &MD, bool Verbose) {
  const auto &MS = MD.MutationSequence();
  const auto &DS = MD.DictionaryEntrySequence();
  Printf("MS: %zd %s", MS.size(), MS.GetString(Verbose).c_str());
  if (!DS.empty())
    Printf(" DE: %s", DS.GetString(Verbose).c_str());
}

}  // namespace fuzzer
