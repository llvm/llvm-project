//===--- Quality.cpp --------------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
#include "Quality.h"
#include "index/Index.h"
#include "clang/Sema/CodeCompleteConsumer.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace clangd {
using namespace llvm;

void SymbolQualitySignals::merge(const CodeCompletionResult &SemaCCResult) {
  SemaCCPriority = SemaCCResult.Priority;

  if (SemaCCResult.Availability == CXAvailability_Deprecated)
    Deprecated = true;
}

void SymbolQualitySignals::merge(const Symbol &IndexResult) {
  References = std::max(IndexResult.References, References);
}

float SymbolQualitySignals::evaluate() const {
  float Score = 1;

  // This avoids a sharp gradient for tail symbols, and also neatly avoids the
  // question of whether 0 references means a bad symbol or missing data.
  if (References >= 3)
    Score *= std::log(References);

  if (SemaCCPriority)
    // Map onto a 0-2 interval, so we don't reward/penalize non-Sema results.
    // Priority 80 is a really bad score.
    Score *= 2 - std::min<float>(80, SemaCCPriority) / 40;

  if (Deprecated)
    Score *= 0.1;

  return Score;
}

raw_ostream &operator<<(raw_ostream &OS, const SymbolQualitySignals &S) {
  OS << formatv("=== Symbol quality: {0}\n", S.evaluate());
  if (S.SemaCCPriority)
    OS << formatv("\tSemaCCPriority: {0}\n", S.SemaCCPriority);
  OS << formatv("\tReferences: {0}\n", S.References);
  OS << formatv("\tDeprecated: {0}\n", S.Deprecated);
  return OS;
}

void SymbolRelevanceSignals::merge(const CodeCompletionResult &SemaCCResult) {
  if (SemaCCResult.Availability == CXAvailability_NotAvailable ||
      SemaCCResult.Availability == CXAvailability_NotAccessible)
    Forbidden = true;
}

float SymbolRelevanceSignals::evaluate() const {
  if (Forbidden)
    return 0;
  return NameMatch;
}
raw_ostream &operator<<(raw_ostream &OS, const SymbolRelevanceSignals &S) {
  OS << formatv("=== Symbol relevance: {0}\n", S.evaluate());
  OS << formatv("\tName match: {0}\n", S.NameMatch);
  OS << formatv("\tForbidden: {0}\n", S.Forbidden);
  return OS;
}

float evaluateSymbolAndRelevance(float SymbolQuality, float SymbolRelevance) {
  return SymbolQuality * SymbolRelevance;
}

// Produces an integer that sorts in the same order as F.
// That is: a < b <==> encodeFloat(a) < encodeFloat(b).
static uint32_t encodeFloat(float F) {
  static_assert(std::numeric_limits<float>::is_iec559, "");
  constexpr uint32_t TopBit = ~(~uint32_t{0} >> 1);

  // Get the bits of the float. Endianness is the same as for integers.
  uint32_t U = FloatToBits(F);
  // IEEE 754 floats compare like sign-magnitude integers.
  if (U & TopBit)    // Negative float.
    return 0 - U;    // Map onto the low half of integers, order reversed.
  return U + TopBit; // Positive floats map onto the high half of integers.
}

std::string sortText(float Score, llvm::StringRef Name) {
  // We convert -Score to an integer, and hex-encode for readability.
  // Example: [0.5, "foo"] -> "41000000foo"
  std::string S;
  llvm::raw_string_ostream OS(S);
  write_hex(OS, encodeFloat(-Score), llvm::HexPrintStyle::Lower,
            /*Width=*/2 * sizeof(Score));
  OS << Name;
  OS.flush();
  return S;
}

} // namespace clangd
} // namespace clang
