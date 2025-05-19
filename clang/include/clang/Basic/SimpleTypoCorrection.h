#ifndef LLVM_CLANG_BASIC_SIMPLETYPOCORRECTION_H
#define LLVM_CLANG_BASIC_SIMPLETYPOCORRECTION_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace clang {

class IdentifierInfo;

class SimpleTypoCorrection {
  StringRef BestCandidate;
  StringRef Typo;

  const unsigned MaxEditDistance;
  unsigned BestEditDistance;
  unsigned BestIndex;
  unsigned NextIndex;

public:
  explicit SimpleTypoCorrection(StringRef Typo)
      : BestCandidate(), Typo(Typo), MaxEditDistance((Typo.size() + 2) / 3),
        BestEditDistance(MaxEditDistance + 1), BestIndex(0), NextIndex(0) {}

  void add(const StringRef Candidate);
  void add(const char *Candidate);
  void add(const IdentifierInfo *Candidate);

  std::optional<StringRef> getCorrection() const;
  bool hasCorrection() const;
  unsigned getCorrectionIndex() const;
};
} // namespace clang

#endif // LLVM_CLANG_BASIC_SIMPLETYPOCORRECTION_H
