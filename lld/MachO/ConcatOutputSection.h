//===- ConcatOutputSection.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_CONCAT_OUTPUT_SECTION_H
#define LLD_MACHO_CONCAT_OUTPUT_SECTION_H

#include "InputSection.h"
#include "OutputSection.h"
#include "Symbols.h"
#include "lld/Common/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"

namespace lld::macho {

// Linking multiple files will inevitably mean resolving sections in different
// files that are labeled with the same segment and section name. This class
// contains all such sections and writes the data from each section sequentially
// in the final binary.
class ConcatOutputSection : public OutputSection {
public:
  explicit ConcatOutputSection(StringRef name,
                               OutputSection::Kind kind = ConcatKind)
      : OutputSection(kind, name) {}

  const ConcatInputSection *firstSection() const { return inputs.front(); }
  const ConcatInputSection *lastSection() const { return inputs.back(); }
  bool isNeeded() const override { return !inputs.empty(); }

  // These accessors will only be valid after finalizing the section
  uint64_t getSize() const override { return size; }
  uint64_t getFileSize() const override { return fileSize; }

  // Assign values to InputSection::outSecOff. In contrast to TextOutputSection,
  // which does this in its implementation of `finalize()`, we can do this
  // without `finalize()`'s sequential guarantees detailed in the block comment
  // of `OutputSection::finalize()`.
  virtual void finalizeContents();

  void addInput(ConcatInputSection *input);
  void writeTo(uint8_t *buf) const override;

  static bool classof(const OutputSection *sec) {
    return sec->kind() == ConcatKind || sec->kind() == TextKind;
  }

  static ConcatOutputSection *getOrCreateForInput(const InputSection *);

  std::vector<ConcatInputSection *> inputs;

protected:
  size_t size = 0;
  uint64_t fileSize = 0;
  void finalizeOne(ConcatInputSection *);

private:
  void finalizeFlags(InputSection *input);
};

// ConcatOutputSections that contain code (text) require special handling to
// support thunk insertion.
class TextOutputSection : public ConcatOutputSection {
public:
  explicit TextOutputSection(StringRef name)
      : ConcatOutputSection(name, TextKind) {}
  void finalizeContents() override {}
  void finalize() override;
  bool needsThunks() const;
  ArrayRef<ConcatInputSection *> getThunks() const { return thunks; }
  void writeTo(uint8_t *buf) const override;

  static bool classof(const OutputSection *sec) {
    return sec->kind() == TextKind;
  }

private:
  uint64_t estimateBranchTargetThresholdVA(size_t callIdx) const;

  std::vector<ConcatInputSection *> thunks;
};

// We maintain one ThunkInfo per real function.
//
// The "active thunk" is represented by the sym/isec pair that
// turns-over during finalize(): as the call-site address advances,
// the active thunk goes out of branch-range, and we create a new
// thunk to take its place.
//
// The remaining members -- bools and counters -- apply to the
// collection of thunks associated with the real function.

struct ThunkInfo {
  // These denote the active thunk:
  Defined *sym = nullptr;             // private-extern symbol for active thunk
  ConcatInputSection *isec = nullptr; // input section for active thunk

  // The following values are cumulative across all thunks on this function
  uint32_t callSiteCount = 0;  // how many calls to the real function?
  uint32_t callSitesUsed = 0;  // how many call sites processed so-far?
  uint32_t thunkCallCount = 0; // how many call sites went to thunk?
  uint8_t sequence = 0;        // how many thunks created so-far?
};

NamePair maybeRenameSection(NamePair key);

// Output sections are added to output segments in iteration order
// of ConcatOutputSection, so must have deterministic iteration order.
extern llvm::MapVector<NamePair, ConcatOutputSection *> concatOutputSections;

// Branch-extension thunks are keyed by both the target referent and the
// branch relocation's addend.  Two call sites that branch to the same
// symbol with different addends (e.g. `bl _func` and `bl _func+8`) target
// distinct addresses and therefore need distinct thunks.
//
// After ICF, multiple Defined symbols may point to the same (isec, value)
// yet remain as distinct Symbol pointers.  The equality predicate below
// canonicalizes Defined symbols by (isec, value) so that ICF-folded copies
// still share a single thunkMap entry when their addends match.
struct ThunkKey {
  Symbol *sym;
  int64_t addend;

  static ThunkKey getEmptyKey() {
    return {llvm::DenseMapInfo<Symbol *>::getEmptyKey(), 0};
  }
  static ThunkKey getTombstoneKey() {
    return {llvm::DenseMapInfo<Symbol *>::getTombstoneKey(), 0};
  }
  bool isSentinel() const {
    return sym == llvm::DenseMapInfo<Symbol *>::getEmptyKey() ||
           sym == llvm::DenseMapInfo<Symbol *>::getTombstoneKey();
  }
  bool operator==(const ThunkKey &other) const {
    if (addend != other.addend)
      return false;
    if (sym == other.sym)
      return true;
    if (isSentinel() || other.isSentinel())
      return false;
    const auto *dl = dyn_cast<Defined>(sym);
    const auto *dr = dyn_cast<Defined>(other.sym);
    if (dl && dr)
      return dl->isec() == dr->isec() && dl->value == dr->value;
    return false;
  }
};

struct ThunkMapKeyInfo {
  static ThunkKey getEmptyKey() { return ThunkKey::getEmptyKey(); }
  static ThunkKey getTombstoneKey() { return ThunkKey::getTombstoneKey(); }
  static unsigned getHashValue(const ThunkKey &k) {
    if (k.isSentinel())
      return llvm::hash_value(k.sym);
    if (const auto *d = dyn_cast<Defined>(k.sym))
      return llvm::hash_combine(d->isec(), d->value, k.addend);
    return llvm::hash_combine(k.sym, k.addend);
  }
  static bool isEqual(const ThunkKey &lhs, const ThunkKey &rhs) {
    return lhs == rhs;
  }
};

extern llvm::DenseMap<ThunkKey, ThunkInfo, ThunkMapKeyInfo> thunkMap;

} // namespace lld::macho

#endif
