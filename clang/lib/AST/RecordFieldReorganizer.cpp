//===----- RecordFieldReorganizer.cpp - Implementation for field reorder -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation for RecordDecl field reordering.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/RecordFieldReorganizer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RandstructSeed.h"

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

// FIXME: Find a better alternative to SmallVector with hardcoded size!

namespace clang {
std::string RandstructSeed = "";
bool RandstructAutoSelect = false;

void RecordFieldReorganizer::reorganizeFields(const ASTContext &C,
                                              const RecordDecl *D) {
  // Save original fields for asserting later that a subclass hasn't
  // sabotaged the RecordDecl by removing or adding fields
  std::set<Decl *> mutateGuard;

  SmallVector<Decl *, 64> fields;
  for (auto f : D->fields()) {
    mutateGuard.insert(f);
    fields.push_back(f);
  }
  // Now allow subclass implementations to reorder the fields
  reorganize(C, D, fields);

  // Assert all fields are still present
  assert(mutateGuard.size() == fields.size() &&
         "Field count altered after reorganization");
  for (auto f : fields) {
    auto found = std::find(std::begin(mutateGuard), std::end(mutateGuard), f);
    assert(found != std::end(mutateGuard) &&
           "Unknown field encountered after reorganization");
  }

  commit(D, fields);
}
void RecordFieldReorganizer::commit(
    const RecordDecl *D, SmallVectorImpl<Decl *> &NewFieldOrder) const {
  Decl *First, *Last;
  std::tie(First, Last) = DeclContext::BuildDeclChain(
      NewFieldOrder, D->hasLoadedFieldsFromExternalStorage());
  D->FirstDecl = First;
  D->LastDecl = Last;
}

/// Bucket to store fields up to size of a cache line during randomization.
class Bucket {
public:
  virtual ~Bucket() = default;
  /// Returns a randomized version of the bucket.
  virtual SmallVector<FieldDecl *, 64> randomize(std::default_random_engine &rng);
  /// Checks if an added element would fit in a cache line.
  virtual bool canFit(size_t size) const;
  /// Adds a field to the bucket.
  void add(FieldDecl *field, size_t size);
  /// Is this bucket for bitfields?
  virtual bool isBitfieldRun() const;
  /// Is this bucket full?
  bool full() const;
  bool empty() const;

protected:
  size_t size;
  SmallVector<FieldDecl *, 64> fields;
};

/// BitfieldRun is a bucket for storing adjacent bitfields that may
/// exceed the size of a cache line.
class BitfieldRun : public Bucket {
public:
  virtual SmallVector<FieldDecl *, 64> randomize(std::default_random_engine &rng) override;
  virtual bool canFit(size_t size) const override;
  virtual bool isBitfieldRun() const override;
};

// FIXME: Is there a way to detect this? (i.e. on 32bit system vs 64?)
const size_t CACHE_LINE = 64;

SmallVector<FieldDecl *, 64> Bucket::randomize(std::default_random_engine &rng) {
  std::shuffle(std::begin(fields), std::end(fields), rng);
  return fields;
}

bool Bucket::canFit(size_t size) const {
  // We will say we can fit any size if the bucket is empty
  // because there are many instances where a field is much
  // larger than 64 bits (i.e., an array, a structure, etc)
  // but it still must be placed into a bucket.
  //
  // Otherwise, if the bucket has elements and we're still
  // trying to create a cache-line sized grouping, we cannot
  // fit a larger field in here.
  return empty() || this->size + size <= CACHE_LINE;
}

void Bucket::add(FieldDecl *field, size_t size) {
  fields.push_back(field);
  this->size += size;
}

bool Bucket::isBitfieldRun() const {
  // The normal bucket is not a bitfieldrun. This is to avoid RTTI.
  return false;
}

bool Bucket::full() const {
  // We're full if our size is a cache line.
  return size >= CACHE_LINE;
}

bool Bucket::empty() const { return size == 0; }

SmallVector<FieldDecl *, 64> BitfieldRun::randomize(std::default_random_engine &rng) {
  // Keep bit fields adjacent, we will not scramble them.
  return fields;
}

bool BitfieldRun::canFit(size_t size) const {
  // We can always fit another adjacent bitfield.
  return true;
}

bool BitfieldRun::isBitfieldRun() const { return true; }

SmallVector<Decl *, 64> Randstruct::randomize(SmallVector<Decl *, 64> fields) {
  std::shuffle(std::begin(fields), std::end(fields), rng);
  return fields;
}

SmallVector<Decl *, 64> Randstruct::perfrandomize(const ASTContext &ctx,
                                      SmallVector<Decl *, 64> fields) {
  // All of the buckets produced by best-effort cache-line algorithm.
  std::vector<std::unique_ptr<Bucket>> buckets;

  // The current bucket of fields that we are trying to fill to a cache-line.
  std::unique_ptr<Bucket> currentBucket = nullptr;
  // The current bucket containing the run of adjacent  bitfields to ensure
  // they remain adjacent.
  std::unique_ptr<Bucket> currentBitfieldRun = nullptr;

  // Tracks the number of fields that we failed to fit to the current bucket,
  // and thus still need to be added later.
  size_t skipped = 0;

  while (!fields.empty()) {
    // If we've skipped more fields than we have remaining to place,
    // that means that they can't fit in our current bucket, and we
    // need to start a new one.
    if (skipped >= fields.size()) {
      skipped = 0;
      buckets.push_back(std::move(currentBucket));
    }

    // Take the first field that needs to be put in a bucket.
    auto field = fields.begin();
    auto *f = llvm::cast<FieldDecl>(*field);

    if (f->isBitField()) {
      // Start a bitfield run if this is the first bitfield
      // we have found.
      if (!currentBitfieldRun) {
        currentBitfieldRun = llvm::make_unique<BitfieldRun>();
      }

      // We've placed the field, and can remove it from the
      // "awaiting buckets" vector called "fields"
      currentBitfieldRun->add(f, 1);
      fields.erase(field);
    } else {
      // Else, current field is not a bitfield
      // If we were previously in a bitfield run, end it.
      if (currentBitfieldRun) {
        buckets.push_back(std::move(currentBitfieldRun));
      }
      // If we don't have a bucket, make one.
      if (!currentBucket) {
        currentBucket = llvm::make_unique<Bucket>();
      }

      auto width = ctx.getTypeInfo(f->getType()).Width;

      // If we can fit, add it.
      if (currentBucket->canFit(width)) {
        currentBucket->add(f, width);
        fields.erase(field);

        // If it's now full, tie off the bucket.
        if (currentBucket->full()) {
          skipped = 0;
          buckets.push_back(std::move(currentBucket));
        }
      } else {
        // We can't fit it in our current bucket.
        // Move to the end for processing later.
        ++skipped; // Mark it skipped.
        fields.push_back(f);
        fields.erase(field);
      }
    }
  }

  // Done processing the fields awaiting a bucket.

  // If we were filling a bucket, tie it off.
  if (currentBucket) {
    buckets.push_back(std::move(currentBucket));
  }

  // If we were processing a bitfield run bucket, tie it off.
  if (currentBitfieldRun) {
    buckets.push_back(std::move(currentBitfieldRun));
  }

  std::shuffle(std::begin(buckets), std::end(buckets), rng);

  // Produce the new ordering of the elements from our buckets.
  SmallVector<Decl *, 64> finalOrder;
  for (auto &bucket : buckets) {
    auto randomized = bucket->randomize(rng);
    finalOrder.insert(finalOrder.end(), randomized.begin(), randomized.end());
  }

  return finalOrder;
}

void Randstruct::reorganize(const ASTContext &C, const RecordDecl *D,
                            SmallVector<Decl *, 64> &NewOrder) {
  SmallVector<Decl *, 64> randomized = perfrandomize(C, NewOrder);
  NewOrder = randomized;
}
bool Randstruct::isTriviallyRandomizable(const RecordDecl *D) {
  for (auto f : D->fields()){
    //If an element of the structure does not have a
    //function type is not a function pointer
    if(f->getFunctionType() == nullptr){ return false; }
  }
  return true;
}
} // namespace clang
