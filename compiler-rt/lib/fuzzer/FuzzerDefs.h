//===- FuzzerDefs.h - Internal header for the Fuzzer ------------*- C++ -* ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Basic definitions.
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_DEFS_H
#define LLVM_FUZZER_DEFS_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace fuzzer {

template <class T> T Min(T a, T b) { return a < b ? a : b; }
template <class T> T Max(T a, T b) { return a > b ? a : b; }

class Random;
struct FuzzingOptions;
class InputCorpus;
struct InputInfo;
struct ExternalFunctions;

// Global interface to functions that may or may not be available.
extern ExternalFunctions *EF;

// We are using a custom allocator to give a different symbol name to STL
// containers in order to avoid ODR violations.
template<typename T>
  class fuzzer_allocator: public std::allocator<T> {
    public:
      fuzzer_allocator() = default;

      template<class U>
      fuzzer_allocator(const fuzzer_allocator<U>&) {}

      template<class Other>
      struct rebind { typedef fuzzer_allocator<Other> other;  };
  };

template<typename T>
using Vector = std::vector<T, fuzzer_allocator<T>>;

template<typename T>
using Set = std::set<T, std::less<T>, fuzzer_allocator<T>>;

typedef Vector<uint8_t> Unit;
typedef Vector<Unit> UnitVector;

// A simple POD sized array of bytes.
template <size_t kMaxSizeT> class FixedWord {
public:
  static const size_t kMaxSize = kMaxSizeT;
  FixedWord() { memset(Data, 0, kMaxSize); }
  FixedWord(const uint8_t *B, size_t S) { Set(B, S); }

  void Set(const uint8_t *B, size_t S) {
    static_assert(kMaxSizeT <= std::numeric_limits<uint8_t>::max(),
                  "FixedWord::kMaxSizeT cannot fit in a uint8_t.");
    assert(S <= kMaxSize);
    memcpy(Data, B, S);
    Size = static_cast<uint8_t>(S);
  }

  bool operator==(const FixedWord<kMaxSize> &w) const {
    return Size == w.Size && 0 == memcmp(Data, w.Data, Size);
  }

  static size_t GetMaxSize() { return kMaxSize; }
  const uint8_t *data() const { return Data; }
  uint8_t size() const { return Size; }

private:
  uint8_t Size = 0;
  uint8_t Data[kMaxSize];
};

typedef FixedWord<64> Word;

typedef int (*UserCallback)(const uint8_t *Data, size_t Size);

int FuzzerDriver(int *argc, char ***argv, UserCallback Callback);

uint8_t *ExtraCountersBegin();
uint8_t *ExtraCountersEnd();
void ClearExtraCounters();

extern bool RunningUserCallback;

}  // namespace fuzzer

#endif  // LLVM_FUZZER_DEFS_H
