//===- MutagenSequence.h - Internal header for the mutagen ------*- C++ -* ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// mutagen::Sequence
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_MUTAGEN_SEQUENCE_H
#define LLVM_FUZZER_MUTAGEN_SEQUENCE_H

#include "FuzzerDefs.h"
#include <sstream>
#include <string>

namespace mutagen {
namespace {

using fuzzer::Vector;

} // namespace

// The Sequence type bundles together a list of items, a string representation,
// and a position in that string suitable for truncating it when overly long,
// e.g. after the tenth item.
template <typename T> class Sequence {
public:
  constexpr static size_t kMaxBriefItems = 10;

  void clear() {
    Items.clear();
    Size = 0;
    Str.clear();
    Brief = 0;
  }

  bool empty() const { return Size == 0; }

  size_t size() const { return Size; }

  void push_back(T t) { Items.push_back(t); }

  typename Vector<T>::const_iterator begin() const { return Items.begin(); }
  typename Vector<T>::iterator begin() { return Items.begin(); }

  typename Vector<T>::const_iterator end() const { return Items.end(); }
  typename Vector<T>::iterator end() { return Items.end(); }

  std::string GetString(bool Verbose = true) const {
    return Verbose ? Str : Str.substr(0, Brief);
  }

  // Constructs the string representation of the sequence, using a callback that
  // converts items to strings.
  template <typename ItemCallback>
  // std::string ItemCallback(T Item);
  void SetString(ItemCallback ConvertToASCII) {
    // No change since last call.
    if (Size == Items.size())
      return;
    Size = Items.size();
    std::ostringstream OSS;
    size_t i = 0;
    for (; i < Size && i < kMaxBriefItems; i++)
      OSS << ConvertToASCII(Items[i]) << "-";
    Brief = static_cast<size_t>(OSS.tellp());
    for (; i < Size; i++)
      OSS << ConvertToASCII(Items[i]) << "-";
    Str = OSS.str();
  }

private:
  Vector<T> Items;
  size_t Size = 0;
  std::string Str;
  size_t Brief = 0;
};

template <typename T>
typename Vector<T>::const_iterator begin(const Sequence<T> &S) {
  return S.begin();
}

template <typename T> typename Vector<T>::iterator begin(Sequence<T> &S) {
  return S.begin();
}

template <typename T>
typename Vector<T>::const_iterator end(const Sequence<T> &S) {
  return S.end();
}

template <typename T> typename Vector<T>::iterator end(Sequence<T> &S) {
  return S.end();
}

} // namespace mutagen

#endif // LLVM_FUZZER_MUTAGEN_SEQUENCE_H
