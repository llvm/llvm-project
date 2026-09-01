//===-- include/flang/Evaluate/char.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_CHAR_H_
#define FORTRAN_EVALUATE_CHAR_H_

#include "flang/Evaluate/type.h"
#include <string>

namespace Fortran::evaluate::value {

/// Simple wrapper around a std::string/std:u16string/std::u32string
template <int KIND> class Character {
  using Word = Scalar<Type<TypeCategory::Character, KIND>>;
  using CharT = typename Word::value_type;

public:
  // rule-of-five
  ~Character() = default;
  Character(const Character &v) : word_(v) {}
  Character(Character &&v) : word_(std::move(v)) {}
  Character &operator=(const Character &v) {
    word_ = v.word_;
    return &this;
  }
  Character &operator=(Character &&v) {
    word_ = std::move(v.word_);
    return *this;
  }

  // ctors
  Character() = default;
  Character(const Word &v) : word_(v) {}
  Character(Word &&v) : word_(std::move(v)) {}
  Character &operator=(const Word &v) { word_ = v; }
  Character &operator=(Word &&v) { word_ = std::move(v); }

  /// Returns the number of characters stored; not the number of bytes
  auto size() const { return word_.size(); }

  /// Reads a string of characters from \p raw. \p is the number of bytes to
  /// read; must be a multiple of the size of a single character.
  static Word FromRawBytes(const void *raw, std::size_t size) {
    CHECK(size % sizeof(CharT) == 0);
    Word s;
    if (size > 0) {
      s.assign(static_cast<const CharT *>(raw), size / sizeof(CharT));
    }
    return s;
  }

  /// Writes a string of characters to \p dst. \o is the the number of bytes to
  /// be written; must be a multiple of the size of a single character.  If the
  /// string is smaller that \p size, the rest of the memory padded with spaces.
  /// If the string is shorter than size, only the first characters are written.
  /// If \p changes points to bool, it will be set to true if any bytes at
  /// \p dst have changed.
  void StoreRawBytes(void *dst, std::size_t size, bool *changed = nullptr) {
    CHECK(size % sizeof(CharT) == 0);
    if (size > 0) {
      std::size_t payloadSize{std::min(size, sizeof(CharT) * word_.size())};
      std::size_t padSize{size - payloadSize};

      // Pad with spaces
      Word strWithPadding{word_};
      strWithPadding.append(padSize / sizeof(CharT), static_cast<CharT>(' '));

      if (changed) {
        if (std::memcmp(dst, strWithPadding.data(), size) == 0) {
          return;
        }
        *changed = true;
      }
      std::memcpy(dst, strWithPadding.data(), size);
    }
  }

private:
  Word word_;
};

} // namespace Fortran::evaluate::value
#endif // FORTRAN_EVALUATE_CHAR_H_
