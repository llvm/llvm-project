//===-- Standalone implementation std::string_view --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_CPP_STRINGVIEW_H
#define LLVM_LIBC_SRC_SUPPORT_CPP_STRINGVIEW_H

#include <stddef.h>

namespace __llvm_libc {
namespace cpp {

// This is very simple alternate of the std::string_view class. There is no
// bounds check performed in any of the methods. The callers are expected to
// do the checks before invoking the methods.
//
// This class will be extended as needed in future.
class StringView {
private:
  const char *Data;
  size_t Len;

  static size_t min(size_t A, size_t B) { return A <= B ? A : B; }

  static int compareMemory(const char *Lhs, const char *Rhs, size_t Length) {
    for (size_t I = 0; I < Length; ++I)
      if (int Diff = (int)Lhs[I] - (int)Rhs[I])
        return Diff;
    return 0;
  }

public:
  // special value equal to the maximum value representable by the type
  // size_type.
  static constexpr size_t npos = -1;

  constexpr StringView() : Data(nullptr), Len(0) {}

  // Assumes Str is a null-terminated string. The length of the string does
  // not include the terminating null character.
  explicit constexpr StringView(const char *Str) : Data(Str), Len(0) {
    if (Str == nullptr)
      return;
    for (const char *D = Data; *D != '\0'; ++D, ++Len)
      ;
    if (Len == 0)
      Data = nullptr;
  }

  explicit constexpr StringView(const char *Str, size_t N)
      : Data(N ? Str : nullptr), Len(Str == nullptr ? 0 : N) {}

  // Ctor for raw literal.
  template <size_t N>
  constexpr StringView(const char (&Str)[N]) : StringView(Str, N - 1) {}

  constexpr const char *data() const { return Data; }

  // Returns the size of the StringView.
  constexpr size_t size() const { return Len; }

  // Returns whether the StringView is empty.
  constexpr bool empty() const { return Len == 0; }

  // Returns an iterator to the first character of the view.
  const char *begin() const { return Data; }

  // Returns an iterator to the character following the last character of the
  // view.
  const char *end() const { return Data + Len; }

  // Returns a const reference to the character at specified location pos.
  // No bounds checking is performed: the behavior is undefined if pos >=
  // size().
  constexpr const char &operator[](size_t Index) const { return Data[Index]; }

  /// compare - Compare two strings; the result is -1, 0, or 1 if this string
  /// is lexicographically less than, equal to, or greater than the \p Other.
  int compare(StringView Other) const {
    // Check the prefix for a mismatch.
    if (int Res = compareMemory(Data, Other.Data, min(Len, Other.Len)))
      return Res < 0 ? -1 : 1;
    // Otherwise the prefixes match, so we only need to check the lengths.
    if (Len == Other.Len)
      return 0;
    return Len < Other.Len ? -1 : 1;
  }

  // An equivalent method is not available in std::string_view.
  bool equals(StringView Other) const {
    return (Len == Other.Len &&
            compareMemory(Data, Other.Data, Other.Len) == 0);
  }

  inline bool operator==(StringView Other) const { return equals(Other); }
  inline bool operator!=(StringView Other) const { return !(*this == Other); }
  inline bool operator<(StringView Other) const { return compare(Other) == -1; }
  inline bool operator<=(StringView Other) const { return compare(Other) != 1; }
  inline bool operator>(StringView Other) const { return compare(Other) == 1; }
  inline bool operator>=(StringView Other) const {
    return compare(Other) != -1;
  }

  // Moves the start of the view forward by n characters.
  // The behavior is undefined if n > size().
  void remove_prefix(size_t N) {
    Len -= N;
    Data += N;
  }

  // Moves the end of the view back by n characters.
  // The behavior is undefined if n > size().
  void remove_suffix(size_t N) { Len -= N; }

  // An equivalent method is not available in std::string_view.
  StringView trim(const char C) const {
    StringView Copy = *this;
    while (Copy.starts_with(C))
      Copy = Copy.drop_front();
    while (Copy.ends_with(C))
      Copy = Copy.drop_back();
    return Copy;
  }

  // Check if this string starts with the given Prefix.
  bool starts_with(StringView Prefix) const {
    return Len >= Prefix.Len &&
           compareMemory(Data, Prefix.Data, Prefix.Len) == 0;
  }

  // Check if this string starts with the given Prefix.
  bool starts_with(const char Prefix) const {
    return !empty() && front() == Prefix;
  }

  // Check if this string ends with the given Prefix.
  bool ends_with(const char Suffix) const {
    return !empty() && back() == Suffix;
  }

  // Check if this string ends with the given Suffix.
  bool ends_with(StringView Suffix) const {
    return Len >= Suffix.Len &&
           compareMemory(end() - Suffix.Len, Suffix.Data, Suffix.Len) == 0;
  }

  // Return a reference to the substring from [Start, Start + N).
  //
  // Start The index of the starting character in the substring; if the index is
  // npos or greater than the length of the string then the empty substring will
  // be returned.
  //
  // N The number of characters to included in the substring. If N exceeds the
  // number of characters remaining in the string, the string suffix (starting
  // with Start) will be returned.
  StringView substr(size_t Start, size_t N = npos) const {
    Start = min(Start, Len);
    return StringView(Data + Start, min(N, Len - Start));
  }

  // Search for the first character matching the character
  //
  // Returns The index of the first character satisfying the character starting
  // from From, or npos if not found.
  size_t find_first_of(const char c, size_t From = 0) const noexcept {
    StringView S = drop_front(From);
    while (!S.empty()) {
      if (S.front() == c)
        return size() - S.size();
      S = S.drop_front();
    }
    return npos;
  }

  // Search for the last character matching the character
  //
  // Return the index of the last character equal to the |c| before End.
  size_t find_last_of(const char c, size_t End = npos) const {
    End = End > size() ? size() : End + 1;
    StringView S = drop_back(size() - End);
    while (!S.empty()) {
      if (S.back() == c)
        return S.size() - 1;
      S = S.drop_back();
    }
    return npos;
  }

  // Search for the first character satisfying the predicate Function
  //
  // Returns The index of the first character satisfying Function starting from
  // From, or npos if not found.
  template <typename F> size_t find_if(F Function, size_t From = 0) const {
    StringView S = drop_front(From);
    while (!S.empty()) {
      if (Function(S.front()))
        return size() - S.size();
      S = S.drop_front();
    }
    return npos;
  }

  // Search for the first character not satisfying the predicate Function
  // Returns The index of the first character not satisfying Function starting
  // from From, or npos if not found.
  template <typename F> size_t find_if_not(F Function, size_t From = 0) const {
    return find_if([Function](char c) { return !Function(c); }, From);
  }

  // front - Get the first character in the string.
  char front() const { return Data[0]; }

  // back - Get the last character in the string.
  char back() const { return Data[Len - 1]; }

  // Return a StringView equal to 'this' but with the first N elements
  // dropped.
  StringView drop_front(size_t N = 1) const { return substr(N); }

  // Return a StringView equal to 'this' but with the last N elements
  // dropped.
  StringView drop_back(size_t N = 1) const { return substr(0, size() - N); }

  // Return a StringView equal to 'this' but with only the first N
  // elements remaining.  If N is greater than the length of the
  // string, the entire string is returned.
  StringView take_front(size_t N = 1) const {
    if (N >= size())
      return *this;
    return drop_back(size() - N);
  }

  // Return a StringView equal to 'this' but with only the last N
  // elements remaining.  If N is greater than the length of the
  // string, the entire string is returned.
  StringView take_back(size_t N = 1) const {
    if (N >= size())
      return *this;
    return drop_front(size() - N);
  }

  // Return the longest prefix of 'this' such that every character
  // in the prefix satisfies the given predicate.
  template <typename F> StringView take_while(F Function) const {
    return substr(0, find_if_not(Function));
  }

  // Return the longest prefix of 'this' such that no character in
  // the prefix satisfies the given predicate.
  template <typename F> StringView take_until(F Function) const {
    return substr(0, find_if(Function));
  }

  // Return a StringView equal to 'this', but with all characters satisfying
  // the given predicate dropped from the beginning of the string.
  template <typename F> StringView drop_while(F Function) const {
    return substr(find_if_not(Function));
  }

  // Return a StringView equal to 'this', but with all characters not
  // satisfying the given predicate dropped from the beginning of the string.
  template <typename F> StringView drop_until(F Function) const {
    return substr(find_if(Function));
  }

  // Returns true if this StringView has the given prefix and removes that
  // prefix.
  bool consume_front(StringView Prefix) {
    if (!starts_with(Prefix))
      return false;

    *this = drop_front(Prefix.size());
    return true;
  }

  // Returns true if this StringView has the given suffix and removes that
  // suffix.
  bool consume_back(StringView Suffix) {
    if (!ends_with(Suffix))
      return false;

    *this = drop_back(Suffix.size());
    return true;
  }
};

} // namespace cpp
} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_CPP_STRINGVIEW_H
