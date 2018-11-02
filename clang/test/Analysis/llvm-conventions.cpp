// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.llvm.Conventions \
// RUN:   -std=c++14 -verify  %s

#include "Inputs/system-header-simulator-cxx.h"

//===----------------------------------------------------------------------===//
// Forward declarations for StringRef tests.
//===----------------------------------------------------------------------===//

using size_type = size_t;

namespace std {

template <class T>
struct numeric_limits { const static bool is_signed; };

} // end of namespace std

namespace llvm {

template <class T>
struct iterator_range;

template <class Func>
struct function_ref;

struct hash_code;

template <class T>
struct SmallVectorImpl;

struct APInt;

class StringRef {
public:
  static const size_t npos = ~size_t(0);
  using iterator = const char *;
  using const_iterator = const char *;
  using size_type = size_t;

  /*implicit*/ StringRef() = default;
  StringRef(std::nullptr_t) = delete;
  /*implicit*/ StringRef(const char *Str);
  /*implicit*/ constexpr StringRef(const char *data, size_t length);
  /*implicit*/ StringRef(const std::string &Str);

  static StringRef withNullAsEmpty(const char *data);
  iterator begin() const;
  iterator end() const;
  const unsigned char *bytes_begin() const;
  const unsigned char *bytes_end() const;
  iterator_range<const unsigned char *> bytes() const;
  const char *data() const;
  bool empty() const;
  size_t size() const;
  char front() const;
  char back() const;
  template <typename Allocator>
  StringRef copy(Allocator &A) const;
  bool equals(StringRef RHS) const;
  bool equals_lower(StringRef RHS) const;
  int compare(StringRef RHS) const;
  int compare_lower(StringRef RHS) const;
  int compare_numeric(StringRef RHS) const;
  unsigned edit_distance(StringRef Other, bool AllowReplacements = true,
                         unsigned MaxEditDistance = 0) const;
  std::string str() const;
  char operator[](size_t Index) const;
  template <typename T>
  typename std::enable_if<std::is_same<T, std::string>::value,
                          StringRef>::type &
  operator=(T &&Str) = delete;
  operator std::string() const;
  bool startswith(StringRef Prefix) const;
  bool startswith_lower(StringRef Prefix) const;
  bool endswith(StringRef Suffix) const;
  bool endswith_lower(StringRef Suffix) const;
  size_t find(char C, size_t From = 0) const;
  size_t find_lower(char C, size_t From = 0) const;
  size_t find_if(function_ref<bool(char)> F, size_t From = 0) const;
  size_t find_if_not(function_ref<bool(char)> F, size_t From = 0) const;
  size_t find(StringRef Str, size_t From = 0) const;
  size_t find_lower(StringRef Str, size_t From = 0) const;
  size_t rfind(char C, size_t From = npos) const;
  size_t rfind_lower(char C, size_t From = npos) const;
  size_t rfind(StringRef Str) const;
  size_t rfind_lower(StringRef Str) const;
  size_t find_first_of(char C, size_t From = 0) const;
  size_t find_first_of(StringRef Chars, size_t From = 0) const;
  size_t find_first_not_of(char C, size_t From = 0) const;
  size_t find_first_not_of(StringRef Chars, size_t From = 0) const;
  size_t find_last_of(char C, size_t From = npos) const;
  size_t find_last_of(StringRef Chars, size_t From = npos) const;
  size_t find_last_not_of(char C, size_t From = npos) const;
  size_t find_last_not_of(StringRef Chars, size_t From = npos) const;
  bool contains(StringRef Other) const;
  bool contains(char C) const;
  bool contains_lower(StringRef Other) const;
  bool contains_lower(char C) const;
  size_t count(char C) const;
  size_t count(StringRef Str) const;
  template <typename T>
  typename std::enable_if<std::numeric_limits<T>::is_signed, bool>::type
  getAsInteger(unsigned Radix, T &Result) const;
  template <typename T>
  typename std::enable_if<!std::numeric_limits<T>::is_signed, bool>::type
  getAsInteger(unsigned Radix, T &Result) const;
  template <typename T>
  typename std::enable_if<std::numeric_limits<T>::is_signed, bool>::type
  consumeInteger(unsigned Radix, T &Result);
  template <typename T>
  typename std::enable_if<!std::numeric_limits<T>::is_signed, bool>::type
  consumeInteger(unsigned Radix, T &Result);
  bool getAsInteger(unsigned Radix, APInt &Result) const;
  bool getAsDouble(double &Result, bool AllowInexact = true) const;
  std::string lower() const;
  std::string upper() const;
  StringRef substr(size_t Start, size_t N = npos) const;
  StringRef take_front(size_t N = 1) const;
  StringRef take_back(size_t N = 1) const;
  StringRef take_while(function_ref<bool(char)> F) const;
  StringRef take_until(function_ref<bool(char)> F) const;
  StringRef drop_front(size_t N = 1) const;
  StringRef drop_back(size_t N = 1) const;
  StringRef drop_while(function_ref<bool(char)> F) const;
  StringRef drop_until(function_ref<bool(char)> F) const;
  bool consume_front(StringRef Prefix);
  bool consume_back(StringRef Suffix);
  StringRef slice(size_t Start, size_t End) const;
  std::pair<StringRef, StringRef> split(char Separator) const;
  std::pair<StringRef, StringRef> split(StringRef Separator) const;
  std::pair<StringRef, StringRef> rsplit(StringRef Separator) const;
  void split(SmallVectorImpl<StringRef> &A,
             StringRef Separator, int MaxSplit = -1,
             bool KeepEmpty = true) const;
  void split(SmallVectorImpl<StringRef> &A, char Separator, int MaxSplit = -1,
             bool KeepEmpty = true) const;
  std::pair<StringRef, StringRef> rsplit(char Separator) const;
  StringRef ltrim(char Char) const;
  StringRef ltrim(StringRef Chars = " \t\n\v\f\r") const;
  StringRef rtrim(char Char) const;
  StringRef rtrim(StringRef Chars = " \t\n\v\f\r") const;
  StringRef trim(char Char) const;
  StringRef trim(StringRef Chars = " \t\n\v\f\r") const;
};

inline bool operator==(StringRef LHS, StringRef RHS);
inline bool operator!=(StringRef LHS, StringRef RHS);
inline bool operator<(StringRef LHS, StringRef RHS);
inline bool operator<=(StringRef LHS, StringRef RHS);
inline bool operator>(StringRef LHS, StringRef RHS);
inline bool operator>=(StringRef LHS, StringRef RHS);
inline std::string &operator+=(std::string &buffer, StringRef string);
hash_code hash_value(StringRef S);
template <typename T> struct isPodLike;
template <> struct isPodLike<StringRef> { static const bool value = true; };

} // end of namespace llvm

//===----------------------------------------------------------------------===//
// Tests for StringRef.
//===----------------------------------------------------------------------===//

void temporarayStringToStringRefAssignmentTest() {
  // TODO: Emit a warning.
  llvm::StringRef Ref = std::string("Yimmy yummy test.");
}

void assigningStringToStringRefWithLongerLifetimeTest() {
  llvm::StringRef Ref;
  {
    // TODO: Emit a warning.
    std::string TmpStr("This is a fine string.");
    Ref = TmpStr;
  }
}

std::string getTemporaryString() {
  return "One two three.";
}

void assigningTempStringFromFunctionToStringRefTest() {
  // TODO: Emit a warning.
  llvm::StringRef Ref = getTemporaryString();
}

//===----------------------------------------------------------------------===//
// Forward declaration for Clang AST nodes.
//===----------------------------------------------------------------------===//

namespace llvm {

template <class T, int Size>
struct SmallVector {};

} // end of namespace llvm

namespace clang {

struct Type;
struct Decl;
struct Stmt;
struct Attr;

} // end of namespace clang

//===----------------------------------------------------------------------===//
// Tests for Clang AST nodes.
//===----------------------------------------------------------------------===//

namespace clang {

struct Type {
  std::string str; // expected-warning{{AST class 'Type' has a field 'str' that allocates heap memory (type std::string)}}
};

} // end of namespace clang

namespace clang {

struct Decl {
  llvm::SmallVector<int, 5> Vec; // expected-warning{{AST class 'Decl' has a field 'Vec' that allocates heap memory (type llvm::SmallVector<int, 5>)}}
};

} // end of namespace clang
