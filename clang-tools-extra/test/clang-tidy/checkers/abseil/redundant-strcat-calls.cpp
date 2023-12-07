// RUN: %check_clang_tidy %s abseil-redundant-strcat-calls %t -- -- -isystem %clang_tidy_headers
#include <string>

int strlen(const char *);

namespace absl {

class string_view {
 public:
  typedef std::char_traits<char> traits_type;

  string_view();
  string_view(const char *);
  string_view(const std::string &);
  string_view(const char *, int);
  string_view(string_view, int);

  template <typename A>
  explicit operator std::basic_string<char, traits_type, A>() const;

  const char *data() const;
  int size() const;
  int length() const;
};

bool operator==(string_view A, string_view B);

struct AlphaNum {
  AlphaNum(int i);
  AlphaNum(double f);
  AlphaNum(const char *c_str);
  AlphaNum(const std::string &str);
  AlphaNum(const string_view &pc);

 private:
  AlphaNum(const AlphaNum &);
  AlphaNum &operator=(const AlphaNum &);
};

std::string StrCat();
std::string StrCat(const AlphaNum &A);
std::string StrCat(const AlphaNum &A, const AlphaNum &B);
std::string StrCat(const AlphaNum &A, const AlphaNum &B, const AlphaNum &C);
std::string StrCat(const AlphaNum &A, const AlphaNum &B, const AlphaNum &C,
                   const AlphaNum &D);

// Support 5 or more arguments
template <typename... AV>
std::string StrCat(const AlphaNum &A, const AlphaNum &B, const AlphaNum &C,
              const AlphaNum &D, const AlphaNum &E, const AV &... args);

void StrAppend(std::string *Dest, const AlphaNum &A);
void StrAppend(std::string *Dest, const AlphaNum &A, const AlphaNum &B);
void StrAppend(std::string *Dest, const AlphaNum &A, const AlphaNum &B,
                    const AlphaNum &C);
void StrAppend(std::string *Dest, const AlphaNum &A, const AlphaNum &B,
                    const AlphaNum &C, const AlphaNum &D);

// Support 5 or more arguments
template <typename... AV>
void StrAppend(std::string *Dest, const AlphaNum &A, const AlphaNum &B,
               const AlphaNum &C, const AlphaNum &D, const AlphaNum &E,
               const AV &... args);

}  // namespace absl

using absl::AlphaNum;
using absl::StrAppend;
using absl::StrCat;

void Positives() {
  std::string S = StrCat(1, StrCat("A", StrCat(1.1)));
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: multiple calls to 'absl::StrCat' can be flattened into a single call
  // CHECK-FIXES: string S = StrCat(1, "A", 1.1);

  S = StrCat(StrCat(StrCat(StrCat(StrCat(1)))));
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: multiple calls to 'absl::StrCat' can be flattened into a single call
  // CHECK-FIXES: S = StrCat(1);

  // TODO: should trigger. The issue here is that in the current
  // implementation we ignore any StrCat with StrCat ancestors. Therefore
  // inserting anything in between calls will disable triggering the deepest
  // ones.
  // s = StrCat(Identity(StrCat(StrCat(1, 2), StrCat(3, 4))));

  StrAppend(&S, 001, StrCat(1, 2, "3"), StrCat("FOO"));
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple calls to 'absl::StrCat' can be flattened into a single call
  // CHECK-FIXES: StrAppend(&S, 001, 1, 2, "3", "FOO");

  StrAppend(&S, 001, StrCat(StrCat(1, 2), "3"), StrCat("FOO"));
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: multiple calls to 'absl::StrCat' can be flattened into a single call
  // CHECK-FIXES: StrAppend(&S, 001, 1, 2, "3", "FOO");

  // Too many args. Ignore for now.
  S = StrCat(1, 2, StrCat(3, 4, 5, 6, 7), 8, 9, 10,
             StrCat(11, 12, 13, 14, 15, 16, 17, 18), 19, 20, 21, 22, 23, 24, 25,
             26, 27);
  // CHECK-MESSAGES: :[[@LINE-3]]:7: warning: multiple calls to 'absl::StrCat' can be flattened into a single call
  StrAppend(&S, StrCat(1, 2, 3, 4, 5), StrCat(6, 7, 8, 9, 10));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: multiple calls to 'absl::StrCat' can be flattened into a single call
  // CHECK-FIXES: StrAppend(&S, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

  StrCat(1, StrCat());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: multiple calls to 'absl::StrCat' can be flattened into a single call
}

void Negatives() {
  // One arg. It is used for conversion. Ignore.
  std::string S = StrCat(1);

#define A_MACRO(x, y, z) StrCat(x, y, z)
  S = A_MACRO(1, 2, StrCat("A", "B"));
}
