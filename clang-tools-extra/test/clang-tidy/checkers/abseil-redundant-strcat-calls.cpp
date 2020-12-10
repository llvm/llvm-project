// RUN: %check_clang_tidy %s abseil-redundant-strcat-calls %t

int strlen(const char *);

// Here we mimic the hierarchy of ::string.
// We need to do so because we are matching on the fully qualified name of the
// methods.
struct __sso_string_base {};
namespace __gnu_cxx {
template <typename A, typename B, typename C, typename D = __sso_string_base>
class __versa_string {
 public:
  const char *c_str() const;
  const char *data() const;
  int size() const;
  int capacity() const;
  int length() const;
  bool empty() const;
  char &operator[](int);
  void clear();
  void resize(int);
  int compare(const __versa_string &) const;
};
}  // namespace __gnu_cxx

namespace std {
template <typename T>
class char_traits {};
template <typename T>
class allocator {};
}  // namespace std

template <typename A, typename B = std::char_traits<A>,
          typename C = std::allocator<A>>
class basic_string : public __gnu_cxx::__versa_string<A, B, C> {
 public:
  basic_string();
  basic_string(const basic_string &);
  basic_string(const char *, C = C());
  basic_string(const char *, int, C = C());
  basic_string(const basic_string &, int, int, C = C());
  ~basic_string();

  basic_string &operator+=(const basic_string &);
};

template <typename A, typename B, typename C>
basic_string<A, B, C> operator+(const basic_string<A, B, C> &,
                                const basic_string<A, B, C> &);
template <typename A, typename B, typename C>
basic_string<A, B, C> operator+(const basic_string<A, B, C> &, const char *);

typedef basic_string<char> string;

bool operator==(const string &, const string &);
bool operator==(const string &, const char *);
bool operator==(const char *, const string &);

bool operator!=(const string &, const string &);
bool operator<(const string &, const string &);
bool operator>(const string &, const string &);
bool operator<=(const string &, const string &);
bool operator>=(const string &, const string &);

namespace std {
template <typename _CharT, typename _Traits = char_traits<_CharT>,
          typename _Alloc = allocator<_CharT>>
class basic_string;

template <typename _CharT, typename _Traits, typename _Alloc>
class basic_string {
 public:
  basic_string();
  basic_string(const basic_string &);
  basic_string(const char *, const _Alloc & = _Alloc());
  basic_string(const char *, int, const _Alloc & = _Alloc());
  basic_string(const basic_string &, int, int, const _Alloc & = _Alloc());
  ~basic_string();

  basic_string &operator+=(const basic_string &);

  unsigned size() const;
  unsigned length() const;
  bool empty() const;
};

typedef basic_string<char> string;
}  // namespace std

namespace absl {

class string_view {
 public:
  typedef std::char_traits<char> traits_type;

  string_view();
  string_view(const char *);
  string_view(const string &);
  string_view(const char *, int);
  string_view(string_view, int);

  template <typename A>
  explicit operator ::basic_string<char, traits_type, A>() const;

  const char *data() const;
  int size() const;
  int length() const;
};

bool operator==(string_view A, string_view B);

struct AlphaNum {
  AlphaNum(int i);
  AlphaNum(double f);
  AlphaNum(const char *c_str);
  AlphaNum(const string &str);
  AlphaNum(const string_view &pc);

 private:
  AlphaNum(const AlphaNum &);
  AlphaNum &operator=(const AlphaNum &);
};

string StrCat();
string StrCat(const AlphaNum &A);
string StrCat(const AlphaNum &A, const AlphaNum &B);
string StrCat(const AlphaNum &A, const AlphaNum &B, const AlphaNum &C);
string StrCat(const AlphaNum &A, const AlphaNum &B, const AlphaNum &C,
              const AlphaNum &D);

// Support 5 or more arguments
template <typename... AV>
string StrCat(const AlphaNum &A, const AlphaNum &B, const AlphaNum &C,
              const AlphaNum &D, const AlphaNum &E, const AV &... args);

void StrAppend(string *Dest, const AlphaNum &A);
void StrAppend(string *Dest, const AlphaNum &A, const AlphaNum &B);
void StrAppend(string *Dest, const AlphaNum &A, const AlphaNum &B,
               const AlphaNum &C);
void StrAppend(string *Dest, const AlphaNum &A, const AlphaNum &B,
               const AlphaNum &C, const AlphaNum &D);

// Support 5 or more arguments
template <typename... AV>
void StrAppend(string *Dest, const AlphaNum &A, const AlphaNum &B,
               const AlphaNum &C, const AlphaNum &D, const AlphaNum &E,
               const AV &... args);

}  // namespace absl

using absl::AlphaNum;
using absl::StrAppend;
using absl::StrCat;

void Positives() {
  string S = StrCat(1, StrCat("A", StrCat(1.1)));
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: multiple calls to 'absl::StrCat' can be flattened into a single call
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
  string S = StrCat(1);

#define A_MACRO(x, y, z) StrCat(x, y, z)
  S = A_MACRO(1, 2, StrCat("A", "B"));
}
