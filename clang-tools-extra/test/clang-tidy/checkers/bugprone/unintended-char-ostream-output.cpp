// RUN: %check_clang_tidy %s bugprone-unintended-char-ostream-output %t

namespace std {

template <class _CharT, class _Traits = void> class basic_ostream {
public:
  basic_ostream &operator<<(int);
  basic_ostream &operator<<(unsigned int);
};

template <class CharT, class Traits>
basic_ostream<CharT, Traits> &operator<<(basic_ostream<CharT, Traits> &, CharT);
template <class CharT, class Traits>
basic_ostream<CharT, Traits> &operator<<(basic_ostream<CharT, Traits> &, char);
template <class _Traits>
basic_ostream<char, _Traits> &operator<<(basic_ostream<char, _Traits> &, char);
template <class _Traits>
basic_ostream<char, _Traits> &operator<<(basic_ostream<char, _Traits> &,
                                          signed char);
template <class _Traits>
basic_ostream<char, _Traits> &operator<<(basic_ostream<char, _Traits> &,
                                          unsigned char);

using ostream = basic_ostream<char>;

} // namespace std

class A : public std::ostream {};

void origin_ostream(std::ostream &os) {
  unsigned char unsigned_value = 9;
  os << unsigned_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'unsigned char' passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned int>(unsigned_value);

  signed char signed_value = 9;
  os << signed_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'signed char' passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<int>(signed_value);

  char char_value = 9;
  os << char_value;
}

void based_on_ostream(A &os) {
  unsigned char unsigned_value = 9;
  os << unsigned_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'unsigned char' passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned int>(unsigned_value);

  signed char signed_value = 9;
  os << signed_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'signed char' passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<int>(signed_value);

  char char_value = 9;
  os << char_value;
}

void based_on_ostream(std::basic_ostream<unsigned char> &os) {
  unsigned char unsigned_value = 9;
  os << unsigned_value;

  signed char signed_value = 9;
  os << signed_value;

  char char_value = 9;
  os << char_value;
}

template <class T> class B : public std::ostream {};
void template_based_on_ostream(B<int> &os) {
  unsigned char unsigned_value = 9;
  os << unsigned_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'unsigned char' passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned int>(unsigned_value);
}

template<class T> void template_fn_1(T &os) {
  unsigned char unsigned_value = 9;
  os << unsigned_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'unsigned char' passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned int>(unsigned_value);
}
template<class T> void template_fn_2(std::ostream &os) {
  T unsigned_value = 9;
  os << unsigned_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'unsigned char' passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned int>(unsigned_value);
}
template<class T> void template_fn_3(std::ostream &os) {
  T unsigned_value = 9;
  os << unsigned_value;
}
void call_template_fn() {
  A a{};
  template_fn_1(a);
  template_fn_2<unsigned char>(a);
  template_fn_3<char>(a);
}

using U8 = unsigned char;
void alias_unsigned_char(std::ostream &os) {
  U8 v = 9;
  os << v;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'U8' (aka 'unsigned char') passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned int>(v);
}

using I8 = signed char;
void alias_signed_char(std::ostream &os) {
  I8 v = 9;
  os << v;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'I8' (aka 'signed char') passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<int>(v);
}

using C8 = char;
void alias_char(std::ostream &os) {
  C8 v = 9;
  os << v;
}


#define MACRO_VARIANT_NAME a
void macro_variant_name(std::ostream &os) {
  unsigned char MACRO_VARIANT_NAME = 9;
  os << MACRO_VARIANT_NAME;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'unsigned char' passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned int>(MACRO_VARIANT_NAME);
}
