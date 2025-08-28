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

using uint8_t = unsigned char;
using int8_t = signed char;

void origin_ostream(std::ostream &os) {
  uint8_t unsigned_value = 9;
  os << unsigned_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'uint8_t' (aka 'unsigned char') passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned int>(unsigned_value);

  int8_t signed_value = 9;
  os << signed_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'int8_t' (aka 'signed char') passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<int>(signed_value);

  char char_value = 9;
  os << char_value;
  unsigned char unsigned_char_value = 9;
  os << unsigned_char_value;
  signed char signed_char_value = 9;
  os << signed_char_value;
}

void explicit_cast_to_char_type(std::ostream &os) {
  enum V : uint8_t {};
  V e{};
  os << static_cast<unsigned char>(e);
  os << (unsigned char)(e);
  os << (static_cast<unsigned char>(e));
}

void based_on_ostream(A &os) {
  uint8_t unsigned_value = 9;
  os << unsigned_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'uint8_t' (aka 'unsigned char') passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned int>(unsigned_value);

  int8_t signed_value = 9;
  os << signed_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'int8_t' (aka 'signed char') passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<int>(signed_value);

  char char_value = 9;
  os << char_value;
}

void other_ostream_template_parameters(std::basic_ostream<uint8_t> &os) {
  uint8_t unsigned_value = 9;
  os << unsigned_value;

  int8_t signed_value = 9;
  os << signed_value;

  char char_value = 9;
  os << char_value;
}

template <class T> class B : public std::ostream {};
void template_based_on_ostream(B<int> &os) {
  uint8_t unsigned_value = 9;
  os << unsigned_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'uint8_t' (aka 'unsigned char') passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned int>(unsigned_value);
}

template<class T> void template_fn_1(T &os) {
  uint8_t unsigned_value = 9;
  os << unsigned_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'uint8_t' (aka 'unsigned char') passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned int>(unsigned_value);
}
template<class T> void template_fn_2(std::ostream &os) {
  T unsigned_value = 9;
  os << unsigned_value;
  // It should be detected as well. But we cannot get the sugared type name for SubstTemplateTypeParmType.
}
template<class T> void template_fn_3(std::ostream &os) {
  T unsigned_value = 9;
  os << unsigned_value;
}
void call_template_fn() {
  A a{};
  template_fn_1(a);
  template_fn_2<uint8_t>(a);
  template_fn_3<char>(a);
}

using C8 = char;
void alias_char(std::ostream &os) {
  C8 v = 9;
  os << v;
}


#define MACRO_VARIANT_NAME a
void macro_variant_name(std::ostream &os) {
  uint8_t MACRO_VARIANT_NAME = 9;
  os << MACRO_VARIANT_NAME;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'uint8_t' (aka 'unsigned char') passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned int>(MACRO_VARIANT_NAME);
}
