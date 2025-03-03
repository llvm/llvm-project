// RUN: %check_clang_tidy %s bugprone-unintended-char-ostream-output %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {bugprone-unintended-char-ostream-output.CastTypeName: "uint8_t"}}"

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
  // CHECK-FIXES: os << static_cast<uint8_t>(unsigned_value);

  signed char signed_value = 9;
  os << signed_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'signed char' passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<uint8_t>(signed_value);

  char char_value = 9;
  os << char_value;
}
