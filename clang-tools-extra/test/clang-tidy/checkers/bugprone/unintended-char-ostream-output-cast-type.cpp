// RUN: %check_clang_tidy %s bugprone-unintended-char-ostream-output %t -- \
// RUN:   -config="{CheckOptions: \
// RUN:             {bugprone-unintended-char-ostream-output.CastTypeName: \"unsigned char\"}}"

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

using uint8_t = unsigned char;
using int8_t = signed char;

void origin_ostream(std::ostream &os) {
  uint8_t unsigned_value = 9;
  os << unsigned_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'uint8_t' (aka 'unsigned char') passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned char>(unsigned_value);

  int8_t signed_value = 9;
  os << signed_value;
  // CHECK-MESSAGES: [[@LINE-1]]:6: warning: 'int8_t' (aka 'signed char') passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES: os << static_cast<unsigned char>(signed_value);

  char char_value = 9;
  os << char_value;
}
