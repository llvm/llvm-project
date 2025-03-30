// RUN: %check_clang_tidy %s bugprone-unintended-char-ostream-output %t -check-suffix=WARN-EXPLICIT-CAST
// RUN: %check_clang_tidy %s bugprone-unintended-char-ostream-output %t \
// RUN:   -config='{CheckOptions: { \
// RUN:       bugprone-unintended-char-ostream-output.WarnOnExplicitCast: false, \
// RUN:   }}' -check-suffix=IGNORE-EXPLICIT-CAST --

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

void gh133425(std::ostream os) {
  int v = 10;
  os << static_cast<unsigned char>(v);
  // CHECK-MESSAGES-WARN-EXPLICIT-CAST: [[@LINE-1]]:6: warning: 'unsigned char' passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES-WARN-EXPLICIT-CAST: os << static_cast<unsigned int>(static_cast<unsigned char>(v));
  os << (unsigned char)(v);
  // CHECK-MESSAGES-WARN-EXPLICIT-CAST: [[@LINE-1]]:6: warning: 'unsigned char' passed to 'operator<<' outputs as character instead of integer
  // CHECK-FIXES-WARN-EXPLICIT-CAST: os << static_cast<unsigned int>((unsigned char)(v));
}
