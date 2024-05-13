#pragma clang system_header

// Implement standard types that are known to be defined as unsigned in some
// implementations like MSVC.
namespace std {
namespace locale {
enum category : int {
  none = 0u,
  collate = 1u << 1u,
  ctype = 1u << 2u,
  monetary = 1u << 3u,
  numeric = 1u << 4u,
  time = 1u << 5u,
  messages = 1u << 6u,
  all = none | collate | ctype | monetary | numeric | time | messages
  // CHECK MESSAGES: [[@LINE-1]]:9: warning: use of a signed integer operand with a binary bitwise operator
};
} // namespace locale

namespace ctype_base {
enum mask : int {
  space,
  print,
  cntrl,
  upper,
  lower,
  alpha,
  digit,
  punct,
  xdigit,
  /* blank, // C++11 */
  alnum = alpha | digit,
  // CHECK MESSAGES: [[@LINE-1]]:11: warning: use of a signed integer operand with a binary bitwise operator
  graph = alnum | punct
  // CHECK MESSAGES: [[@LINE-1]]:11: warning: use of a signed integer operand with a binary bitwise operator
};
} // namespace ctype_base

namespace ios_base {
enum fmtflags : int {
  dec = 0u,
  oct = 1u << 2u,
  hex = 1u << 3u,
  basefield = dec | oct | hex | 0u,
  // CHECK MESSAGES: [[@LINE-1]]:15: warning: use of a signed integer operand with a binary bitwise operator
  left = 1u << 4u,
  right = 1u << 5u,
  internal = 1u << 6u,
  adjustfield = left | right | internal,
  // CHECK MESSAGES: [[@LINE-1]]:17: warning: use of a signed integer operand with a binary bitwise operator
  scientific = 1u << 7u,
  fixed = 1u << 8u,
  floatfield = scientific | fixed | (scientific | fixed) | 0u,
  // CHECK MESSAGES: [[@LINE-1]]:16: warning: use of a signed integer operand with a binary bitwise operator
  // CHECK MESSAGES: [[@LINE-2]]:38: warning: use of a signed integer operand with a binary bitwise operator
  boolalpha = 1u << 9u,
  showbase = 1u << 10u,
  showpoint = 1u << 11u,
  showpos = 1u << 12u,
  skipws = 1u << 13u,
  unitbuf = 1u << 14u,
  uppercase = 1u << 15u
};

enum iostate : int {
  goodbit = 0u,
  badbit = 1u << 1u,
  failbit = 1u << 2u,
  eofbit = 1u << 3u
};

enum openmode : int {
  app = 0u,
  binary = 0u << 1u,
  in = 0u << 2u,
  out = 0u << 3u,
  trunc = 0u << 4u,
  ate = 0u << 5u
};
} // namespace ios_base
} // namespace std
