// RUN: clang-tidy %s -checks='-*,hicpp-signed-bitwise' -- -std=c++11
// FIXME: Make the test work in all language modes.

#include "signed-bitwise-standard-types.h"

void pure_bitmask_types() {
  // std::locale::category
  int SResult = 0;
  std::locale::category C = std::locale::category::ctype;

  SResult = std::locale::category::none | std::locale::category::collate;
  SResult|= std::locale::category::collate;
  SResult = std::locale::category::ctype & std::locale::category::monetary;
  SResult&= std::locale::category::monetary;
  SResult = std::locale::category::numeric ^ std::locale::category::time;
  SResult^= std::locale::category::time;
  SResult = std::locale::category::messages | std::locale::category::all;

  SResult = std::locale::category::all & C;
  SResult&= std::locale::category::all;
  SResult = std::locale::category::all | C;
  SResult|= std::locale::category::all;
  SResult = std::locale::category::all ^ C;
  SResult^= std::locale::category::all;

  // std::ctype_base::mask
  std::ctype_base::mask M = std::ctype_base::mask::punct;

  SResult = std::ctype_base::mask::space | std::ctype_base::mask::print;
  SResult = std::ctype_base::mask::cntrl & std::ctype_base::mask::upper;
  SResult = std::ctype_base::mask::lower ^ std::ctype_base::mask::alpha;
  SResult|= std::ctype_base::mask::digit | std::ctype_base::mask::punct;
  SResult&= std::ctype_base::mask::xdigit & std::ctype_base::mask::alnum;
  SResult^= std::ctype_base::mask::alnum ^ std::ctype_base::mask::graph;

  SResult&= std::ctype_base::mask::space & M;
  SResult|= std::ctype_base::mask::space | M;
  SResult^= std::ctype_base::mask::space ^ M;

  // std::ios_base::fmtflags
  std::ios_base::fmtflags F = std::ios_base::fmtflags::floatfield;

  SResult = std::ios_base::fmtflags::dec | std::ios_base::fmtflags::oct;
  SResult = std::ios_base::fmtflags::hex & std::ios_base::fmtflags::basefield;
  SResult = std::ios_base::fmtflags::left ^ std::ios_base::fmtflags::right;
  SResult|= std::ios_base::fmtflags::internal | std::ios_base::fmtflags::adjustfield;
  SResult&= std::ios_base::fmtflags::scientific & std::ios_base::fmtflags::fixed;
  SResult^= std::ios_base::fmtflags::floatfield ^ std::ios_base::fmtflags::boolalpha;
  SResult = std::ios_base::fmtflags::showbase | std::ios_base::fmtflags::showpoint;
  SResult = std::ios_base::fmtflags::showpos & std::ios_base::fmtflags::skipws;
  SResult = std::ios_base::fmtflags::unitbuf ^ std::ios_base::fmtflags::uppercase;

  SResult|= std::ios_base::fmtflags::unitbuf | F;
  SResult&= std::ios_base::fmtflags::unitbuf & F;
  SResult^= std::ios_base::fmtflags::unitbuf ^ F;

  // std::ios_base::iostate
  std::ios_base::iostate S = std::ios_base::iostate::goodbit;

  SResult^= std::ios_base::iostate::goodbit | std::ios_base::iostate::badbit;
  SResult|= std::ios_base::iostate::failbit & std::ios_base::iostate::eofbit;
  SResult&= std::ios_base::iostate::failbit ^ std::ios_base::iostate::eofbit;

  SResult = std::ios_base::iostate::goodbit | S;
  SResult = std::ios_base::iostate::goodbit & S;
  SResult = std::ios_base::iostate::goodbit ^ S;

  // std::ios_base::openmode
  std::ios_base::openmode B = std::ios_base::openmode::binary;

  SResult = std::ios_base::openmode::app | std::ios_base::openmode::binary;
  SResult = std::ios_base::openmode::in & std::ios_base::openmode::out;
  SResult = std::ios_base::openmode::trunc ^ std::ios_base::openmode::ate;

  SResult&= std::ios_base::openmode::trunc | B;
  SResult^= std::ios_base::openmode::trunc & B;
  SResult|= std::ios_base::openmode::trunc ^ B;
}

void still_forbidden() {
  // std::locale::category
  unsigned int UResult = 0u;
  int SResult = 0;

  SResult = std::ctype_base::mask::print ^ 8u;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  SResult = std::ctype_base::mask::cntrl | 8;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  SResult = std::ctype_base::mask::upper & 8;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  SResult = std::ctype_base::mask::lower ^ -8;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  
  // Staying within the allowed standard types is ok for bitwise assignment
  // operations.
  std::ctype_base::mask var = std::ctype_base::mask::print;
  SResult<<= std::ctype_base::mask::upper;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  SResult>>= std::ctype_base::mask::upper;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  SResult &= std::ctype_base::mask::upper;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  SResult |= std::ctype_base::mask::upper;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  SResult ^= std::ctype_base::mask::upper;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator

  UResult = std::locale::category::collate << 1u;
  UResult = std::locale::category::ctype << 1;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::locale::category::monetary >> 1u;
  UResult = std::locale::category::numeric >> 1;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator

  UResult = ~std::locale::category::messages;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a unary bitwise operator
  SResult = ~std::locale::category::all;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a unary bitwise operator

  // std::ctype_base::mask
  UResult = std::ctype_base::mask::space | 8;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ctype_base::mask::print & 8u;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ctype_base::mask::cntrl ^ -8;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator

  UResult = std::ctype_base::mask::upper << 1;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ctype_base::mask::lower << 1u;
  UResult = std::ctype_base::mask::alpha >> 1u;
  UResult = std::ctype_base::mask::digit >> 1u;

  UResult = ~std::ctype_base::mask::punct;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a unary bitwise operator
  SResult = ~std::ctype_base::mask::xdigit;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a unary bitwise operator

  // std::ios_base::fmtflags
  UResult = std::ios_base::fmtflags::dec | 1;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ios_base::fmtflags::oct & 1u;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ios_base::fmtflags::hex ^ -1;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator

  UResult = std::ios_base::fmtflags::basefield >> 1;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ios_base::fmtflags::left >> 1u;
  UResult = std::ios_base::fmtflags::right << 1;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ios_base::fmtflags::internal << 1u;

  UResult = ~std::ios_base::fmtflags::adjustfield;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a unary bitwise operator
  SResult = ~std::ios_base::fmtflags::scientific;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a unary bitwise operator

  // std::ios_base::iostate
  UResult = std::ios_base::iostate::goodbit | 8;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ios_base::iostate::badbit & 8u;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ios_base::iostate::failbit ^ -8;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator

  UResult = std::ios_base::iostate::eofbit << 1;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ios_base::iostate::goodbit << 1u;
  UResult = std::ios_base::iostate::badbit >> 1;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ios_base::iostate::failbit >> 1u;

  UResult = ~std::ios_base::iostate::eofbit;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a unary bitwise operator
  SResult = ~std::ios_base::iostate::goodbit;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a unary bitwise operator

  // std::ios_base::openmode
  UResult = std::ios_base::app | 8;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ios_base::binary & 8u;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ios_base::in ^ -8;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator

  UResult = std::ios_base::out >> 1;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ios_base::trunc >> 1u;
  UResult = std::ios_base::ate << 1;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = std::ios_base::ate << 1u;

  UResult = ~std::ios_base::openmode::app;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a unary bitwise operator
  SResult = ~std::ios_base::openmode::binary;
  // CHECK-MESSAGES: [[@LINE-1]]:13: warning: use of a signed integer operand with a unary bitwise operator
}
