int is_debug() {
#ifndef NDEBUG
  return 1; // in debug mode
#else
  return 0;
#endif
}

/* This comment gets lexed along with the input above! We just don't CHECK it.

RUN: clang-pseudo -source %s -print-source | FileCheck %s -check-prefix=SOURCE --strict-whitespace
     SOURCE: int is_debug() {
SOURCE-NEXT: #ifndef NDEBUG
SOURCE-NEXT:   return 1; // in debug mode
SOURCE-NEXT: #else
SOURCE-NEXT:  return 0;
SOURCE-NEXT: #end
SOURCE-NEXT: }

RUN: clang-pseudo -source %s -print-tokens | FileCheck %s -check-prefix=TOKEN
     TOKEN:   0: raw_identifier   0:0 "int" flags=1
TOKEN-NEXT: raw_identifier   0:0 "is_debug"
TOKEN-NEXT: l_paren          0:0 "("
TOKEN-NEXT: r_paren          0:0 ")"
TOKEN-NEXT: l_brace          0:0 "{"
TOKEN-NEXT: hash             1:0 "#" flags=1
TOKEN-NEXT: raw_identifier   1:0 "ifndef"
TOKEN-NEXT: raw_identifier   1:0 "NDEBUG"
TOKEN-NEXT: raw_identifier   2:2 "return" flags=1
TOKEN-NEXT: numeric_constant 2:2 "1"
TOKEN-NEXT: semi             2:2 ";"
TOKEN-NEXT: comment          2:2 "// in debug mode"
TOKEN-NEXT: hash             3:0 "#" flags=1
TOKEN-NEXT: raw_identifier   3:0 "else"
TOKEN-NEXT: raw_identifier   4:2 "return" flags=1
TOKEN-NEXT: numeric_constant 4:2 "0"
TOKEN-NEXT: semi             4:2 ";"
TOKEN-NEXT: hash             5:0 "#" flags=1
TOKEN-NEXT: raw_identifier   5:0 "endif"
TOKEN-NEXT: r_brace          6:0 "}" flags=1

RUN: clang-pseudo -source %s -print-pp-structure | FileCheck %s -check-prefix=PPS --strict-whitespace
     PPS: code (5 tokens)
PPS-NEXT: #ifndef (3 tokens)
PPS-NEXT:   code (4 tokens)
PPS-NEXT: #else (2 tokens)
PPS-NEXT:   code (3 tokens)
PPS-NEXT: #endif (2 tokens)
PPS-NEXT: code (2 tokens)
                ^ including this block comment

*******************************************************************************/
