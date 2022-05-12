#include <stdio.h>
int main() {
#error This was inevitable...
#if HELLO
  printf("hello, world\n");
  return 0;
#else
  abort();
#endif
}

/* This comment gets lexed along with the input above! We just don't CHECK it.

RUN: clang-pseudo -source %s -print-directive-tree | FileCheck %s -check-prefix=PPT --strict-whitespace
     PPT: #include (7 tokens)
PPT-NEXT: code (5 tokens)
PPT-NEXT: #error (6 tokens)
PPT-NEXT: #if (3 tokens) TAKEN
PPT-NEXT:   code (8 tokens)
PPT-NEXT: #else (2 tokens)
PPT-NEXT:   code (4 tokens)
PPT-NEXT: #endif (2 tokens)
PPT-NEXT: code (2 tokens)
                ^ including this block comment

RUN: clang-pseudo -source %s -strip-directives -print-source | FileCheck %s --strict-whitespace
     CHECK: int main() {
CHECK-NEXT:   printf("hello, world\n");
CHECK-NEXT:   return 0;
CHECK-NEXT: }

RUN: clang-pseudo -source %s -strip-directives -print-tokens | FileCheck %s --check-prefix=TOKEN
     TOKEN: 0: raw_identifier 1:0 "int" flags=1
TOKEN-NEXT: raw_identifier    1:0 "main"
TOKEN-NEXT: l_paren           1:0 "("
TOKEN-NEXT: r_paren           1:0 ")"
TOKEN-NEXT: l_brace           1:0 "{"
TOKEN-NEXT: raw_identifier    4:2 "printf" flags=1
TOKEN-NEXT: l_paren           4:2 "("
TOKEN-NEXT: string_literal    4:2 "\22hello, world\\n\22"
TOKEN-NEXT: r_paren            4:2 ")"
TOKEN-NEXT: semi              4:2 ";"
TOKEN-NEXT: raw_identifier    5:2 "return" flags=1
TOKEN-NEXT: numeric_constant  5:2 "0"
TOKEN-NEXT: semi              5:2 ";"
TOKEN-NEXT: r_brace           9:0 "}" flags=1

*******************************************************************************/

