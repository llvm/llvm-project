// RUN: printf -- "-DX=A\\\\\nTHIS_SHOULD_NOT_EXIST_IN_THE_OUTPUT\n" | xargs %clang -E %s | FileCheck -strict-whitespace %s

// Per GCC -D semantics, \n and anything that follows is ignored.

// CHECK: {{^START A END$}}
START X END
