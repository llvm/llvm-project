// RUN: grep -Ev "// *[A-Z0-9_]+:" %s \
// RUN:   | clang-format -style=LLVM -assume-filename=foo.js \
// RUN:   | FileCheck -strict-whitespace -check-prefix=CHECK1 %s
// RUN: grep -Ev "// *[A-Z0-9_]+:" %s \
// RUN:   | clang-format -style=LLVM -assume-filename=foo.cpp \
// RUN:   | FileCheck -strict-whitespace -check-prefix=CHECK2 %s
// CHECK1: {{^a >>>= b;$}}
// CHECK2: {{^a >> >= b;$}}
a >>>= b;
