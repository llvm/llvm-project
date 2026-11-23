void foo(void) {}
// CHECK-OK: trailing-comma.c:1:1: note: "root" binds here
// CHECK-ERR-COMMA: Invalid token <,> found when looking for a value.

// RUN: clang-query -c "match \
// RUN:   functionDecl( \
// RUN:     hasName( \
// RUN:       \"foo\", \
// RUN:     ), \
// RUN:   ) \
// RUN: " %s | FileCheck --check-prefix=CHECK-OK %s

// Same with \n tokens
// RUN: echo "match functionDecl( hasName( \"foo\" , ) , )" | sed "s/ /\n/g" >%t
// RUN: clang-query -f %t %s | FileCheck --check-prefix=CHECK-OK %s

// RUN: not clang-query -c "match functionDecl(hasName(\"foo\"),,)" %s \
// RUN:   | FileCheck --check-prefix=CHECK-ERR-COMMA %s

// RUN: not clang-query -c "match functionDecl(,)" %s \
// RUN:   | FileCheck --check-prefix=CHECK-ERR-COMMA %s
