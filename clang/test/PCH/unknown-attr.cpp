// A retained UnknownAttr, including its interned argument text, survives PCH
// serialization and deserialization. This works because the argument is stored
// as text (a StringArgument), which TableGen serializes automatically; a source
// range would not survive being read back into a fresh compilation.

// RUN: %clang_cc1 -std=c++17 -Wno-unknown-attributes -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++17 -Wno-unknown-attributes -include-pch %t %s \
// RUN:   -ast-dump-all 2>&1 | FileCheck %s

#ifndef HEADER
#define HEADER

struct X {
  int x [[ns::transient(a, b)]];
};

#else

// CHECK: FieldDecl {{.*}} x 'int'
// CHECK-NEXT: UnknownAttr {{.*}} ns::transient "(a, b)"

#endif
