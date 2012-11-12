// RUN: %clang_cc1 -emit-pch -x c -o %t.pch %S/Inputs/c-index-pch.h
// RUN: %clang_cc1 -include-pch %t.pch -x c -emit-pch -o %t.ast %s
// RUN: c-index-test -test-load-tu %t.ast all | FileCheck -check-prefix=ALL %s
// ALL: FunctionDecl=foo
// ALL: VarDecl=bar
// ALL: FunctionDecl=wibble
// ALL: FunctionDecl=wonka
void wibble(int i);
void wonka(float);
