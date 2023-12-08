// RUN: c-index-test -write-pch %t.pch -x c %S/Inputs/c-index-pch.h
// RUN: c-index-test -write-pch %t.ast -Xclang -include-pch -Xclang %t.pch -x c %s
// RUN: c-index-test -test-load-tu %t.ast all | FileCheck -check-prefix=ALL %s
// ALL: FunctionDecl=foo
// ALL: VarDecl=bar
// ALL: FunctionDecl=wibble
// ALL: FunctionDecl=wonka
void wibble(int i);
void wonka(float);
