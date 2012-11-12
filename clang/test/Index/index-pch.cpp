// RUN: c-index-test -write-pch %t.pch -fshort-wchar %s
// RUN: c-index-test -index-tu %t.pch | FileCheck %s

const wchar_t *wideStr = L"123";

// CHECK: [indexDeclaration]: kind: variable | name: wideStr
