// RUN: c-index-test -write-pch %t.pch -fwchar-type=short -fno-signed-wchar %s
// RUN: env LIBCLANG_NOTHREADS=1 c-index-test -index-tu %t.pch | FileCheck %s

// CHECK: [indexDeclaration]: kind: variable | name: wideStr
const wchar_t *wideStr = L"123";

// CHECK: [indexDeclaration]: kind: struct | name: __is_void
struct __is_void {};
