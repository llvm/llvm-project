//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Test exporting the symbol: "__cxa_deleted_virtual" in macosx
// But don't expect the symbol to be exported in previous versions.
//
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.{{9|10|11|12|13}}

struct S { virtual void f() = delete; virtual ~S() {} };
int main(int, char**) {
  S *s = new S;
  delete s;

  return 0;
}
