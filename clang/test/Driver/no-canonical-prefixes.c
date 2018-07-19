// Due to ln -sf:
// REQUIRES: shell
// RUN: mkdir -p %t.real
// RUN: cd %t.real
// RUN: ln -sf %clang test-clang
// RUN: cd ..
// RUN: ln -sf %t.real %t.fake
// RUN: cd %t.fake
// RUN: ./test-clang -v -S %s 2>&1 | FileCheck --check-prefix=CANONICAL %s
// RUN: ./test-clang -v -S %s -no-canonical-prefixes 2>&1 | FileCheck --check-prefix=NON-CANONICAL %s
//
// FIXME: This should really be '.real'.
// CANONICAL: InstalledDir: {{.*}}.fake
// CANONICAL: {{[/|\\]*}}clang{{.*}}" -cc1
//
// NON-CANONICAL: InstalledDir: .{{$}}
// NON-CANONICAL: test-clang" -cc1
