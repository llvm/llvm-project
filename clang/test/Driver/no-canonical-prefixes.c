// Due to ln -sf:
// REQUIRES: shell
// RUN: mkdir -p %t.real
// RUN: cd %t.real
// RUN: ln -sf %clang test-clang
// RUN: cd ..
// Important to remove %t.fake: If it already is a symlink to %t.real when
// `ln -sf %t.real %t.fake` runs, then that would symlink %t.real to itself,
// forming a cycle.
// RUN: rm -f %t.fake
// RUN: ln -sf %t.real %t.fake
// RUN: cd %t.fake
// RUN: ./test-clang -v -S %s 2>&1 \
// RUN:     | FileCheck --check-prefix=CANONICAL %s
// RUN: ./test-clang -v -S %s 2>&1 \
// RUN:     -no-canonical-prefixes \
// RUN:     | FileCheck --check-prefix=NON-CANONICAL %s
// RUN: ./test-clang -v -S %s 2>&1 \
// RUN:     -no-canonical-prefixes \
// RUN:     -canonical-prefixes \
// RUN:     | FileCheck --check-prefix=CANONICAL %s
// RUN: ./test-clang -v -S %s 2>&1 \
// RUN:     -no-canonical-prefixes \
// RUN:     -canonical-prefixes \
// RUN:     -no-canonical-prefixes \
// RUN:     | FileCheck --check-prefix=NON-CANONICAL %s

// FIXME: This should really be '.real'.
// CANONICAL: InstalledDir: {{.*}}.fake
// CANONICAL: {{[/|\\]*}}clang{{.*}}" -cc1

// NON-CANONICAL: InstalledDir: .{{$}}
// NON-CANONICAL: test-clang" -cc1
