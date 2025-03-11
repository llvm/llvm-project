// Test the current working directory C APIs.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: c-index-test core -scan-deps -working-dir %S -- %clang \
// RUN:   -c %t/main.c -fmodules -fmodules-cache-path=%t/module-cache \
// RUN:   2>&1 > %t/no_cwd_opt.txt
// RUN: cat %t/no_cwd_opt.txt | FileCheck %s --check-prefix=NO-CWD-OPT


// RUN: c-index-test core -scan-deps -working-dir %S -optimize-cwd -- \
// RUN:   %clang \
// RUN:   -c %t/main.c -fmodules -fmodules-cache-path=%t/module-cache \
// RUN:   2>&1 > %t/cwd_opt.txt
// RUN: cat %t/cwd_opt.txt | FileCheck %s --check-prefix=CWD-OPT

//--- module.modulemap
module Mod { header "Mod.h" }

//--- Mod.h
int foo();

//--- main.c
#include "Mod.h"

int main() {
  return foo();
}

// NO-CWD-OPT: modules:
// NO-CWD-OPT-NEXT:   module:
// NO-CWD-OPT-NEXT:     name: Mod
// NO-CWD-OPT-NEXT:     context-hash:{{.*}}
// NO-CWD-OPT-NEXT:     cwd-ignored: 0


// CWD-OPT: modules:
// CWD-OPT-NEXT:   module:
// CWD-OPT-NEXT:     name: Mod
// CWD-OPT-NEXT:     context-hash:{{.*}}
// CWD-OPT-NEXT:     cwd-ignored: 1
