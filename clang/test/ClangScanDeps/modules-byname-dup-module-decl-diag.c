// Test duplicating module decls discovered during by-name module scanning with
// a shared compiler instance.
// This tests covers the case where the current modulemap we are loading
// contains a module decl that satisfies two conditions:
// 1. The compiler has seen the module decl during previous lookups.
// 2. The previous decl comes from a different modulemap.
// In this case, an error is produced with no dependency information returned.
// Specifically, we have the following setup:
//   - "A" is a framework module whose modulemap also declares an empty "B".
//   - A separate include path has its own B/module.modulemap that declares "B"
//     with a header depending on module "Dep"
// We scan B first, and during A's scan, the compiler should report an error.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: sed "s|DIR|%/t|g" %t/cdb.json.template > %t/cdb.json

// RUN: not clang-scan-deps -compilation-database %t/cdb.json -format \
// RUN:   experimental-full -module-names=B,A 2>&1 | \
// RUN:   sed 's:\\\\\?:/:g' | FileCheck -DPREFIX=%/t %s

// CHECK: A.framework/Modules/module.modulemap:7:8: error: redefinition of module 'B'
// CHECK: include/B/module.modulemap:1:8: note: previously defined here

//--- frameworks/A.framework/Modules/module.modulemap
framework module A [system] {
  umbrella header "A.h"
  export *
  module * { export * }
}

module B {
  export *
}

//--- frameworks/A.framework/Headers/A.h
// Framework A umbrella header
void a_func(void);

//--- include/B/module.modulemap
module B {
  header "B.h"
  export *
}

//--- include/B/B.h
#include "Dep.h"
void b_func(void);

//--- include/Dep/module.modulemap
module Dep {
  header "Dep.h"
  export *
}

//--- include/Dep/Dep.h
// Module Dep header
void dep_func(void);

//--- cdb.json.template
[{
  "file": "",
  "directory": "DIR",
  "command": "clang -fmodules -fmodules-cache-path=DIR/cache -I DIR/include/B -F DIR/frameworks -I DIR/include/Dep -x c"
}]
