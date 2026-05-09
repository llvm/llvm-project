// Test that implicit module builds diagnose redefinition of a module when two
// different modulemaps define the same module name.
//
// Module "b" is defined in both first/module.modulemap and
// second/module.modulemap.

// RUN: rm -rf %t
// RUN: split-file %s %t

// We detect the error when we read a's modulemap in first
// and discovers another b, which we have seen earlier in
// second/module.modulemap.
// RUN: not %clang_cc1 -x objective-c -fmodules -fimplicit-module-maps \
// RUN:   -I %t/second -I %t/first \
// RUN:   -fmodules-cache-path=%t/cache \
// RUN:   %t/sourcefile.c 2>&1 | FileCheck %s

// CHECK: first{{[/\\]}}module.modulemap:5:8: error: redefinition of module 'b'
// CHECK: second{{[/\\]}}module.modulemap:1:8: note: previously defined here

// On the other hand, we do NOT detect error if we load a's modulemap first.
// Checking duplicating module decls in general is too expensive, since
// it requires loading all the modulemaps. The command below should succeed.
// RUN: %clang_cc1 -x objective-c -fmodules -fimplicit-module-maps \
// RUN:   -I %t/first -I %t/second \
// RUN:   -fmodules-cache-path=%t/cache \
// RUN:   %t/sourcefile.c

//--- first/module.modulemap
module a {
  header "a.h"
}

module b {
  export *
}

//--- first/a.h
// empty

//--- second/module.modulemap
module b {
  header "b.h"
}

//--- second/b.h
// empty


//--- sourcefile.c
@import b;
@import a;

int main() {
  return 0;
}
