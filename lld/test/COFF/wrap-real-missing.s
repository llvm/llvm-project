// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-win32-gnu %s -o %t.obj

// RUN: not lld-link -lldmingw -out:%t.exe %t.obj -entry:entry -subsystem:console -wrap:foo 2>&1 | FileCheck %s

// Check that we error out properly with an undefined symbol, if
// __real_foo is referenced and missing, even if the -lldmingw flag is set
// (which otherwise tolerates certain cases of references to missing
// sections, to tolerate certain GCC peculiarities).

// CHECK: error: undefined symbol: foo

.global entry
entry:
  call foo
  ret

.global __wrap_foo
__wrap_foo:
  call __real_foo
  ret
