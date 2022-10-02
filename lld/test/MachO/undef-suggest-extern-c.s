# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.o

## The reference is mangled while the definition is not, suggest a missing
## extern "C".
# RUN: echo 'call __Z3fooi' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t1.o
# RUN: not %lld %t.o %t1.o -demangle -o /dev/null 2>&1 | FileCheck %s

# CHECK:      error: undefined symbol: foo(int)
# CHECK-NEXT: >>> referenced by {{.*}}
# CHECK-NEXT: >>> did you mean: extern "C" _foo

## Don't suggest for nested names like F::foo() and foo::foo().
# RUN: echo 'call __ZN1F3fooEv; call __ZN3fooC1Ev' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t2.o
# RUN: not ld.lld %t.o %t2.o -o /dev/null 2>&1 | FileCheck /dev/null --implicit-check-not='did you mean'

.globl _start, _foo
_start:
_foo:
