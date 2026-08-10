# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## The definition is mangled while the reference is not, suggest an arbitrary
## C++ overload.
# RUN: echo '.globl _Z3fooi; _Z3fooi:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: not ld.lld %t.o %t1.o -o /dev/null 2>&1 | FileCheck %s

## Check that we can suggest a local definition.
# RUN: echo '_Z3fooi: call foo' | llvm-mc -filetype=obj -triple=x86_64 - -o %t2.o
# RUN: not ld.lld %t2.o -o /dev/null 2>&1 | FileCheck %s

# CHECK:      error: undefined symbol: foo
# CHECK-NEXT: >>> referenced by {{.*}}
# CHECK-NEXT: >>> did you mean to declare foo(int) as extern "C"?

## Don't suggest nested names whose base name is "foo", e.g. F::foo().
# RUN: echo '.globl _ZN1F3fooEv; _ZN1F3fooEv:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t3.o
# RUN: not ld.lld %t.o %t3.o -o /dev/null 2>&1 | FileCheck /dev/null --implicit-check-not='did you mean'

call foo
