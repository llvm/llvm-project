# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t.o

## The definition is mangled while the reference is not, suggest an arbitrary
## C++ overload.
# RUN: echo '.globl __Z3fooi; __Z3fooi:' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t1.o
# RUN: not %lld %t.o %t1.o -demangle -o /dev/null 2>&1 | FileCheck %s

## Check that we can suggest a local definition.
# RUN: echo '__Z3fooi: call _foo' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t2.o
# RUN: not %lld %t2.o -demangle -o /dev/null 2>&1 | FileCheck %s

# CHECK:      error: undefined symbol: _foo
# CHECK-NEXT: >>> referenced by {{.*}}
# CHECK-NEXT: >>> did you mean to declare foo(int) as extern "C"?

## Don't suggest nested names whose base name is "foo", e.g. F::foo().
# RUN: echo '.globl __ZN1F3fooEv; __ZN1F3fooEv:' | llvm-mc -filetype=obj -triple=x86_64-apple-macos - -o %t3.o
# RUN: not %lld %t.o %t3.o -o /dev/null 2>&1 | FileCheck /dev/null --implicit-check-not='did you mean'

call _foo
