# REQUIRES: x86
## FIXME: Paths on windows have both `\` and '/', as a result, they are in a different
## order when sorted. Maybe create a separate test for that?
# UNSUPPORTED: system-windows
#
# RUN: rm -rf %t
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/foo.o %t/foo.s
# RUN: %lld -dylib -o %t/libfoo.dylib %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/bar.o %t/bar.s
# RUN: llvm-ar csr  %t/bar.a %t/bar.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main.o %t/main.s

# RUN: %lld %t/main.o %t/bar.a %t/libfoo.dylib -lSystem -o %t/test.out -dependency_info %t/deps_info.out
# RUN: %python %S/Inputs/DependencyDump.py %t/deps_info.out | FileCheck %s

# CHECK: lld-version: {{.*}}LLD {{.*}}
# CHECK-DAG: input-file: {{.*}}/bar.a
# CHECK-DAG: input-file: {{.*}}/libfoo.dylib
# CHECK-DAG: input-file: {{.*}}/libSystem.tbd
# CHECK-DAG: input-file: {{.*}}/main.o
# CHECK-DAG: input-file: {{.*}}bar.o

# CHECK-NEXT: not-found: {{.*}}/libdyld.dylib
## There could be more not-found here but we are not checking those because it's brittle.

# CHECK: output-file: {{.*}}/test.out

#--- foo.s
.globl __Z3foo
__Z3foo:
  ret

#--- bar.s
.globl _bar
_bar:
  callq __Z3foo
  ret

#--- main.s
.globl _main
_main:
  callq _bar
  callq __Z3foo
  ret
