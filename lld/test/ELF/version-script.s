# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
# RUN: ld.lld -shared %t2.o -soname shared -o %t2.so

# RUN: echo "{ global: foo1; foo3; local: *; };" > %t.script
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld --version-script %t.script -shared %t.o %t2.so -o %t.so --fatal-warnings
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck --check-prefix=DSO %s

# RUN: echo "# comment" > %t3.script
# RUN: echo "{ local: *; # comment" >> %t3.script
# RUN: echo -n "}; # comment" >> %t3.script
# RUN: ld.lld --version-script %t3.script -shared %t.o %t2.so -o %t3.so
# RUN: llvm-readelf --dyn-syms %t3.so | FileCheck --check-prefix=DSO2 %s

## Also check that both "global:" and "global :" forms are accepted
# RUN: echo "VERSION_1.0 { global : foo1; local : *; };" > %t4.script
# RUN: echo "VERSION_2.0 { global: foo3; local: *; };" >> %t4.script
# RUN: ld.lld --version-script %t4.script -shared %t.o %t2.so -o %t4.so --fatal-warnings
# RUN: llvm-readelf --dyn-syms %t4.so | FileCheck --check-prefix=VERDSO %s

# RUN: echo "VERSION_1.0 { global: foo1; local: *; };" > %t5.script
# RUN: echo "{ global: foo3; local: *; };" >> %t5.script
# RUN: not ld.lld --version-script %t5.script -shared %t.o %t2.so -o /dev/null 2>&1 | \
# RUN:   FileCheck -check-prefix=ERR1 %s
# ERR1: anonymous version definition is used in combination with other version definitions

# RUN: echo "{ global: foo1; local: *; };" > %t5.script
# RUN: echo "VERSION_2.0 { global: foo3; local: *; };" >> %t5.script
# RUN: not ld.lld --version-script %t5.script -shared %t.o %t2.so -o /dev/null 2>&1 | \
# RUN:   FileCheck -check-prefix=ERR2 %s
# ERR2: EOF expected, but got VERSION_2.0

# RUN: echo "{ foo1; foo2; };" > %t.list
# RUN: ld.lld --version-script %t.script --dynamic-list %t.list %t.o %t2.so -o %t2
# RUN: llvm-readobj %t2 > /dev/null

## Check that we can handle multiple "--version-script" options.
# RUN: echo "VERSION_1.0 { global : foo1; local : *; };" > %t7a.script
# RUN: echo "VERSION_2.0 { global: foo3; local: *; };" > %t7b.script
# RUN: ld.lld --version-script %t7a.script --version-script %t7b.script -shared %t.o %t2.so -o %t7.so
# RUN: llvm-readelf --dyn-syms %t7.so | FileCheck --check-prefix=VERDSO %s

# DSO:      bar{{$}}
# DSO-NEXT: foo1{{$}}
# DSO-NEXT: foo3{{$}}
# DSO-NOT:  {{.}}

# DSO2:      bar{{$}}
# DSO2-NOT:  {{.}}

# VERDSO:      bar{{$}}
# VERDSO-NEXT: foo1@@VERSION_1.0
# VERDSO-NEXT: foo3@@VERSION_2.0
# VERDSO-NOT:  {{.}}

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld --hash-style=sysv -shared %t.o %t2.so -o %t.so
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck --check-prefix=ALL %s

# RUN: echo "{ global: foo1; foo3; };" > %t2.script
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld --hash-style=sysv --version-script %t2.script -shared %t.o %t2.so -o %t.so
# RUN: llvm-readelf --dyn-syms %t.so | FileCheck --check-prefix=ALL %s

# ALL:      _start{{$}}
# ALL-NEXT: bar{{$}}
# ALL-NEXT: foo1{{$}}
# ALL-NEXT: foo2{{$}}
# ALL-NEXT: foo3{{$}}
# ALL-NOT:  {{.}}

# RUN: echo "VERSION_1.0 { global: foo1; foo1; local: *; };" > %t8.script
# RUN: ld.lld --version-script %t8.script -shared %t.o -o /dev/null --fatal-warnings

.globl foo1
foo1:
  call bar@PLT
  ret

.globl foo2
foo2:
  ret

.globl foo3
foo3:
  call foo2@PLT
  ret

.globl _start
_start:
  ret
