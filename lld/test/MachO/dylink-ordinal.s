# REQUIRES: x86

## --no-leading-lines needed for .tbd files.
# RUN: rm -rf %t; split-file --no-leading-lines %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o
# RUN: %lld -o %t/main -L%t -lFoo -lBar -lSystem %t/main.o
# RUN: llvm-objdump --lazy-bind -d --no-show-raw-insn %t/main | FileCheck %s

# CHECK: callq 0x[[#%x,FOO_OFF:]]
# CHECK-NEXT: callq 0x[[#%x,BAR_OFF:]]

# CHECK: [[#%x,BAR_OFF]]: jmpq {{.*}} ## 0x[[#%x,BAR_BIND:]]
# CHECK: [[#%x,FOO_OFF]]: jmpq {{.*}} ## 0x[[#%x,FOO_BIND:]]

# CHECK-LABEL: Lazy bind table:
# CHECK-DAG: __DATA __la_symbol_ptr 0x[[#%x,FOO_BIND]] Foo _foo
# CHECK-DAG: __DATA __la_symbol_ptr 0x[[#%x,BAR_BIND]] Foo _bar

# RUN: llvm-nm -m %t/main | FileCheck --check-prefix=NM %s

# NM-DAG: _bar (from Foo)
# NM-DAG: _foo (from Foo)

# RUN: llvm-otool -L %t/main | FileCheck %s --check-prefix=LOAD

# LOAD: Foo.dylib
# LOAD-NOT: Foo.dylib

#--- libFoo.tbd
--- !tapi-tbd
tbd-version:      4
targets:          [ x86_64-macos ]
uuids:
  - target:       x86_64-macos
    value:        00000000-0000-0000-0000-000000000000
install-name:     'Foo.dylib'
current-version:  0001.001.1
exports:
  - targets:      [ x86_64-macos ]
    symbols:      [ _foo ]

#--- libBar.tbd
--- !tapi-tbd
tbd-version:      4
targets:          [ x86_64-macos ]
uuids:
  - target:       x86_64-macos
    value:        00000000-0000-0000-0000-000000000000
## Also uses Foo.dylib as install-name!
## Normally, this would happen conditionally via an $ld$ symbol.
install-name:     'Foo.dylib'
current-version:  0001.001.1
exports:
  - targets:      [ x86_64-macos ]
    symbols:      [ _bar ]

#--- main.s
.section __TEXT,__text
.globl _main, _foo, _bar

_main:
  callq _foo
  callq _bar
  ret
