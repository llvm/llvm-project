# REQUIRES: x86
## FIXME: We appear to emit LC_LOAD_DYLIBs in a different order on Windows.
# UNSUPPORTED: system-windows
# RUN: rm -rf %t; split-file %s %t
# RUN: mkdir -p %t/usr/lib/system

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libfoo.s -o %t/libfoo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libtoplevel.s -o %t/libtoplevel.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libsublevel.s -o %t/libsublevel.o
## libunused will be used to verify that we load implicit dylibs even if we
## don't use any symbols they contain.
# RUN: echo "" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/libunused.o
# RUN: echo "" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/reexporter.o

# RUN: %lld -dylib -lSystem %t/libfoo.o -o %t/libfoo.dylib
# RUN: %lld -dylib -lSystem %t/libtoplevel.o -o %t/usr/lib/libtoplevel.dylib -install_name /usr/lib/libtoplevel.dylib
# RUN: %lld -dylib -lSystem %t/libsublevel.o -o %t/usr/lib/system/libsublevel.dylib -install_name /usr/lib/system/libsublevel.dylib
# RUN: %lld -dylib -lSystem %t/libunused.o -o %t/usr/lib/libunused.dylib -install_name /usr/lib/libunused.dylib
# RUN: %lld -dylib -syslibroot %t \
# RUN:   -lc++ -ltoplevel -lunused %t/usr/lib/system/libsublevel.dylib %t/libfoo.dylib \
# RUN:   -sub_library libc++ -sub_library libfoo -sub_library libtoplevel \
# RUN:   -sub_library libsublevel -sub_library libunused \
# RUN:   %t/reexporter.o -o %t/libreexporter.dylib

# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: %lld -syslibroot %t -o %t/test -lSystem -L%t -lreexporter %t/test.o
# RUN: llvm-objdump --bind --no-show-raw-insn -d %t/test | FileCheck %s
# CHECK:     Bind table:
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libreexporter _foo
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libtoplevel   _toplevel
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libreexporter _sublevel
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libc++abi     ___gxx_personality_v0

# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s \
# RUN:   --check-prefix=LOAD -DDIR=%t --implicit-check-not LC_LOAD_DYLIB
## Check that we don't create duplicate LC_LOAD_DYLIBs.
# RUN: %lld -syslibroot %t -o %t/test -lSystem -L%t -lreexporter -ltoplevel %t/test.o
# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s \
# RUN:   --check-prefix=LOAD -DDIR=%t --implicit-check-not LC_LOAD_DYLIB

# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT: cmdsize
# LOAD-NEXT:    name /usr/lib/libSystem.B.dylib
# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT: cmdsize
# LOAD-NEXT:    name /usr/lib/libc++abi.dylib
# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT: cmdsize
# LOAD-NEXT:    name /usr/lib/libc++.dylib
# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT: cmdsize
# LOAD-NEXT:    name /usr/lib/libtoplevel.dylib
# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT: cmdsize
# LOAD-NEXT:    name /usr/lib/libunused.dylib
# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT: cmdsize
# LOAD-NEXT:    name [[DIR]]/libreexporter.dylib

# RUN: %lld -no_implicit_dylibs -syslibroot %t -o %t/no-implicit -lSystem -L%t -lreexporter %t/test.o
# RUN: llvm-objdump --bind --no-show-raw-insn -d %t/no-implicit | FileCheck %s --check-prefix=NO-IMPLICIT
# NO-IMPLICIT:     Bind table:
# NO-IMPLICIT-DAG: __DATA __data {{.*}} pointer 0 libreexporter _foo
# NO-IMPLICIT-DAG: __DATA __data {{.*}} pointer 0 libreexporter _toplevel
# NO-IMPLICIT-DAG: __DATA __data {{.*}} pointer 0 libreexporter _sublevel
# NO-IMPLICIT-DAG: __DATA __data {{.*}} pointer 0 libreexporter ___gxx_personality_v0

#--- libfoo.s
.data
.globl _foo
_foo:

#--- libtoplevel.s
.data
.globl _toplevel
_toplevel:

#--- libsublevel.s
.data
.globl _sublevel
_sublevel:

#--- test.s
.text
.globl _main

_main:
  ret

.data
  .quad _foo
  .quad _toplevel
  .quad _sublevel
  .quad ___gxx_personality_v0
