# REQUIRES: x86
## FIXME: We appear to emit LC_LOAD_DYLIBs in a different order on Windows.
# UNSUPPORTED: system-windows
# RUN: rm -rf %t; split-file %s %t
# RUN: mkdir -p %t/usr/lib/system
# RUN: mkdir -p %t/System/Library/Frameworks/Foo.framework/Versions/A
# RUN: mkdir -p %t/System/Library/Frameworks/Foo.framework/Frameworks/Bar.framework/Versions/A
# RUN: mkdir -p %t/Baz.framework/Versions/A

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libfoo.s -o %t/libfoo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libtoplevel.s -o %t/libtoplevel.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/libsublevel.s -o %t/libsublevel.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/framework-foo.s -o %t/framework-foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/framework-bar.s -o %t/framework-bar.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/framework-baz.s -o %t/framework-baz.o
## libunused will be used to verify that we load implicit dylibs even if we
## don't use any symbols they contain.
# RUN: echo "" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/libunused.o
# RUN: echo "" | llvm-mc -filetype=obj -triple=x86_64-apple-darwin -o %t/reexporter.o

# RUN: %lld -dylib -lSystem %t/libfoo.o -o %t/libfoo.dylib
# RUN: %lld -dylib -lSystem %t/libtoplevel.o -o %t/usr/lib/libtoplevel.dylib -install_name /usr/lib/libtoplevel.dylib
# RUN: %lld -dylib -lSystem %t/libsublevel.o -o %t/usr/lib/system/libsublevel.dylib -install_name /usr/lib/system/libsublevel.dylib
# RUN: %lld -dylib -lSystem %t/libunused.o -o %t/usr/lib/libunused.dylib -install_name /usr/lib/libunused.dylib

## Bar.framework is nested within Foo.framework.
# RUN: %lld -dylib -lSystem %t/framework-bar.o -o %t/System/Library/Frameworks/Foo.framework/Frameworks/Bar.framework/Versions/A/Bar \
# RUN:   -install_name /System/Library/Frameworks/Foo.framework/Frameworks/Bar.framework/Versions/A/Bar
# RUN: ln -sf Versions/A/Bar \
# RUN:   %t/System/Library/Frameworks/Foo.framework/Frameworks/Bar.framework/Bar

## Have Foo re-export Bar.
# RUN: %lld -dylib -F %t/System/Library/Frameworks/Foo.framework/Frameworks \
# RUN:   -framework Bar -sub_umbrella Bar -lSystem %t/framework-foo.o -o %t/System/Library/Frameworks/Foo.framework/Versions/A/Foo \
# RUN:   -install_name /System/Library/Frameworks/Foo.framework/Versions/A/Foo
# RUN: ln -sf Versions/A/Foo %t/System/Library/Frameworks/Foo.framework/Foo

# RUN: %lld -dylib -lSystem %t/framework-baz.o -o %t/Baz.framework/Versions/A/Baz \
# RUN:   -install_name %t/Baz.framework/Versions/A/Baz
# RUN: ln -sf Versions/A/Baz %t/Baz.framework/Baz

# RUN: %lld -dylib -syslibroot %t -framework Foo -F %t -framework Baz \
# RUN:   -lc++ -ltoplevel -lunused %t/usr/lib/system/libsublevel.dylib %t/libfoo.dylib \
# RUN:   -sub_library libc++ -sub_library libfoo -sub_library libtoplevel \
# RUN:   -sub_library libsublevel -sub_library libunused \
# RUN:   -sub_umbrella Baz -sub_umbrella Foo \
# RUN:   %t/reexporter.o -o %t/libreexporter.dylib

# RUN: llvm-mc -filetype obj -triple x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: %lld -syslibroot %t -o %t/test -lSystem -L%t -lreexporter %t/test.o
# RUN: llvm-objdump --bind --no-show-raw-insn -d %t/test | FileCheck %s
# CHECK:     Bind table:
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libreexporter _foo
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libtoplevel   _toplevel
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libreexporter _sublevel
# CHECK-DAG: __DATA __data {{.*}} pointer 0 Foo           _framework_foo
# CHECK-DAG: __DATA __data {{.*}} pointer 0 Foo           _framework_bar
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libreexporter _framework_baz
# CHECK-DAG: __DATA __data {{.*}} pointer 0 libc++abi     ___gxx_personality_v0

# RUN: llvm-otool -l %t/test | FileCheck %s \
# RUN:   --check-prefix=LOAD -DDIR=%t --implicit-check-not LC_LOAD_DYLIB
## Check that we don't create duplicate LC_LOAD_DYLIBs.
# RUN: %lld -syslibroot %t -o %t/test -lSystem -L%t -lreexporter -ltoplevel %t/test.o
# RUN: llvm-otool -l %t/test | FileCheck %s \
# RUN:   --check-prefix=LOAD -DDIR=%t --implicit-check-not LC_LOAD_DYLIB

# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT: cmdsize
# LOAD-NEXT:    name /usr/lib/libSystem.dylib
# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT: cmdsize
# LOAD-NEXT:    name [[DIR]]/libreexporter.dylib
# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT: cmdsize
# LOAD-NEXT:    name /System/Library/Frameworks/Foo.framework/Versions/A/Foo
# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT: cmdsize
# LOAD-NEXT:    name /usr/lib/libc++abi.dylib
# LOAD:          cmd LC_LOAD_DYLIB
# LOAD-NEXT: cmdsize
# LOAD-NEXT:    name /usr/lib/libtoplevel.dylib

# RUN: %lld -no_implicit_dylibs -syslibroot %t -o %t/no-implicit -lSystem -L%t -lreexporter %t/test.o
# RUN: llvm-objdump --bind --no-show-raw-insn -d %t/no-implicit | FileCheck %s --check-prefix=NO-IMPLICIT
# NO-IMPLICIT:     Bind table:
# NO-IMPLICIT-DAG: __DATA __data {{.*}} pointer 0 libreexporter _foo
# NO-IMPLICIT-DAG: __DATA __data {{.*}} pointer 0 libreexporter _toplevel
# NO-IMPLICIT-DAG: __DATA __data {{.*}} pointer 0 libreexporter _sublevel
# NO-IMPLICIT-DAG: __DATA __data {{.*}} pointer 0 libreexporter _framework_foo
# NO-IMPLICIT-DAG: __DATA __data {{.*}} pointer 0 libreexporter _framework_bar
# NO-IMPLICIT-DAG: __DATA __data {{.*}} pointer 0 libreexporter _framework_baz
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

#--- framework-foo.s
.data
.globl _framework_foo
_framework_foo:

#--- framework-bar.s
.data
.globl _framework_bar
_framework_bar:

#--- framework-baz.s
.data
.globl _framework_baz
_framework_baz:

#--- test.s
.text
.globl _main

_main:
  ret

.data
  .quad _foo
  .quad _toplevel
  .quad _sublevel
  .quad _framework_foo
  .quad _framework_bar
  .quad _framework_baz
  .quad ___gxx_personality_v0
