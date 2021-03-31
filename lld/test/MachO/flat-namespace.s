# REQUIRES: x86
# RUN: rm -rf %t
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/foo.o %t/foo.s
# RUN: %lld -dylib -o %t/foo.dylib %t/foo.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/bar.o %t/bar.s
# RUN: %lld -lSystem -dylib -o %t/bar.dylib %t/bar.o %t/foo.dylib

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/baz.o %t/baz.s
# RUN: %lld -lSystem -dylib -o %t/baz.dylib %t/baz.o %t/bar.dylib

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main.o %t/main.s

# With flat_namespace, the linker automatically looks in foo.dylib and
# bar.dylib too, but it doesn't add a LC_LOAD_DYLIB for it.
# RUN: %lld -flat_namespace -lSystem %t/main.o %t/baz.dylib -o %t/out
# RUN: llvm-objdump --macho --all-headers %t/out \
# RUN:     | FileCheck --check-prefix=HEADERBITS %s
# RUN: llvm-objdump --macho --bind --lazy-bind --weak-bind %t/out \
# RUN:     | FileCheck --check-prefix=FLAT %s
# RUN: llvm-nm -m %t/out | FileCheck --check-prefix=FLATSYM %s
# RUN: llvm-readobj --syms %t/out | FileCheck --check-prefix=FLATSYM-READOBJ %s

# HEADERBITS-NOT: NOUNDEFS
# HEADERBITS-NOT: TWOLEVEL
# HEADERBITS: DYLDLINK
# HEADERBITS-NOT: foo.dylib
# HEADERBITS-NOT: bar.dylib

# FLAT: Bind table:
# FLAT: __DATA_CONST __got          0x{{[0-9a-f]*}} pointer         0 flat-namespace   dyld_stub_binder
# FLAT: Lazy bind table:
# FLAT-DAG: __DATA   __la_symbol_ptr    0x{{[0-9a-f]*}} flat-namespace   _bar
# FLAT-DAG: __DATA   __la_symbol_ptr    0x{{[0-9a-f]*}} flat-namespace   _baz
# FLAT-DAG: __DATA   __la_symbol_ptr    0x{{[0-9a-f]*}} flat-namespace   _foo

# No "(dynamically looked up)" because llvm-nm -m doesn't print that
# for files without MH_TWOLEVEL for some reason.
# FLATSYM: (undefined) external _bar
# FLATSYM: (undefined) external _baz
# FLATSYM: (undefined) external _foo

# ...but `llvm-readobj --syms` does, so verify we put the right thing there.
# FLATSYM-READOBJ: Flags [ (0xFE00)

# Undefined symbols should still cause errors by default.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos \
# RUN:     -o %t/main-with-undef.o %t/main-with-undef.s
# RUN: not %lld -flat_namespace -lSystem %t/main-with-undef.o %t/bar.dylib \
# RUN:     -o %t/out 2>&1 | FileCheck --check-prefix=UNDEF %s
# UNDEF: error: undefined symbol: _quux

#--- foo.s
.globl _foo
_foo:
  ret

#--- bar.s
.globl _bar
_bar:
  callq _foo
  ret

#--- baz.s
.globl _baz
_baz:
  callq _bar
  ret

#--- main.s
.globl _main
_main:
  callq _foo
  callq _bar
  callq _baz
  ret

#--- main-with-undef.s
.globl _main
_main:
  callq _foo
  callq _bar
  callq _baz
  callq _quux
  ret
