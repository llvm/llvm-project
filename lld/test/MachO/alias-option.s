# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main.o %t/main.s

# RUN: %lld -alias _foo _bar -u _bar -exported_symbol _bar %t/main.o %t/foo.o -o %t/bar.o
# RUN: llvm-nm %t/bar.o | FileCheck %s --check-prefix=BAR

# BAR: [[#%x,FOO_ADDR:]] T _bar
# BAR: [[#FOO_ADDR]]     t _foo

# RUN: not %lld -alias _missing _bar -alias _missing2 _baz %t/main.o -o %t/missing.o %s 2>&1 | FileCheck %s --check-prefix=MISSING

# MISSING-DAG: undefined base symbol '_missing' for alias '_bar'
# MISSING-DAG: undefined base symbol '_missing2' for alias '_baz'

# RUN: %lld -alias _foo _main %t/foo.o -o %t/main_rename.o
# RUN: llvm-nm %t/main_rename.o | FileCheck %s --check-prefix=MAIN

# MAIN: [[#%x,FOO_ADDR:]] T _foo
# MAIN: [[#FOO_ADDR]]     T _main

## Verify dead stripping doesn't remove the aliased symbol. This behavior differs
## from ld64 where it actually does dead strip only the alias, not the original symbol.
# RUN: %lld -dead_strip -alias _foo _bar -alias _main _fake_main %t/main.o %t/foo.o -o %t/multiple.o
# RUN: llvm-nm %t/multiple.o | FileCheck %s --check-prefix=MULTIPLE

# MULTIPLE: [[#%x,FOO_ADDR:]]  T _bar
# MULTIPLE: [[#%x,MAIN_ADDR:]] T _fake_main
# MULTIPLE: [[#FOO_ADDR]]      T _foo
# MULTIPLE: [[#MAIN_ADDR]]     T _main

#--- foo.s
.subsections_via_symbols
.globl _foo
_foo:
  ret

#--- main.s
.subsections_via_symbols
.globl _main
_main:
  ret
