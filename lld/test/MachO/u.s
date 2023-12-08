# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-ar csr %t/lib.a %t/foo.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main.o %t/main.s

# RUN: %lld -lSystem %t/main.o %t/lib.a -o /dev/null -why_load | count 0

# RUN: %lld -lSystem %t/main.o %t/lib.a -u _foo -o /dev/null -why_load | \
# RUN:     FileCheck %s --check-prefix=FOO

# RUN: not %lld %t/main.o %t/lib.a -u _asdf -u _fdsa -o /dev/null 2>&1 | \
# RUN:     FileCheck %s --check-prefix=UNDEF

# RUN: %lld -lSystem %t/main.o %t/lib.a -u _asdf -undefined dynamic_lookup -o %t/dyn-lookup
# RUN: llvm-objdump --macho --syms %t/dyn-lookup | FileCheck %s --check-prefix=DYN

# FOO: _foo forced load of {{.+}}lib.a(foo.o)
# UNDEF:      error: undefined symbol: _asdf
# UNDEF-NEXT: >>> referenced by -u
# UNDEF:      error: undefined symbol: _fdsa
# UNDEF-NEXT: >>> referenced by -u
# DYN: *UND* _asdf

#--- foo.s
.globl _foo
_foo:
  ret

#--- main.s
.globl _main
_main:
  ret

