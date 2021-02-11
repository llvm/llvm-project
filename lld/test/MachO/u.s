# REQUIRES: x86
# RUN: rm -rf %t
# RUN: split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-ar csr  %t/lib.a %t/foo.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos -o %t/main.o %t/main.s

# RUN: %lld %t/main.o %t/lib.a -o /dev/null -why_load | \
# RUN:     FileCheck %s --check-prefix=NOFOO --allow-empty

# RUN: %lld %t/main.o %t/lib.a -u _foo -o /dev/null -why_load | \
# RUN:     FileCheck %s --check-prefix=FOO

# RUN: not %lld %t/main.o %t/lib.a -u _asdf -o /dev/null 2>&1 | \
# RUN:     FileCheck %s --check-prefix=UNDEF

# NOFOO-NOT: _foo forced load of lib.a(foo.o)
# FOO: _foo forced load of lib.a(foo.o)
# UNDEF: error: undefined symbol: _asdf

#--- foo.s
.globl _foo
_foo:
  ret

#--- main.s
.globl _main
_main:
  ret

