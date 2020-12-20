# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: rm -f %t/foo.a
# RUN: llvm-ar rcs %t/foo.a %t/foo.o

# FOO-FIRST: <_foo>:
# FOO-FIRST: <_main>:

# FOO-SECOND: <_main>:
# FOO-SECOND: <_foo>:

# RUN: %lld -lSystem -o %t/test-1 %t/test.o %t/foo.o -order_file %t/ord-1
# RUN: llvm-objdump -d %t/test-1 | FileCheck %s --check-prefix=FOO-FIRST
## Output should be the same regardless of the command-line order of object files
# RUN: %lld -lSystem -o %t/test-1 %t/foo.o %t/test.o -order_file %t/ord-1
# RUN: llvm-objdump -d %t/test-1 | FileCheck %s --check-prefix=FOO-FIRST

# RUN: %lld -lSystem -o %t/test-2 %t/test.o %t/foo.o -order_file %t/ord-2
# RUN: llvm-objdump -d %t/test-2 | FileCheck %s --check-prefix=FOO-SECOND
# RUN: %lld -lSystem -o %t/test-2 %t/foo.o %t/test.o -order_file %t/ord-2
# RUN: llvm-objdump -d %t/test-2 | FileCheck %s --check-prefix=FOO-SECOND

# RUN: %lld -lSystem -o %t/test-file-match %t/test.o %t/foo.o -order_file %t/ord-file-match
# RUN: llvm-objdump -d %t/test-file-match | FileCheck %s --check-prefix=FOO-FIRST
## Output should be the same regardless of the command-line order of object files
# RUN: %lld -lSystem -o %t/test-file-match %t/foo.o %t/test.o -order_file %t/ord-file-match
# RUN: llvm-objdump -d %t/test-file-match | FileCheck %s --check-prefix=FOO-FIRST

# RUN: %lld -lSystem -o %t/test-file-nomatch %t/test.o %t/foo.o -order_file %t/ord-file-nomatch
# RUN: llvm-objdump -d %t/test-file-nomatch | FileCheck %s --check-prefix=FOO-SECOND
# RUN: %lld -lSystem -o %t/test-file-nomatch %t/foo.o %t/test.o -order_file %t/ord-file-nomatch
# RUN: llvm-objdump -d %t/test-file-nomatch | FileCheck %s --check-prefix=FOO-SECOND

# RUN: %lld -lSystem -o %t/test-arch-match %t/test.o %t/foo.o -order_file %t/ord-arch-match
# RUN: llvm-objdump -d %t/test-arch-match | FileCheck %s --check-prefix=FOO-FIRST
# RUN: %lld -lSystem -o %t/test-arch-match %t/foo.o %t/test.o -order_file %t/ord-arch-match
# RUN: llvm-objdump -d %t/test-arch-match | FileCheck %s --check-prefix=FOO-FIRST

# RUN: %lld -lSystem -o %t/test-arch-nomatch %t/test.o %t/foo.o -order_file %t/ord-arch-nomatch
# RUN: llvm-objdump -d %t/test-arch-nomatch | FileCheck %s --check-prefix=FOO-SECOND
# RUN: %lld -lSystem -o %t/test-arch-nomatch %t/foo.o %t/test.o -order_file %t/ord-arch-nomatch
# RUN: llvm-objdump -d %t/test-arch-nomatch | FileCheck %s --check-prefix=FOO-SECOND

# RUN: %lld -lSystem -o %t/test-arch-match %t/test.o %t/foo.o -order_file %t/ord-arch-match
# RUN: llvm-objdump -d %t/test-arch-match | FileCheck %s --check-prefix=FOO-FIRST
# RUN: %lld -lSystem -o %t/test-arch-match %t/foo.o %t/test.o -order_file %t/ord-arch-match
# RUN: llvm-objdump -d %t/test-arch-match | FileCheck %s --check-prefix=FOO-FIRST

## Test archives

# RUN: %lld -lSystem -o %t/test-archive-1 %t/test.o %t/foo.a -order_file %t/ord-1
# RUN: llvm-objdump -d %t/test-archive-1 | FileCheck %s --check-prefix=FOO-FIRST
# RUN: %lld -lSystem -o %t/test-archive-1 %t/foo.a %t/test.o -order_file %t/ord-1
# RUN: llvm-objdump -d %t/test-archive-1 | FileCheck %s --check-prefix=FOO-FIRST

# RUN: %lld -lSystem -o %t/test-archive-file-no-match %t/test.o %t/foo.a -order_file %t/ord-file-nomatch
# RUN: llvm-objdump -d %t/test-archive-file-no-match | FileCheck %s --check-prefix=FOO-SECOND
# RUN: %lld -lSystem -o %t/test-archive %t/foo.a %t/test.o -order_file %t/ord-file-nomatch
# RUN: llvm-objdump -d %t/test-archive-file-no-match | FileCheck %s --check-prefix=FOO-SECOND

## The following tests check that if an address is matched by multiple order
## file entries, it should always use the lowest-ordered match.

# RUN: %lld -lSystem -o %t/test-1 %t/test.o %t/foo.o -order_file %t/ord-multiple-1
# RUN: llvm-objdump -d %t/test-1 | FileCheck %s --check-prefix=FOO-FIRST
# RUN: %lld -lSystem -o %t/test-1 %t/foo.o %t/test.o -order_file %t/ord-multiple-1
# RUN: llvm-objdump -d %t/test-1 | FileCheck %s --check-prefix=FOO-FIRST

# RUN: %lld -lSystem -o %t/test-2 %t/test.o %t/foo.o -order_file %t/ord-multiple-2
# RUN: llvm-objdump -d %t/test-2 | FileCheck %s --check-prefix=FOO-FIRST
# RUN: %lld -lSystem -o %t/test-2 %t/foo.o %t/test.o -order_file %t/ord-multiple-2
# RUN: llvm-objdump -d %t/test-2 | FileCheck %s --check-prefix=FOO-FIRST

# RUN: %lld -lSystem -o %t/test-3 %t/test.o %t/foo.o -order_file %t/ord-multiple-3
# RUN: llvm-objdump -d %t/test-3 | FileCheck %s --check-prefix=FOO-FIRST
# RUN: %lld -lSystem -o %t/test-3 %t/foo.o %t/test.o -order_file %t/ord-multiple-3
# RUN: llvm-objdump -d %t/test-3 | FileCheck %s --check-prefix=FOO-FIRST

# RUN: %lld -lSystem -o %t/test-4 %t/test.o %t/foo.o -order_file %t/ord-multiple-4
# RUN: llvm-objdump -d %t/test-4 | FileCheck %s --check-prefix=FOO-FIRST
# RUN: %lld -lSystem -o %t/test-4 %t/foo.o %t/test.o -order_file %t/ord-multiple-4
# RUN: llvm-objdump -d %t/test-4 | FileCheck %s --check-prefix=FOO-FIRST

## _foo and _bar both point to the same location. When both symbols appear in
## an order file, the location in question should be ordered according to the
## lowest-ordered symbol that references it.

# RUN: %lld -lSystem -o %t/test-alias %t/test.o %t/foo.o -order_file %t/ord-alias
# RUN: llvm-objdump -d %t/test-alias | FileCheck %s --check-prefix=FOO-FIRST
# RUN: %lld -lSystem -o %t/test-alias %t/foo.o %t/test.o -order_file %t/ord-alias
# RUN: llvm-objdump -d %t/test-alias | FileCheck %s --check-prefix=FOO-FIRST

#--- ord-1
_foo # just a comment
_main # another comment

#--- ord-2
_main # just a comment
_foo # another comment

#--- ord-file-match
foo.o:_foo
_main

#--- ord-file-nomatch
bar.o:_foo
_main
_foo

#--- ord-arch-match
x86_64:_foo
_main

#--- ord-arch-nomatch
ppc:_foo
_main
_foo

#--- ord-arch-file-match
x86_64:bar.o:_foo
_main

#--- ord-multiple-1
_foo
_main
foo.o:_foo

#--- ord-multiple-2
foo.o:_foo
_main
_foo

#--- ord-multiple-3
_foo
_main
_foo

#--- ord-multiple-4
foo.o:_foo
_main
foo.o:_foo

#--- ord-alias
_bar
_main
_foo

#--- foo.s
.globl _foo
_foo:
_bar:
  ret

#--- test.s
.globl _main

_main:
  callq _foo
  ret
