# REQUIRES: x86
## For a relative pathname in INPUT() or GROUP(), the parent directory of
## the current linker script has priority over current working directory and -L.

# RUN: rm -rf %t.dir && mkdir %t.dir && cd %t.dir

# RUN: mkdir dir
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o a.o
# RUN: echo '.globl b, cwd; b: cwd:' | llvm-mc -filetype=obj -triple=x86_64 - -o b.o
# RUN: echo '.globl b, dir; b: dir:' | llvm-mc -filetype=obj -triple=x86_64 - -o dir/b.o
# RUN: llvm-ar rc libb.a b.o
# RUN: llvm-ar rc dir/libb.a dir/b.o

## A relative pathname is relative to the parent directory of the current linker script.
## The directory has priority over current working directory and -L.
# RUN: echo 'INPUT(libb.a)' > dir/relative.lds
# RUN: ld.lld -L. a.o dir/relative.lds -o - | llvm-nm - | FileCheck --check-prefix=DIR %s
## GROUP() uses the same search order.
# RUN: echo 'GROUP(libb.a)' > dir/relative1.lds
# RUN: ld.lld -L. a.o dir/relative1.lds -o - | llvm-nm - | FileCheck --check-prefix=DIR %s

# DIR: T dir

## -l does not use the special rule.
# RUN: echo 'INPUT(-lb)' > dir/cwd.lds
# RUN: ld.lld -L. a.o dir/cwd.lds -o - | llvm-nm - | FileCheck --check-prefix=CWD %s
# RUN: echo 'GROUP(-lb)' > dir/cwd1.lds
# RUN: ld.lld -L. a.o dir/cwd1.lds -o - | llvm-nm - | FileCheck --check-prefix=CWD %s

# CWD: T cwd

## The rules does not apply to an absolute path.
# RUN: echo 'INPUT(/libb.a)' > dir/absolute.lds
# RUN: not ld.lld a.o dir/absolute.lds -o /dev/null

## If the parent directory of the current linker script does not contain the file,
## fall back to the current working directory.
# RUN: cp libb.a libc.a
# RUN: echo 'INPUT(libc.a)' > dir/fallback.lds
# RUN: ld.lld a.o dir/fallback.lds -o /dev/null

.globl _start
_start:
  call b
