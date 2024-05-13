# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/a_b.s -o %t/a_b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/b.s -o %t/b.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %t/err.s -o %t/err.o
# RUN: llvm-ar rc %t/a.a %t/a.o
# RUN: llvm-ar rc %t/a_b.a %t/a_b.o
# RUN: llvm-ar rc %t/b.a %t/b.o
# RUN: cd %t

## Nothing is extracted from an archive. The file is created with just a header.
# RUN: ld.lld main.o a.o b.a -o /dev/null --why-extract=why1.txt
# RUN: FileCheck %s --input-file=why1.txt --check-prefix=CHECK1 --match-full-lines --strict-whitespace

#      CHECK1:reference	extracted	symbol
#  CHECK1-NOT:{{.}}

## Some archive members are extracted.
# RUN: ld.lld main.o a_b.a b.a -o /dev/null --why-extract=why2.txt
# RUN: FileCheck %s --input-file=why2.txt --check-prefix=CHECK2 --match-full-lines --strict-whitespace

## A relocation error does not suppress the output.
# RUN: rm -f why2.txt && not ld.lld main.o a_b.a b.a err.o -o /dev/null --why-extract=why2.txt
# RUN: FileCheck %s --input-file=why2.txt --check-prefix=CHECK2 --match-full-lines --strict-whitespace

#      CHECK2:reference	extracted	symbol
# CHECK2-NEXT:main.o	a_b.a(a_b.o)	a
# CHECK2-NEXT:a_b.a(a_b.o)	b.a(b.o)	b()

## An undefined symbol error does not suppress the output.
# RUN: not ld.lld main.o a_b.a -o /dev/null --why-extract=why3.txt
# RUN: FileCheck %s --input-file=why3.txt --check-prefix=CHECK3 --match-full-lines --strict-whitespace

## Check that backward references are supported.
## - means stdout.
# RUN: ld.lld b.a a_b.a main.o -o /dev/null --why-extract=- | FileCheck %s --check-prefix=CHECK4

#      CHECK3:reference	extracted	symbol
# CHECK3-NEXT:main.o	a_b.a(a_b.o)	a

#      CHECK4:reference	extracted	symbol
# CHECK4-NEXT:a_b.a(a_b.o)	b.a(b.o)	b()
# CHECK4-NEXT:main.o	a_b.a(a_b.o)	a

# RUN: ld.lld main.o a_b.a b.a -o /dev/null --no-demangle --why-extract=- | FileCheck %s --check-prefix=MANGLED

# MANGLED: a_b.a(a_b.o)	b.a(b.o)	_Z1bv

# RUN: ld.lld main.o a.a b.a -o /dev/null -u _Z1bv --why-extract=- | FileCheck %s --check-prefix=UNDEFINED

## We insert -u symbol before processing other files, so its name is <internal>.
## This is not ideal.
# UNDEFINED: <internal>	b.a(b.o)	b()

# RUN: ld.lld main.o a.a b.a -o /dev/null --undefined-glob '_Z1b*' --why-extract=- | FileCheck %s --check-prefix=UNDEFINED_GLOB

# UNDEFINED_GLOB: --undefined-glob	b.a(b.o)	b()

# RUN: ld.lld main.o a.a b.a -o /dev/null -e _Z1bv --why-extract=- | FileCheck %s --check-prefix=ENTRY

# ENTRY: --entry	b.a(b.o)	b()

# RUN: ld.lld main.o b.a -o /dev/null -T a.lds --why-extract=- | FileCheck %s --check-prefix=SCRIPT

# SCRIPT: <internal>	b.a(b.o)	b()

# RUN: ld.lld main.o --start-lib a_b.o b.o --end-lib -o /dev/null --why-extract=- | FileCheck %s --check-prefix=LAZY

# LAZY: main.o	a_b.o	a
# LAZY: a_b.o	b.o	b()

# RUN: not ld.lld -shared main.o -o /dev/null --why-extract=/ 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: cannot open --why-extract= file /: {{.*}}

#--- main.s
.globl _start
_start:
  call a

#--- a.s
.globl a
a:

#--- a_b.s
.globl a
a:
  call _Z1bv

#--- b.s
.globl _Z1bv
_Z1bv:

#--- a.lds
a = _Z1bv;

#--- err.s
.reloc ., R_X86_64_RELATIVE, 0
