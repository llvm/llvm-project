# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %t/a.s -o %t/a.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %t/a_b.s -o %t/a_b.o
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %t/b.s -o %t/b.o
# RUN: llvm-ar rc %t/a.a %t/a.o
# RUN: llvm-ar rc %t/a_b.a %t/a_b.o
# RUN: llvm-ar rc %t/b.a %t/b.o
# RUN: cd %t

## Nothing is extracted from an archive. The file is created with just a header.
# RUN: wasm-ld main.o a.o b.a -o /dev/null --why-extract=why1.txt
# RUN: FileCheck %s --input-file=why1.txt --check-prefix=CHECK1 --match-full-lines --strict-whitespace

#      CHECK1:reference	extracted	symbol
#  CHECK1-NOT:{{.}}

## Some archive members are extracted.
# RUN: wasm-ld main.o a_b.a b.a -o /dev/null --why-extract=why2.txt
# RUN: FileCheck %s --input-file=why2.txt --check-prefix=CHECK2 --match-full-lines --strict-whitespace

#      CHECK2:reference	extracted	symbol
# CHECK2-NEXT:main.o	a_b.a(a_b.o)	a
# CHECK2-NEXT:a_b.a(a_b.o)	b.a(b.o)	b()

## An undefined symbol error does not suppress the output.
# RUN: not wasm-ld main.o a_b.a -o /dev/null --why-extract=why3.txt
# RUN: FileCheck %s --input-file=why3.txt --check-prefix=CHECK3 --match-full-lines --strict-whitespace

## Check that backward references are supported.
## - means stdout.
# RUN: wasm-ld b.a a_b.a main.o -o /dev/null --why-extract=- | FileCheck %s --check-prefix=CHECK4

#      CHECK3:reference	extracted	symbol
# CHECK3-NEXT:main.o	a_b.a(a_b.o)	a

#      CHECK4:reference	extracted	symbol
# CHECK4-NEXT:a_b.a(a_b.o)	b.a(b.o)	b()
# CHECK4-NEXT:main.o	a_b.a(a_b.o)	a

# RUN: wasm-ld main.o a_b.a b.a -o /dev/null --no-demangle --why-extract=- | FileCheck %s --check-prefix=MANGLED

# MANGLED: a_b.a(a_b.o)	b.a(b.o)	_Z1bv

# RUN: wasm-ld main.o a.a b.a -o /dev/null -u _Z1bv --why-extract=- | FileCheck %s --check-prefix=UNDEFINED

## We insert -u symbol before processing other files, so its name is <internal>.
## This is not ideal.
# UNDEFINED: <internal>	b.a(b.o)	b()

# RUN: wasm-ld main.o a.a b.a -o /dev/null -e _Z1bv --why-extract=- | FileCheck %s --check-prefix=ENTRY

# ENTRY: --entry	b.a(b.o)	b()

# SCRIPT: <internal>	b.a(b.o)	b()

# RUN: not wasm-ld -shared main.o -o /dev/null --why-extract=/ 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: cannot open --why-extract= file /: {{.*}}

#--- main.s
.globl _start
.functype a () -> ()
_start:
  .functype _start () -> ()
  call a
  end_function

#--- a.s
.globl a
a:
  .functype a () -> ()
  end_function

#--- a_b.s
.functype _Z1bv () -> ()
.globl a
a:
  .functype a () -> ()
  call _Z1bv
  end_function

#--- b.s
.globl _Z1bv
_Z1bv:
  .functype _Z1bv () -> ()
  end_function
