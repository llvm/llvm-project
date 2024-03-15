# REQUIRES: x86

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 main.s -o main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 def.s -o def.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 def-hidden.s -o def-hidden.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 ref.s -o ref.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o && ld.lld -shared a.o -o a.so
# RUN: cp a.so b.so
# RUN: llvm-mc -filetype=obj -triple=x86_64 empty.s -o empty.o && ld.lld -shared empty.o -o empty.so

# RUN: ld.lld --allow-shlib-undefined main.o a.so -o /dev/null
# RUN: not ld.lld --no-allow-shlib-undefined main.o a.so -o /dev/null 2>&1 | FileCheck %s
## Executable linking defaults to --no-allow-shlib-undefined.
# RUN: not ld.lld main.o a.so -o /dev/null 2>&1 | FileCheck %s
# RUN: ld.lld main.o a.so --noinhibit-exec -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: ld.lld main.o a.so --warn-unresolved-symbols -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
## -shared linking defaults to --allow-shlib-undefined.
# RUN: ld.lld -shared main.o a.so -o /dev/null

## DSO with undefines should link with or without any of these options.
# RUN: ld.lld -shared --allow-shlib-undefined a.o -o /dev/null
# RUN: ld.lld -shared --no-allow-shlib-undefined a.o -o /dev/null

## Perform checking even if an unresolved symbol is first seen in a regular object file.
# RUN: not ld.lld --gc-sections main.o ref.o a.so -o /dev/null 2>&1 | FileCheck %s

## Check that the error is reported for each shared library where the symbol
## is referenced.
# RUN: not ld.lld main.o a.so empty.so b.so -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK2

## Test some cases when a relocatable object file provides a non-exported definition.
# RUN: not ld.lld main.o a.so def-hidden.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=NONEXPORTED
# RUN: not ld.lld main.o a.so def-hidden.o -shared --no-allow-shlib-undefined -o /dev/null 2>&1 | FileCheck %s --check-prefix=NONEXPORTED
# RUN: ld.lld main.o a.so def-hidden.o --allow-shlib-undefined --fatal-warnings -o /dev/null
## Test a relocatable object file definition that is converted to STB_LOCAL.
# RUN: not ld.lld main.o a.so def-hidden.o --version-script=local.ver -o /dev/null 2>&1 | FileCheck %s --check-prefix=NONEXPORTED

## The section containing the definition is discarded, and we report an error.
# RUN: not ld.lld --gc-sections main.o a.so def-hidden.o -o /dev/null 2>&1 | FileCheck %s
## The definition def.so is ignored.
# RUN: ld.lld -shared def.o -o def.so
# RUN: ld.lld --gc-sections main.o a.so def.so def-hidden.o --fatal-warnings -o /dev/null

# CHECK-NOT:   error:
# CHECK:       error: undefined reference: x1{{$}}
# CHECK-NEXT:  >>> referenced by a.so (disallowed by --no-allow-shlib-undefined){{$}}
# CHECK-NOT:   {{.}}

# CHECK2-NOT:  error:
# CHECK2:      error: undefined reference: x1
# CHECK2-NEXT: >>> referenced by a.so (disallowed by --no-allow-shlib-undefined)
# CHECK2:      error: undefined reference: x1
# CHECK2-NEXT: >>> referenced by b.so (disallowed by --no-allow-shlib-undefined)
# CHECK2-NOT:  {{.}}

# WARN:        warning: undefined reference: x1
# WARN-NEXT:   >>> referenced by a.so (disallowed by --no-allow-shlib-undefined)

# NONEXPORTED-NOT: error:
# NONEXPORTED:     error: non-exported symbol 'x1' in 'def-hidden.o' is referenced by DSO 'a.so'
# NONEXPORTED-NOT: {{.}}

#--- main.s
.globl _start
_start:
  callq shared@PLT
#--- ref.s
  callq x1@PLT
#--- def.s
.globl x1
x1:
#--- def-hidden.s
.globl x1
.hidden x1
x1:

#--- a.s
.globl shared
.weak x2
shared:
  callq x1@PLT
  movq x2@GOTPCREL(%rip), %rax

#--- empty.s
#--- local.ver
v1 { local: x1; };
