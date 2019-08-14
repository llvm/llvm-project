# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -shared -soname=t.so -o %t.so
# RUN: echo '{ foo; };' > %t.list
# RUN: ld.lld %t.o %t.so -shared --dynamic-list %t.list -o %t
# RUN: llvm-readelf --dyn-syms %t | FileCheck --check-prefix=SYM %s
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=REL %s

## foo and bar interpose symbols in another DSO, so both are exported,
## even if --dynamic-list specifies only foo.

# SYM-DAG: bar
# SYM-DAG: foo

## bar is not specified in --dynamic-list, so it is not preemptable when
## producing a DSO, and its PLT does not have an associated JUMP_SLOT.

# REL:      .rela.plt {
# REL-NEXT:   R_X86_64_JUMP_SLOT foo 0x0
# REL-NEXT: }

.globl foo, bar
foo:
bar:
  ret

call foo@PLT
call bar@PLT
