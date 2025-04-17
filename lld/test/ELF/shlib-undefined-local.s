# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-linux-gnu -o %t1.o %S/Inputs/shlib-undefined-ref.s
# RUN: ld.lld -shared -o %t.so %t1.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-linux-gnu -o %t2.o %s
# RUN: echo "{ local: *; };" > %t.script
# RUN: not ld.lld -version-script %t.script %t2.o %t.so -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

# ERR: error: non-exported symbol 'should_not_be_exported' in '{{.*}}tmp2.o' is referenced by DSO '{{.*}}tmp.so'

.globl should_not_be_exported
should_not_be_exported:
	ret

.globl _start
_start:
	ret
