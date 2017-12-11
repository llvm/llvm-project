# REQUIRES: x86, gnustat

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld -r %t.o -o %t1.o
# RUN: stat -c %%A %t1.o | FileCheck --check-prefix=CHECK-RELOC %s
# CHECK-RELOC: -rw-r--r--
# RUN: ld.lld -shared %t.o -o %t2.so
# RUN: stat -c %%A %t2.so | FileCheck --check-prefix=CHECK-SHARED %s
# CHECK-SHARED: -rwxr-xr-x
# RUN: ld.lld %t.o -o %t3
# RUN: stat -c %%A %t3 | FileCheck --check-prefix=CHECK-EXEC %s
# CHECK-EXEC: -rwxr-xr-x

.global _start
_start:
  nop
