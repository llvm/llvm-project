# REQUIRES: ppc
## Test that some save and restore functions can be synthesized.
## The code sequences are tested by ppc64-restgpr*.s and ppc64-savegpr*.s

# RUN: llvm-mc -filetype=obj -triple=ppc64le %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readelf -s %t.so | FileCheck --check-prefix=NM %s
# RUN: llvm-objdump -d %t.so | FileCheck %s

## The synthesized symbols are not exported.
# NM:      FUNC LOCAL HIDDEN {{.*}} _restgpr0_30
# NM-NEXT: FUNC LOCAL HIDDEN {{.*}} _restgpr1_30
# NM-NEXT: FUNC LOCAL HIDDEN {{.*}} _savegpr0_30
# NM-NEXT: FUNC LOCAL HIDDEN {{.*}} _savegpr1_30

# CHECK: 00000000000[[#%x,RESTGPR0:]] <_restgpr0_30>:
# CHECK: 00000000000[[#%x,RESTGPR1:]] <_restgpr1_30>:
# CHECK: 00000000000[[#%x,SAVEGPR0:]] <_savegpr0_30>:
# CHECK: 00000000000[[#%x,SAVEGPR1:]] <_savegpr1_30>:
# CHECK-LABEL: <_start>:
# CHECK-NEXT:    bl 0x[[#RESTGPR0]]
# CHECK-NEXT:    bl 0x[[#RESTGPR1]]
# CHECK-NEXT:    bl 0x[[#SAVEGPR0]]
# CHECK-NEXT:    bl 0x[[#SAVEGPR1]]

.globl _start
_start:
  bl _restgpr0_30
  bl _restgpr1_30
  bl _savegpr0_30
  bl _savegpr1_30
