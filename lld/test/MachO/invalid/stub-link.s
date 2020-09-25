## FIXME: This test seems to be failing on some Google Mac buildbots for
## unclear reasons, so it's disabled for now. See D85404 for details.
# UNSUPPORTED: darwin
# REQUIRES: x86

# RUN: mkdir -p %t
#
# RUN: llvm-mc -filetype obj -triple x86_64-apple-ios %s -o %t/test.o
# RUN: not lld -flavor darwinnew -o %t/test -syslibroot %S/../Inputs/iPhoneSimulator.sdk -lSystem %t/test.o 2>&1 | FileCheck %s

# CHECK-DAG: error: undefined symbol __cache_handle_memory_pressure_event
# CHECK-DAG: error: undefined symbol _from_non_reexported_tapi_dylib

.section __TEXT,__text
.global _main

_main:
  movq __cache_handle_memory_pressure_event@GOTPCREL(%rip), %rax
  movq _from_non_reexported_tapi_dylib@GOTPCREL(%rip), %rax
  ret
