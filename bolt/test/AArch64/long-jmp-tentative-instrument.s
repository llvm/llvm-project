## Check that the LongJmp tentative layout matches the emitted layout for the
## functions injected by instrumentation, which are emitted into their own code
## section after .text, or before it with --hot-functions-at-end.

# REQUIRES: system-linux,asserts

## The runtime library configured for the test suite is built for the host, so
## link against a stub AArch64 one instead.
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux --defsym RT=1 %s -o %t.rt.o
# RUN: ld.lld %t.o -o %t.exe -q --entry=F

# RUN: llvm-bolt %t.exe -o %t.bolt --instrument \
# RUN:   --runtime-instrumentation-lib=%t.rt.o \
# RUN:   --instrumentation-sleep-time=1 -debug-only=longjmp 2> %t.log
# RUN: echo "EMITTED:" >> %t.log
# RUN: llvm-nm %t.bolt >> %t.log
# RUN: FileCheck %s --input-file=%t.log

# RUN: llvm-bolt %t.exe -o %t.hfe.bolt --instrument \
# RUN:   --runtime-instrumentation-lib=%t.rt.o \
# RUN:   --instrumentation-sleep-time=1 --hot-functions-at-end \
# RUN:   -debug-only=longjmp 2> %t.hfe.log
# RUN: echo "EMITTED:" >> %t.hfe.log
# RUN: llvm-nm %t.hfe.bolt >> %t.hfe.log
# RUN: FileCheck %s --input-file=%t.hfe.log

# CHECK-DAG: {{^}}F tentative: [[#%x,F:]]
# CHECK-DAG: {{^}}G tentative: [[#%x,G:]]
# CHECK-DAG: {{^}}__bolt_instr_ind_call_handler tentative: [[#%x,HANDLER:]]
# CHECK-DAG: {{^}}__bolt_num_counters_getter tentative: [[#%x,GETTER:]]
# CHECK:     EMITTED:
# CHECK-DAG: [[#%.16x,F]] T F{{$}}
# CHECK-DAG: [[#%.16x,G]] T G{{$}}
# CHECK-DAG: [[#%.16x,HANDLER]] {{[tT]}} __bolt_instr_ind_call_handler{{$}}
# CHECK-DAG: [[#%.16x,GETTER]] {{[tT]}} __bolt_num_counters_getter{{$}}

.ifdef RT
## Stub instrumentation runtime library.
  .text
  .globl __bolt_instr_start
  .type __bolt_instr_start, %function
__bolt_instr_start:
  ret
  .size __bolt_instr_start, .-__bolt_instr_start

  .globl __bolt_instr_fini
  .type __bolt_instr_fini, %function
__bolt_instr_fini:
  ret
  .size __bolt_instr_fini, .-__bolt_instr_fini

  .globl __bolt_instr_clear_counters
  .type __bolt_instr_clear_counters, %function
__bolt_instr_clear_counters:
  ret
  .size __bolt_instr_clear_counters, .-__bolt_instr_clear_counters

  .globl __bolt_instr_data_dump
  .type __bolt_instr_data_dump, %function
__bolt_instr_data_dump:
  ret
  .size __bolt_instr_data_dump, .-__bolt_instr_data_dump
.else
  .text
  .globl F
  .type F, %function
F:
  .rept 20
  add x0, x0, #1
  .endr
  ret
  .size F, .-F

  .globl G
  .type G, %function
G:
  ret
  .size G, .-G

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
.endif
