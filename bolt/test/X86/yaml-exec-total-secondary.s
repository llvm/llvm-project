## Raw external entry counts must stay separate from computed block totals.
# REQUIRES: system-linux
# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -data %t.fdata -w %t.yaml -o %t.out \
# RUN:   --print-cfg --print-only=main 2>&1 | FileCheck %s --check-prefix=CFG
# RUN: FileCheck %s --check-prefix=YAML < %t.yaml
# RUN: llvm-bolt %t.exe -data %t.yaml -o %t.read \
# RUN:   --print-cfg --print-only=main 2>&1 | FileCheck %s --check-prefix=CFG
# RUN: sed '/exec-total:/d' %t.yaml > %t.legacy.yaml
# RUN: llvm-bolt %t.exe -data %t.legacy.yaml -o %t.legacy \
# RUN:   --print-cfg --print-only=main 2>&1 | FileCheck %s --check-prefix=CFG

# RUN: link_fdata %s %t.exe %t.preagg PREAGG
# RUN: perf2bolt %t.exe -pa -p %t.preagg -profile-format=yaml -o %t.aggregate.yaml
# RUN: FileCheck %s --check-prefix=YAML < %t.aggregate.yaml

# YAML:      - name:            main
# YAML:        blocks:
# YAML:          - bid:             0
# YAML:            exec:            90
# YAML:            exec-total:      90
# YAML:          - bid:             1
# YAML:            exec:            10
# YAML-NEXT:       exec-total:      100
# CFG: Binary Function "main" after building cfg {
# CFG:   Exec Count : 90
# CFG:   Exec Count : 100
# CFG-NOT: Exec Count : 190

  .text
  .globl main
  .type main,@function
main:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 main 0 0 90
# PREAGG: B X:0 #main# 90 0
Ljump:
  jmp secondary_entry
# FDATA: 1 main #Ljump# 1 main #secondary_entry# 0 90
# PREAGG: B #main# #secondary_entry# 90 0
  .globl secondary_entry
secondary_entry:
# FDATA: 0 [unknown] 0 1 secondary_entry 0 0 10
# PREAGG: B X:0 #secondary_entry# 10 0
  retq
  .cfi_endproc
  .size main, .-main
