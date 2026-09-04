; REQUIRES: host-supports-inter-bmg
; RUN: inter-opt %S/../Emit/scratch-exdesc.mlir --inter-resource-info -o %t.mlir
; RUN: inter-translate %t.mlir --xemachine-to-zebin -o %t.bin
; RUN: inter-runner %t.bin scratch_exdesc 32
