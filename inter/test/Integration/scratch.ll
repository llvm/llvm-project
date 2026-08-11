; REQUIRES: host-supports-inter-bmg
; RUN: inter-translate %S/../Emit/scratch-exdesc.mlir --xemachine-to-zebin -o %t.bin
; RUN: inter-runner %t.bin scratch_exdesc 32
