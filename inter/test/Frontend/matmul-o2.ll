; RUN: inter-translate %s --import-llvm | inter-opt --inter-import-llvm | FileCheck %s
; RUN: inter-translate %s --import-llvm | inter-opt \
; RUN:   --inter-import-llvm --lift-cf-to-scf --inter-verify-structured \
; RUN:   --inter-convert-llvm-to-xw --inter-refine-distribution \
; RUN:   --canonicalize --inter-expand-arithmetic --canonicalize --cse \
; RUN:   --inter-infer-memory-tokens \
; RUN:   --inter-select-to-machine --verify-each | \
; RUN:   FileCheck %s --check-prefix=SELECT
; RUN: inter-translate %s --import-llvm | inter-opt --verify-each \
; RUN:   --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' | \
; RUN:   FileCheck %s --check-prefix=BACKEND
; RUN: inter-translate %s --import-llvm | inter-opt --verify-each \
; RUN:   --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' -o %t.xemachine.mlir
; RUN: inter-translate --xemachine-to-asm %t.xemachine.mlir -o %t.asm
; RUN: inter-translate --xemachine-to-zebin %t.xemachine.mlir -o %t.zebin
; Generated with opt -S -passes='default<O2>' from Inputs/matmul.ll.
;
; CHECK: module attributes {dlti.dl_spec = #dlti.dl_spec<
; CHECK-SAME: !llvm.ptr<1> = dense<64>
; CHECK-SAME: "dlti.global_memory_space" = 1 : ui64
; CHECK-SAME: llvm.target_triple = "spir64-unknown-unknown"
; CHECK: func.func @matmul(
; CHECK-SAME: attributes {
; CHECK-SAME: xw.kernel
; CHECK-SAME: xw.kernel_args = [{access = "read_only", address_space = 1 : i32, alignment = 8 : i64, kind = "pointer", offset = 24 : i64, size = 8 : i64}
; CHECK-SAME: {access = "read_only", address_space = 1 : i32, alignment = 8 : i64, kind = "pointer", offset = 32 : i64, size = 8 : i64}
; CHECK-SAME: {access = "write_only", address_space = 1 : i32, alignment = 8 : i64, kind = "pointer", offset = 40 : i64, size = 8 : i64}
; CHECK: cf.cond_br
; CHECK: llvm.freeze
; CHECK: llvm.fmul
; CHECK: cf.cond_br
; CHECK: return
; CHECK: llvm.func {{.*}}spir_funccc @_Z13get_global_idj(i32) -> i64
; SELECT-LABEL: func.func @matmul
; SELECT: xemachine.shr
; SELECT: xemachine.cmp
; SELECT: xemachine.exec_if
; SELECT: xemachine.uniform_loop
; SELECT: xemachine.eot
; SELECT-NOT: xw.
; BACKEND-LABEL: func.func @matmul
; BACKEND-SAME: xemachine.grf_used =
; BACKEND-SAME: xemachine.regalloc_iterations =
; BACKEND: !xemachine.arf<f, 2, {{[01]}}>
; BACKEND: xemachine.eot
; BACKEND-NOT: !xemachine.arf<f, 2, -1>
; BACKEND-NOT: !xemachine.reg<{{[0-9]+}}, -1>
;
; ModuleID = 'inter/test/Frontend/Inputs/matmul.ll'
source_filename = "inter/test/Frontend/Inputs/matmul.ll"
target datalayout = "e-p:64:64-p1:64:64-i64:64-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @matmul(ptr addrspace(1) noalias readonly captures(none) %a, ptr addrspace(1) noalias readonly captures(none) %b, ptr addrspace(1) noalias nofree writeonly captures(none) %c, i64 %m, i64 %n) local_unnamed_addr {
entry:
  %gid = tail call spir_func i64 @_Z13get_global_idj(i32 0)
  %elements = mul i64 %n, %m
  %active = icmp ult i64 %gid, %elements
  br i1 %active, label %setup, label %exit

setup:                                            ; preds = %entry
  %gid.frozen = freeze i64 %gid
  %n.frozen = freeze i64 %n
  %row = udiv i64 %gid.frozen, %n.frozen
  %0 = mul i64 %row, %n.frozen
  %col.decomposed = sub i64 %gid.frozen, %0
  %.idx = shl i64 %row, 8
  %1 = getelementptr i8, ptr addrspace(1) %a, i64 %.idx
  %invariant.gep = getelementptr [4 x i8], ptr addrspace(1) %b, i64 %col.decomposed
  br label %reduce

reduce:                                           ; preds = %reduce, %setup
  %index = phi i64 [ 0, %setup ], [ %next, %reduce ]
  %acc = phi float [ 0.000000e+00, %setup ], [ %sum, %reduce ]
  %b.row = mul i64 %index, %n
  %a.ptr = getelementptr [4 x i8], ptr addrspace(1) %1, i64 %index
  %gep = getelementptr [4 x i8], ptr addrspace(1) %invariant.gep, i64 %b.row
  %a.value = load float, ptr addrspace(1) %a.ptr, align 4
  %b.value = load float, ptr addrspace(1) %gep, align 4
  %product = fmul fast float %b.value, %a.value
  %sum = fadd fast float %product, %acc
  %next = add nuw nsw i64 %index, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %store, label %reduce, !llvm.loop !0

store:                                            ; preds = %reduce
  %c.ptr = getelementptr inbounds [4 x i8], ptr addrspace(1) %c, i64 %gid
  store float %sum, ptr addrspace(1) %c.ptr, align 4
  br label %exit

exit:                                             ; preds = %store, %entry
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32) local_unnamed_addr

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
