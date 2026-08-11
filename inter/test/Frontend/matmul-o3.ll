; RUN: inter-translate %s --import-llvm | inter-opt --inter-normalize-cf | FileCheck %s
; RUN: inter-translate %s --import-llvm | inter-opt --inter-normalize-cf --inter-normalize-pointers | FileCheck %s --check-prefix=PTR
; Generated with opt -S -passes='default<O3>' from Inputs/matmul.ll.
;
; CHECK: module attributes {dlti.dl_spec = #dlti.dl_spec<
; CHECK-SAME: !llvm.ptr<1> = dense<64>
; CHECK-SAME: "dlti.global_memory_space" = 1 : ui64
; CHECK-SAME: llvm.target_triple = "spir64-unknown-unknown"
; CHECK: func.func @matmul(
; CHECK-SAME: !llvm.ptr<1> {llvm.noalias{{[^}]*}}llvm.readonly
; CHECK-SAME: !llvm.ptr<1> {llvm.noalias{{[^}]*}}llvm.readonly
; CHECK-SAME: !llvm.ptr<1> {llvm.noalias
; CHECK-SAME: attributes {
; CHECK-SAME: xemachine.kernel
; CHECK-SAME: xemachine.llvm_func_properties
; CHECK: cf.cond_br
; CHECK: llvm.freeze
; CHECK: llvm.fmul
; CHECK: cf.cond_br
; CHECK: return
; CHECK: llvm.func {{.*}}spir_funccc @_Z13get_global_idj(i32) -> i64
; PTR-LABEL: func.func @matmul
; PTR: xw.ptradd
; PTR: xw.ptradd {{.*}} {gep_flags = 3 : i32}
; PTR-NOT: llvm.getelementptr
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
