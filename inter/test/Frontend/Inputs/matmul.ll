; Reduced from Lighthouse's pipeline-check matmul for optimized import tests.

target datalayout = "e-p:64:64-p1:64:64-i64:64-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"

define spir_kernel void @matmul(ptr addrspace(1) noalias readonly %a,
                               ptr addrspace(1) noalias readonly %b,
                               ptr addrspace(1) noalias writeonly %c,
                               i64 %m, i64 %n) {
entry:
  %gid = call spir_func i64 @_Z13get_global_idj(i32 0)
  %elements = mul i64 %m, %n
  %active = icmp ult i64 %gid, %elements
  br i1 %active, label %setup, label %exit

setup:
  %row = udiv i64 %gid, %n
  %col = urem i64 %gid, %n
  br label %reduce

reduce:
  %index = phi i64 [ 0, %setup ], [ %next, %reduce ]
  %acc = phi float [ 0.0, %setup ], [ %sum, %reduce ]
  %a.row = mul i64 %row, 64
  %a.index = add i64 %a.row, %index
  %b.row = mul i64 %index, %n
  %b.index = add i64 %b.row, %col
  %a.ptr = getelementptr inbounds float, ptr addrspace(1) %a, i64 %a.index
  %b.ptr = getelementptr inbounds float, ptr addrspace(1) %b, i64 %b.index
  %a.value = load float, ptr addrspace(1) %a.ptr, align 4
  %b.value = load float, ptr addrspace(1) %b.ptr, align 4
  %product = fmul fast float %a.value, %b.value
  %sum = fadd fast float %acc, %product
  %next = add nuw i64 %index, 1
  %done = icmp eq i64 %next, 64
  br i1 %done, label %store, label %reduce, !llvm.loop !0

store:
  %c.ptr = getelementptr inbounds float, ptr addrspace(1) %c, i64 %gid
  store float %sum, ptr addrspace(1) %c.ptr, align 4
  br label %exit

exit:
  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
