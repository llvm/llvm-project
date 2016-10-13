; ModuleID = 'builtins.bc'
source_filename = "builtins.ll"
target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"

; Function Attrs: nounwind readnone
declare i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workgroup.id.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workgroup.id.y() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workgroup.id.z() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.y() #0

; Function Attrs: nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.z() #0

; Function Attrs: convergent nounwind
declare void @llvm.amdgcn.s.barrier() #1

; Function Attrs: alwaysinline
define i32 @llvm_amdgcn_read_local_size_x() #2 {
  %dispatch_ptr = call noalias nonnull dereferenceable(64) i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  %dispatch_ptr_i32 = bitcast i8 addrspace(2)* %dispatch_ptr to i32 addrspace(2)*
  %size_xy_ptr = getelementptr inbounds i32, i32 addrspace(2)* %dispatch_ptr_i32, i64 1
  %size_xy = load i32, i32 addrspace(2)* %size_xy_ptr, align 4, !invariant.load !0
  %1 = and i32 %size_xy, 65535
  ret i32 %1
}

; Function Attrs: alwaysinline nounwind
define i32 @llvm_amdgcn_read_num_groups_x() #3 {
  %1 = tail call i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #0
  %2 = getelementptr inbounds i8, i8 addrspace(2)* %1, i64 12
  %3 = bitcast i8 addrspace(2)* %2 to i32 addrspace(2)*
  %4 = load i32, i32 addrspace(2)* %3, align 4, !tbaa !1
  %5 = getelementptr inbounds i8, i8 addrspace(2)* %1, i64 4
  %6 = bitcast i8 addrspace(2)* %5 to i16 addrspace(2)*
  %7 = load i16, i16 addrspace(2)* %6, align 4, !tbaa !10
  %8 = zext i16 %7 to i32
  %9 = udiv i32 %4, %8
  %10 = mul i32 %9, %8
  %11 = icmp ugt i32 %4, %10
  %12 = zext i1 %11 to i32
  %13 = add i32 %12, %9
  ret i32 %13
}

attributes #0 = { nounwind readnone }
attributes #1 = { convergent nounwind }
attributes #2 = { alwaysinline }
attributes #3 = { alwaysinline nounwind }

!0 = !{}
!1 = !{!2, !6, i64 12}
!2 = !{!"hsa_kernel_dispatch_packet_s", !3, i64 0, !3, i64 2, !3, i64 4, !3, i64 6, !3, i64 8, !3, i64 10, !6, i64 12, !6, i64 16, !6, i64 20, !6, i64 24, !6, i64 28, !7, i64 32, !8, i64 40, !6, i64 48, !7, i64 56, !9, i64 64}
!3 = !{!"short", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!"int", !4, i64 0}
!7 = !{!"long", !4, i64 0}
!8 = !{!"any pointer", !4, i64 0}
!9 = !{!"hsa_signal_s", !7, i64 0}
!10 = !{!2, !3, i64 4}
