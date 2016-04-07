declare i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr() #0

define i32 @get_local_size(i32 %dim) #1 {
  %dispatch_ptr = call noalias nonnull dereferenceable(64) i8 addrspace(2)* @llvm.amdgcn.dispatch.ptr()
  %dispatch_ptr_i32 = bitcast i8 addrspace(2)* %dispatch_ptr to i32 addrspace(2)*
  %xy_size_ptr = getelementptr inbounds i32, i32 addrspace(2)* %dispatch_ptr_i32, i64 1
  %xy_size = load i32, i32 addrspace(2)* %xy_size_ptr, align 4, !invariant.load !0
  switch i32 %dim, label %default [
    i32 0, label %x_dim
    i32 1, label %y_dim
    i32 2, label %z_dim
  ]

x_dim:
  %x_size = and i32 %xy_size, 65535
  ret i32 %x_size

y_dim:
  %y_size = lshr i32 %xy_size, 16
  ret i32 %y_size

z_dim:
  %z_size_ptr = getelementptr inbounds i32, i32 addrspace(2)* %dispatch_ptr_i32, i64 2
  %z_size = load i32, i32 addrspace(2)* %z_size_ptr, align 4, !invariant.load !0, !range !1
  ret i32 %z_size

default:
  ret i32 1
}

attributes #0 = { nounwind readnone }
attributes #1 = { alwaysinline norecurse nounwind readonly }

!0 = !{}
!1 = !{ i32 0, i32 257 }
