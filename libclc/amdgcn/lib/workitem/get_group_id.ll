declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0

define i32 @get_group_id(i32 %dim) #1 {
  switch i32 %dim, label %default [
    i32 0, label %x_dim
    i32 1, label %y_dim
    i32 2, label %z_dim
  ]

x_dim:
  %x = tail call i32 @llvm.amdgcn.workgroup.id.x()
  ret i32 %x

y_dim:
  %y = tail call i32 @llvm.amdgcn.workgroup.id.y()
  ret i32 %y

z_dim:
  %z = tail call i32 @llvm.amdgcn.workgroup.id.z()
  ret i32 %z

default:
  ret i32 0
}

attributes #0 = { nounwind readnone }
attributes #1 = { alwaysinline norecurse nounwind readnone }
