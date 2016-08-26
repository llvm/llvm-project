declare i32 @llvm.r600.read.ngroups.x() nounwind readnone
declare i32 @llvm.r600.read.ngroups.y() nounwind readnone
declare i32 @llvm.r600.read.ngroups.z() nounwind readnone

define i32 @get_num_groups(i32 %dim) nounwind readnone alwaysinline {
  switch i32 %dim, label %default [i32 0, label %x_dim i32 1, label %y_dim i32 2, label %z_dim]
x_dim:
  %x = call i32 @llvm.r600.read.ngroups.x() nounwind readnone
  ret i32 %x
y_dim:
  %y = call i32 @llvm.r600.read.ngroups.y() nounwind readnone
  ret i32 %y
z_dim:
  %z = call i32 @llvm.r600.read.ngroups.z() nounwind readnone
  ret i32 %z
default:
  ret i32 1
}
