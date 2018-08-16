declare i32 @llvm.r600.read.global.size.x() nounwind readnone
declare i32 @llvm.r600.read.global.size.y() nounwind readnone
declare i32 @llvm.r600.read.global.size.z() nounwind readnone

target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"

define i64 @get_global_size(i32 %dim) nounwind readnone alwaysinline {
  switch i32 %dim, label %default [i32 0, label %x_dim i32 1, label %y_dim i32 2, label %z_dim]
x_dim:
  %x = call i32 @llvm.r600.read.global.size.x()
  %x.ext = zext i32 %x to i64
  ret i64 %x.ext
y_dim:
  %y = call i32 @llvm.r600.read.global.size.y()
  %y.ext = zext i32 %y to i64
  ret i64 %y.ext
z_dim:
  %z = call i32 @llvm.r600.read.global.size.z()
  %z.ext = zext i32 %z to i64
  ret i64 %z.ext
default:
  ret i64 1
}
