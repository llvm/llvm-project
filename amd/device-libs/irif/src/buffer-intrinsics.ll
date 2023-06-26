target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

declare <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32>, i32, i32, i32, i32 immarg) #0
declare <4 x half> @llvm.amdgcn.struct.buffer.load.format.v4f16(<4 x i32>, i32, i32, i32, i32 immarg) #0
declare void @llvm.amdgcn.struct.buffer.store.format.v4f32(<4 x float>, <4 x i32>, i32, i32, i32, i32 immarg) #1
declare void @llvm.amdgcn.struct.buffer.store.format.v4f16(<4 x half>, <4 x i32>, i32, i32, i32, i32 immarg) #1


define <4 x float> @__llvm_amdgcn_struct_buffer_load_format_v4f32(<4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset) #0 {
  %1 = call <4 x float> @llvm.amdgcn.struct.buffer.load.format.v4f32(<4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset, i32 0)
  ret <4 x float> %1
}

define <4 x half> @__llvm_amdgcn_struct_buffer_load_format_v4f16(<4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset) #0 {
  %1 = call <4 x half> @llvm.amdgcn.struct.buffer.load.format.v4f16(<4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset, i32 0)
  ret <4 x half> %1
}

define void @__llvm_amdgcn_struct_buffer_store_format_v4f32(<4 x float> %vdata, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset) #1 {
  call void @llvm.amdgcn.struct.buffer.store.format.v4f32(<4 x float> %vdata, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset, i32 0)
  ret void
}

define void @__llvm_amdgcn_struct_buffer_store_format_v4f16(<4 x half> %vdata, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset) #1 {
  call void @llvm.amdgcn.struct.buffer.store.format.v4f16(<4 x half> %vdata, <4 x i32> %rsrc, i32 %vindex, i32 %voffset, i32 %soffset, i32 0)
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(read) }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(write) }
