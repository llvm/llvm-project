declare i8 @llvm.ctlz.i8(i8, i1)
declare i16 @llvm.ctlz.i16(i16, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i64 @llvm.ctlz.i64(i64, i1)

define i8 @__clc_clz_impl_s8(i8 %x) nounwind readnone alwaysinline {
  %call = call i8 @llvm.ctlz.i8(i8 %x, i1 0)
  ret i8 %call
}

define i8 @__clc_clz_impl_u8(i8 %x) nounwind readnone alwaysinline {
  %call = call i8 @llvm.ctlz.i8(i8 %x, i1 0)
  ret i8 %call
}

define i16 @__clc_clz_impl_s16(i16 %x) nounwind readnone alwaysinline {
  %call = call i16 @llvm.ctlz.i16(i16 %x, i1 0)
  ret i16 %call
}

define i16 @__clc_clz_impl_u16(i16 %x) nounwind readnone alwaysinline {
  %call = call i16 @llvm.ctlz.i16(i16 %x, i1 0)
  ret i16 %call
}

define i32 @__clc_clz_impl_s32(i32 %x) nounwind readnone alwaysinline {
  %call = call i32 @llvm.ctlz.i32(i32 %x, i1 0)
  ret i32 %call
}

define i32 @__clc_clz_impl_u32(i32 %x) nounwind readnone alwaysinline {
  %call = call i32 @llvm.ctlz.i32(i32 %x, i1 0)
  ret i32 %call
}

define i64 @__clc_clz_impl_s64(i64 %x) nounwind readnone alwaysinline {
  %call = call i64 @llvm.ctlz.i64(i64 %x, i1 0)
  ret i64 %call
}

define i64 @__clc_clz_impl_u64(i64 %x) nounwind readnone alwaysinline {
  %call = call i64 @llvm.ctlz.i64(i64 %x, i1 0)
  ret i64 %call
}
