declare i8 @__clc_clz_impl_s8(i8 %x)

define i8 @__clc_clz_s8(i8 %x) nounwind readnone alwaysinline {
  %call = call i8 @__clc_clz_impl_s8(i8 %x)
  ret i8 %call
}

declare i8 @__clc_clz_impl_u8(i8 %x)

define i8 @__clc_clz_u8(i8 %x) nounwind readnone alwaysinline {
  %call = call i8 @__clc_clz_impl_u8(i8 %x)
  ret i8 %call
}

declare i16 @__clc_clz_impl_s16(i16 %x)

define i16 @__clc_clz_s16(i16 %x) nounwind readnone alwaysinline {
  %call = call i16 @__clc_clz_impl_s16(i16 %x)
  ret i16 %call
}

declare i16 @__clc_clz_impl_u16(i16 %x)

define i16 @__clc_clz_u16(i16 %x) nounwind readnone alwaysinline {
  %call = call i16 @__clc_clz_impl_u16(i16 %x)
  ret i16 %call
}

declare i32 @__clc_clz_impl_s32(i32 %x)

define i32 @__clc_clz_s32(i32 %x) nounwind readnone alwaysinline {
  %call = call i32 @__clc_clz_impl_s32(i32 %x)
  ret i32 %call
}

declare i32 @__clc_clz_impl_u32(i32 %x)

define i32 @__clc_clz_u32(i32 %x) nounwind readnone alwaysinline {
  %call = call i32 @__clc_clz_impl_u32(i32 %x)
  ret i32 %call
}

declare i64 @__clc_clz_impl_s64(i64 %x)

define i64 @__clc_clz_s64(i64 %x) nounwind readnone alwaysinline {
  %call = call i64 @__clc_clz_impl_s64(i64 %x)
  ret i64 %call
}

declare i64 @__clc_clz_impl_u64(i64 %x)

define i64 @__clc_clz_u64(i64 %x) nounwind readnone alwaysinline {
  %call = call i64 @__clc_clz_impl_u64(i64 %x)
  ret i64 %call
}
