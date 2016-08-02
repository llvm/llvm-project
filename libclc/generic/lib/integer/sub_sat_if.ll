declare i8 @__clc_sub_sat_impl_s8(i8 %x, i8 %y)

define i8 @__clc_sub_sat_s8(i8 %x, i8 %y) nounwind readnone alwaysinline {
  %call = call i8 @__clc_sub_sat_impl_s8(i8 %x, i8 %y)
  ret i8 %call
}

declare i8 @__clc_sub_sat_impl_u8(i8 %x, i8 %y)

define i8 @__clc_sub_sat_u8(i8 %x, i8 %y) nounwind readnone alwaysinline {
  %call = call i8 @__clc_sub_sat_impl_u8(i8 %x, i8 %y)
  ret i8 %call
}

declare i16 @__clc_sub_sat_impl_s16(i16 %x, i16 %y)

define i16 @__clc_sub_sat_s16(i16 %x, i16 %y) nounwind readnone alwaysinline {
  %call = call i16 @__clc_sub_sat_impl_s16(i16 %x, i16 %y)
  ret i16 %call
}

declare i16 @__clc_sub_sat_impl_u16(i16 %x, i16 %y)

define i16 @__clc_sub_sat_u16(i16 %x, i16 %y) nounwind readnone alwaysinline {
  %call = call i16 @__clc_sub_sat_impl_u16(i16 %x, i16 %y)
  ret i16 %call
}

declare i32 @__clc_sub_sat_impl_s32(i32 %x, i32 %y)

define i32 @__clc_sub_sat_s32(i32 %x, i32 %y) nounwind readnone alwaysinline {
  %call = call i32 @__clc_sub_sat_impl_s32(i32 %x, i32 %y)
  ret i32 %call
}

declare i32 @__clc_sub_sat_impl_u32(i32 %x, i32 %y)

define i32 @__clc_sub_sat_u32(i32 %x, i32 %y) nounwind readnone alwaysinline {
  %call = call i32 @__clc_sub_sat_impl_u32(i32 %x, i32 %y)
  ret i32 %call
}

declare i64 @__clc_sub_sat_impl_s64(i64 %x, i64 %y)

define i64 @__clc_sub_sat_s64(i64 %x, i64 %y) nounwind readnone alwaysinline {
  %call = call i64 @__clc_sub_sat_impl_s64(i64 %x, i64 %y)
  ret i64 %call
}

declare i64 @__clc_sub_sat_impl_u64(i64 %x, i64 %y)

define i64 @__clc_sub_sat_u64(i64 %x, i64 %y) nounwind readnone alwaysinline {
  %call = call i64 @__clc_sub_sat_impl_u64(i64 %x, i64 %y)
  ret i64 %call
}
