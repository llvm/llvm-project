declare {i8, i1} @llvm.sadd.with.overflow.i8(i8, i8)
declare {i8, i1} @llvm.uadd.with.overflow.i8(i8, i8)

define i8 @__clc_add_sat_impl_s8(i8 %x, i8 %y) nounwind readnone alwaysinline {
  %call = call {i8, i1} @llvm.sadd.with.overflow.i8(i8 %x, i8 %y)
  %res = extractvalue {i8, i1} %call, 0
  %over = extractvalue {i8, i1} %call, 1
  %x.msb = ashr i8 %x, 7
  %x.limit = xor i8 %x.msb, 127
  %sat = select i1 %over, i8 %x.limit, i8 %res
  ret i8 %sat
}

define i8 @__clc_add_sat_impl_u8(i8 %x, i8 %y) nounwind readnone alwaysinline {
  %call = call {i8, i1} @llvm.uadd.with.overflow.i8(i8 %x, i8 %y)
  %res = extractvalue {i8, i1} %call, 0
  %over = extractvalue {i8, i1} %call, 1
  %sat = select i1 %over, i8 -1, i8 %res
  ret i8 %sat
}

declare {i16, i1} @llvm.sadd.with.overflow.i16(i16, i16)
declare {i16, i1} @llvm.uadd.with.overflow.i16(i16, i16)

define i16 @__clc_add_sat_impl_s16(i16 %x, i16 %y) nounwind readnone alwaysinline {
  %call = call {i16, i1} @llvm.sadd.with.overflow.i16(i16 %x, i16 %y)
  %res = extractvalue {i16, i1} %call, 0
  %over = extractvalue {i16, i1} %call, 1
  %x.msb = ashr i16 %x, 15
  %x.limit = xor i16 %x.msb, 32767
  %sat = select i1 %over, i16 %x.limit, i16 %res
  ret i16 %sat
}

define i16 @__clc_add_sat_impl_u16(i16 %x, i16 %y) nounwind readnone alwaysinline {
  %call = call {i16, i1} @llvm.uadd.with.overflow.i16(i16 %x, i16 %y)
  %res = extractvalue {i16, i1} %call, 0
  %over = extractvalue {i16, i1} %call, 1
  %sat = select i1 %over, i16 -1, i16 %res
  ret i16 %sat
}

declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32)
declare {i32, i1} @llvm.uadd.with.overflow.i32(i32, i32)

define i32 @__clc_add_sat_impl_s32(i32 %x, i32 %y) nounwind readnone alwaysinline {
  %call = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %x, i32 %y)
  %res = extractvalue {i32, i1} %call, 0
  %over = extractvalue {i32, i1} %call, 1
  %x.msb = ashr i32 %x, 31
  %x.limit = xor i32 %x.msb, 2147483647
  %sat = select i1 %over, i32 %x.limit, i32 %res
  ret i32 %sat
}

define i32 @__clc_add_sat_impl_u32(i32 %x, i32 %y) nounwind readnone alwaysinline {
  %call = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %x, i32 %y)
  %res = extractvalue {i32, i1} %call, 0
  %over = extractvalue {i32, i1} %call, 1
  %sat = select i1 %over, i32 -1, i32 %res
  ret i32 %sat
}

declare {i64, i1} @llvm.sadd.with.overflow.i64(i64, i64)
declare {i64, i1} @llvm.uadd.with.overflow.i64(i64, i64)

define i64 @__clc_add_sat_impl_s64(i64 %x, i64 %y) nounwind readnone alwaysinline {
  %call = call {i64, i1} @llvm.sadd.with.overflow.i64(i64 %x, i64 %y)
  %res = extractvalue {i64, i1} %call, 0
  %over = extractvalue {i64, i1} %call, 1
  %x.msb = ashr i64 %x, 63
  %x.limit = xor i64 %x.msb, 9223372036854775807
  %sat = select i1 %over, i64 %x.limit, i64 %res
  ret i64 %sat
}

define i64 @__clc_add_sat_impl_u64(i64 %x, i64 %y) nounwind readnone alwaysinline {
  %call = call {i64, i1} @llvm.uadd.with.overflow.i64(i64 %x, i64 %y)
  %res = extractvalue {i64, i1} %call, 0
  %over = extractvalue {i64, i1} %call, 1
  %sat = select i1 %over, i64 -1, i64 %res
  ret i64 %sat
}
