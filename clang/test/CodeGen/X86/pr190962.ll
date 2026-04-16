; RUN: %clang -O1 -mapx-features=ndd --target=x86_64-pc-windows-gnu -S %s -o /dev/null

;; Check no crash when building below IR with Clang.

define i32 @foo(ptr %0, ptr %1, ptr %2, i64 %3, i64 %4, i64 %5) {
  %7 = call i64 @"_ZZN3jxl15PatchDictionary6DecodeEP22JxlMemoryManagerStructPNS_9BitReaderEyyyPbENK3$_0clEy"()
  %8 = mul i64 %3, %4
  %9 = icmp ugt i64 1, %8
  br i1 %9, label %common.ret1, label %10

common.ret1:                                      ; preds = %26, %23, %16, %6
  %common.ret1.op = phi i32 [ 0, %23 ], [ 0, %16 ], [ 0, %26 ], [ 0, %6 ]
  ret i32 %common.ret1.op

10:                                               ; preds = %6
  %11 = load volatile i64, ptr null, align 8
  %12 = call i64 @"_ZZN3jxl15PatchDictionary6DecodeEP22JxlMemoryManagerStructPNS_9BitReaderEyyyPbENK3$_0clEy"()
  %13 = load volatile i64, ptr null, align 8
  %14 = or i64 %11, %5
  %15 = icmp ugt i64 %14, 0
  br i1 %15, label %16, label %19

16:                                               ; preds = %10
  call void @_ZN3jxl6StatusC2ENS_10StatusCodeE()
  %17 = load i32, ptr null, align 4
  %18 = call i32 (i32, ptr, ...) @_ZN3jxl13StatusMessageENS_6StatusEPKcz(i32 %17, ptr null, ptr null, i32 0)
  call void @_ZN3jxl6StatusC2ENS_10StatusCodeE()
  br label %common.ret1

19:                                               ; preds = %10
  %20 = call i64 @_ZNK3jxl11ImageBundle5ysizeEv(ptr %1)
  %21 = or i64 %12, %13
  %22 = icmp ugt i64 %21, 0
  br i1 %22, label %23, label %26

23:                                               ; preds = %19
  call void @_ZN3jxl6StatusC2ENS_10StatusCodeE()
  %24 = load i32, ptr null, align 4
  %25 = call i32 (i32, ptr, ...) @_ZN3jxl13StatusMessageENS_6StatusEPKcz(i32 %24, ptr null, ptr null, i32 1)
  call void @_ZN3jxl6StatusC2ENS_10StatusCodeE()
  br label %common.ret1

26:                                               ; preds = %19
  %27 = icmp ugt i64 1, %3
  br i1 %27, label %common.ret1, label %28

28:                                               ; preds = %26
  store i32 0, ptr %0, align 4
  %29 = call i32 (i32, ptr, ...) @_ZN3jxl13StatusMessageENS_6StatusEPKcz(i32 0, ptr null, ptr null, i32 0, i64 0, i64 0, i64 %4)
  unreachable
}

declare i32 @_ZN3jxl13StatusMessageENS_6StatusEPKcz(i32, ptr, ...)

declare i64 @"_ZZN3jxl15PatchDictionary6DecodeEP22JxlMemoryManagerStructPNS_9BitReaderEyyyPbENK3$_0clEy"()

declare void @_ZN3jxl6StatusC2ENS_10StatusCodeE()

declare i64 @_ZNK3jxl11ImageBundle5ysizeEv(ptr)

; uselistorder directives
uselistorder ptr @_ZN3jxl13StatusMessageENS_6StatusEPKcz, { 2, 1, 0 }
uselistorder ptr @"_ZZN3jxl15PatchDictionary6DecodeEP22JxlMemoryManagerStructPNS_9BitReaderEyyyPbENK3$_0clEy", { 1, 0 }
uselistorder ptr @_ZN3jxl6StatusC2ENS_10StatusCodeE, { 3, 2, 1, 0 }
