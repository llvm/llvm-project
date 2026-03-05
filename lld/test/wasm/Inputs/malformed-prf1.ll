; ModuleID = 'malformed-prf1.c'
source_filename = "malformed-prf1.c"
target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-unknown"

$__llvm_profile_runtime_user = comdat any

$__covrec_5CF8C24CDB18BDACu = comdat any

@__covrec_5CF8C24CDB18BDACu = linkonce_odr hidden constant <{ i64, i32, i64, i64, [9 x i8] }> <{ i64 6699318081062747564, i32 9, i64 0, i64 -8255375002284483420, [9 x i8] c"\01\01\00\01\01\03\0C\02\02" }>, section "__llvm_covfun", comdat, align 8
@__llvm_coverage_mapping = private constant { { i32, i32, i32, i32 }, [76 x i8] } { { i32, i32, i32, i32 } { i32 0, i32 76, i32 0, i32 6 }, [76 x i8] c"\02EIx\DA\0D\CA\D1\09\800\0C\05\C0\15\\D\1F\E2\14\8E\11\DA\94\22MS\92T\D7\B7\9F\07w\A1\AA0f%\F3\0E\E3\A1h\ED\95}\98>\9Cb!#\D8\03\1F\B9\E0\EEc\86oB\AD\A8\09\E7\D5\CAy\A4\1F\85E\19\B0" }, section "__llvm_covmap", align 8
@__llvm_profile_runtime = external hidden global i32
@__profc_foo = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_foo = private global { i64, i64, i32, i32, ptr, ptr, i32, [3 x i16], i32 } { i64 6699318081062747564, i64 0, i32 sub (i32 ptrtoint (ptr @__profc_foo to i32), i32 ptrtoint (ptr @__profd_foo to i32)), i32 0, ptr null, ptr null, i32 1, [3 x i16] zeroinitializer, i32 0 }, section "__llvm_prf_data", align 8
@__llvm_prf_nm = private constant [13 x i8] c"\03\0Bx\DAK\CB\CF\07\00\02\82\01E", section "__llvm_prf_names", align 1
@llvm.used = appending global [5 x ptr] [ptr @__covrec_5CF8C24CDB18BDACu, ptr @__llvm_coverage_mapping, ptr @__llvm_profile_runtime_user, ptr @__profd_foo, ptr @__llvm_prf_nm], section "llvm.metadata"

; Function Attrs: noinline nounwind optnone
define hidden void @foo() #0 {
  %1 = load i64, ptr @__profc_foo, align 8
  %2 = add i64 %1, 1
  store i64 %2, ptr @__profc_foo, align 8
  call void @bar()
  ret void
}

; Function Attrs: nounwind
declare void @llvm.instrprof.increment(ptr, i64, i32, i32) #1

declare void @bar(...) #2

; Function Attrs: noinline
define linkonce_odr hidden i32 @__llvm_profile_runtime_user() #3 comdat {
  %1 = load i32, ptr @__llvm_profile_runtime, align 4
  ret i32 %1
}

attributes #0 = { noinline nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+bulk-memory,+bulk-memory-opt,+call-indirect-overlong,+multivalue,+mutable-globals,+nontrapping-fptoint,+reference-types,+sign-ext" }
attributes #1 = { nounwind }
attributes #2 = { "no-prototype" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+bulk-memory,+bulk-memory-opt,+call-indirect-overlong,+multivalue,+mutable-globals,+nontrapping-fptoint,+reference-types,+sign-ext" }
attributes #3 = { noinline }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"EnableValueProfiling", i32 0}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{!"clang version 21.1.6"}
