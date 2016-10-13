; ModuleID = 'get_ptr_as_int.bc'
source_filename = "get_ptr_as_int.ll"
target datalayout = "e-p:32:32-p1:64:64-p2:64:64-p3:32:32-p4:64:64-p5:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64"
target triple = "amdgcn--amdhsa"

%class.omptarget_nvptx_ThreadPrivateContext = type { [524288 x %class.omptarget_nvptx_TaskDescr], [524288 x %class.omptarget_nvptx_TaskDescr*], [524288 x i16], [524288 x i64], [524288 x i32], [524288 x i64], [524288 x i64], [524288 x i64], [524288 x i64], %class.omptarget_nvptx_TeamDescr*, %struct.omptarget_nvptx_GlobalICV* }
%class.omptarget_nvptx_TaskDescr = type { %union.anon, %class.omptarget_nvptx_TaskDescr* }
%union.anon = type { %"struct.omptarget_nvptx_TaskDescr::(anonymous union)::TaskDescr_items" }
%"struct.omptarget_nvptx_TaskDescr::(anonymous union)::TaskDescr_items" = type { i8, i8, i16, i16, i16, i16, i64 }
%class.omptarget_nvptx_TeamDescr = type <{ %class.omptarget_nvptx_TaskDescr, %class.omptarget_nvptx_WorkDescr, i32, [4 x i8] }>
%class.omptarget_nvptx_WorkDescr = type <{ %class.omptarget_nvptx_CounterGroup, %class.omptarget_nvptx_TaskDescr, i8, [7 x i8] }>
%class.omptarget_nvptx_CounterGroup = type { i64, i64, i64 }
%struct.omptarget_nvptx_GlobalICV = type { double, i8 }

@omptarget_nvptx_threadPrivateContext = external addrspace(1) global %class.omptarget_nvptx_ThreadPrivateContext*, align 8
@omptarget_nvptx_device_teamContexts = external addrspace(1) global [16 x [512 x %class.omptarget_nvptx_TeamDescr]], align 8
@omptarget_nvptx_device_globalICV = external addrspace(1) global [16 x %struct.omptarget_nvptx_GlobalICV], align 8

; Function Attrs: convergent nounwind
define i64 @get_ptr_as_int_device_threadPrivateContext(i32) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load %class.omptarget_nvptx_ThreadPrivateContext*, %class.omptarget_nvptx_ThreadPrivateContext* addrspace(1)* @omptarget_nvptx_threadPrivateContext, align 8
  %4 = load i32, i32* %2, align 4
  %5 = sext i32 %4 to i64
  %6 = getelementptr inbounds %class.omptarget_nvptx_ThreadPrivateContext, %class.omptarget_nvptx_ThreadPrivateContext* %3, i64 %5
  %7 = ptrtoint %class.omptarget_nvptx_ThreadPrivateContext* %6 to i64
  ret i64 %7
}

; Function Attrs: convergent nounwind
define i64 @get_ptr_as_int_device_teamContexts(i32, i32) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  store i32 %1, i32* %4, align 4
  %5 = load i32, i32* %3, align 4
  %6 = sext i32 %5 to i64
  %7 = getelementptr inbounds [16 x [512 x %class.omptarget_nvptx_TeamDescr]], [16 x [512 x %class.omptarget_nvptx_TeamDescr]] addrspace(1)* @omptarget_nvptx_device_teamContexts , i64 0, i64 %6
  %8 = load i32, i32* %4, align 4
  %9 = sext i32 %8 to i64
  %10 = getelementptr inbounds [512 x %class.omptarget_nvptx_TeamDescr], [512 x %class.omptarget_nvptx_TeamDescr] addrspace(1)* %7, i64 0, i64 %9
  %11 = ptrtoint %class.omptarget_nvptx_TeamDescr addrspace(1)* %10 to i64
  ret i64 %11
}

; Function Attrs: convergent nounwind
define i64 @get_ptr_as_int_device_globalICV(i32) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = sext i32 %3 to i64
  %5 = getelementptr inbounds [16 x %struct.omptarget_nvptx_GlobalICV], [16 x %struct.omptarget_nvptx_GlobalICV] addrspace(1)* @omptarget_nvptx_device_globalICV , i64 0, i64 %4
  %6 = ptrtoint %struct.omptarget_nvptx_GlobalICV addrspace(1)* %5 to i64
  ret i64 %6
}

attributes #0 = { convergent nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="kaveri" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}
!nvvm.internalize.after.link = !{}
!nvvmir.version = !{!2}
!nvvm.annotations = !{!3, !4, !3, !5, !5, !5, !5, !6, !6, !5}

!0 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!1 = !{!"clang version 4.0.0 "}
!2 = !{i32 1, i32 2}
!3 = !{null, !"align", i32 8}
!4 = !{null, !"align", i32 8, !"align", i32 65544, !"align", i32 131080}
!5 = !{null, !"align", i32 16}
!6 = !{null, !"align", i32 16, !"align", i32 65552, !"align", i32 131088}
