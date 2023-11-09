; REQUIRES: arm-registered-target
; RUN: %clang --target=arm-none-eabi -mcpu=cortex-m85 -mfloat-abi=hard -save-temps=obj -S -o - %s | FileCheck %s
; RUN: %clang --target=arm-none-eabi -mcpu=cortex-m55 -mfloat-abi=hard -save-temps=obj -S -o - %s | FileCheck %s
; RUN: %clang --target=arm-none-eabi -mcpu=cortex-m85 -mfloat-abi=hard -O2 -c -mthumb -save-temps=obj %s
; RUN: %clang --target=arm-none-eabi -mcpu=cortex-m55 -mfloat-abi=hard -O2 -c -mthumb -save-temps=obj %s
; CHECK: .fpu   fpv5-d16
; CHECK-NEXT  .arch_extension mve.fp

%struct.dummy_t = type { float, float, float, float }

define dso_local signext i8 @foo(ptr noundef %handle) #0 {
entry:
  %handle.addr = alloca ptr, align 4
  store ptr %handle, ptr %handle.addr, align 4
  %0 = load ptr, ptr %handle.addr, align 4
  %a = getelementptr inbounds %struct.dummy_t, ptr %0, i32 0, i32 0
  %1 = load float, ptr %a, align 4
  %sub = fsub float 0x3F5439DE40000000, %1
  %2 = load ptr, ptr %handle.addr, align 4
  %a1 = getelementptr inbounds %struct.dummy_t, ptr %2, i32 0, i32 0
  %3 = load float, ptr %a1, align 4
  %4 = call float @llvm.fmuladd.f32(float 0x3F847AE140000000, float %sub, float %3)
  store float %4, ptr %a1, align 4
  %5 = load ptr, ptr %handle.addr, align 4
  %b = getelementptr inbounds %struct.dummy_t, ptr %5, i32 0, i32 1
  %6 = load float, ptr %b, align 4
  %sub2 = fsub float 0x3F5439DE40000000, %6
  %7 = load ptr, ptr %handle.addr, align 4
  %b3 = getelementptr inbounds %struct.dummy_t, ptr %7, i32 0, i32 1
  %8 = load float, ptr %b3, align 4
  %9 = call float @llvm.fmuladd.f32(float 0x3F947AE140000000, float %sub2, float %8)
  store float %9, ptr %b3, align 4
  %10 = load ptr, ptr %handle.addr, align 4
  %c = getelementptr inbounds %struct.dummy_t, ptr %10, i32 0, i32 2
  %11 = load float, ptr %c, align 4
  %sub4 = fsub float 0x3F5439DE40000000, %11
  %12 = load ptr, ptr %handle.addr, align 4
  %c5 = getelementptr inbounds %struct.dummy_t, ptr %12, i32 0, i32 2
  %13 = load float, ptr %c5, align 4
  %14 = call float @llvm.fmuladd.f32(float 0x3F9EB851E0000000, float %sub4, float %13)
  store float %14, ptr %c5, align 4
  %15 = load ptr, ptr %handle.addr, align 4
  %d = getelementptr inbounds %struct.dummy_t, ptr %15, i32 0, i32 3
  %16 = load float, ptr %d, align 4
  %sub6 = fsub float 0x3F5439DE40000000, %16
  %17 = load ptr, ptr %handle.addr, align 4
  %d7 = getelementptr inbounds %struct.dummy_t, ptr %17, i32 0, i32 3
  %18 = load float, ptr %d7, align 4
  %19 = call float @llvm.fmuladd.f32(float 0x3FA47AE140000000, float %sub6, float %18)
  store float %19, ptr %d7, align 4
  ret i8 0
}

declare float @llvm.fmuladd.f32(float, float, float) #1
