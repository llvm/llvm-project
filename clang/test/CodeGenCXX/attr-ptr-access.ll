; ModuleID = '/home/nikolas/source/clang/clang/test/CodeGenCXX/attr-ptr-access.cpp'
source_filename = "/home/nikolas/source/clang/clang/test/CodeGenCXX/attr-ptr-access.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z5func1Pi(ptr noundef readnone %0) #0 {
entry:
  %.addr = alloca ptr, align 8
  store ptr %0, ptr %.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z5func2Ri(ptr noundef nonnull readnone align 4 dereferenceable(4) %0) #0 {
entry:
  %.addr = alloca ptr, align 8
  store ptr %0, ptr %.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z5func3Pi(ptr noundef readonly %0) #0 {
entry:
  %.addr = alloca ptr, align 8
  store ptr %0, ptr %.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z5func4Ri(ptr noundef nonnull readonly align 4 dereferenceable(4) %0) #0 {
entry:
  %.addr = alloca ptr, align 8
  store ptr %0, ptr %.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z5func5Pi(ptr noundef writeonly %0) #0 {
entry:
  %.addr = alloca ptr, align 8
  store ptr %0, ptr %.addr, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone
define dso_local void @_Z5func6Ri(ptr noundef nonnull writeonly align 4 dereferenceable(4) %0) #0 {
entry:
  %.addr = alloca ptr, align 8
  store ptr %0, ptr %.addr, align 8
  ret void
}

attributes #0 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 23.0.0git (git@github.com:philnik777/llvm-project.git 4d55fb46624afb17e8bb70b308cf0d0e8a48e6cd)"}
