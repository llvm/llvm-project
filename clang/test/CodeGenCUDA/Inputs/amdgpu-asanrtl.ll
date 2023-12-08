; Sample code for amdgpu address sanitizer runtime.

; Note the runtime functions need to have weak linkage and default
; visibility, otherwise they may be internalized and removed by GlobalOptPass.

define weak void @__amdgpu_device_library_preserve_asan_functions() {
  tail call void @__asan_report_load1(i64 0)
  ret void
}

define weak void @__asan_report_load1(i64 %0) {
  ret void
}
