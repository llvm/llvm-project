! Test predefined macros for WebAssembly architectures.
! REQUIRES: webassembly-registered-target

! RUN: %flang_fc1 -triple wasm32-unknown-emscripten -cpp -E %s | FileCheck %s --check-prefixes=COMMON,WASM32
! RUN: %flang_fc1 -triple wasm64-unknown-emscripten -cpp -E %s | FileCheck %s --check-prefixes=COMMON,WASM64

! COMMON: integer :: wasm = 1
! COMMON: integer :: wasm_underscores = 1
! WASM32: integer :: wasm32 = 1
! WASM32: integer :: wasm32_underscores = 1
! WASM32-NOT: integer :: wasm64
! WASM64: integer :: wasm64 = 1
! WASM64: integer :: wasm64_underscores = 1
! WASM64-NOT: integer :: wasm32

#if __wasm
  integer :: wasm = __wasm
#endif
#if __wasm__
  integer :: wasm_underscores = __wasm__
#endif
#if __wasm32
  integer :: wasm32 = __wasm32
#endif
#if __wasm32__
  integer :: wasm32_underscores = __wasm32__
#endif
#if __wasm64
  integer :: wasm64 = __wasm64
#endif
#if __wasm64__
  integer :: wasm64_underscores = __wasm64__
#endif
end program
