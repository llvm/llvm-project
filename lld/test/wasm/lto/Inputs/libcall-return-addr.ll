target datalayout = "e-m:e-p:32:32-p10:8:8-p20:8:8-i64:64-i128:128-f128:64-n32:64-S128-ni:1:10:20"
target triple = "wasm32-unknown-emscripten"

define ptr @emscripten_return_address() {
  ret ptr null
}
