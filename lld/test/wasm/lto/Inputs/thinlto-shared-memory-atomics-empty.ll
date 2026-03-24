
target triple = "wasm32-unknown-emscripten"

; Thread-local variable that forces generation of TLS layout
@my_tls = thread_local global i32 42, align 4

; This function will be removed by dropDeadSymbols because it's unused,
; taking its target-features attribute block along with it.
define void @unused() #0 {
entry:
  ret void
}

attributes #0 = { "target-features"="+atomics,+bulk-memory,+mutable-globals,+sign-ext" }
