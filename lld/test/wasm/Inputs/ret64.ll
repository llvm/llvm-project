target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

define i64 @ret64(double %arg) local_unnamed_addr #0 {
entry:
    ret i64 1
}
