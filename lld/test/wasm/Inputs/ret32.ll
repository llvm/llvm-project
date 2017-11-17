target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

; Function Attrs: norecurse nounwind readnone
define hidden i32 @ret32(float %arg) #0 {
entry:
    ret i32 0
     ; ptrtoint (i32 (float)* @ret32 to i32)
}
