target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

@indirect_bar = hidden local_unnamed_addr global i32 ()* @bar, align 4

; Function Attrs: norecurse nounwind readnone
define i32 @bar() #0 {
entry:
  ret i32 1
}

; Function Attrs: nounwind
define void @call_bar_indirect() local_unnamed_addr #1 {
entry:
  %0 = load i32 ()*, i32 ()** @indirect_bar, align 4
  %call = tail call i32 %0() #2
  ret void
}
