target triple = "wasm32-unknown-unknown"

define hidden void @myctor() {
entry:
  ret void
}

define hidden void @mydtor() {
entry:
  %ptr = alloca i32
  ret void
}

@llvm.global_ctors = appending global [3 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 2002, ptr @myctor, ptr null },
  { i32, ptr, ptr } { i32 101, ptr @myctor, ptr null },
  { i32, ptr, ptr } { i32 202, ptr @myctor, ptr null }
]

@llvm.global_dtors = appending global [3 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 2002, ptr @mydtor, ptr null },
  { i32, ptr, ptr } { i32 101, ptr @mydtor, ptr null },
  { i32, ptr, ptr } { i32 202, ptr @mydtor, ptr null }
]
