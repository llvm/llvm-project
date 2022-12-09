target triple = "wasm32-unknown-unknown"

$foo = comdat any

@constantData = constant [3 x i8] c"abc", comdat($foo)

define i32 @comdatFn() comdat($foo) {
  ret i32 ptrtoint (ptr @constantData to i32)
}

define internal void @do_init() comdat($foo) {
  ret void
}

%0 = type { i32, ptr, ptr }
@llvm.global_ctors = appending global [1 x %0 ] [%0 { i32 65535, ptr @do_init, ptr null }]

; Everything above this is part of the `foo` comdat group

define i32 @callComdatFn1() {
    ret i32 ptrtoint (ptr @comdatFn to i32)
}
