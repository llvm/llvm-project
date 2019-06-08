target triple = "wasm32-unknown-unknown"

$foo = comdat any

@constantData = constant [3 x i8] c"abc", comdat($foo)

define i32 @comdatFn() comdat($foo) {
  ret i32 ptrtoint ([3 x i8]* @constantData to i32)
}

define internal void @do_init() comdat($foo) {
  ret void
}

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void
()*, i8* } { i32 65535, void ()* @do_init, i8* null }]

; Everything above this is part of the `foo` comdat group

define i32 @callComdatFn1() {
    ret i32 ptrtoint (i32 ()* @comdatFn to i32)
}
