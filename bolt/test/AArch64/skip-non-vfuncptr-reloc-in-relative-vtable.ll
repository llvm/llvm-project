; Test the fix that BOLT should skip speical handling of any non virtual
; function pointer relocations in relative vtable.

; RUN: llc --filetype=obj -mtriple=aarch64-none-linux-gnu \
; RUN:     --relocation-model=pic -o %t.o %s
; RUN: %clang %cxxflags -fuse-ld=lld %t.o -o %t.so -Wl,-q
; RUN: llvm-bolt %t.so -o %t.bolted.so

$_fake_rtti_data = comdat any
@_fake_rtti_data = internal unnamed_addr constant [16 x i8] c"_FAKE_RTTI_DATA_", comdat, align 8

@_ZTV3gooE = internal unnamed_addr constant { { [3 x i32] } } { { [3 x i32] } { [3 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr @_fake_rtti_data to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV3gooE, i32 0, i32 0, i32 2) to i64)) to i32), i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @foo to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [3 x i32] }, ptr @_ZTV3gooE, i32 0, i32 0, i32 2) to i64)) to i32)] } }, align 4

define internal ptr @foo(ptr %this) {
  %1 = load ptr, ptr @_ZTV3gooE, align 8
  ret ptr %1
}
