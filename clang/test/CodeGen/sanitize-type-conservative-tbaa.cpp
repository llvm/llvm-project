// RUN: %clang --target=x86_64-linux-gnu -S -fsanitize=type -emit-llvm -o - %s \
// RUN:     | FileCheck %s --implicit-check-not='!{!"float"'


// With TySan enabled, when "omnipotent char" TBAA would be emitted,
// instead it should emit TysanConservativeTBAA. This gets recorded
// by TySan, and it emits a specially tagged type descriptor.

// CHECK: @__tysan_v1_TysanConservativeTBAA = linkonce_odr constant { i64, i64, ptr, i64, [22 x i8] } { i64 3, i64 1, ptr @__tysan_v1_omnipotent_20char, i64 0, [22 x i8] c"TysanConservativeTBAA\00" }, comdat

struct S{
    float forceConservativeTBAAPath[1];
};

S s;

// CHECK: !{!"TysanConservativeTBAA",
