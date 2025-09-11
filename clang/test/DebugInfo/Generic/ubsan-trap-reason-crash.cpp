// FIXME: We should emit a trap message for this case too.
// But sometimes Clang will emit a ubsan trap into the prologue of a function,
// at which point the debug-info locations haven't been set up yet and
// can't hook up our artificial inline frame. [Issue #150707]

// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -O0 -debug-info-kind=standalone -dwarf-version=5 \
// RUN: -fsanitize=null -fsanitize-trap=null -emit-llvm %s -o - | FileCheck %s

struct Foo {
  void target() {}
} f;

void caller() {
  f.target();
}


// CHECK-LABEL: @_Z6callerv
// CHECK: call void @llvm.ubsantrap(i8 22){{.*}}!nosanitize
// CHECK-NOT: __clang_trap_msg
