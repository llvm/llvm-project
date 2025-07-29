// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited \
// RUN:   -fsanitize=null %s -o - | FileCheck %s

// Check that santizer check calls have a !dbg location.
// CHECK: define {{.*}}acquire{{.*}} !dbg
// CHECK-NOT: define
// CHECK: call void {{.*}}@__ubsan_handle_null_pointer_use
// CHECK-SAME: !dbg

struct SourceLocation {
  SourceLocation acquire() { return {}; };
};
extern "C" void __ubsan_handle_null_pointer_use(SourceLocation *Loc);
static void handleNullPointerUseImpl(SourceLocation *Loc) { Loc->acquire(); }
void __ubsan_handle_null_pointer_use(SourceLocation *Loc) {
  handleNullPointerUseImpl(Loc);
}
