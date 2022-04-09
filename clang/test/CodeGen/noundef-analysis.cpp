// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-darwin -enable-noundef-analysis -emit-llvm -o - %s | FileCheck %s -check-prefix ENABLED
// RUN: %clang_cc1 -no-opaque-pointers -triple arm64-darwin -no-enable-noundef-analysis -emit-llvm -o - %s | FileCheck %s -check-prefix DISABLED

union u1 {
  int val;
};

struct s1 {
  int val;
};

int indirect_callee_int(int a) { return a; }
union u1 indirect_callee_union(union u1 a) {
  return a;
}

static int sink;

static void examineValue(int x) { sink = x; }

// ENABLED-LABEL: @main(
// ENABLED:    [[CALL:%.*]] = call noundef {{.*}}i32 @_Z19indirect_callee_inti(i32 noundef {{.*}}0)
// ENABLED:    [[CALL1:%.*]] = call i32 @_Z21indirect_callee_union2u1(i64 {{.*}})
// ENABLED:    [[CALL2:%.*]] = call noalias noundef nonnull i8* @_Znwm(i64 noundef 4) #[[ATTR4:[0-9]+]]
// ENABLED:    call void @_ZL12examineValuei(i32 noundef {{.*}})
// DISABLED-LABEL: @main(
// DISABLED:    [[CALL:%.*]] = call {{.*}}i32 @_Z19indirect_callee_inti(i32 {{.*}}0)
// DISABLED:    [[CALL1:%.*]] = call i32 @_Z21indirect_callee_union2u1(i64 {{.*}})
// DISABLED:    [[CALL2:%.*]] = call noalias nonnull i8* @_Znwm(i64 4) #[[ATTR4:[0-9]+]]
// DISABLED:    call void @_ZL12examineValuei(i32 {{.*}})
int main() {
  indirect_callee_int(0);
  indirect_callee_union((union u1){0});

  auto s = new s1;
  examineValue(s->val);

  return 0;
}
