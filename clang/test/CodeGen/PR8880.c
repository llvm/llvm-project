// RUN: %clang_cc1 -Wno-gcc-compat -emit-llvm -o - %s | FileCheck %s

void pr8880_cg_1(int *iptr) {
// CHECK-LABEL: define {{.*}}void @pr8880_cg_1(
  int i, j;
// CHECK: br label %[[OUTER_COND:[0-9A-Za-z$._]+]]
  for (i = 2; i != 10 ; i++ )
// CHECK: [[OUTER_COND]]
// CHECK: label %[[OUTER_BODY:[0-9A-Za-z$._]+]], label %[[OUTER_END:[0-9A-Za-z$._]+]]
// CHECK: [[OUTER_BODY]]
// CHECK: br label %[[INNER_COND:[0-9A-Za-z$._]+]]
    for (j = 3 ; j < 22; (void)({ ++j; break; j;})) {
// CHECK: [[INNER_COND]]
// CHECK: label %[[INNER_BODY:[0-9A-Za-z$._]+]], label %[[INNER_END:[0-9A-Za-z$._]+]]
// CHECK: [[INNER_BODY]]
      *iptr = 7;
// CHECK: store i32 7,
// CHECK: br label %[[INNER_INC:[0-9A-Za-z$._]+]]
// CHECK: [[INNER_INC]]

// break in 3rd expression of inner loop causes branch to end of inner loop

// CHECK: br label %[[INNER_END:[0-9A-Za-z$._]+]]
// CHECK: [[INNER_END]]
    }
// CHECK: br label %[[OUTER_INC:[0-9A-Za-z$._]+]]
// CHECK: [[OUTER_INC]]
// CHECK: br label %[[OUTER_COND]]
// CHECK: [[OUTER_END]]
// CHECK: ret
}

void pr8880_cg_2(int *iptr) {
// CHECK-LABEL: define {{.*}}void @pr8880_cg_2(
  int i, j;
// CHECK: br label %[[OUTER_COND:[0-9A-Za-z$._]+]]
  for (i = 2; i != 10 ; i++ )
// CHECK: [[OUTER_COND]]
// CHECK: label %[[OUTER_BODY:[0-9A-Za-z$._]+]], label %[[OUTER_END:[0-9A-Za-z$._]+]]
// CHECK: [[OUTER_BODY]]
// CHECK: br label %[[INNER_COND:[0-9A-Za-z$._]+]]
    for (j = 3 ; j < 22; (void)({ ++j; continue; j;})) {
// CHECK: [[INNER_COND]]
// CHECK: label %[[INNER_BODY:[0-9A-Za-z$._]+]], label %[[INNER_END:[0-9A-Za-z$._]+]]
// CHECK: [[INNER_BODY]]
      *iptr = 7;
// CHECK: store i32 7,
// CHECK: br label %[[INNER_INC:[0-9A-Za-z$._]+]]
// CHECK: [[INNER_INC]]

// continue in 3rd expression of inner loop causes branch to inc of inner loop

// CHECK: br label %[[INNER_INC]]
// CHECK: [[INNER_END]]
    }
// CHECK: br label %[[OUTER_INC:[0-9A-Za-z$._]+]]
// CHECK: [[OUTER_INC]]
// CHECK: br label %[[OUTER_COND]]
// CHECK: [[OUTER_END]]
// CHECK: ret
}

void pr8880_cg_3(int *iptr) {
// CHECK-LABEL: define {{.*}}void @pr8880_cg_3(
  int i, j;
// CHECK: br label %[[OUTER_COND:[0-9A-Za-z$._]+]]
  for (i = 2 ; i != 10 ; i++ )
// CHECK: [[OUTER_COND]]
// CHECK: label %[[OUTER_BODY:[0-9A-Za-z$._]+]], label %[[OUTER_END:[0-9A-Za-z$._]+]]
// CHECK: [[OUTER_BODY]]
// CHECK: br label %[[INNER_COND:[0-9A-Za-z$._]+]]
    for (j = 3 ; ({break; j;}); j++) {

// break in 2nd expression of inner loop causes branch to end of inner loop

// CHECK: [[INNER_COND]]
// CHECK: br label %[[INNER_END:[0-9A-Za-z$._]+]]
// CHECK: label %[[INNER_BODY:[0-9A-Za-z$._]+]], label %[[INNER_END:[0-9A-Za-z$._]+]]
// CHECK: [[INNER_BODY]]
      *iptr = 7;
// CHECK: store i32 7,
// CHECK: br label %[[INNER_INC:[0-9A-Za-z$._]+]]
// CHECK: [[INNER_INC]]
// CHECK: br label %[[INNER_COND]]
    }
// CHECK: [[INNER_END]]
// CHECK: br label %[[OUTER_INC:[0-9A-Za-z$._]+]]
// CHECK: [[OUTER_INC]]
// CHECK: br label %[[OUTER_COND]]
// CHECK: [[OUTER_END]]
// CHECK: ret
}

void pr8880_cg_4(int *iptr) {
// CHECK-LABEL: define {{.*}}void @pr8880_cg_4(
  int i, j;
// CHECK: br label %[[OUTER_COND:[0-9A-Za-z$._]+]]
  for (i = 2 ; i != 10 ; i++ )
// CHECK: [[OUTER_COND]]
// CHECK: label %[[OUTER_BODY:[0-9A-Za-z$._]+]], label %[[OUTER_END:[0-9A-Za-z$._]+]]
// CHECK: [[OUTER_BODY]]
// CHECK: br label %[[INNER_COND:[0-9A-Za-z$._]+]]
    for (j = 3 ; ({continue; j;}); j++) {

// continue in 2nd expression of inner loop causes branch to inc of inner loop

// CHECK: [[INNER_COND]]
// CHECK: br label %[[INNER_INC:[0-9A-Za-z$._]+]]
// CHECK: label %[[INNER_BODY:[0-9A-Za-z$._]+]], label %[[INNER_END:[0-9A-Za-z$._]+]]
// CHECK: [[INNER_BODY]]
      *iptr = 7;
// CHECK: store i32 7,
// CHECK: br label %[[INNER_INC]]
// CHECK: [[INNER_INC]]
// CHECK: br label %[[INNER_COND]]
    }
// CHECK: [[INNER_END]]
// CHECK: br label %[[OUTER_INC:[0-9A-Za-z$._]+]]
// CHECK: [[OUTER_INC]]
// CHECK: br label %[[OUTER_COND]]
// CHECK: [[OUTER_END]]
// CHECK: ret
}

void pr8880_cg_5(int x, int *iptr) {
// CHECK-LABEL: define {{.*}}void @pr8880_cg_5(
  int y = 5;
// CHECK: br label %[[OUTER_COND:[0-9A-Za-z$._]+]]
// CHECK: [[OUTER_COND]]
  while(--x) {
// CHECK: label %[[OUTER_BODY:[0-9A-Za-z$._]+]], label %[[OUTER_END:[0-9A-Za-z$._]+]]
// CHECK: [[OUTER_BODY]]
// CHECK: br label %[[INNER_COND:[0-9A-Za-z$._]+]]
    while(({ break; --y; })) {
// CHECK: [[INNER_COND]]
// CHECK: br label %[[INNER_END:[0-9A-Za-z$._]+]]
// CHECK: label %[[INNER_BODY:[0-9A-Za-z$._]+]], label %[[INNER_END:[0-9A-Za-z$._]+]]
// CHECK: [[INNER_BODY]]
      *iptr = 7;
// CHECK: store i32 7,
    }
// CHECK: br label %[[INNER_COND]]
  }
// CHECK: [[INNER_END]]
// CHECK: br label %[[OUTER_COND]]
// CHECK: [[OUTER_END]]
// CHECK: ret void
}

void pr8880_cg_6(int x, int *iptr) {
// CHECK-LABEL: define {{.*}}void @pr8880_cg_6(
  int y = 5;
// CHECK: br label %[[OUTER_COND:[0-9A-Za-z$._]+]]
// CHECK: [[OUTER_COND]]
  while(--x) {
// CHECK: label %[[OUTER_BODY:[0-9A-Za-z$._]+]], label %[[OUTER_END:[0-9A-Za-z$._]+]]
// CHECK: [[OUTER_BODY]]
// CHECK: br label %[[INNER_BODY:[0-9A-Za-z$._]+]]
// CHECK: [[INNER_BODY]]
    do {
// CHECK: store i32 7,
      *iptr = 7;
// CHECK: br label %[[INNER_COND:[0-9A-Za-z$._]+]]
    } while(({ break; --y; }));
// CHECK: [[INNER_COND]]
// CHECK: br label %[[INNER_END:[0-9A-Za-z$._]+]]
// CHECK: label %[[INNER_BODY:[0-9A-Za-z$._]+]], label %[[INNER_END]]
  }
// CHECK: [[INNER_END]]
// CHECK: br label %[[OUTER_COND]]
// CHECK: [[OUTER_END]]
// CHECK: ret void
}
