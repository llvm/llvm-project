// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

int g(void);
void use(int);

// The goto to L skips the initialization of x, so the load of x after L must
// not be replaced with the initializer.
void skip_init_with_goto(int c) {
  if (c)
    goto L;
  {
    const int x = g();
  L:
    if (!c)
      use(x);
  }
}

// CIR-LABEL: cir.func {{.*}} @skip_init_with_goto
// CIR: %[[X:.*]] = cir.alloca "x" {{.*}} const
// CIR: cir.label "L"
// CIR: %[[V:.*]] = cir.load {{.*}} %[[X]]
// CIR: cir.call @use(%[[V]])

// LLVM-LABEL: define {{.*}} void @skip_init_with_goto(
// LLVM: %[[X:.*]] = {{.*}}call i32 @g()
// LLVM: call void @use(i32 noundef %[[X]])

// The switch enters case 1 without initializing x.
void skip_init_with_case(int c) {
  switch (c) {
  case 0: {
    const int x = g();
  case 1:
    if (c == 0)
      use(x);
  }
  }
}

// CIR-LABEL: cir.func {{.*}} @skip_init_with_case
// CIR: %[[X:.*]] = cir.alloca "x" {{.*}} const
// CIR: cir.case(equal, [#cir.int<1> : !s32i])
// CIR: %[[V:.*]] = cir.load {{.*}} %[[X]]
// CIR: cir.call @use(%[[V]])

// LLVM-LABEL: define {{.*}} void @skip_init_with_case(
// LLVM: %[[X:.*]] = {{.*}}call i32 @g()
// LLVM: %[[V:.*]] = phi i32 [ %[[X]], %{{.*}} ]
// LLVM: call void @use(i32 noundef %[[V]])
