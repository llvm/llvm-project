// RUN: %clang_cc1 -triple thumbv8m.base-none-eabi -mcmse -O1 -emit-llvm %s -o - 2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -triple thumbebv8m.base-none-eabi -mcmse -O1 -emit-llvm %s -o - 2>&1 | \
// RUN: FileCheck %s --check-prefix=CHECK

typedef void fn_t(void);
fn_t s;
fn_t *p0 __attribute__((cmse_nonsecure_call));

typedef fn_t *pfn_t __attribute__((cmse_nonsecure_call));
pfn_t p1;
pfn_t a0[4];
extern pfn_t a1[];

typedef void (*pfn1_t)(int) __attribute__((cmse_nonsecure_call));
pfn1_t p2;

typedef fn_t *apfn_t[4] __attribute__((cmse_nonsecure_call));
apfn_t a2;

typedef pfn_t apfn1_t[4] __attribute__((cmse_nonsecure_call));
apfn1_t a3;

typedef void (*apfn2_t[4])(void) __attribute__((cmse_nonsecure_call));
apfn2_t a4;

void (*b[4])(int) __attribute__((cmse_nonsecure_call));

void f(int i) {
  s();
// CHECK: call void @s() #[[#A1:]]

  p0();
// CHECK: %[[#P0:]] = load {{.*}} @p0
// CHECK: call void %[[#P0]]() #[[#A2:]]

  p1();
// CHECK: %[[#P1:]] = load {{.*}} @p1
// CHECK: call void %[[#P1]]() #[[#A2]]

  p2(i);
// CHECK: %[[#P2:]] = load {{.*}} @p2
// CHECK: call void %[[#P2]](i32 %i) #[[#A2]]

  a0[i]();
// CHECK: %[[EP0:.*]] = getelementptr {{.*}} @a0
// CHECK: %[[#E0:]] = load {{.*}} %[[EP0]]
// CHECK: call void %[[#E0]]() #[[#A2]]

  a1[i]();
// CHECK: %[[EP1:.*]] = getelementptr {{.*}} @a1
// CHECK: %[[#E1:]] = load {{.*}} %[[EP1]]
// CHECK: call void %[[#E1]]() #[[#A2]]

  a2[i]();
// CHECK: %[[EP2:.*]] = getelementptr {{.*}} @a2
// CHECK: %[[#E2:]] = load {{.*}} %[[EP2]]
// CHECK: call void %[[#E2]]() #[[#A2]]

  a3[i]();
// CHECK: %[[EP3:.*]] = getelementptr {{.*}} @a3
// CHECK: %[[#E3:]] = load {{.*}} %[[EP3]]
// CHECK: call void %[[#E3]]() #[[#A2]]

  a4[i]();
// CHECK: %[[EP4:.*]] = getelementptr {{.*}} @a4
// CHECK: %[[#E4:]] = load {{.*}} %[[EP4]]
// CHECK: call void %[[#E4]]() #[[#A2]]

  b[i](i);
// CHECK: %[[EP5:.*]] = getelementptr {{.*}} @b
// CHECK: %[[#E5:]] = load {{.*}} %[[EP5]]
// CHECK: call void %[[#E5]](i32 %i) #[[#A2]]
}

// CHECK: attributes #[[#A1]] = { nounwind }
// CHECK: attributes #[[#A2]] = { nounwind "cmse_nonsecure_call"
