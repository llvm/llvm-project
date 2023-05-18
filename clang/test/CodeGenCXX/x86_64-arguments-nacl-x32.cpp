// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-nacl -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -triple=x86_64-unknown-linux-gnux32 -emit-llvm -o - %s | FileCheck %s

struct test_struct {};
typedef int test_struct::* test_struct_mdp;
typedef int (test_struct::*test_struct_mfp)();

// CHECK-LABEL: define{{.*}} i32 @{{.*}}f_mdp{{.*}}(i32 %a)
test_struct_mdp f_mdp(test_struct_mdp a) { return a; }

// CHECK-LABEL: define {{.*}} @{{.*}}f_mfp{{.*}}(i64 %a.coerce)
test_struct_mfp f_mfp(test_struct_mfp a) { return a; }

// A struct with <= 12 bytes before a member data pointer should still
// be allowed in registers, since the member data pointer is only 4 bytes.
// CHECK-LABEL: define{{.*}} void @{{.*}}f_struct_with_mdp{{.*}}(i64 %a.coerce0, i64 %a.coerce1)
struct struct_with_mdp { char *a; char *b; char *c; test_struct_mdp d; };
void f_struct_with_mdp(struct_with_mdp a) { (void)a; }

struct struct_with_mdp_too_much {
  char *a; char *b; char *c; char *d; test_struct_mdp e;
};
// CHECK-LABEL: define{{.*}} void @{{.*}}f_struct_with_mdp_too_much{{.*}}({{.*}} byval({{.*}} {{.*}} %a)
void f_struct_with_mdp_too_much(struct_with_mdp_too_much a) {
  (void)a;
}

// A struct with <= 8 bytes before a member function pointer should still
// be allowed in registers, since the member function pointer is only 8 bytes.
// CHECK-LABEL: define{{.*}} void @{{.*}}f_struct_with_mfp_0{{.*}}(i64 %a.coerce0, i32 %a.coerce1)
struct struct_with_mfp_0 { char *a; test_struct_mfp b; };
void f_struct_with_mfp_0(struct_with_mfp_0 a) { (void)a; }

// CHECK-LABEL: define{{.*}} void @{{.*}}f_struct_with_mfp_1{{.*}}(i64 %a.coerce0, i64 %a.coerce1)
struct struct_with_mfp_1 { char *a; char *b; test_struct_mfp c; };
void f_struct_with_mfp_1(struct_with_mfp_1 a) { (void)a; }

// CHECK-LABEL: define{{.*}} void @{{.*}}f_struct_with_mfp_too_much{{.*}}({{.*}} byval({{.*}}) {{.*}} %a, i32 noundef %x)
struct struct_with_mfp_too_much {
  char *a; char *b; char *c; test_struct_mfp d;
};
void f_struct_with_mfp_too_much(struct_with_mfp_too_much a, int x) {
  (void)a;
}

/* Struct containing an empty struct */
typedef struct { int* a; test_struct x; double *b; } struct_with_empty;

// CHECK-LABEL: define{{.*}} void @{{.*}}f_pass_struct_with_empty{{.*}}(i64 %x{{.*}}, ptr %x
void f_pass_struct_with_empty(struct_with_empty x) {
  (void) x;
}

// CHECK-LABEL: define{{.*}} { i64, ptr } @{{.*}}f_return_struct_with_empty
struct_with_empty f_return_struct_with_empty() {
  return {0, {}, 0};
}
