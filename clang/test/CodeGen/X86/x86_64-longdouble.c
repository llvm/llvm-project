// RUN: %clang_cc1 -triple x86_64-linux-android -emit-llvm -O -o - %s \
// RUN:    | FileCheck %s --check-prefix=ANDROID --check-prefix=CHECK
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -O -o - %s \
// RUN:    | FileCheck %s --check-prefix=GNU --check-prefix=CHECK
// RUN: %clang_cc1 -triple x86_64 -emit-llvm -O -o - %s \
// RUN:    | FileCheck %s --check-prefix=GNU --check-prefix=CHECK
// NaCl is an example of a target for which long double is the same as double.
// RUN: %clang_cc1 -triple x86_64-nacl -emit-llvm -O -o - %s \
// RUN:    | FileCheck %s --check-prefix=NACL --check-prefix=CHECK

// Android uses fp128 for long double but other x86_64 targets use x86_fp80.

long double dataLD = 1.0L;
// ANDROID: @dataLD ={{.*}} local_unnamed_addr global fp128 0xL00000000000000003FFF000000000000, align 16
// GNU: @dataLD ={{.*}} local_unnamed_addr global x86_fp80 0xK3FFF8000000000000000, align 16

long double _Complex dataLDC = {1.0L, 1.0L};
// ANDROID: @dataLDC ={{.*}} local_unnamed_addr global { fp128, fp128 } { fp128 0xL00000000000000003FFF000000000000, fp128 0xL00000000000000003FFF000000000000 }, align 16
// GNU: @dataLDC ={{.*}} local_unnamed_addr global { x86_fp80, x86_fp80 } { x86_fp80 0xK3FFF8000000000000000, x86_fp80 0xK3FFF8000000000000000 }, align 16

long double TestLD(long double x) {
  return x * x;
// ANDROID: define{{.*}} fp128 @TestLD(fp128 %x)
// GNU: define{{.*}} x86_fp80 @TestLD(x86_fp80 %x)
// NACL: define{{.*}} double @TestLD(double %x)
}

long double _Complex TestLDC(long double _Complex x) {
  return x * x;
// ANDROID: define{{.*}} void @TestLDC({ fp128, fp128 }* {{.*}}, { fp128, fp128 }* {{.*}} %x)
// GNU: define{{.*}} { x86_fp80, x86_fp80 } @TestLDC({ x86_fp80, x86_fp80 }* {{.*}} %x)
// NACL: define{{.*}} { double, double } @TestLDC(double %x{{.*}}, double %x{{.*}})
}

typedef __builtin_va_list va_list;

int TestGetVarInt(va_list ap) {
  return __builtin_va_arg(ap, int);
// Since int can be passed in memory or register there are two branches.
// CHECK:   define{{.*}} i32 @TestGetVarInt(
// CHECK:   br label
// CHECK:   br label
// CHECK:   = phi
// CHECK:   ret i32
}

double TestGetVarDouble(va_list ap) {
  return __builtin_va_arg(ap, double);
// Since double can be passed in memory or register there are two branches.
// CHECK:   define{{.*}} double @TestGetVarDouble(
// CHECK:   br label
// CHECK:   br label
// CHECK:   = phi
// CHECK:   ret double
}

long double TestGetVarLD(va_list ap) {
  return __builtin_va_arg(ap, long double);
// fp128 and double can be passed in memory or in register, but x86_fp80 is in
// memory.
// ANDROID: define{{.*}} fp128 @TestGetVarLD(
// GNU:     define{{.*}} x86_fp80 @TestGetVarLD(
// NACL:     define{{.*}} double @TestGetVarLD(
// ANDROID: br label
// ANDROID: br label
// NACL: br
// ANDROID: = phi
// GNU-NOT: br
// GNU-NOT: = phi
// NACL: = phi
// ANDROID: ret fp128
// GNU:     ret x86_fp80
}

long double _Complex TestGetVarLDC(va_list ap) {
  return __builtin_va_arg(ap, long double _Complex);
// Pair of fp128 or x86_fp80 are passed as struct in memory.
// ANDROID:   define{{.*}} void @TestGetVarLDC({ fp128, fp128 }* {{.*}}, %struct.__va_list_tag*
// GNU:       define{{.*}} { x86_fp80, x86_fp80 } @TestGetVarLDC(
// Pair of double can go in SSE registers or memory
// NACL:       define{{.*}} { double, double } @TestGetVarLDC(
// ANDROID-NOT: br
// GNU-NOT: br
// NACL: br
// ANDROID-NOT: phi
// GNU-NOT: phi
// NACL: phi
// ANDROID:   ret void
// GNU:       ret { x86_fp80, x86_fp80 }
// NACL:       ret { double, double }
}

void TestVarArg(const char *s, ...);

void TestPassVarInt(int x) {
  TestVarArg("A", x);
// CHECK: define{{.*}} void @TestPassVarInt(i32 %x)
// CHECK: call {{.*}} @TestVarArg(i8* {{.*}}, i32 %x)
}

void TestPassVarFloat(float x) {
  TestVarArg("A", x);
// CHECK: define{{.*}} void @TestPassVarFloat(float %x)
// CHECK: call {{.*}} @TestVarArg(i8* {{.*}}, double %
}

void TestPassVarDouble(double x) {
  TestVarArg("A", x);
// CHECK: define{{.*}} void @TestPassVarDouble(double %x)
// CHECK: call {{.*}} @TestVarArg(i8* {{.*}}, double %x
}

void TestPassVarLD(long double x) {
  TestVarArg("A", x);
// ANDROID: define{{.*}} void @TestPassVarLD(fp128 %x)
// ANDROID: call {{.*}} @TestVarArg(i8* {{.*}}, fp128 %x
// GNU: define{{.*}} void @TestPassVarLD(x86_fp80 %x)
// GNU: call {{.*}} @TestVarArg(i8* {{.*}}, x86_fp80 %x
// NACL: define{{.*}} void @TestPassVarLD(double %x)
// NACL: call {{.*}} @TestVarArg(i8* {{.*}}, double %x
}

void TestPassVarLDC(long double _Complex x) {
  TestVarArg("A", x);
// ANDROID:      define{{.*}} void @TestPassVarLDC({ fp128, fp128 }* {{.*}} %x)
// ANDROID:      store fp128 %{{.*}}, fp128* %
// ANDROID-NEXT: store fp128 %{{.*}}, fp128* %
// ANDROID-NEXT: call {{.*}} @TestVarArg(i8* {{.*}}, { fp128, fp128 }* {{.*}} %
// GNU:          define{{.*}} void @TestPassVarLDC({ x86_fp80, x86_fp80 }* {{.*}} %x)
// GNU:          store x86_fp80 %{{.*}}, x86_fp80* %
// GNU-NEXT:     store x86_fp80 %{{.*}}, x86_fp80* %
// GNU-NEXT:   call {{.*}} @TestVarArg(i8* {{.*}}, { x86_fp80, x86_fp80 }* {{.*}} %
// NACL:      define{{.*}} void @TestPassVarLDC(double %x{{.*}}, double %x{{.*}})
// NACL: call {{.*}} @TestVarArg(i8* {{.*}}, double %x{{.*}}, double %x{{.*}})
}
