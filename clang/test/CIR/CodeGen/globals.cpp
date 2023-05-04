// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int a = 3;
const int b = 4; // unless used wont be generated

unsigned long int c = 2;
float y = 3.4;
double w = 4.3;
char x = '3';
unsigned char rgb[3] = {0, 233, 33};
char alpha[4] = "abc";
const char *s = "example";
const char *s1 = "example1";
const char *s2 = "example";

void use_global() {
  int li = a;
}

void use_global_string() {
  unsigned char c = s2[0];
}

template <typename T>
T func() {
  return T();
}

int use_func() { return func<int>(); }

// CHECK: module {{.*}} {
// CHECK-NEXT: cir.global external @a = 3 : i32
// CHECK-NEXT: cir.global external @c = 2 : i64
// CHECK-NEXT: cir.global external @y = 3.400000e+00 : f32
// CHECK-NEXT: cir.global external @w = 4.300000e+00 : f64
// CHECK-NEXT: cir.global external @x = 51 : i8
// CHECK-NEXT: cir.global external @rgb = #cir.const_array<[0 : i8, -23 : i8, 33 : i8]> : !cir.array<i8 x 3>
// CHECK-NEXT: cir.global external @alpha = #cir.const_array<[97 : i8, 98 : i8, 99 : i8, 0 : i8]> : !cir.array<i8 x 4>

// CHECK-NEXT: cir.global "private" constant internal @".str" = #cir.const_array<"example\00" : !cir.array<i8 x 8>> : !cir.array<i8 x 8> {alignment = 1 : i64}
// CHECK-NEXT: cir.global external @s = @".str": !cir.ptr<i8>

// CHECK-NEXT: cir.global "private" constant internal @".str1" = #cir.const_array<"example1\00" : !cir.array<i8 x 9>> : !cir.array<i8 x 9> {alignment = 1 : i64}
// CHECK-NEXT: cir.global external @s1 = @".str1": !cir.ptr<i8>

// CHECK-NEXT: cir.global external @s2 = @".str": !cir.ptr<i8>

//      CHECK: cir.func @_Z10use_globalv() {
// CHECK-NEXT:     %0 = cir.alloca i32, cir.ptr <i32>, ["li", init] {alignment = 4 : i64}
// CHECK-NEXT:     %1 = cir.get_global @a : cir.ptr <i32>
// CHECK-NEXT:     %2 = cir.load %1 : cir.ptr <i32>, i32
// CHECK-NEXT:     cir.store %2, %0 : i32, cir.ptr <i32>

//      CHECK: cir.func @_Z17use_global_stringv() {
// CHECK-NEXT:   %0 = cir.alloca i8, cir.ptr <i8>, ["c", init] {alignment = 1 : i64}
// CHECK-NEXT:   %1 = cir.get_global @s2 : cir.ptr <!cir.ptr<i8>>
// CHECK-NEXT:   %2 = cir.load %1 : cir.ptr <!cir.ptr<i8>>, !cir.ptr<i8>
// CHECK-NEXT:   %3 = cir.const(0 : i32) : i32
// CHECK-NEXT:   %4 = cir.ptr_stride(%2 : !cir.ptr<i8>, %3 : i32), !cir.ptr<i8>
// CHECK-NEXT:   %5 = cir.load %4 : cir.ptr <i8>, i8
// CHECK-NEXT:   cir.store %5, %0 : i8, cir.ptr <i8>

//      CHECK:  cir.func linkonce_odr @_Z4funcIiET_v() -> i32 {
// CHECK-NEXT:    %0 = cir.alloca i32, cir.ptr <i32>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:    %1 = cir.const(0 : i32) : i32
// CHECK-NEXT:    cir.store %1, %0 : i32, cir.ptr <i32>
// CHECK-NEXT:    %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT:    cir.return %2 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  cir.func @_Z8use_funcv() -> i32 {
// CHECK-NEXT:    %0 = cir.alloca i32, cir.ptr <i32>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:    %1 = cir.call @_Z4funcIiET_v() : () -> i32
// CHECK-NEXT:    cir.store %1, %0 : i32, cir.ptr <i32>
// CHECK-NEXT:    %2 = cir.load %0 : cir.ptr <i32>, i32
// CHECK-NEXT:    cir.return %2 : i32
// CHECK-NEXT:  }


char string[] = "whatnow";
// CHECK: cir.global external @string = #cir.const_array<[119 : i8, 104 : i8, 97 : i8, 116 : i8, 110 : i8, 111 : i8, 119 : i8, 0 : i8]> : !cir.array<i8 x 8>
unsigned uint[] = {255};
// CHECK: cir.global external @uint = #cir.const_array<[255 : i32]> : !cir.array<i32 x 1>
short sshort[] = {11111, 22222};
// CHECK: cir.global external @sshort = #cir.const_array<[11111 : i16, 22222 : i16]> : !cir.array<i16 x 2>
int sint[] = {123, 456, 789};
// CHECK: cir.global external @sint = #cir.const_array<[123 : i32, 456 : i32, 789 : i32]> : !cir.array<i32 x 3>
long long ll[] = {999999999, 0, 0, 0};
// CHECK: cir.global external @ll = #cir.const_array<[999999999, 0, 0, 0]> : !cir.array<i64 x 4>

void get_globals() {
  // CHECK: cir.func @_Z11get_globalsv()
  char *s = string;
  // CHECK: %[[RES:[0-9]+]] = cir.get_global @string : cir.ptr <!cir.array<i8 x 8>>
  // CHECK: %{{[0-9]+}} = cir.cast(array_to_ptrdecay, %[[RES]] : !cir.ptr<!cir.array<i8 x 8>>), !cir.ptr<i8>
  unsigned *u = uint;
  // CHECK: %[[RES:[0-9]+]] = cir.get_global @uint : cir.ptr <!cir.array<i32 x 1>>
  // CHECK: %{{[0-9]+}} = cir.cast(array_to_ptrdecay, %[[RES]] : !cir.ptr<!cir.array<i32 x 1>>), !cir.ptr<i32>
  short *ss = sshort;
  // CHECK: %[[RES:[0-9]+]] = cir.get_global @sshort : cir.ptr <!cir.array<i16 x 2>>
  // CHECK: %{{[0-9]+}} = cir.cast(array_to_ptrdecay, %[[RES]] : !cir.ptr<!cir.array<i16 x 2>>), !cir.ptr<i16>
  int *si = sint;
  // CHECK: %[[RES:[0-9]+]] = cir.get_global @sint : cir.ptr <!cir.array<i32 x 3>>
  // CHECK: %{{[0-9]+}} = cir.cast(array_to_ptrdecay, %[[RES]] : !cir.ptr<!cir.array<i32 x 3>>), !cir.ptr<i32>
  long long *l = ll;
  // CHECK: %[[RES:[0-9]+]] = cir.get_global @ll : cir.ptr <!cir.array<i64 x 4>>
  // CHECK: %{{[0-9]+}} = cir.cast(array_to_ptrdecay, %[[RES]] : !cir.ptr<!cir.array<i64 x 4>>), !cir.ptr<i64>
}
