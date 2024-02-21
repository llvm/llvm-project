// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int a = 3;
const int b = 4; // unless used wont be generated

unsigned long int c = 2;
int d = a;
bool e;
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
// CHECK-NEXT: cir.global external @a = #cir.int<3> : !s32i
// CHECK-NEXT: cir.global external @c = #cir.int<2> : !u64i
// CHECK-NEXT: cir.global external @d = #cir.int<0> : !s32i

// CHECK-NEXT: cir.func internal private @__cxx_global_var_init()
// CHECK-NEXT:   [[TMP0:%.*]] = cir.get_global @d : cir.ptr <!s32i>
// CHECK-NEXT:   [[TMP1:%.*]] = cir.get_global @a : cir.ptr <!s32i>
// CHECK-NEXT:   [[TMP2:%.*]] = cir.load [[TMP1]] : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:   cir.store [[TMP2]], [[TMP0]] : !s32i, cir.ptr <!s32i>

// CHECK: cir.global external @e = #false
// CHECK-NEXT: cir.global external @y = #cir.fp<3.400000e+00> : !cir.float
// CHECK-NEXT: cir.global external @w = #cir.fp<4.300000e+00> : !cir.double
// CHECK-NEXT: cir.global external @x = #cir.int<51> : !s8i
// CHECK-NEXT: cir.global external @rgb = #cir.const_array<[#cir.int<0> : !u8i, #cir.int<233> : !u8i, #cir.int<33> : !u8i]> : !cir.array<!u8i x 3>
// CHECK-NEXT: cir.global external @alpha = #cir.const_array<"abc\00" : !cir.array<!s8i x 4>> : !cir.array<!s8i x 4>

// CHECK-NEXT: cir.global "private" constant internal @".str" = #cir.const_array<"example\00" : !cir.array<!s8i x 8>> : !cir.array<!s8i x 8> {alignment = 1 : i64}
// CHECK-NEXT: cir.global external @s = #cir.global_view<@".str"> : !cir.ptr<!s8i>

// CHECK-NEXT: cir.global "private" constant internal @".str1" = #cir.const_array<"example1\00" : !cir.array<!s8i x 9>> : !cir.array<!s8i x 9> {alignment = 1 : i64}
// CHECK-NEXT: cir.global external @s1 = #cir.global_view<@".str1"> : !cir.ptr<!s8i>

// CHECK-NEXT: cir.global external @s2 = #cir.global_view<@".str"> : !cir.ptr<!s8i>

//      CHECK: cir.func @_Z10use_globalv()
// CHECK-NEXT:     %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["li", init] {alignment = 4 : i64}
// CHECK-NEXT:     %1 = cir.get_global @a : cir.ptr <!s32i>
// CHECK-NEXT:     %2 = cir.load %1 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:     cir.store %2, %0 : !s32i, cir.ptr <!s32i>

//      CHECK: cir.func @_Z17use_global_stringv()
// CHECK-NEXT:   %0 = cir.alloca !u8i, cir.ptr <!u8i>, ["c", init] {alignment = 1 : i64}
// CHECK-NEXT:   %1 = cir.get_global @s2 : cir.ptr <!cir.ptr<!s8i>>
// CHECK-NEXT:   %2 = cir.load %1 : cir.ptr <!cir.ptr<!s8i>>, !cir.ptr<!s8i>
// CHECK-NEXT:   %3 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK-NEXT:   %4 = cir.ptr_stride(%2 : !cir.ptr<!s8i>, %3 : !s32i), !cir.ptr<!s8i>
// CHECK-NEXT:   %5 = cir.load %4 : cir.ptr <!s8i>, !s8i
// CHECK-NEXT:   %6 = cir.cast(integral, %5 : !s8i), !u8i
// CHECK-NEXT:   cir.store %6, %0 : !u8i, cir.ptr <!u8i>
// CHECK-NEXT:   cir.return

//      CHECK:  cir.func linkonce_odr @_Z4funcIiET_v() -> !s32i
// CHECK-NEXT:    %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:    %1 = cir.const(#cir.int<0> : !s32i) : !s32i
// CHECK-NEXT:    cir.store %1, %0 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:    %2 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:    cir.return %2 : !s32i
// CHECK-NEXT:  }
// CHECK-NEXT:  cir.func @_Z8use_funcv() -> !s32i
// CHECK-NEXT:    %0 = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"] {alignment = 4 : i64}
// CHECK-NEXT:    %1 = cir.call @_Z4funcIiET_v() : () -> !s32i
// CHECK-NEXT:    cir.store %1, %0 : !s32i, cir.ptr <!s32i>
// CHECK-NEXT:    %2 = cir.load %0 : cir.ptr <!s32i>, !s32i
// CHECK-NEXT:    cir.return %2 : !s32i
// CHECK-NEXT:  }


char string[] = "whatnow";
// CHECK: cir.global external @string = #cir.const_array<"whatnow\00" : !cir.array<!s8i x 8>> : !cir.array<!s8i x 8>
unsigned uint[] = {255};
// CHECK: cir.global external @uint = #cir.const_array<[#cir.int<255> : !u32i]> : !cir.array<!u32i x 1>
short sshort[] = {11111, 22222};
// CHECK: cir.global external @sshort = #cir.const_array<[#cir.int<11111> : !s16i, #cir.int<22222> : !s16i]> : !cir.array<!s16i x 2>
int sint[] = {123, 456, 789};
// CHECK: cir.global external @sint = #cir.const_array<[#cir.int<123> : !s32i, #cir.int<456> : !s32i, #cir.int<789> : !s32i]> : !cir.array<!s32i x 3>
long long ll[] = {999999999, 0, 0, 0};
// CHECK: cir.global external @ll = #cir.const_array<[#cir.int<999999999> : !s64i, #cir.int<0> : !s64i, #cir.int<0> : !s64i, #cir.int<0> : !s64i]> : !cir.array<!s64i x 4>

void get_globals() {
  // CHECK: cir.func @_Z11get_globalsv()
  char *s = string;
  // CHECK: %[[RES:[0-9]+]] = cir.get_global @string : cir.ptr <!cir.array<!s8i x 8>>
  // CHECK: %{{[0-9]+}} = cir.cast(array_to_ptrdecay, %[[RES]] : !cir.ptr<!cir.array<!s8i x 8>>), !cir.ptr<!s8i>
  unsigned *u = uint;
  // CHECK: %[[RES:[0-9]+]] = cir.get_global @uint : cir.ptr <!cir.array<!u32i x 1>>
  // CHECK: %{{[0-9]+}} = cir.cast(array_to_ptrdecay, %[[RES]] : !cir.ptr<!cir.array<!u32i x 1>>), !cir.ptr<!u32i>
  short *ss = sshort;
  // CHECK: %[[RES:[0-9]+]] = cir.get_global @sshort : cir.ptr <!cir.array<!s16i x 2>>
  // CHECK: %{{[0-9]+}} = cir.cast(array_to_ptrdecay, %[[RES]] : !cir.ptr<!cir.array<!s16i x 2>>), !cir.ptr<!s16i>
  int *si = sint;
  // CHECK: %[[RES:[0-9]+]] = cir.get_global @sint : cir.ptr <!cir.array<!s32i x 3>>
  // CHECK: %{{[0-9]+}} = cir.cast(array_to_ptrdecay, %[[RES]] : !cir.ptr<!cir.array<!s32i x 3>>), !cir.ptr<!s32i>
  long long *l = ll;
  // CHECK: %[[RES:[0-9]+]] = cir.get_global @ll : cir.ptr <!cir.array<!s64i x 4>>
  // CHECK: %{{[0-9]+}} = cir.cast(array_to_ptrdecay, %[[RES]] : !cir.ptr<!cir.array<!s64i x 4>>), !cir.ptr<!s64i>
}

// Should generate extern global variables.
extern int externVar;
int testExternVar(void) { return externVar; }
// CHECK: cir.global "private" external @externVar : !s32i
// CHECK: cir.func @{{.+}}testExternVar
// CHECK:   cir.get_global @externVar : cir.ptr <!s32i>

// Should constant initialize global with constant address.
int var = 1;
int *constAddr = &var;
// CHECK-DAG: cir.global external @constAddr = #cir.global_view<@var> : !cir.ptr<!s32i>
