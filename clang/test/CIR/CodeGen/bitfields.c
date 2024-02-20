// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

struct __long {
  struct __attribute__((__packed__)) {
      unsigned __is_long_ : 1;
      unsigned __cap_ : sizeof(unsigned) * 8 - 1;
  };
  unsigned __size_;
  unsigned *__data_;
};

void m() {
  struct __long l;
}

typedef struct {
  int a : 4;
  int b : 5;
  int c;
} D;

typedef struct {
  int a : 4;
  int b : 27;
  int c : 17;
  int d : 2;
  int e : 15;
  unsigned f; // type other than int above, not a bitfield
} S; 

typedef struct {
  int a : 3;  // one bitfield with size < 8
  unsigned b;
} T; 

typedef struct {
    char a;
    char b;
    char c;
  
    // startOffset 24 bits, new storage from here
    int d: 2;  
    int e: 2;
    int f: 4;
    int g: 25;
    int h: 3;
    int i: 4;  
    int j: 3;
    int k: 8;

    int l: 14; // need to be a part of the new storage
               // because (tail - startOffset) is 65 after 'l' field   
} U;

// CHECK: !ty_22D22 = !cir.struct<struct "D" {!cir.int<u, 16>, !cir.int<s, 32>}>
// CHECK: !ty_22S22 = !cir.struct<struct "S" {!cir.int<u, 32>, !cir.int<u, 32>, !cir.int<u, 16>, !cir.int<u, 32>}>
// CHECK: !ty_22T22 = !cir.struct<struct "T" {!cir.int<u, 8>, !cir.int<u, 32>} #cir.record.decl.ast>
// CHECK: !ty_22U22 = !cir.struct<struct "U" {!cir.int<s, 8>, !cir.int<s, 8>, !cir.int<s, 8>, !cir.int<u, 64>, !cir.int<u, 16>}>
// CHECK: !ty_22anon2E122 = !cir.struct<struct "anon.1" {!cir.int<u, 32>} #cir.record.decl.ast>
// CHECK: !ty_anon_struct = !cir.struct<struct  {!cir.int<u, 8>, !cir.int<u, 8>, !cir.int<s, 32>}>
// CHECK: !ty_22__long22 = !cir.struct<struct "__long" {!cir.struct<struct "anon.1" {!cir.int<u, 32>} #cir.record.decl.ast>, !cir.int<u, 32>, !cir.ptr<!cir.int<u, 32>>}>

// CHECK: cir.func {{.*@store_field}}
// CHECK:   [[TMP0:%.*]] = cir.alloca !ty_22S22, cir.ptr <!ty_22S22>
// CHECK:   [[TMP1:%.*]] = cir.const(#cir.int<3> : !s32i) : !s32i
// CHECK:   [[TMP2:%.*]] = cir.cast(bitcast, [[TMP0]] : !cir.ptr<!ty_22S22>), !cir.ptr<!u32i>
// CHECK:   [[TMP3:%.*]] = cir.cast(integral, [[TMP1]] : !s32i), !u32i
// CHECK:   [[TMP4:%.*]] = cir.load [[TMP2]] : cir.ptr <!u32i>, !u32i
// CHECK:   [[TMP5:%.*]] = cir.const(#cir.int<15> : !u32i) : !u32i
// CHECK:   [[TMP6:%.*]] = cir.binop(and, [[TMP3]], [[TMP5]]) : !u32i
// CHECK:   [[TMP7:%.*]] = cir.const(#cir.int<4294967280> : !u32i) : !u32i
// CHECK:   [[TMP8:%.*]] = cir.binop(and, [[TMP4]], [[TMP7]]) : !u32i
// CHECK:   [[TMP9:%.*]] = cir.binop(or, [[TMP8]], [[TMP6]]) : !u32i
// CHECK:   cir.store [[TMP9]], [[TMP2]] : !u32i, cir.ptr <!u32i>
void store_field() {
  S s;
  s.a = 3;
}

// CHECK: cir.func {{.*@store_neg_field}}
// CHECK:  [[TMP0:%.*]]  = cir.alloca !ty_22S22, cir.ptr <!ty_22S22>
// CHECK:  [[TMP1:%.*]]  = cir.const(#cir.int<1> : !s32i) : !s32i
// CHECK:  [[TMP2:%.*]]  = cir.unary(minus, [[TMP1]]) : !s32i, !s32i
// CHECK:  [[TMP3:%.*]]  = cir.get_member [[TMP0]][1] {name = "d"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!u32i>
// CHECK:  [[TMP4:%.*]]  = cir.cast(integral, [[TMP2]] : !s32i), !u32i
// CHECK:  [[TMP5:%.*]]  = cir.load [[TMP3]] : cir.ptr <!u32i>, !u32i
// CHECK:  [[TMP6:%.*]]  = cir.const(#cir.int<3> : !u32i) : !u32i
// CHECK:  [[TMP7:%.*]]  = cir.binop(and, [[TMP4]], [[TMP6]]) : !u32i
// CHECK:  [[TMP8:%.*]]  = cir.const(#cir.int<17> : !u32i) : !u32i
// CHECK:  [[TMP9:%.*]] = cir.shift(left, [[TMP7]] : !u32i, [[TMP8]] : !u32i) -> !u32i
// CHECK:  [[TMP10:%.*]] = cir.const(#cir.int<4294574079> : !u32i) : !u32i
// CHECK:  [[TMP11:%.*]] = cir.binop(and, [[TMP5]], [[TMP10]]) : !u32i
// CHECK:  [[TMP12:%.*]] = cir.binop(or, [[TMP11]], [[TMP9]]) : !u32i
// CHECK:  cir.store [[TMP12]], [[TMP3]] : !u32i, cir.ptr <!u32i>
void store_neg_field() {
  S s;
  s.d = -1;
}

// CHECK: cir.func {{.*@load_field}}
// CHECK:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!ty_22S22>, cir.ptr <!cir.ptr<!ty_22S22>>
// CHECK:   [[TMP2:%.*]] = cir.load [[TMP0]] : cir.ptr <!cir.ptr<!ty_22S22>>, !cir.ptr<!ty_22S22>
// CHECK:   [[TMP3:%.*]] = cir.get_member [[TMP2]][1] {name = "d"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!u32i>
// CHECK:   [[TMP4:%.*]] = cir.load [[TMP3]] : cir.ptr <!u32i>, !u32i
// CHECK:   [[TMP5:%.*]] = cir.cast(integral, [[TMP4]] : !u32i), !s32i
// CHECK:   [[TMP6:%.*]] = cir.const(#cir.int<13> : !s32i) : !s32i
// CHECK:   [[TMP7:%.*]] = cir.shift(left, [[TMP5]] : !s32i, [[TMP6]] : !s32i) -> !s32i
// CHECK:   [[TMP8:%.*]] = cir.const(#cir.int<30> : !s32i) : !s32i
// CHECK:   [[TMP9:%.*]] = cir.shift( right, [[TMP7]] : !s32i, [[TMP8]] : !s32i) -> !s32i
// CHECK:   [[TMP10:%.*]] = cir.cast(integral, [[TMP9]] : !s32i), !s32i
// CHECK:   cir.store [[TMP10]], [[TMP1]] : !s32i, cir.ptr <!s32i>
// CHECK:   [[TMP11:%.*]] = cir.load [[TMP1]] : cir.ptr <!s32i>, !s32i
int load_field(S* s) {
  return s->d;
}

// CHECK: cir.func {{.*@load_non_bitfield}}
// CHECK:   cir.get_member {{%.}}[3] {name = "f"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!u32i>
unsigned load_non_bitfield(S *s) {
  return s->f;
}

// just create a usage of T type 
// CHECK: cir.func {{.*@load_one_bitfield}}
int load_one_bitfield(T* t) {
  return t->a;
}

// CHECK: cir.func {{.*@createU}}
void createU() {
  U u;
}

// for this struct type we create an anon structure with different storage types in initialization
// CHECK: cir.func {{.*@createD}}
// CHECK:   %0 = cir.alloca !ty_22D22, cir.ptr <!ty_22D22>, ["d"] {alignment = 4 : i64}
// CHECK:   %1 = cir.cast(bitcast, %0 : !cir.ptr<!ty_22D22>), !cir.ptr<!ty_anon_struct>
// CHECK:   %2 = cir.const(#cir.const_struct<{#cir.int<33> : !u8i, #cir.int<0> : !u8i, #cir.int<3> : !s32i}> : !ty_anon_struct) : !ty_anon_struct
// CHECK:   cir.store %2, %1 : !ty_anon_struct, cir.ptr <!ty_anon_struct>
void createD() {
  D d = {1,2,3};
}
