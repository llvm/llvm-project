// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

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

// CHECK: !ty_22anon22 = !cir.struct<struct "anon" {!u32i} #cir.recdecl.ast>
// CHECK: !ty_22__long22 = !cir.struct<struct "__long" {!ty_22anon22, !u32i, !cir.ptr<!u32i>}>

typedef struct {
  int a : 4;
  int b : 27;
  int c : 17;
  int d : 2;
  int e : 15;
} S; // 65 bits in total, i.e. more than 64

// CHECK: cir.func {{.*@store_field}}
// CHECK:   [[TMP0:%.*]] = cir.alloca !ty_22S22, cir.ptr <!ty_22S22>, 
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
// CHECK:  [[TMP0:%.*]]  = cir.alloca !ty_22S22, cir.ptr <!ty_22S22>, 
// CHECK:  [[TMP1:%.*]]  = cir.const(#cir.int<1> : !s32i) : !s32i 
// CHECK:  [[TMP2:%.*]]  = cir.unary(minus, [[TMP1]]) : !s32i, !s32i 
// CHECK:  [[TMP3:%.*]]  = cir.get_member [[TMP0]][1] {name = "d"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!s32i> 
// CHECK:  [[TMP4:%.*]]  = cir.cast(bitcast, [[TMP3]] : !cir.ptr<!s32i>), !cir.ptr<!u32i> 
// CHECK:  [[TMP5:%.*]]  = cir.cast(integral, [[TMP2]] : !s32i), !u32i
// CHECK:  [[TMP6:%.*]]  = cir.load [[TMP4]] : cir.ptr <!u32i>, !u32i 
// CHECK:  [[TMP7:%.*]]  = cir.const(#cir.int<3> : !u32i) : !u32i
// CHECK:  [[TMP8:%.*]]  = cir.binop(and, [[TMP5]], [[TMP7]]) : !u32i 
// CHECK:  [[TMP9:%.*]]  = cir.const(#cir.int<17> : !u32i) : !u32i 
// CHECK:  [[TMP10:%.*]] = cir.shift(left, [[TMP8]] : !u32i, [[TMP9]] : !u32i) -> !u32i 
// CHECK:  [[TMP11:%.*]] = cir.const(#cir.int<4294574079> : !u32i) : !u32i 
// CHECK:  [[TMP12:%.*]] = cir.binop(and, [[TMP6]], [[TMP11]]) : !u32i 
// CHECK:  [[TMP13:%.*]] = cir.binop(or, [[TMP12]], [[TMP10]]) : !u32i 
// CHECK:  cir.store [[TMP13]], [[TMP4]] : !u32i, cir.ptr <!u32i> 
void store_neg_field() {
  S s;
  s.d = -1;
}

// CHECK: cir.func {{.*@load_field}}
// CHECK:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!ty_22S22>, cir.ptr <!cir.ptr<!ty_22S22>>
// CHECK:   [[TMP2:%.*]] = cir.load [[TMP0]] : cir.ptr <!cir.ptr<!ty_22S22>>, !cir.ptr<!ty_22S22>
// CHECK:   [[TMP3:%.*]] = cir.get_member [[TMP2]][1] {name = "d"} : !cir.ptr<!ty_22S22> -> !cir.ptr<!s32i> 
// CHECK:   [[TMP4:%.*]] = cir.cast(bitcast, [[TMP3]] : !cir.ptr<!s32i>), !cir.ptr<!u32i> 
// CHECK:   [[TMP5:%.*]] = cir.load [[TMP4]] : cir.ptr <!u32i>, !u32i 
// CHECK:   [[TMP6:%.*]] = cir.cast(integral, [[TMP5]] : !u32i), !s32i 
// CHECK:   [[TMP7:%.*]] = cir.const(#cir.int<13> : !s32i) : !s32i 
// CHECK:   [[TMP8:%.*]] = cir.shift(left, [[TMP6]] : !s32i, [[TMP7]] : !s32i) -> !s32i 
// CHECK:   [[TMP9:%.*]] = cir.const(#cir.int<30> : !s32i) : !s32i 
// CHECK:   [[TMP10:%.*]] = cir.shift( right, [[TMP8]] : !s32i, [[TMP9]] : !s32i) -> !s32i 
// CHECK:   [[TMP11:%.*]] = cir.cast(integral, [[TMP10]] : !s32i), !s32i 
// CHECK:   cir.store [[TMP11]], [[TMP1]] : !s32i, cir.ptr <!s32i> 
// CHECK:   [[TMP12:%.*]] = cir.load [[TMP1]] : cir.ptr <!s32i>, !s32i 
int load_field(S* s) {
  return s->d;
}
