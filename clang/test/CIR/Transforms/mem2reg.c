// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir 
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=BEFORE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fclangir-mem2reg %s -o %t.cir 
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=MEM2REG

int return_42() {
  int y = 42;
  return y;  
}

// BEFORE: cir.func {{.*@return_42}}
// BEFORE:   %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// BEFORE:   %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init] {alignment = 4 : i64}
// BEFORE:   %2 = cir.const #cir.int<42> : !s32i
// BEFORE:   cir.store %2, %1 : !s32i, !cir.ptr<!s32i> 
// BEFORE:   %3 = cir.load %1 : !cir.ptr<!s32i>, !s32i
// BEFORE:   cir.store %3, %0 : !s32i, !cir.ptr<!s32i>
// BEFORE:   %4 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// BEFORE:   cir.return %4 : !s32i

// MEM2REG:  cir.func {{.*@return_42()}}
// MEM2REG:    %0 = cir.const #cir.int<42> : !s32i
// MEM2REG:    cir.return %0 : !s32i

void alloca_in_loop(int* ar, int n) {
  for (int i = 0; i < n; ++i) {
    int a = 4;
    ar[i] = a;
  }
}

// BEFORE:  cir.func {{.*@alloca_in_loop}}
// BEFORE:    %0 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["ar", init] {alignment = 8 : i64}
// BEFORE:    %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["n", init] {alignment = 4 : i64}
// BEFORE:    cir.store %arg0, %0 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// BEFORE:    cir.store %arg1, %1 : !s32i, !cir.ptr<!s32i>
// BEFORE:    cir.scope {
// BEFORE:      %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// BEFORE:      %3 = cir.const #cir.int<0> : !s32i
// BEFORE:      cir.store %3, %2 : !s32i, !cir.ptr<!s32i>
// BEFORE:      cir.for : cond {
// BEFORE:        %4 = cir.load %2 : !cir.ptr<!s32i>, !s32i
// BEFORE:        %5 = cir.load %1 : !cir.ptr<!s32i>, !s32i
// BEFORE:        %6 = cir.cmp(lt, %4, %5) : !s32i, !s32i
// BEFORE:        %7 = cir.cast(int_to_bool, %6 : !s32i), !cir.bool
// BEFORE:        cir.condition(%7)
// BEFORE:      } body {
// BEFORE:        cir.scope {
// BEFORE:          %4 = cir.alloca !s32i, !cir.ptr<!s32i>, ["a", init] {alignment = 4 : i64}
// BEFORE:          %5 = cir.const #cir.int<4> : !s32i
// BEFORE:          cir.store %5, %4 : !s32i, !cir.ptr<!s32i>
// BEFORE:          %6 = cir.load %4 : !cir.ptr<!s32i>, !s32i
// BEFORE:          %7 = cir.load %0 : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// BEFORE:          %8 = cir.load %2 : !cir.ptr<!s32i>, !s32i
// BEFORE:          %9 = cir.ptr_stride(%7 : !cir.ptr<!s32i>, %8 : !s32i), !cir.ptr<!s32i>
// BEFORE:          cir.store %6, %9 : !s32i, !cir.ptr<!s32i>
// BEFORE:        }
// BEFORE:        cir.yield
// BEFORE:      } step {
// BEFORE:        %4 = cir.load %2 : !cir.ptr<!s32i>, !s32i
// BEFORE:        %5 = cir.unary(inc, %4) : !s32i, !s32i
// BEFORE:        cir.store %5, %2 : !s32i, !cir.ptr<!s32i>
// BEFORE:        cir.yield
// BEFORE:      }
// BEFORE:    }
// BEFORE:    cir.return  

// MEM2REG:  cir.func {{.*@alloca_in_loop}}
// MEM2REG:    cir.br ^bb1
// MEM2REG:  ^bb1:  // pred: ^bb0
// MEM2REG:    %0 = cir.const #cir.int<0> : !s32i
// MEM2REG:    cir.br ^bb2(%0 : !s32i)
// MEM2REG:  ^bb2(%1: !s32i{{.*}}):  // 2 preds: ^bb1, ^bb6
// MEM2REG:    %2 = cir.cmp(lt, %1, %arg1) : !s32i, !s32i
// MEM2REG:    %3 = cir.cast(int_to_bool, %2 : !s32i), !cir.bool
// MEM2REG:    cir.brcond %3 ^bb3, ^bb7
// MEM2REG:  ^bb3:  // pred: ^bb2
// MEM2REG:    cir.br ^bb4
// MEM2REG:  ^bb4:  // pred: ^bb3
// MEM2REG:    %4 = cir.const #cir.int<4> : !s32i
// MEM2REG:    %5 = cir.ptr_stride(%arg0 : !cir.ptr<!s32i>, %1 : !s32i), !cir.ptr<!s32i>
// MEM2REG:    cir.store %4, %5 : !s32i, !cir.ptr<!s32i>
// MEM2REG:    cir.br ^bb5
// MEM2REG:  ^bb5:  // pred: ^bb4
// MEM2REG:    cir.br ^bb6
// MEM2REG:  ^bb6:  // pred: ^bb5
// MEM2REG:    %6 = cir.unary(inc, %1) : !s32i, !s32i
// MEM2REG:    cir.br ^bb2(%6 : !s32i)
// MEM2REG:  ^bb7:  // pred: ^bb2
// MEM2REG:    cir.br ^bb8
// MEM2REG:  ^bb8:  // pred: ^bb7
// MEM2REG:    cir.return


int alloca_in_ifelse(int x) {
  int y = 0;
  if (x > 42) {
    int z = 2;
    y = x * z;
  } else  {
    int z = 3;
    y = x * z;
  }

  y = y + 1;
  return y;
}

// BEFORE:  cir.func {{.*@alloca_in_ifelse}}
// BEFORE:    %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// BEFORE:    %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// BEFORE:    %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init] {alignment = 4 : i64}
// BEFORE:    cir.store %arg0, %0 : !s32i, !cir.ptr<!s32i>
// BEFORE:    %3 = cir.const #cir.int<0> : !s32i
// BEFORE:    cir.store %3, %2 : !s32i, !cir.ptr<!s32i>
// BEFORE:    cir.scope {
// BEFORE:      %9 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// BEFORE:      %10 = cir.const #cir.int<42> : !s32i
// BEFORE:      %11 = cir.cmp(gt, %9, %10) : !s32i, !s32i
// BEFORE:      %12 = cir.cast(int_to_bool, %11 : !s32i), !cir.bool
// BEFORE:      cir.if %12 {
// BEFORE:        %13 = cir.alloca !s32i, !cir.ptr<!s32i>, ["z", init] {alignment = 4 : i64}
// BEFORE:        %14 = cir.const #cir.int<2> : !s32i
// BEFORE:        cir.store %14, %13 : !s32i, !cir.ptr<!s32i>
// BEFORE:        %15 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// BEFORE:        %16 = cir.load %13 : !cir.ptr<!s32i>, !s32i
// BEFORE:        %17 = cir.binop(mul, %15, %16) nsw : !s32i
// BEFORE:        cir.store %17, %2 : !s32i, !cir.ptr<!s32i>
// BEFORE:      } else {
// BEFORE:        %13 = cir.alloca !s32i, !cir.ptr<!s32i>, ["z", init] {alignment = 4 : i64}
// BEFORE:        %14 = cir.const #cir.int<3> : !s32i
// BEFORE:        cir.store %14, %13 : !s32i, !cir.ptr<!s32i>
// BEFORE:        %15 = cir.load %0 : !cir.ptr<!s32i>, !s32i
// BEFORE:        %16 = cir.load %13 : !cir.ptr<!s32i>, !s32i
// BEFORE:        %17 = cir.binop(mul, %15, %16) nsw : !s32i
// BEFORE:        cir.store %17, %2 : !s32i, !cir.ptr<!s32i>
// BEFORE:     }
// BEFORE:    }
// BEFORE:    %4 = cir.load %2 : !cir.ptr<!s32i>, !s32i
// BEFORE:    %5 = cir.const #cir.int<1> : !s32i
// BEFORE:    %6 = cir.binop(add, %4, %5) nsw : !s32i
// BEFORE:    cir.store %6, %2 : !s32i, !cir.ptr<!s32i>
// BEFORE:    %7 = cir.load %2 : !cir.ptr<!s32i>, !s32i
// BEFORE:    cir.store %7, %1 : !s32i, !cir.ptr<!s32i>
// BEFORE:    %8 = cir.load %1 : !cir.ptr<!s32i>, !s32i
// BEFORE:    cir.return %8 : !s32i

// MEM2REG:  cir.func {{.*@alloca_in_ifelse}}
// MEM2REG:    %0 = cir.const #cir.int<0> : !s32i
// MEM2REG:    cir.br ^bb1
// MEM2REG:  ^bb1:  // pred: ^bb0
// MEM2REG:    %1 = cir.const #cir.int<42> : !s32i
// MEM2REG:    %2 = cir.cmp(gt, %arg0, %1) : !s32i, !s32i
// MEM2REG:    %3 = cir.cast(int_to_bool, %2 : !s32i), !cir.bool
// MEM2REG:    cir.brcond %3 ^bb3, ^bb2
// MEM2REG:  ^bb2:  // pred: ^bb1
// MEM2REG:    %4 = cir.const #cir.int<3> : !s32i
// MEM2REG:    %5 = cir.binop(mul, %arg0, %4) nsw : !s32i
// MEM2REG:    cir.br ^bb4(%5 : !s32i)
// MEM2REG:  ^bb3:  // pred: ^bb1
// MEM2REG:    %6 = cir.const #cir.int<2> : !s32i
// MEM2REG:    %7 = cir.binop(mul, %arg0, %6) nsw : !s32i
// MEM2REG:    cir.br ^bb4(%7 : !s32i)
// MEM2REG:  ^bb4(%8: !s32i{{.*}}):  // 2 preds: ^bb2, ^bb3
// MEM2REG:    cir.br ^bb5
// MEM2REG:  ^bb5:  // pred: ^bb4
// MEM2REG:    %9 = cir.const #cir.int<1> : !s32i
// MEM2REG:    %10 = cir.binop(add, %8, %9) nsw : !s32i
// MEM2REG:    cir.return %10 : !s32i
// MEM2REG:  }




typedef __SIZE_TYPE__ size_t;
void *alloca(size_t size);
  
void test_bitcast(size_t n) {
  int *c1 = alloca(n);
}

// BEFORE:  cir.func {{.*@test_bitcast}}
// BEFORE:    %0 = cir.alloca !u64i, !cir.ptr<!u64i>, ["n", init] {alignment = 8 : i64}
// BEFORE:    %1 = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["c1", init] {alignment = 8 : i64}
// BEFORE:    cir.store %arg0, %0 : !u64i, !cir.ptr<!u64i>
// BEFORE:    %2 = cir.load %0 : !cir.ptr<!u64i>, !u64i
// BEFORE:    %3 = cir.alloca !u8i, !cir.ptr<!u8i>, %2 : !u64i, ["bi_alloca"] {alignment = 16 : i64}
// BEFORE:    %4 = cir.cast(bitcast, %3 : !cir.ptr<!u8i>), !cir.ptr<!void>
// BEFORE:    %5 = cir.cast(bitcast, %4 : !cir.ptr<!void>), !cir.ptr<!s32i>
// BEFORE:    cir.store %5, %1 : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// BEFORE:    cir.return
  
// MEM2REG:  cir.func {{.*@test_bitcast}}
// MEM2REG:    cir.return
// MEM2REG:  }