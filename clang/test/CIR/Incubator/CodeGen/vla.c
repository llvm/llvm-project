// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CHECK --input-file=%t.cir %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// CHECK:  cir.func {{.*}} @f0(%arg0: !s32i
// CHECK:    [[TMP0:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["len", init] {alignment = 4 : i64}
// CHECK:    [[TMP1:%.*]] = cir.alloca !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>, ["saved_stack"] {alignment = 8 : i64}
// CHECK:    cir.store{{.*}} %arg0, [[TMP0]] : !s32i, !cir.ptr<!s32i>
// CHECK:    [[TMP2:%.*]] = cir.load{{.*}} [[TMP0]] : !cir.ptr<!s32i>, !s32i
// CHECK:    [[TMP3:%.*]] = cir.cast integral [[TMP2]] : !s32i -> !u64i
// CHECK:    [[TMP4:%.*]] = cir.stack_save : !cir.ptr<!u8i>
// CHECK:    cir.store{{.*}} [[TMP4]], [[TMP1]] : !cir.ptr<!u8i>, !cir.ptr<!cir.ptr<!u8i>>
// CHECK:    [[TMP5:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, [[TMP3]] : !u64i, ["vla"] {alignment = 16 : i64}
// CHECK:    [[TMP6:%.*]] = cir.load{{.*}} [[TMP1]] : !cir.ptr<!cir.ptr<!u8i>>, !cir.ptr<!u8i>
// CHECK:    cir.stack_restore [[TMP6]] : !cir.ptr<!u8i>
void f0(int len) {
    int a[len];
}

//     CHECK: cir.func {{.*}} @f1
// CHECK-NOT:   cir.stack_save
// CHECK-NOT:   cir.stack_restore
//     CHECK:   cir.return
int f1(int n) {
  return sizeof(int[n]);
}

// CHECK: cir.func {{.*}} @f2
// CHECK:   cir.stack_save
// DONT_CHECK:   cir.stack_restore
// CHECK:   cir.return
int f2(int x) {
  int vla[x];
  return vla[x-1];
}

// CHECK: cir.func {{.*}} @f3
// CHECK:   cir.stack_save
// CHECK:   cir.stack_restore
// CHECK:   cir.return
void f3(int count) {
  int a[count];

  do {  } while (0);
  if (a[0] != 3) {}
}


//     CHECK: cir.func {{.*}} @f4
// CHECK-NOT:   cir.stack_save
// CHECK-NOT:   cir.stack_restore
//     CHECK:   cir.return
void f4(int count) {
  // Make sure we emit sizes correctly in some obscure cases
  int (*a[5])[count];
  int (*b)[][count];
}

// FIXME(cir): the test is commented due to stack_restore operation
// is not emitted for the if branch
// void f5(unsigned x) {
//   while (1) {
//     char s[x];
//     if (x > 5) //: stack restore here is missed
//       break;
//   }
// }

// Check no errors happen
void function1(short width, int data[][width]) {}
void function2(short width, int data[][width][width]) {}
void f6(void) {
     int bork[4][13][15];

     function1(1, bork[2]);
     function2(1, bork);
}

static int GLOB;
int f7(int n)
{
  GLOB = 0;
  char b[1][n+3];

  __typeof__(b[GLOB++]) c;
  return GLOB;
}

double f8(int n, double (*p)[n][5]) {
    return p[1][2][3];
}

int f9(unsigned n, char (*p)[n][n+1][6]) {
    __typeof(p) p2 = (p + n/2) - n/4;

  return p2 - p;
}

long f10(int n) {
    int (*p)[n];
    int (*q)[n];
    return q - p;
}
// CHECK-LABEL: cir.func {{.*}} @f10
// CHECK: %[[Q_VAL:[0-9]+]] = cir.load {{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[P_VAL:[0-9]+]] = cir.load {{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[PTRDIFF:[0-9]+]] = cir.ptr_diff %[[Q_VAL]], %[[P_VAL]] : !cir.ptr<!s32i> -> !s64i
// CHECK: %[[N_S64:[0-9]+]] = cir.cast integral %{{.*}} : !u64i -> !s64i
// CHECK: %[[DIV:[0-9]+]] = cir.binop(div, %[[PTRDIFF]], %[[N_S64]]) : !s64i

// LLVM-LABEL: @f10(
// LLVM: %[[QI:[0-9]+]] = ptrtoint ptr %{{.*}} to i64
// LLVM: %[[PI:[0-9]+]] = ptrtoint ptr %{{.*}} to i64
// LLVM: %[[DIFF_BYTES:[0-9]+]] = sub i64 %[[QI]], %[[PI]]
// LLVM: %[[PTRDIFF_INTS:[0-9]+]] = sdiv i64 %[[DIFF_BYTES]], 4
// LLVM: %[[RESULT:[0-9]+]] = sdiv i64 %[[PTRDIFF_INTS]], %{{.*}}

// OGCG-LABEL: @f10(
// OGCG: %{{.*}} = ptrtoint ptr %{{.*}} to i64
// OGCG: %{{.*}} = ptrtoint ptr %{{.*}} to i64
// OGCG: %{{.*}} = sub i64 %{{.*}}, %{{.*}}
// OGCG: %{{.*}} = mul nuw i64 4, %{{.*}}
// OGCG: %{{.*}} = sdiv exact i64 %{{.*}}, %{{.*}}

long f11(int n, int m) {
    int (*p)[n][m];
    int (*q)[n][m];
    return q - p;
}
// CHECK-LABEL: cir.func {{.*}} @f11

// # allocas
// CHECK: %[[N_ADDR:[0-9]+]] = cir.alloca !s32i, !cir.ptr<!s32i>
// CHECK: %[[M_ADDR:[0-9]+]] = cir.alloca !s32i, !cir.ptr<!s32i>
// CHECK: %[[RET:[0-9]+]] = cir.alloca !s64i, !cir.ptr<!s64i>
// CHECK: %[[P:[0-9]+]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CHECK: %[[Q:[0-9]+]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>

// # store n, m
// CHECK: cir.store %arg0, %[[N_ADDR]] : !s32i, !cir.ptr<!s32i>
// CHECK: cir.store %arg1, %[[M_ADDR]] : !s32i, !cir.ptr<!s32i>

// # load n and cast to u64
// CHECK: %[[N_LOAD:[0-9]+]] = cir.load {{.*}} %[[N_ADDR]] : !cir.ptr<!s32i>, !s32i
// CHECK: %[[N_U64:[0-9]+]] = cir.cast integral %[[N_LOAD]] : !s32i -> !u64i

// # load m and cast to u64
// CHECK: %[[M_LOAD:[0-9]+]] = cir.load {{.*}} %[[M_ADDR]] : !cir.ptr<!s32i>, !s32i
// CHECK: %[[M_U64:[0-9]+]] = cir.cast integral %[[M_LOAD]] : !s32i -> !u64i

// # load q and p
// CHECK: %[[Q_VAL:[0-9]+]] = cir.load {{.*}} %[[Q]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// CHECK: %[[P_VAL:[0-9]+]] = cir.load {{.*}} %[[P]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
// 
// # ptrdiff → (byte_diff / 4)
// CHECK: %[[PTRDIFF:[0-9]+]] = cir.ptr_diff %[[Q_VAL]], %[[P_VAL]] : !cir.ptr<!s32i> -> !s64i

// # compute n*m
// CHECK: %[[NM_U64:[0-9]+]] = cir.binop(mul, %[[N_U64]], %[[M_U64]]) : !u64i
// CHECK: %[[NM_S64:[0-9]+]] = cir.cast integral %[[NM_U64]] : !u64i -> !s64i

// # divide ptrdiff_ints by (n*m)
// CHECK: %[[RESULT:[0-9]+]] = cir.binop(div, %[[PTRDIFF]], %[[NM_S64]]) : !s64i

// # store + return
// CHECK: cir.store{{.*}} %[[RESULT]], %[[RET]] : !s64i, !cir.ptr<!s64i>
// CHECK: %[[RETVAL:[0-9]+]] = cir.load{{.*}} %[[RET]] : !cir.ptr<!s64i>, !s64i
// CHECK: cir.return %[[RETVAL]] : !s64i


// LLVM-LABEL: @f11(
// # load q and p
// LLVM: %[[QI:[0-9]+]] = ptrtoint ptr %{{.*}} to i64
// LLVM: %[[PI:[0-9]+]] = ptrtoint ptr %{{.*}} to i64
// LLVM: %[[DIFF_BYTES:[0-9]+]] = sub i64 %[[QI]], %[[PI]]
// LLVM: %[[PTRDIFF_INTS:[0-9]+]] = sdiv i64 %[[DIFF_BYTES]], 4
// LLVM: %[[NM:[0-9]+]] = mul i64 %{{.*}}, %{{.*}}
// LLVM: %[[RESULT:[0-9]+]] = sdiv i64 %[[PTRDIFF_INTS]], %[[NM]]

// OGCG-LABEL: @f11(
// OGCG: %{{.*}} = ptrtoint ptr %{{.*}} to i64
// OGCG: %{{.*}} = ptrtoint ptr %{{.*}} to i64
// OGCG: %{{.*}} = sub i64 %{{.*}}, %{{.*}}
// OGCG: %{{.*}} = mul nuw i64 %{{.*}}, %{{.*}}
// OGCG: %{{.*}} = mul nuw i64 4, %{{.*}}
// OGCG: %{{.*}} = sdiv exact i64 %{{.*}}, %{{.*}}
