// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef struct __Base {
  unsigned long id;
  unsigned int a;
  unsigned int n;
  unsigned char x;
  unsigned short u;
} Base;

struct w {
  Base _base;
  const void * ref;
};

typedef struct w *wPtr;

void field_access(wPtr item) {
  __atomic_exchange_n((&item->ref), (((void*)0)), 5);
}

// CHECK: ![[W:.*]] = !cir.struct<struct "w"
// CHECK: cir.func @field_access
// CHECK-NEXT: %[[WADDR:.*]] = cir.alloca !cir.ptr<![[W]]>, {{.*}} {alignment = 8 : i64}
// CHECK: %[[FIELD:.*]] = cir.load %[[WADDR]]
// CHECK: %[[MEMBER:.*]] = cir.get_member %[[FIELD]][1] {name = "ref"}
// CHECK: cir.atomic.xchg(%[[MEMBER]] : !cir.ptr<!cir.ptr<!void>>, {{.*}} : !u64i, seq_cst)

// LLVM: define void @field_access
// LLVM: = alloca ptr, i64 1, align 8
// LLVM: %[[VAL_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: %[[RES_ADDR:.*]] = alloca ptr, i64 1, align 8

// LLVM: %[[MEMBER:.*]] = getelementptr %struct.w, ptr {{.*}}, i32 0, i32 1
// LLVM: store ptr null, ptr %[[VAL_ADDR]], align 8
// LLVM: %[[VAL:.*]] = load i64, ptr %[[VAL_ADDR]], align 8
// LLVM: %[[RES:.*]] = atomicrmw xchg ptr %[[MEMBER]], i64 %[[VAL]] seq_cst, align 8
// LLVM: store i64 %[[RES]], ptr %4, align 8
// LLVM: load ptr, ptr %[[RES_ADDR]], align 8
// LLVM: ret void