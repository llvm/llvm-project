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
// CHECK-LABEL: @field_access
// CHECK-NEXT: %[[WADDR:.*]] = cir.alloca !cir.ptr<![[W]]>, {{.*}} {alignment = 8 : i64}
// CHECK: %[[FIELD:.*]] = cir.load %[[WADDR]]
// CHECK: %[[MEMBER:.*]] = cir.get_member %[[FIELD]][1] {name = "ref"}
// CHECK: cir.atomic.xchg(%[[MEMBER]] : !cir.ptr<!cir.ptr<!void>>, {{.*}} : !u64i, seq_cst)

// LLVM-LABEL: @field_access
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

void structAtomicExchange(unsigned referenceCount, wPtr item) {
  __atomic_compare_exchange_n((&item->_base.a), (&referenceCount), (referenceCount + 1), 1 , 5, 5);
}

// CHECK-LABEL: @structAtomicExchange
// CHECK:  cir.atomic.cmp_xchg({{.*}} : !cir.ptr<!u32i>, {{.*}} : <!u32i>, {{.*}} : !cir.ptr<!u32i>, success = seq_cst, failure = seq_cst) weak : !cir.bool

// LLVM-LABEL: @structAtomicExchange
// LLVM:   load i32
// LLVM:   add i32
// LLVM:   store i32
// LLVM:   %[[EXP:.*]] = load i32
// LLVM:   %[[DES:.*]] = load i32
// LLVM:   %[[RES:.*]] = cmpxchg weak ptr %9, i32 %[[EXP]], i32 %[[DES]] seq_cst seq_cst
// LLVM:   %[[OLD:.*]] = extractvalue { i32, i1 } %[[RES]], 0
// LLVM:   %[[CMP:.*]] = extractvalue { i32, i1 } %[[RES]], 1
// LLVM:   br i1 %[[CMP]], label %[[CONTINUE:.*]], label %[[STORE_OLD:.*]],
// LLVM: [[CONTINUE]]:
// LLVM:   zext i1 %[[CMP]] to i8
// LLVM:   ret void

// LLVM: [[STORE_OLD]]:
// LLVM:   store i32 %[[OLD]], ptr
// LLVM:   br label %[[CONTINUE]]