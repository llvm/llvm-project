// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -mconstructor-aliases -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -mconstructor-aliases -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Check that `delete p` calls the aligned (de)allocation function when the
// object's type is over-aligned and the program uses C++17 aligned allocation.

typedef decltype(sizeof(0)) size_t;
namespace std { enum class align_val_t : size_t {}; }

#define OVERALIGNED alignas(__STDCPP_DEFAULT_NEW_ALIGNMENT__ * 2)

// Global aligned operator delete.
// ===============================
struct OVERALIGNED A { A(); int n[128]; };

void a2(A *p) { delete p; }

// CIR: cir.func {{.*}} @_Z2a2P1A
// CIR:   %[[P:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CIR:   %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR:   %[[NOT_NULL:.*]] = cir.cmp ne %[[P]], %[[NULL]]
// CIR:   cir.if %[[NOT_NULL]] {
// CIR:     cir.cleanup.scope {
// CIR:       cir.yield
// CIR:     } cleanup normal {
// CIR:       %[[P_CAST:.*]] = cir.cast bitcast %[[P]] : !cir.ptr<!rec_A> -> !cir.ptr<!void>
// CIR:       %[[OBJ_SIZE:.*]] = cir.const #cir.int<512> : !u64i
// CIR:       %[[ALIGNMENT:.*]] = cir.const #cir.int<32> : !u64i
// CIR:       cir.call @_ZdlPvmSt11align_val_t(%[[P_CAST]], %[[OBJ_SIZE]], %[[ALIGNMENT]])
// CIR:       cir.yield
// CIR:     }
// CIR:   }

// LLVM: define {{.*}} void @_Z2a2P1A
// LLVM:   %[[P:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[NOT_NULL:.*]] = icmp ne ptr %[[P]], null
// LLVM:   br i1 %[[NOT_NULL]], label %[[DELETE_NOTNULL:.*]], label %{{.*}}
// LLVM: [[DELETE_NOTNULL]]:
// LLVM:   call void @_ZdlPvmSt11align_val_t(ptr noundef %[[P]], i64 noundef 512, i64 noundef 32)

// OGCG: define {{.*}} void @_Z2a2P1A(
// OGCG:   %[[P:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[ISNULL:.*]] = icmp eq ptr %[[P]], null
// OGCG:   br i1 %[[ISNULL]], label %{{.*}}, label %[[DELETE_NOTNULL:.*]]
// OGCG: [[DELETE_NOTNULL]]:
// OGCG:   call void @_ZdlPvmSt11align_val_t(ptr noundef %[[P]], i64 noundef 512, i64 noundef 32)


// Class-specific aligned operator delete.
// =======================================
struct OVERALIGNED B {
  B();
  // These are just a distraction. We should ignore them.
  void *operator new(size_t);
  void operator delete(void*, size_t);

  void *operator new(size_t, std::align_val_t);
  void operator delete(void*, std::align_val_t);

  int n[128];
};

void b2(B *p) { delete p; }

// CIR: cir.func {{.*}} @_Z2b2P1B
// CIR:   %[[P:.*]] = cir.load{{.*}} %{{.*}} : !cir.ptr<!cir.ptr<!rec_B>>, !cir.ptr<!rec_B>
// CIR:   %[[NULL:.*]] = cir.const #cir.ptr<null>
// CIR:   %[[NOT_NULL:.*]] = cir.cmp ne %[[P]], %[[NULL]]
// CIR:   cir.if %[[NOT_NULL]] {
// CIR:     cir.cleanup.scope {
// CIR:       cir.yield
// CIR:     } cleanup normal {
// CIR:       %[[P_CAST:.*]] = cir.cast bitcast %[[P]] : !cir.ptr<!rec_B> -> !cir.ptr<!void>
// CIR:       %[[ALIGNMENT:.*]] = cir.const #cir.int<32> : !u64i
// CIR:       cir.call @_ZN1BdlEPvSt11align_val_t(%[[P_CAST]], %[[ALIGNMENT]])
// CIR:       cir.yield
// CIR:     }
// CIR:   }

// LLVM: define {{.*}} void @_Z2b2P1B
// LLVM:   %[[P:.*]] = load ptr, ptr %{{.*}}
// LLVM:   %[[NOT_NULL:.*]] = icmp ne ptr %[[P]], null
// LLVM:   br i1 %[[NOT_NULL]], label %[[DELETE_NOTNULL:.*]], label %{{.*}}
// LLVM: [[DELETE_NOTNULL]]:
// LLVM:   call void @_ZN1BdlEPvSt11align_val_t(ptr noundef %[[P]], i64 noundef 32)

// OGCG: define {{.*}} void @_Z2b2P1B(
// OGCG:   %[[P:.*]] = load ptr, ptr %{{.*}}
// OGCG:   %[[ISNULL:.*]] = icmp eq ptr %[[P]], null
// OGCG:   br i1 %[[ISNULL]], label %{{.*}}, label %[[DELETE_NOTNULL:.*]]
// OGCG: [[DELETE_NOTNULL]]:
// OGCG:   call void @_ZN1BdlEPvSt11align_val_t(ptr noundef %[[P]], i64 noundef 32)
