// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=OGCG %s

struct Base1 {
  int base1_data;
};

struct Base2 {
  int base2_data;
};

struct Derived : Base1, Base2 {
  int derived_data;
};

auto base_to_derived(int Base2::*ptr) -> int Derived::* {
  return ptr;
}

// CIR: cir.func {{.*}} @_Z15base_to_derivedM5Base2i
// CIR:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   %[[RET:.*]] = cir.derived_data_member %[[PTR]] : !cir.data_member<!s32i in !rec_Base2> [4] -> !cir.data_member<!s32i in !rec_Derived>

// LLVM: define {{.*}} i64 @_Z15base_to_derivedM5Base2i
// LLVM:   %[[PTR:.*]] = load i64, ptr %{{.*}}
// LLVM:   %[[IS_NULL:.*]] = icmp eq i64 %[[PTR]], -1
// LLVM:   %[[DERIVED:.*]] = add nsw i64 %[[PTR]], 4
// LLVM:   %[[RET:.*]] = select i1 %[[IS_NULL]], i64 %[[PTR]], i64 %[[DERIVED]]

// OGCG: define {{.*}} i64 @_Z15base_to_derivedM5Base2i
// OGCG:   %[[PTR:.*]] = load i64, ptr %{{.*}}
// OGCG:   %[[DERIVED:.*]] = add nsw i64 %[[PTR]], 4
// OGCG:   %[[IS_NULL:.*]] = icmp eq i64 %[[PTR]], -1
// OGCG:   %[[RET:.*]] = select i1 %[[IS_NULL]], i64 %[[PTR]], i64 %[[DERIVED]]

auto derived_to_base(int Derived::*ptr) -> int Base2::* {
  return static_cast<int Base2::*>(ptr);
}

// CIR: cir.func {{.*}} @_Z15derived_to_baseM7Derivedi
// CIR:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   %[[RET:.*]] = cir.base_data_member %[[PTR]] : !cir.data_member<!s32i in !rec_Derived> [4] -> !cir.data_member<!s32i in !rec_Base2>

// LLVM: define {{.*}} i64 @_Z15derived_to_baseM7Derivedi
// LLVM:   %[[PTR:.*]] = load i64, ptr %{{.*}}
// LLVM:   %[[IS_NULL:.*]] = icmp eq i64 %[[PTR]], -1
// LLVM:   %[[BASE:.*]] = sub nsw i64 %[[PTR]], 4
// LLVM:   %[[RET:.*]] = select i1 %[[IS_NULL]], i64 %[[PTR]], i64 %[[BASE]]

// OGCG: define {{.*}} i64 @_Z15derived_to_baseM7Derivedi
// OGCG:   %[[PTR:.*]] = load i64, ptr %{{.*}}
// OGCG:   %[[BASE:.*]] = sub nsw i64 %[[PTR]], 4
// OGCG:   %[[IS_NULL:.*]] = icmp eq i64 %[[PTR]], -1
// OGCG:   %[[RET:.*]] = select i1 %[[IS_NULL]], i64 %[[PTR]], i64 %[[BASE]]

auto base_to_derived_zero_offset(int Base1::*ptr) -> int Derived::* {
  return ptr;
}

// CIR: cir.func {{.*}} @_Z27base_to_derived_zero_offsetM5Base1i
// CIR:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   %[[RET:.*]] = cir.derived_data_member %[[PTR]] : !cir.data_member<!s32i in !rec_Base1> [0] -> !cir.data_member<!s32i in !rec_Derived>

// No LLVM instructions emitted for performing a zero-offset cast.

// LLVM:      define {{.*}} i64 @_Z27base_to_derived_zero_offsetM5Base1i
// LLVM-NEXT:   %[[PTR_ADDR:.*]] = alloca i64
// LLVM-NEXT:   %[[RETVAL:.*]] = alloca i64
// LLVM-NEXT:   store i64 %{{.*}}, ptr %[[PTR_ADDR]]
// LLVM-NEXT:   %[[TEMP:.*]] = load i64, ptr %[[PTR_ADDR]]
// LLVM-NEXT:   store i64 %[[TEMP]], ptr %[[RETVAL]]
// LLVM-NEXT:   %[[RET:.*]] = load i64, ptr %[[RETVAL]]
// LLVM-NEXT:   ret i64 %[[RET]]

// OGCG:      define {{.*}} i64 @_Z27base_to_derived_zero_offsetM5Base1i
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[PTR_ADDR:.*]] = alloca i64
// OGCG-NEXT:   store i64 %{{.*}}, ptr %[[PTR_ADDR]]
// OGCG-NEXT:   %[[RET:.*]] = load i64, ptr %[[PTR_ADDR]]
// OGCG-NEXT:   ret i64 %[[RET]]

auto derived_to_base_zero_offset(int Derived::*ptr) -> int Base1::* {
  return static_cast<int Base1::*>(ptr);
}

// CIR: cir.func {{.*}} @_Z27derived_to_base_zero_offsetM7Derivedi
// CIR:   %[[PTR:.*]] = cir.load{{.*}} %{{.*}}
// CIR:   %[[RET:.*]] = cir.base_data_member %[[PTR]] : !cir.data_member<!s32i in !rec_Derived> [0] -> !cir.data_member<!s32i in !rec_Base1>

// No LLVM instructions emitted for performing a zero-offset cast.

// LLVM:      define {{.*}} i64 @_Z27derived_to_base_zero_offsetM7Derivedi
// LLVM-NEXT:   %[[PTR_ADDR:.*]] = alloca i64
// LLVM-NEXT:   %[[RETVAL:.*]] = alloca i64
// LLVM-NEXT:   store i64 %{{.*}}, ptr %[[PTR_ADDR]]
// LLVM-NEXT:   %[[TEMP:.*]] = load i64, ptr %[[PTR_ADDR]]
// LLVM-NEXT:   store i64 %[[TEMP]], ptr %[[RETVAL]]
// LLVM-NEXT:   %[[RET:.*]] = load i64, ptr %[[RETVAL]]
// LLVM-NEXT:   ret i64 %[[RET]]

// OGCG:      define {{.*}} i64 @_Z27derived_to_base_zero_offsetM7Derivedi
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[PTR_ADDR:.*]] = alloca i64
// OGCG-NEXT:   store i64 %{{.*}}, ptr %[[PTR_ADDR]]
// OGCG-NEXT:   %[[RET:.*]] = load i64, ptr %[[PTR_ADDR]]
// OGCG-NEXT:   ret i64 %[[RET]]
