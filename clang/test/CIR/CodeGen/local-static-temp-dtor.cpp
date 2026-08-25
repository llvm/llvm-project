// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefix=CIR-BEFORE
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// The same lifetime-extended reference temporaries as global-temp-dtor.cpp, but
// extended by a function-local static. Those are initialized and destroyed
// in-function under their guard variable, so the temporary's destructor belongs
// in the cir.local_init's dtor region rather than on the cir.global.

struct NonTrivial {
  NonTrivial();
  ~NonTrivial();
  int x;
};

typedef NonTrivial NonTrivialArr[2];

void use(const NonTrivial &);

void ref() {
  static const NonTrivial &r = NonTrivial();
  use(r);
}

// CIR-BEFORE-LABEL: cir.func {{.*}}@_Z3refv
// CIR-BEFORE:         cir.local_init static_local @_ZZ3refvE1r ctor {
// CIR-BEFORE:           %[[VAR:.*]] = cir.get_global static_local @_ZZ3refvE1r
// CIR-BEFORE:           %[[TEMP:.*]] = cir.get_global @_ZGRZ3refvE1r_
// CIR-BEFORE:           cir.call @_ZN10NonTrivialC1Ev(%[[TEMP]])
// CIR-BEFORE:           cir.store{{.*}} %[[TEMP]], %[[VAR]]
// CIR-BEFORE:         } dtor {
// CIR-BEFORE:           %[[TEMP_DTOR:.*]] = cir.get_global @_ZGRZ3refvE1r_
// CIR-BEFORE:           cir.call @_ZN10NonTrivialD1Ev(%[[TEMP_DTOR]])
// CIR-BEFORE:         }

// LLVM-DAG: @_ZGRZ3refvE1r_ = internal global %struct.NonTrivial zeroinitializer
// LLVM-DAG: @_ZGRZ3arrvE1a_ = internal global [2 x %struct.NonTrivial] zeroinitializer

// LLVM-LABEL: define dso_local void @_Z3refv()
// LLVM:         call i32 @__cxa_guard_acquire(ptr @_ZGVZ3refvE1r)
// LLVMCIR:      call void @_ZN10NonTrivialC1Ev(ptr {{.*}} @_ZGRZ3refvE1r_)
// LLVMCIR:      store ptr @_ZGRZ3refvE1r_, ptr @_ZZ3refvE1r
// LLVM:         call i32 @__cxa_atexit(ptr @_ZN10NonTrivialD1Ev, ptr @_ZGRZ3refvE1r_, ptr @__dso_handle)
// OGCG:         store ptr @_ZGRZ3refvE1r_, ptr @_ZZ3refvE1r
// LLVM:         call void @__cxa_guard_release(ptr @_ZGVZ3refvE1r)

void arr() {
  static const NonTrivialArr &a = NonTrivialArr{};
  use(a[0]);
}

// CIR-BEFORE-LABEL: cir.func {{.*}}@_Z3arrv
// CIR-BEFORE:         cir.local_init static_local @_ZZ3arrvE1a ctor {
// CIR-BEFORE:         } dtor {
// CIR-BEFORE:           %[[ARR_TEMP:.*]] = cir.get_global @_ZGRZ3arrvE1a_
// CIR-BEFORE:           cir.array.dtor %[[ARR_TEMP]] : !cir.ptr<!cir.array<!rec_NonTrivial x 2>> {
// CIR-BEFORE:           ^bb0(%[[ELEMENT:.*]]: !cir.ptr<!rec_NonTrivial>):
// CIR-BEFORE:             cir.call @_ZN10NonTrivialD1Ev(%[[ELEMENT]])
// CIR-BEFORE:           }
// CIR-BEFORE:         }

// The array destroyer is hoisted into a helper either way, and the two arms
// differ in what they hand it: classic codegen bakes the array's address into
// the helper and ignores the incoming parameter, so it registers a null
// __cxa_atexit argument, while CIR's helper destroys whatever it is given and
// so is registered with the address. Both are correct today, and this is not
// specific to a static local -- CIR emits a namespace-scope array temporary
// the same way.
//
// TODO(cir): we want the classic shape here. Besides being the output these
// tests are written against, it settles a duplication CIR currently pays for
// and does not use: its helper is parameterized, so it is the same function
// for every array of the same element type and count, yet one is still emitted
// per global (@__cxx_global_array_dtor, @__cxx_global_array_dtor.1, ...).
// Closing over the array instead would let these two checks become one.

// LLVM-LABEL: define dso_local void @_Z3arrv()
// LLVM:         call i32 @__cxa_guard_acquire(ptr @_ZGVZ3arrvE1a)
// LLVMCIR:      call i32 @__cxa_atexit(ptr @__cxx_global_array_dtor, ptr @_ZGRZ3arrvE1a_, ptr @__dso_handle)
// OGCG:         call i32 @__cxa_atexit(ptr @__cxx_global_array_dtor, ptr null, ptr @__dso_handle)
// LLVM:         call void @__cxa_guard_release(ptr @_ZGVZ3arrvE1a)
