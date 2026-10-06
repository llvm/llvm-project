// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

// Minimal in-source declarations of the standard-library bits needed for a
// destroying operator delete so this test does not depend on a system header.
namespace std {
struct destroying_delete_t {
  explicit destroying_delete_t() = default;
};
inline constexpr destroying_delete_t destroying_delete{};
} // namespace std

// The deleting destructor calls the destroying operator delete and does not
// call the complete destructor. The operator delete destroys the object.
struct A {
  virtual ~A();
  void operator delete(A *, std::destroying_delete_t);
};

A::~A() {}

void A::operator delete(A *, std::destroying_delete_t) {}

// CIR-LABEL: cir.func {{.*}} @_ZN1AD0Ev
// CIR: %[[THIS_ADDR:.*]] = cir.alloca "this"
// CIR: %[[TAG:.*]] = cir.alloca "destroying.delete.tag"
// CIR: cir.store %[[ARG:.*]], %[[THIS_ADDR]]
// CIR: %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR: cir.load{{.*}} %[[TAG]]
// CIR: cir.call @_ZN1AdlEPS_St19destroying_delete_t(%[[THIS]])
// CIR-NOT: cir.call @_ZN1AD{{[12]}}Ev
// CIR: cir.return

// LLVM-LABEL: define {{.*}} void @_ZN1AD0Ev(
// LLVM: %[[THIS_ADDR:.*]] = alloca ptr
// LLVMCIR: alloca %"struct.std::destroying_delete_t"
// OGCG-NOT: alloca %"struct.std::destroying_delete_t"
// LLVM: store ptr %[[ARG:.*]], ptr %[[THIS_ADDR]]
// LLVM: %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM: call void @_ZN1AdlEPS_St19destroying_delete_t(ptr noundef %[[THIS]])
// LLVM-NOT: call {{.*}}@_ZN1AD{{[12]}}Ev
// LLVM: ret void

// B inherits A's destroying operator delete. The deleting destructor adjusts
// 'this' to the A subobject and calls that operator delete, and does not call
// B's complete destructor.
struct Padding {
  virtual void f();
};

struct B : Padding, A {
  ~B() override;
};

B::~B() {}

// CIR-LABEL: cir.func {{.*}} @_ZN1BD0Ev
// CIR: %[[THIS_ADDR:.*]] = cir.alloca "this"
// CIR: cir.store %[[ARG:.*]], %[[THIS_ADDR]]
// CIR: %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR: %[[A:.*]] = cir.base_class_addr {{.*}} %[[THIS]] [8]
// CIR: cir.call @_ZN1AdlEPS_St19destroying_delete_t(%[[A]])
// CIR-NOT: cir.call @_ZN1BD{{[12]}}Ev
// CIR-NOT: cir.call @_ZN1AD{{[12]}}Ev
// CIR: cir.return

// LLVM-LABEL: define {{.*}} void @_ZN1BD0Ev(
// LLVM: %[[THIS_ADDR:.*]] = alloca ptr
// LLVM: store ptr %[[ARG:.*]], ptr %[[THIS_ADDR]]
// LLVM: %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM: %[[A:.*]] = getelementptr {{.*}}i8, ptr %[[THIS]], {{i32|i64}} 8
// LLVM: call void @_ZN1AdlEPS_St19destroying_delete_t(ptr noundef %[[A]])
// LLVM-NOT: call {{.*}}@_ZN1BD{{[12]}}Ev
// LLVM-NOT: call {{.*}}@_ZN1AD{{[12]}}Ev
// LLVM: ret void
