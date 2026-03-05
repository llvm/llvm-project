// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

struct B { B(); };

struct A {
  B a;
  int b;
  char c;
};

struct C {
  C(int a, int b): a(a), b(b) {}
  template <unsigned>
  friend const int &get(const C&);
 private:
  int a;
  int b;
};

template <>
const int &get<0>(const C& c) { return c.a; }
template <>
const int &get<1>(const C& c) { return c.b; }

namespace std {

template <typename>
struct tuple_size;

template <>
struct tuple_size<C> { constexpr inline static unsigned value = 2; };

template <unsigned, typename>
struct tuple_element;

template <unsigned I>
struct tuple_element<I, C> { using type = const int; };

}


// binding to data members
void f(A &a) {
  // CIR: @_Z1fR1A
  // LLVM: @_Z1fR1A

  auto &[x, y, z] = a;
  (x, y, z);
  // CIR: %[[a:.*]] = cir.load %1 : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
  // CIR: {{.*}} = cir.get_member %[[a]][0] {name = "a"} : !cir.ptr<!rec_A> -> !cir.ptr<!rec_B>
  // CIR: %[[a:.*]] = cir.load %1 : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
  // CIR: {{.*}} = cir.get_member %[[a]][2] {name = "b"} : !cir.ptr<!rec_A> -> !cir.ptr<!s32i>
  // CIR: %[[a:.*]] = cir.load %1 : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
  // CIR: {{.*}} = cir.get_member %[[a]][3] {name = "c"} : !cir.ptr<!rec_A> -> !cir.ptr<!s8i>
  // LLVM: {{.*}} = getelementptr %struct.A, ptr {{.*}}, i32 0, i32 0
  // LLVM: {{.*}} = getelementptr %struct.A, ptr {{.*}}, i32 0, i32 2
  // LLVM: {{.*}} = getelementptr %struct.A, ptr {{.*}}, i32 0, i32 3

  auto [x2, y2, z2] = a;
  (x2, y2, z2);
  // CIR: cir.copy %[[a:.*]] to %2 : !cir.ptr<!rec_A>
  // CIR: {{.*}} = cir.get_member %2[0] {name = "a"} : !cir.ptr<!rec_A> -> !cir.ptr<!rec_B>
  // CIR: {{.*}} = cir.get_member %2[2] {name = "b"} : !cir.ptr<!rec_A> -> !cir.ptr<!s32i>
  // CIR: {{.*}} = cir.get_member %2[3] {name = "c"} : !cir.ptr<!rec_A> -> !cir.ptr<!s8i>

  // for the rest, just expect the codegen does't crash
  auto &&[x3, y3, z3] = a;
  (x3, y3, z3);

  const auto &[x4, y4, z4] = a;
  (x4, y4, z4);

  const auto [x5, y5, z5] = a;
  (x5, y5, z5);
}

// binding to a tuple-like type
void g(C &c) {
  // CIR: @_Z1gR1C
  // LLVM: @_Z1gR1C

  auto [x8, y8] = c;
  (x8, y8);
  // CIR: cir.copy %7 to %[[c:.*]] : !cir.ptr<!rec_C>
  // CIR: %[[x8:.*]] = cir.call @_Z3getILj0EERKiRK1C(%[[c]]) : (!cir.ptr<!rec_C>) -> !cir.ptr<!s32i>
  // CIR: cir.store{{.*}} %[[x8]], %[[x8p:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: %[[x9:.*]] = cir.call @_Z3getILj1EERKiRK1C(%[[c]]) : (!cir.ptr<!rec_C>) -> !cir.ptr<!s32i>
  // CIR: cir.store{{.*}} %[[x9]], %[[x9p:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: {{.*}} = cir.load %[[x8p]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: {{.*}} = cir.load %[[x9p]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // LLVM: call void @llvm.memcpy.p0.p0.i32(ptr {{.*}}, ptr {{.*}}, i32 8, i1 false)
  // LLVM: {{.*}} = call ptr @_Z3getILj0EERKiRK1C(ptr {{.*}})
  // LLVM: {{.*}} = call ptr @_Z3getILj1EERKiRK1C(ptr {{.*}})

  auto &[x9, y9] = c;
  (x9, y9);
  // CIR: cir.store{{.*}} %12, %[[cp:.*]] : !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>
  // CIR: %[[c:.*]] = cir.load %[[cp]] : !cir.ptr<!cir.ptr<!rec_C>>, !cir.ptr<!rec_C>
  // CIR: %[[x8:.*]] = cir.call @_Z3getILj0EERKiRK1C(%[[c]]) : (!cir.ptr<!rec_C>) -> !cir.ptr<!s32i>
  // CIR: cir.store{{.*}} %[[x8]], %[[x8p:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: %[[c:.*]] = cir.load %[[cp]] : !cir.ptr<!cir.ptr<!rec_C>>, !cir.ptr<!rec_C>
  // CIR: %[[x9:.*]] = cir.call @_Z3getILj1EERKiRK1C(%[[c]]) : (!cir.ptr<!rec_C>) -> !cir.ptr<!s32i>
  // CIR: cir.store{{.*}} %[[x9]], %[[x9p:.*]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // CIR: {{.*}} = cir.load %[[x8p]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i>
  // CIR: {{.*}} = cir.load %[[x9p]] : !cir.ptr<!cir.ptr<!s32i>>, !cir.ptr<!s32i> 
}

// TODO: add test case for binding to an array type
// after ArrayInitLoopExpr is supported
