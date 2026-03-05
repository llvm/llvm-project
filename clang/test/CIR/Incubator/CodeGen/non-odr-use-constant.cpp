// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck %s --input-file=%t.ll --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ogcg.ll
// RUN: FileCheck %s --input-file=%t.ogcg.ll --check-prefix=OGCG
//
// Test non-ODR-use constant expressions

namespace llvm {
  template<typename ValueTy> class StringMapEntry {};
  template<typename ValueTy> class StringMapIterBase {
  public:
    StringMapEntry<ValueTy>& operator*() const;
    StringMapIterBase& operator++();
    friend bool operator!=(const StringMapIterBase& LHS, const StringMapIterBase& RHS);
  };
  template<typename ValueTy> class StringMap {
  public:
    StringMapIterBase<ValueTy> begin();
    StringMapIterBase<ValueTy> end();
  };
  struct EmptyStringSetTag {};
  template<class AllocatorTy = int> class StringSet : public StringMap<EmptyStringSetTag> {};
}

namespace clang {
  // Static variable that will be referenced without ODR-use in range-for
  static llvm::StringSet<> BuiltinClasses;

  void EmitBuiltins() {
    // This range-for iterates over BuiltinClasses without constituting an ODR-use
    // because it's used in an unevaluated context for the range-for desugaring
    for (const auto &Entry : BuiltinClasses) {
    }
  }
}

// CIR: cir.global "private" internal dso_local @_ZN5clangL14BuiltinClassesE
// CIR: cir.func {{.*}}@_ZN5clang12EmitBuiltinsEv()
// CIR:   %{{.*}} = cir.const #cir.global_view<@_ZN5clangL14BuiltinClassesE>

// LLVM: @_ZN5clangL14BuiltinClassesE = internal global
// LLVM: define {{.*}}@_ZN5clang12EmitBuiltinsEv()
// LLVM:   %{{.*}} = alloca ptr
// LLVM:   store ptr @_ZN5clangL14BuiltinClassesE

// OGCG: @_ZN5clangL14BuiltinClassesE = internal global
// OGCG: define {{.*}}@_ZN5clang12EmitBuiltinsEv()
// OGCG:   %{{.*}} = alloca ptr
// OGCG:   store ptr @_ZN5clangL14BuiltinClassesE

// Test non-reference type NOUR_Constant (local constexpr in lambda)
struct A { int x, y[2]; int arr[3]; };
// CIR-DAG: @__const._Z1fi.a
// LLVM-DAG: @__const._Z1fi.a
// OGCG-DAG: @__const._Z1fi.a
int f(int i) {
  constexpr A a = {1, 2, 3, 4, 5, 6};
  return [] (int n, int A::*p) {
    return (n >= 0 ? a.arr[n] : (n == -1 ? a.*p : a.y[2 - n]));
  }(i, &A::x);
}
// CIR: cir.get_global @__const._Z1fi.a
// LLVM: getelementptr {{.*}} @__const._Z1fi.a
// OGCG: getelementptr inbounds {{.*}} @__const._Z1fi.a
