// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -O1 -fno-rtti -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -O1 -fno-rtti -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -O1 -fno-rtti -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

namespace ConstructionVTableThunk {
struct Base {
  virtual int f(int x);
};
struct Left : virtual Base {};
struct Right : virtual Base {
  int f(int x) override;
};
struct Middle : Left, Right {};
struct Derived : Middle {
  int f(int x) override;
};
int Derived::f(int x) { return x; }
} // namespace ConstructionVTableThunk

// The Middle-in-Derived construction vtable reaches Right::f through a virtual
// thunk, and this translation unit only declares Right::f.
// CIR: cir.global "private" constant external @_ZTCN23ConstructionVTableThunk7DerivedE0_NS_6MiddleE = #cir.vtable<{{{.*}}#cir.global_view<@_ZTv0_n24_N23ConstructionVTableThunk5Right1fEi>

// Emitting that thunk creates the Right::f declaration on demand.  It must
// become a sibling of the thunk rather than nesting inside its body, so the
// declarations are pinned to the ops immediately following the thunk.
// CIR-LABEL: cir.func available_externally @_ZTv0_n24_N23ConstructionVTableThunk5Right1fEi
// CIR-NOT:     cir.func
// CIR:         cir.call @_ZN23ConstructionVTableThunk5Right1fEi(
// CIR:         cir.return %{{.+}} : !s32i
// CIR-NEXT:  }
// CIR-NEXT:  cir.func private @_ZN23ConstructionVTableThunk5Right1fEi(
// CIR-NEXT:  cir.func private @_ZN23ConstructionVTableThunk4Base1fEi(

// LLVM: define available_externally noundef i32 @_ZTv0_n24_N23ConstructionVTableThunk5Right1fEi(ptr noundef %{{.+}}, i32 noundef %{{.+}})
// LLVM:   {{(tail )?}}call noundef i32 @_ZN23ConstructionVTableThunk5Right1fEi(ptr noundef nonnull align 8 dereferenceable(8) %{{.+}}, i32 noundef %{{.+}})
// LLVM: declare noundef i32 @_ZN23ConstructionVTableThunk5Right1fEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef)
