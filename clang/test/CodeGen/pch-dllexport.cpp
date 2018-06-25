// Build PCH without object file, then use it.
// RUN: %clang_cc1 -triple i686-pc-win32 -fms-extensions -emit-pch -o %t %s
// RUN: %clang_cc1 -triple i686-pc-win32 -fms-extensions -emit-obj -emit-llvm -include-pch %t -o - %s | FileCheck -check-prefix=PCH %s

// Build PCH with object file, then use it.
// RUN: %clang_cc1 -triple i686-pc-win32 -fms-extensions -emit-pch -building-pch-with-obj -o %t %s
// RUN: %clang_cc1 -triple i686-pc-win32 -fms-extensions -emit-obj -emit-llvm -include-pch %t -building-pch-with-obj -o - %s | FileCheck -check-prefix=OBJ %s
// RUN: %clang_cc1 -triple i686-pc-win32 -fms-extensions -emit-obj -emit-llvm -include-pch %t -o - %s | FileCheck -check-prefix=PCHWITHOBJ %s

// Check for vars separately to avoid having to reorder the check statements.
// RUN: %clang_cc1 -triple i686-pc-win32 -fms-extensions -emit-obj -emit-llvm -include-pch %t -o - %s | FileCheck -check-prefix=PCHWITHOBJVARS %s

#ifndef IN_HEADER
#define IN_HEADER

inline void __declspec(dllexport) foo() {}
// OBJ: define weak_odr dso_local dllexport void @"?foo@@YAXXZ"
// PCH: define weak_odr dso_local dllexport void @"?foo@@YAXXZ"
// PCHWITHOBJ-NOT: define {{.*}}foo


// This function is referenced, so gets emitted as usual.
inline void __declspec(dllexport) baz() {}
// OBJ: define weak_odr dso_local dllexport void @"?baz@@YAXXZ"
// PCH: define weak_odr dso_local dllexport void @"?baz@@YAXXZ"
// PCHWITHOBJ: define weak_odr dso_local dllexport void @"?baz@@YAXXZ"


struct __declspec(dllexport) S {
  void bar() {}
// OBJ: define weak_odr dso_local dllexport x86_thiscallcc void @"?bar@S@@QAEXXZ"
// PCH: define weak_odr dso_local dllexport x86_thiscallcc void @"?bar@S@@QAEXXZ"
// PCHWITHOBJ-NOT: define {{.*}}bar
};

// This isn't dllexported, attribute((used)) or referenced, so not emitted.
inline void quux() {}
// OBJ-NOT: define {{.*}}quux
// PCH-NOT: define {{.*}}quux
// PCHWITHOBJ-NOT: define {{.*}}quux

// Referenced non-dllexport function.
inline void referencedNonExported() {}
// OBJ: define {{.*}}referencedNonExported
// PCH: define {{.*}}referencedNonExported
// PCHWITHOBJ: define {{.*}}referencedNonExported

template <typename T> void __declspec(dllexport) implicitInstantiation(T) {}

template <typename T> inline void __declspec(dllexport) explicitSpecialization(T) {}

template <typename T> void __declspec(dllexport) explicitInstantiationDef(T) {}

template <typename T> void __declspec(dllexport) explicitInstantiationDefAfterDecl(T) {}
extern template void explicitInstantiationDefAfterDecl<int>(int);

template <typename T> T __declspec(dllexport) variableTemplate;
extern template int variableTemplate<int>;

#else

void use() {
  baz();
  referencedNonExported();
}

// Templates can be tricky. None of the definitions below come from the PCH.

void useTemplate() { implicitInstantiation(42); }
// PCHWITHOBJ: define weak_odr dso_local dllexport void @"??$implicitInstantiation@H@@YAXH@Z"

template<> inline void __declspec(dllexport) explicitSpecialization<int>(int) {}
// PCHWITHOBJ: define weak_odr dso_local  dllexport void @"??$explicitSpecialization@H@@YAXH@Z"

template void __declspec(dllexport) explicitInstantiationDef<int>(int);
// PCHWITHOBJ: define weak_odr dso_local dllexport void @"??$explicitInstantiationDef@H@@YAXH@Z"

template void __declspec(dllexport) explicitInstantiationDefAfterDecl<int>(int);
// PCHWITHOBJ: define weak_odr dso_local dllexport void @"??$explicitInstantiationDefAfterDecl@H@@YAXH@Z"(i32)

template int __declspec(dllexport) variableTemplate<int>;
// PCHWITHOBJVARS: @"??$variableTemplate@H@@3HA" = weak_odr dso_local dllexport global

#endif
