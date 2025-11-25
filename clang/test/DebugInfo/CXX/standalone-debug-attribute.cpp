// RUN: %clang_cc1 -DSETATTR=0 -triple x86_64-unknown-linux-gnu -emit-llvm -debug-info-kind=constructor %s -o - | FileCheck %s --check-prefix=DEBUG
// RUN: %clang_cc1 -DSETATTR=1 -triple x86_64-unknown-linux-gnu -emit-llvm -debug-info-kind=constructor %s -o - | FileCheck %s --check-prefix=WITHATTR
// Use -debug-info-kind=constructor because it includes all the optimizations.

#if SETATTR
#define STANDALONEDEBUGATTR __attribute__((standalone_debug))
#else
#define STANDALONEDEBUGATTR
#endif

struct STANDALONEDEBUGATTR StructWithConstructor {
  StructWithConstructor() {}
};
void f(StructWithConstructor s) {}
// DEBUG:  !DICompositeType({{.*}}name: "StructWithConstructor"
// DEBUG-SAME:              flags: {{.*}}DIFlagFwdDecl
// WITHATTR: !DICompositeType({{.*}}name: "StructWithConstructor"
// WITHATTR-NOT: DIFlagFwdDecl

union STANDALONEDEBUGATTR UnionWithConstructor {
  UnionWithConstructor() {}
};
void f(UnionWithConstructor u) {}
// DEBUG:  !DICompositeType({{.*}}name: "UnionWithConstructor"
// DEBUG-SAME:              flags: {{.*}}DIFlagFwdDecl
// WITHATTR: !DICompositeType({{.*}}name: "UnionWithConstructor"
// WITHATTR-NOT: DIFlagFwdDecl

template <typename T> struct ExternTemplate {
  ExternTemplate() {}
  T x;
};
extern template struct STANDALONEDEBUGATTR ExternTemplate<int>;
void f(ExternTemplate<int> s) {}
// DEBUG: !DICompositeType({{.*}}name: "ExternTemplate<int>"
// DEBUG-SAME:             flags: {{.*}}DIFlagFwdDecl
// WITHATTR: !DICompositeType({{.*}}name: "ExternTemplate<int>"
// WITHATTR-NOT: DIFlagFwdDecl

struct STANDALONEDEBUGATTR CompleteTypeRequired {};
void f(CompleteTypeRequired &s) {}
// DEBUG: !DICompositeType({{.*}}name: "CompleteTypeRequired"
// DEBUG-SAME:             flags: {{.*}}DIFlagFwdDecl
// WITHATTR: !DICompositeType({{.*}}name: "CompleteTypeRequired"
// WITHATTR-NOT: DIFlagFwdDecl

struct STANDALONEDEBUGATTR Redecl;
struct Redecl {};
void f(Redecl &s) {}
// DEBUG: !DICompositeType({{.*}}name: "Redecl"
// DEBUG-SAME:             flags: {{.*}}DIFlagFwdDecl
// WITHATTR: !DICompositeType({{.*}}name: "Redecl"
// WITHATTR-NOT: DIFlagFwdDecl

