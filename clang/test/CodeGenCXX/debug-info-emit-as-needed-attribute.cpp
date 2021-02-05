// RUN: %clang_cc1 -DSETATTR=0 -emit-llvm -std=c++14 -debug-info-kind=constructor %s -o - | FileCheck %s --check-prefix=DEBUG
// RUN: %clang_cc1 -DSETATTR=1 -emit-llvm -std=c++14 -debug-info-kind=constructor %s -o - | FileCheck %s --check-prefix=WITHATTR

#if SETATTR
#define DEBUGASNEEDED __attribute__((debug_type_info_as_needed))
#else
#define DEBUGASNEEDED
#endif

// Struct that isn't constructed, so its full type info should be omitted with
// -debug-info-kind=constructor.
struct DEBUGASNEEDED some_struct {
  some_struct() {}
};

void func1(some_struct s) {}
// void func2() { func1(); }
// DEBUG:  !DICompositeType({{.*}}name: "some_struct"
// DEBUG-SAME:              flags: {{.*}}DIFlagFwdDecl
// WITHATTR: !DICompositeType({{.*}}name: "some_struct"
// WITHATTR-NOT: DIFlagFwdDecl

