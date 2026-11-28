// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++17 -O2 -emit-llvm -o - %t/sink.cpp | FileCheck %s --check-prefix=KEEP
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++17 -O2 -emit-llvm -fno-module-local-hints -o - %t/sink.cpp | FileCheck %s --check-prefix=KEEP
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++17 -O2 -emit-llvm -o - %t/main.cpp | FileCheck %s --check-prefix=MAIN
// RUN: %clang_cc1 -triple arm64-apple-macosx14.0.0 -std=c++17 -O2 -disable-llvm-passes -emit-llvm -o - %t/sink.cpp | FileCheck %s --check-prefix=ATTRS

// KEEP: define {{.*}} @_Z4sinki(
// KEEP: call {{.*}} @_Z15crossTUTemplateIiET_S0_(
// KEEP: define linkonce_odr {{.*}} @_Z15crossTUTemplateIiET_S0_(

// MAIN: declare {{.*}} @_Z15crossTUTemplateIiET_S0_(

// Functions declared in decl-only.h aren't marked, those only declared in
// sink.cpp are.
// ATTRS-DAG: define linkonce_odr {{.*}} @_Z15crossTUTemplateIiET_S0_({{.*}}) #[[OFF:[0-9]+]]
// ATTRS-DAG: define linkonce_odr {{.*}} @_Z13crossTUInlinei({{.*}}) #[[OFFINL:[0-9]+]]
// ATTRS-DAG: define linkonce_odr {{.*}} @_ZN7HdrType6methodEv({{.*}}) #[[OFFINL]]
// ATTRS-DAG: define linkonce_odr {{.*}} @_Z13localTemplateIiET_S0_({{.*}}) #[[ON:[0-9]+]]
// ATTRS-DAG: define linkonce_odr {{.*}} @_Z11localInlinei({{.*}}) #[[ONINL:[0-9]+]]
// ATTRS-DAG: attributes #[[OFF]] = { mustprogress nounwind {{.*}}}
// ATTRS-DAG: attributes #[[OFFINL]] = { inlinehint mustprogress nounwind {{.*}}}
// ATTRS-DAG: attributes #[[ON]] = { likely_module_local mustprogress nounwind {{.*}}}
// ATTRS-DAG: attributes #[[ONINL]] = { inlinehint likely_module_local mustprogress nounwind {{.*}}}

//--- decl-only.h
// Declared only; the definitions live in sink.cpp.
template <typename T> T crossTUTemplate(T x);
inline int crossTUInline(int x);
struct HdrType {
  int method();
};
int sink(int);
int sinkMore(int);

//--- sink.cpp
#include "decl-only.h"
void opaque(int);
// Big enough to stay out of line without the sole-call bonus.
template <typename T> T crossTUTemplate(T x) {
#define S(i) opaque(x + i); x ^= x << (i % 7); opaque(x * i);
#define S10(i) S(i##0) S(i##1) S(i##2) S(i##3) S(i##4) \
               S(i##5) S(i##6) S(i##7) S(i##8) S(i##9)
  S10(1) S10(2) S10(3) S10(4)
  return x;
}
int sink(int v) { return crossTUTemplate(v); }

inline int crossTUInline(int x) { return x + 1; }
inline int HdrType::method() { return 2; }
template <typename T> T localTemplate(T x) { return x + 3; }
inline int localInline(int x) { return x + 4; }
int sinkMore(int v) {
  return crossTUInline(v) + HdrType().method() + localTemplate(v) +
         localInline(v);
}

//--- main.cpp
#include "decl-only.h"
int use(int v) { return sink(v) + crossTUTemplate(v); }
