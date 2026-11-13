// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -emit-llvm -o - %s | FileCheck %s --check-prefix=C
// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -emit-llvm -x c++ -o - %s \
// RUN:   | FileCheck %s --check-prefix=CXX

void __attribute__((preserve_hlx)) f(int *);

// C-LABEL: define dso_local void @g(
// C: call hexagon_preserve_hlx void @f(
// CXX-LABEL: define dso_local void @_Z1gPi(
// CXX: call hexagon_preserve_hlx void @_Z1fPi(
void g(int *a) { f(a); }

// C: declare hexagon_preserve_hlx void @f(
// CXX: declare hexagon_preserve_hlx void @_Z1fPi(
// C: define dso_local hexagon_preserve_hlx void @h(
// CXX: define dso_local hexagon_preserve_hlx void @_Z1hPi(
void __attribute__((preserve_hlx)) h(int *a) { f(a); }
