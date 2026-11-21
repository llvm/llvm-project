// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -emit-llvm -o - %s | FileCheck %s --check-prefix=C
// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -emit-llvm -x c++ -o - %s \
// RUN:   | FileCheck %s --check-prefix=CXX

void __attribute__((hlxcc)) f(int *);

// C-LABEL: define dso_local void @g(
// C: call hexagon_hlxcc void @f(
// CXX-LABEL: define dso_local void @_Z1gPi(
// CXX: call hexagon_hlxcc void @_Z1fPi(
void g(int *a) { f(a); }

// C: declare hexagon_hlxcc void @f(
// CXX: declare hexagon_hlxcc void @_Z1fPi(
// C: define dso_local hexagon_hlxcc void @h(
// CXX: define dso_local hexagon_hlxcc void @_Z1hPi(
void __attribute__((hlxcc)) h(int *a) { f(a); }

// Calls through a function pointer use the calling convention of its type.
// C-LABEL: define dso_local void @call_fp(
// C: call hexagon_hlxcc void %{{.*}}(
// CXX-LABEL: define dso_local void @_Z7call_fp
// CXX: call hexagon_hlxcc void %{{.*}}(
void call_fp(void (*fp)(int *) __attribute__((hlxcc)), int *a) { fp(a); }
