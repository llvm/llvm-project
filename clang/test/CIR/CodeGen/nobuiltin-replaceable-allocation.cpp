// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-cir -std=c++17 %s -o %t.cir
// RUN: FileCheck -check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fclangir -emit-llvm -std=c++17 %s -o %t-cir.ll
// RUN: FileCheck -check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -std=c++17 %s -o %t.ll
// RUN: FileCheck -check-prefix=LLVM --input-file=%t.ll %s

extern "C" void *malloc(unsigned long);
extern "C" void free(void *);

void *operator new(unsigned long s) {
  return malloc(s);
}

void operator delete(void *p) noexcept {
  free(p);
}

// CIR: cir.func {{.*}}@_Znwm{{.*}} attributes {{.*}}nobuiltin{{.*}} {
// CIR: cir.func {{.*}}@_ZdlPv{{.*}} attributes {{.*}}nobuiltin{{.*}} {

// LLVM: define dso_local noundef nonnull ptr @_Znwm(i64 noundef %{{.*}}) #[[OGCG_NEW_ATTRS:[0-9]+]]
// LLVM: define dso_local void @_ZdlPv(ptr noundef %{{.*}}) #[[OGCG_DEL_ATTRS:[0-9]+]]
// LLVM: attributes #[[OGCG_NEW_ATTRS]] = { {{.*}}nobuiltin{{.*}} }
// LLVM: attributes #[[OGCG_DEL_ATTRS]] = { {{.*}}nobuiltin{{.*}} }
