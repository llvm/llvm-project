// RUN: %clang_cc1 -triple x86_64-windows-msvc -fexceptions -fcxx-exceptions %s -emit-llvm -o - | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fexceptions -fcxx-exceptions -fsjlj-exceptions %s -emit-llvm -o - | FileCheck %s --check-prefix=SJLJ
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fexceptions -fcxx-exceptions -fseh-exceptions %s -emit-llvm -o - | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -triple x86_64-windows-msvc -fexceptions -fcxx-exceptions -fdwarf-exceptions %s -emit-llvm -o - | FileCheck %s --check-prefix=DWARF

// MSVC: define dso_local void @f(){{.*}}@__CxxFrameHandler3
// SJLJ: define dso_local void @f(){{.*}}@__gxx_personality_sj0
// DWARF: define dso_local void @f(){{.*}}@__gxx_personality_v0

struct Cleanup {
  Cleanup();
  ~Cleanup();
  int x = 0;
};

void g();
extern "C" void f() {
  Cleanup c;
  g();
}
