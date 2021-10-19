// RUN: %clang_cc1 %s -fexceptions -exception-model=seh -emit-llvm -triple x86_64-w64-windows-gnu -o - | FileCheck %s --check-prefix=X64
// RUN: %clang_cc1 %s -fexceptions -exception-model=dwarf -emit-llvm -triple i686-w64-windows-gnu -o - | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 %s -fexceptions -emit-llvm -triple i686-w64-windows-gnu -o - | FileCheck %s --check-prefix=X86

extern "C" void foo();
extern "C" void bar();

struct Cleanup {
  ~Cleanup() {
    bar();
  }
};

extern "C" void test() {
  Cleanup x;
  foo();
}

// X64: define dso_local void @test()
// X64-SAME: personality i8* bitcast (i32 (...)* @__gxx_personality_seh0 to i8*)
// X64: invoke void @foo()
// X64: landingpad { i8*, i32 }

// X86: define dso_local void @test()
// X86-SAME: personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
// X86: invoke void @foo()
// X86: landingpad { i8*, i32 }
