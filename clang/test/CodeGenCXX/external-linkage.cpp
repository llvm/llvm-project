// RUN: %clang_cc1 -triple i686-windows-msvc   -fno-rtti -fno-threadsafe-statics -fms-extensions -fms-compatibility-version=18.00 -emit-llvm -std=c++1y -O1 -disable-llvm-passes -o - %s -DMSABI -w | FileCheck --check-prefix=MO1 --check-prefix=MO2 %s

// RUN: %clang_cc1 -triple i686-windows-msvc -fno-rtti -fno-threadsafe-statics -fms-extensions -fms-compatibility-version=18.00 -emit-llvm -std=c++1y -o - %s -DMSABI -w | FileCheck --check-prefix=MO3 --check-prefix=MO4 %s

// MO1-DAG:@"\01??_8B@@7B@" = available_externally dllimport unnamed_addr constant [2 x i32] [i32 0, i32 4]
// MO2-DAG: define available_externally dllimport x86_thiscallcc %struct.B* @"\01??0B@@QAE@XZ"

struct __declspec(dllimport) A {
  virtual ~A();
};
struct __declspec(dllimport) B : virtual A {
  virtual ~B();
};
void f() { B b; }

// MO3-DAG: declare dllimport x86_thiscallcc %struct.B* @"\01??0B@@QAE@XZ"
// MO4-DAG: declare dllimport x86_thiscallcc void @"\01??_DB@@QAEXXZ"
