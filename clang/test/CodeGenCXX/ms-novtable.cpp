// RUN: %clang_cc1 -triple i386-pc-win32 %s -emit-llvm -fms-extensions -fms-compatibility -fno-rtti -o - | FileCheck %s

// CHECK-NOT: @"??_7C@@6B@"

// CHECK-DAG: @"??_7A2@@6B@"

// CHECK-DAG: @"??_7B2@@6B@"

// CHECK-NOT: @"??_7B1@@6B@"

// CHECK-NOT: @"??_7A1@@6B@"

struct __declspec(novtable) A1 {
  virtual void a();
} a1;
struct                      A2 {
  virtual void a();
};
struct __declspec(novtable) B1 : virtual A1 {} b1;
struct                      B2 : virtual A1 {} b2;
struct __declspec(novtable) C  : virtual A2 {} c;
