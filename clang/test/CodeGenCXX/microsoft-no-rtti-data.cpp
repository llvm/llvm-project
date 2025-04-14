// RUN: %clang_cc1 %s -fno-rtti-data -triple=i386-pc-win32 -o - -emit-llvm | FileCheck %s

// vftable shouldn't have RTTI data in it.
// CHECK-NOT: @"??_R4S@@6B@"
// CHECK: @"??_7S@@6B@" = linkonce_odr unnamed_addr constant { [1 x ptr] } { [1 x ptr] [ptr @"??_GS@@UAEPAXI@Z"] }, comdat

struct type_info;
namespace std { using ::type_info; }

struct S {
  virtual ~S();
} s;

struct U : S {
  virtual ~U();
};

extern S *getS();

const std::type_info &ti = typeid(*getS());
const U &u = dynamic_cast<U &>(*getS());
// CHECK: call ptr @__RTDynamicCast(ptr %{{.+}}, i32 0, ptr @"??_R0?AUS@@@8", ptr @"??_R0?AUU@@@8", i32 1)
