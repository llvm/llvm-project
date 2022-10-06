// RUN: %clang_cc1 -triple i686-pc-win32 -fms-extensions -emit-llvm -O1 -disable-llvm-passes %s -o - | FileCheck %s

class __declspec(dllimport) A {
  virtual void m_fn1();
};
template <typename>
class B : virtual A {};

extern template class __declspec(dllimport) B<int>;
class __declspec(dllexport) C : B<int> {};

// CHECK-DAG: @[[VTABLE_C:.*]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4C@@6B@", ptr @"?m_fn1@A@@EAEXXZ"] }
// CHECK-DAG: @[[VTABLE_B:.*]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4?$B@H@@6B@", ptr @"?m_fn1@A@@EAEXXZ"] }, comdat($"??_S?$B@H@@6B@")
// CHECK-DAG: @[[VTABLE_A:.*]] = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"??_R4A@@6B@", ptr @"?m_fn1@A@@EAEXXZ"] }, comdat($"??_SA@@6B@")

// CHECK-DAG: @"??_7C@@6B@" = dllexport unnamed_addr alias ptr, getelementptr inbounds ({ [2 x ptr] }, ptr @[[VTABLE_C]], i32 0, i32 0, i32 1)
// CHECK-DAG: @"??_S?$B@H@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [2 x ptr] }, ptr @[[VTABLE_B]], i32 0, i32 0, i32 1)
// CHECK-DAG: @"??_SA@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [2 x ptr] }, ptr @[[VTABLE_A]], i32 0, i32 0, i32 1)

// CHECK-DAG: @"??_8?$B@H@@7B@" = available_externally dllimport unnamed_addr constant [2 x i32] [i32 0, i32 4]
