// RUN: %clang_cc1 %s -I%S -triple=x86_64-pc-windows-msvc -fstrict-vtable-pointers -disable-llvm-passes -O1 -emit-llvm -o %t.ll
// RUN: FileCheck %s < %t.ll

struct A {
  virtual ~A();
};
struct B : virtual A {};
class C : B {};
C foo;

// CHECK-LABEL: define {{.*}} @"??0C@@QEAA@XZ"(ptr {{.*}} %this, i32 {{.*}} %is_most_derived)
// CHECK: ctor.init_vbases:
// CHECK-NEXT: %0 = getelementptr inbounds i8, ptr %this1, i64 0
// CHECK-NEXT: store ptr @"??_8C@@7B@", ptr %0
// CHECK-NEXT: %1 = call ptr @llvm.launder.invariant.group.p0(ptr %this1)
// CHECK-NEXT: %2 = getelementptr inbounds i8, ptr %1, i64 8
// CHECK-NEXT: %call = call noundef ptr @"??0A@@QEAA@XZ"(ptr {{.*}} %2) #2
// CHECK-NEXT: br label %ctor.skip_vbases
// CHECK-EMPTY:
// CHECK-NEXT: ctor.skip_vbases:
// CHECK-NEXT: %3 = call ptr @llvm.launder.invariant.group.p0(ptr %this1)
// CHECK-NEXT: %call3 = call noundef ptr @"??0B@@QEAA@XZ"(ptr {{.*}} %3, i32 noundef 0) #2
