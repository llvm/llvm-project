// RUN: %clang_cc1 -no-opaque-pointers -std=c++17 -emit-llvm %s -triple x86_64-windows-msvc -o - | FileCheck %s

// Make sure virtual base base destructors get referenced and emitted if
// necessary when the complete ("vbase") destructor is emitted. In this case,
// clang previously did not emit ~DefaultedDtor.
struct HasDtor { ~HasDtor(); };
struct DefaultedDtor {
  ~DefaultedDtor() = default;
  HasDtor o;
};
struct HasCompleteDtor : virtual DefaultedDtor {
  ~HasCompleteDtor();
};
void useCompleteDtor(HasCompleteDtor *p) { delete p; }

// CHECK-LABEL: define dso_local void @"?useCompleteDtor@@YAXPEAUHasCompleteDtor@@@Z"(%struct.HasCompleteDtor* noundef %p)
// CHECK: call void @"??_DHasCompleteDtor@@QEAAXXZ"({{.*}})

// CHECK-LABEL: define linkonce_odr dso_local void @"??_DHasCompleteDtor@@QEAAXXZ"(%struct.HasCompleteDtor* {{[^,]*}} %this)
// CHECK: call void @"??1HasCompleteDtor@@QEAA@XZ"({{.*}})
// CHECK: call void @"??1DefaultedDtor@@QEAA@XZ"({{.*}})

// CHECK-LABEL: define linkonce_odr dso_local void @"??1DefaultedDtor@@QEAA@XZ"(%struct.DefaultedDtor* {{[^,]*}} %this)
// CHECK: call void @"??1HasDtor@@QEAA@XZ"(%struct.HasDtor* {{[^,]*}} %{{.*}})

