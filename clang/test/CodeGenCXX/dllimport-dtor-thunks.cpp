// RUN: %clang_cc1 -mconstructor-aliases %s -triple x86_64-windows-msvc -fms-extensions -emit-llvm -o - | FileCheck %s

// FIXME: We should really consider removing -mconstructor-aliases for MS C++
// ABI. The risk of bugs introducing ABI incompatibility under
// -mno-constructor-aliases is too high.

// PR32990

// Introduces the virtual destructor. We should use the base destructor
// directly, no thunk needed.
struct __declspec(dllimport) ImportIntroVDtor {
  virtual ~ImportIntroVDtor() {}
};

struct BaseClass {
  virtual ~BaseClass() {}
};

// Non-virtually inherits from a non-dllimport base class. We should again call
// the derived base constructor directly. No need for the complete (aka vbase)
// destructor.
struct __declspec(dllimport) ImportOverrideVDtor : public BaseClass {
  virtual ~ImportOverrideVDtor() {}
};

// Virtually inherits from a non-dllimport base class. This time we need to call
// the complete destructor and emit it inline. It's not exported from the DLL,
// and it must be emitted.
struct __declspec(dllimport) ImportVBaseOverrideVDtor
    : public virtual BaseClass {
  virtual ~ImportVBaseOverrideVDtor() {}
};

extern "C" void testit() {
  ImportIntroVDtor t1;
  ImportOverrideVDtor t2;
  ImportVBaseOverrideVDtor t3;
}

// The destructors are called in reverse order of construction. Only the third
// needs the complete destructor (_D).
// CHECK-LABEL: define dso_local void @testit()
// CHECK:  call void @"\01??_DImportVBaseOverrideVDtor@@QEAAXXZ"(%struct.ImportVBaseOverrideVDtor* %{{.*}})
// CHECK:  call void @"\01??1ImportOverrideVDtor@@UEAA@XZ"(%struct.ImportOverrideVDtor* %{{.*}})
// CHECK:  call void @"\01??1ImportIntroVDtor@@UEAA@XZ"(%struct.ImportIntroVDtor* %{{.*}})

// CHECK-LABEL: define linkonce_odr dso_local void @"\01??_DImportVBaseOverrideVDtor@@QEAAXXZ"
// CHECK-LABEL: declare dllimport void @"\01??1ImportOverrideVDtor@@UEAA@XZ"
// CHECK-LABEL: declare dllimport void @"\01??1ImportIntroVDtor@@UEAA@XZ"
