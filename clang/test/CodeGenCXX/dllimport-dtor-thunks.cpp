// RUN: %clang_cc1 -mconstructor-aliases %s -triple x86_64-windows-msvc -fms-extensions -emit-llvm -o - | FileCheck %s

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

// Virtually inherits from a non-dllimport base class. In this case, we can
// expect the DLL to provide a definition of the complete dtor. See
// dllexport-dtor-thunks.cpp.
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
// CHECK:  call void @"??_DImportVBaseOverrideVDtor@@QEAAXXZ"(ptr {{[^,]*}} %{{.*}})
// CHECK:  call void @"??1ImportOverrideVDtor@@UEAA@XZ"(ptr {{[^,]*}} %{{.*}})
// CHECK:  call void @"??1ImportIntroVDtor@@UEAA@XZ"(ptr {{[^,]*}} %{{.*}})

// CHECK-LABEL: declare dllimport void @"??_DImportVBaseOverrideVDtor@@QEAAXXZ"
// CHECK-LABEL: declare dllimport void @"??1ImportOverrideVDtor@@UEAA@XZ"
// CHECK-LABEL: declare dllimport void @"??1ImportIntroVDtor@@UEAA@XZ"
