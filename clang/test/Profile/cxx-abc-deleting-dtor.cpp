// RUN: %clang_cc1 -emit-llvm %s -std=c++11 -o - -fno-rtti \
// RUN:     -fprofile-instrument=clang -fcoverage-mapping -disable-llvm-passes \
// RUN:     -triple=x86_64-windows-msvc | FileCheck %s --check-prefix=MSVC
// RUN: %clang_cc1 -emit-llvm %s -std=c++11 -o - -fno-rtti \
// RUN:     -fprofile-instrument=clang -fcoverage-mapping -disable-llvm-passes \
// RUN:     -triple=x86_64-linux-gnu | FileCheck %s --check-prefix=LINUX

// Check that clang doesn't emit counters or __profn_ variables for deleting
// destructor variants in both C++ ABIs.

struct ABC {
  virtual ~ABC() = default;
  virtual void pure() = 0;
};
struct DerivedABC : ABC {
  ~DerivedABC() override = default;
  void pure() override {}
};
DerivedABC *useABCVTable() { return new DerivedABC(); }

// MSVC-NOT: @"__profn_??_G{{.*}}" =
// MSVC: @"__profn_??1DerivedABC@@{{.*}}" =
// MSVC-NOT: @"__profn_??_G{{.*}}" =
// MSVC: @"__profn_??1ABC@@{{.*}}" =
// MSVC-NOT: @"__profn_??_G{{.*}}" =

// MSVC-LABEL: define linkonce_odr dso_local noundef ptr @"??_GDerivedABC@@UEAAPEAXI@Z"(ptr {{[^,]*}} %this, {{.*}})
// MSVC-NOT:   call void @llvm.instrprof.increment({{.*}})
// MSVC:   call void @"??1DerivedABC@@UEAA@XZ"({{.*}})
// MSVC:   ret void

// MSVC-LABEL: define linkonce_odr dso_local noundef ptr @"??_GABC@@UEAAPEAXI@Z"(ptr {{[^,]*}} %this, {{.*}})
// MSVC-NOT:   call void @llvm.instrprof.increment({{.*}})
// MSVC:   call void @llvm.trap()
// MSVC-NEXT:   unreachable

// MSVC-LABEL: define linkonce_odr dso_local void @"??1DerivedABC@@UEAA@XZ"({{.*}})
// MSVC:   call void @llvm.instrprof.increment({{.*}})
// MSVC:   call void @"??1ABC@@UEAA@XZ"({{.*}})
// MSVC:   ret void

// MSVC-LABEL: define linkonce_odr dso_local void @"??1ABC@@UEAA@XZ"({{.*}})
// MSVC:   call void @llvm.instrprof.increment({{.*}})
// MSVC:   ret void


// D2 is the base, D1 and D0 are deleting and complete dtors.

// LINUX-NOT: @__profn_{{.*D[01]Ev}} =
// LINUX: @__profn__ZN10DerivedABCD2Ev =
// LINUX-NOT: @__profn_{{.*D[01]Ev}} =
// LINUX: @__profn__ZN3ABCD2Ev =
// LINUX-NOT: @__profn_{{.*D[01]Ev}} =

// LINUX-LABEL: define linkonce_odr void @_ZN10DerivedABCD1Ev(ptr {{[^,]*}} %this)
// LINUX-NOT:   call void @llvm.instrprof.increment({{.*}})
// LINUX:   call void @_ZN10DerivedABCD2Ev({{.*}})
// LINUX:   ret void

// LINUX-LABEL: define linkonce_odr void @_ZN10DerivedABCD0Ev(ptr {{[^,]*}} %this)
// LINUX-NOT:   call void @llvm.instrprof.increment({{.*}})
// LINUX:   call void @_ZN10DerivedABCD1Ev({{.*}})
// LINUX:   call void @_ZdlPv({{.*}})
// LINUX:   ret void

// LINUX-LABEL: define linkonce_odr void @_ZN3ABCD1Ev(ptr {{[^,]*}} %this)
// LINUX-NOT:   call void @llvm.instrprof.increment({{.*}})
// LINUX:   call void @llvm.trap()
// LINUX-NEXT:   unreachable

// LINUX-LABEL: define linkonce_odr void @_ZN3ABCD0Ev(ptr {{[^,]*}} %this)
// LINUX-NOT:   call void @llvm.instrprof.increment({{.*}})
// LINUX:   call void @llvm.trap()
// LINUX-NEXT:   unreachable

// LINUX-LABEL: define linkonce_odr void @_ZN10DerivedABCD2Ev(ptr {{[^,]*}} %this)
// LINUX:   call void @llvm.instrprof.increment({{.*}})
// LINUX:   call void @_ZN3ABCD2Ev({{.*}})
// LINUX:   ret void

// LINUX-LABEL: define linkonce_odr void @_ZN3ABCD2Ev(ptr {{[^,]*}} %this)
// LINUX:   call void @llvm.instrprof.increment({{.*}})
// LINUX:   ret void
