// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=ITANIUM,LINUX
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-macos -emit-llvm %s -o - | FileCheck %s --check-prefixes=ITANIUM,DARWIN
// RUN: %clang_cc1 -std=c++11 -triple x86_64-windows-pc -emit-llvm %s -o - | FileCheck %s --check-prefix=WINDOWS

struct S {
  int __attribute__((target("sse4.2"))) foo(int) { return 0; }
  int __attribute__((target("arch=sandybridge"))) foo(int);
  int __attribute__((target("arch=ivybridge"))) foo(int) { return 1; }
  int __attribute__((target("default"))) foo(int) { return 2; }

  S &__attribute__((target("arch=ivybridge"))) operator=(const S &) {
    return *this;
  }
  S &__attribute__((target("default"))) operator=(const S &) {
    return *this;
  }
};

struct ConvertTo {
  __attribute__((target("arch=ivybridge"))) operator S() const {
    return S{};
  }
  __attribute__((target("default"))) operator S() const {
    return S{};
  }
};

int bar() {
  S s;
  S s2;
  s2 = s;

  ConvertTo C;
  s2 = static_cast<S>(C);

  return s.foo(0);
}

struct S2 {
  int __attribute__((target("sse4.2"))) foo(int);
  int __attribute__((target("arch=sandybridge"))) foo(int);
  int __attribute__((target("arch=ivybridge"))) foo(int);
  int __attribute__((target("default"))) foo(int);
};

int bar2() {
  S2 s;
  return s.foo(0);
}

int __attribute__((target("sse4.2"))) S2::foo(int) { return 0; }
int __attribute__((target("arch=ivybridge"))) S2::foo(int) { return 1; }
int __attribute__((target("default"))) S2::foo(int) { return 2; }

template<typename T>
struct templ {
  int __attribute__((target("sse4.2"))) foo(int) { return 0; }
  int __attribute__((target("arch=sandybridge"))) foo(int);
  int __attribute__((target("arch=ivybridge"))) foo(int) { return 1; }
  int __attribute__((target("default"))) foo(int) { return 2; }
};

int templ_use() {
  templ<int> a;
  templ<double> b;
  return a.foo(1) + b.foo(2);
}

// DARWIN-NOT: comdat

// ITANIUM: @_ZN1SaSERKS_.ifunc = weak_odr ifunc ptr (ptr, ptr), ptr @_ZN1SaSERKS_.resolver
// ITANIUM: @_ZNK9ConvertTocv1SEv.ifunc = weak_odr ifunc void (ptr), ptr @_ZNK9ConvertTocv1SEv.resolver
// ITANIUM: @_ZN1S3fooEi.ifunc = weak_odr ifunc i32 (ptr, i32), ptr @_ZN1S3fooEi.resolver
// ITANIUM: @_ZN2S23fooEi.ifunc = weak_odr ifunc i32 (ptr, i32), ptr @_ZN2S23fooEi.resolver
// Templates:
// ITANIUM: @_ZN5templIiE3fooEi.ifunc = weak_odr ifunc i32 (ptr, i32), ptr @_ZN5templIiE3fooEi.resolver
// ITANIUM: @_ZN5templIdE3fooEi.ifunc = weak_odr ifunc i32 (ptr, i32), ptr @_ZN5templIdE3fooEi.resolver

// ITANIUM: define{{.*}} i32 @_Z3barv()
// ITANIUM: %s = alloca %struct.S, align 1
// ITANIUM: %s2 = alloca %struct.S, align 1
// ITANIUM: %C = alloca %struct.ConvertTo, align 1
// ITANIUM: call noundef nonnull align 1 dereferenceable(1) ptr @_ZN1SaSERKS_.ifunc(ptr {{[^,]*}} %s2
// ITANIUM: call void @_ZNK9ConvertTocv1SEv.ifunc(ptr {{[^,]*}} %C)
// ITANIUM: call noundef nonnull align 1 dereferenceable(1) ptr @_ZN1SaSERKS_.ifunc(ptr {{[^,]*}} %s2
// ITANIUM: call noundef i32 @_ZN1S3fooEi.ifunc(ptr {{[^,]*}} %s, i32 noundef 0)

// WINDOWS: define dso_local noundef i32 @"?bar@@YAHXZ"()
// WINDOWS: %s = alloca %struct.S, align 1
// WINDOWS: %s2 = alloca %struct.S, align 1
// WINDOWS: %C = alloca %struct.ConvertTo, align 1
// WINDOWS: call noundef nonnull align 1 dereferenceable(1) ptr @"??4S@@QEAAAEAU0@AEBU0@@Z.resolver"(ptr {{[^,]*}} %s2
// WINDOWS: call void @"??BConvertTo@@QEBA?AUS@@XZ.resolver"(ptr {{[^,]*}} %C
// WINDOWS: call noundef nonnull align 1 dereferenceable(1) ptr @"??4S@@QEAAAEAU0@AEBU0@@Z.resolver"(ptr {{[^,]*}} %s2
// WINDOWS: call noundef i32 @"?foo@S@@QEAAHH@Z.resolver"(ptr {{[^,]*}} %s, i32 noundef 0)

// ITANIUM: define weak_odr ptr @_ZN1SaSERKS_.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_ZN1SaSERKS_.arch_ivybridge
// ITANIUM: ret ptr @_ZN1SaSERKS_

// WINDOWS: define weak_odr dso_local ptr @"??4S@@QEAAAEAU0@AEBU0@@Z.resolver"(ptr %0, ptr %1)
// WINDOWS: call ptr @"??4S@@QEAAAEAU0@AEBU0@@Z.arch_ivybridge"
// WINDOWS: call ptr @"??4S@@QEAAAEAU0@AEBU0@@Z"

// ITANIUM: define weak_odr ptr @_ZNK9ConvertTocv1SEv.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_ZNK9ConvertTocv1SEv.arch_ivybridge
// ITANIUM: ret ptr @_ZNK9ConvertTocv1SEv

// WINDOWS: define weak_odr dso_local void @"??BConvertTo@@QEBA?AUS@@XZ.resolver"(ptr %0, ptr %1)
// WINDOWS: call void @"??BConvertTo@@QEBA?AUS@@XZ.arch_ivybridge"
// WINDOWS: call void @"??BConvertTo@@QEBA?AUS@@XZ"

// ITANIUM: define weak_odr ptr @_ZN1S3fooEi.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_ZN1S3fooEi.arch_sandybridge
// ITANIUM: ret ptr @_ZN1S3fooEi.arch_ivybridge
// ITANIUM: ret ptr @_ZN1S3fooEi.sse4.2
// ITANIUM: ret ptr @_ZN1S3fooEi

// WINDOWS: define weak_odr dso_local i32 @"?foo@S@@QEAAHH@Z.resolver"(ptr %0, i32 %1)
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.arch_sandybridge"
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.arch_ivybridge"
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.sse4.2"
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z"

// ITANIUM: define{{.*}} i32 @_Z4bar2v()
// ITANIUM: call noundef i32 @_ZN2S23fooEi.ifunc

// WINDOWS: define dso_local noundef i32 @"?bar2@@YAHXZ"()
// WINDOWS: call noundef i32 @"?foo@S2@@QEAAHH@Z.resolver"

// ITANIUM: define weak_odr ptr @_ZN2S23fooEi.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_ZN2S23fooEi.arch_sandybridge
// ITANIUM: ret ptr @_ZN2S23fooEi.arch_ivybridge
// ITANIUM: ret ptr @_ZN2S23fooEi.sse4.2
// ITANIUM: ret ptr @_ZN2S23fooEi

// WINDOWS: define weak_odr dso_local i32 @"?foo@S2@@QEAAHH@Z.resolver"(ptr %0, i32 %1)
// WINDOWS: call i32 @"?foo@S2@@QEAAHH@Z.arch_sandybridge"
// WINDOWS: call i32 @"?foo@S2@@QEAAHH@Z.arch_ivybridge"
// WINDOWS: call i32 @"?foo@S2@@QEAAHH@Z.sse4.2"
// WINDOWS: call i32 @"?foo@S2@@QEAAHH@Z"

// ITANIUM: define{{.*}} i32 @_ZN2S23fooEi.sse4.2(ptr {{[^,]*}} %this, i32 noundef %0)
// ITANIUM: define{{.*}} i32 @_ZN2S23fooEi.arch_ivybridge(ptr {{[^,]*}} %this, i32 noundef %0)
// ITANIUM: define{{.*}} i32 @_ZN2S23fooEi(ptr {{[^,]*}} %this, i32 noundef %0)

// WINDOWS: define dso_local noundef i32 @"?foo@S2@@QEAAHH@Z.sse4.2"(ptr {{[^,]*}} %this, i32 noundef %0)
// WINDOWS: define dso_local noundef i32 @"?foo@S2@@QEAAHH@Z.arch_ivybridge"(ptr {{[^,]*}} %this, i32 noundef %0)
// WINDOWS: define dso_local noundef i32 @"?foo@S2@@QEAAHH@Z"(ptr {{[^,]*}} %this, i32 noundef %0)

// ITANIUM: define{{.*}} i32 @_Z9templ_usev()
// ITANIUM: call noundef i32 @_ZN5templIiE3fooEi.ifunc
// ITANIUM: call noundef i32 @_ZN5templIdE3fooEi.ifunc

// WINDOWS: define dso_local noundef i32 @"?templ_use@@YAHXZ"()
// WINDOWS: call noundef i32 @"?foo@?$templ@H@@QEAAHH@Z.resolver"
// WINDOWS: call noundef i32 @"?foo@?$templ@N@@QEAAHH@Z.resolver"

// ITANIUM: define weak_odr ptr @_ZN5templIiE3fooEi.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_ZN5templIiE3fooEi.arch_sandybridge
// ITANIUM: ret ptr @_ZN5templIiE3fooEi.arch_ivybridge
// ITANIUM: ret ptr @_ZN5templIiE3fooEi.sse4.2
// ITANIUM: ret ptr @_ZN5templIiE3fooEi

// WINDOWS: define weak_odr dso_local i32 @"?foo@?$templ@H@@QEAAHH@Z.resolver"(ptr %0, i32 %1)
// WINDOWS: call i32 @"?foo@?$templ@H@@QEAAHH@Z.arch_sandybridge"
// WINDOWS: call i32 @"?foo@?$templ@H@@QEAAHH@Z.arch_ivybridge"
// WINDOWS: call i32 @"?foo@?$templ@H@@QEAAHH@Z.sse4.2"
// WINDOWS: call i32 @"?foo@?$templ@H@@QEAAHH@Z"

// ITANIUM: define weak_odr ptr @_ZN5templIdE3fooEi.resolver()
// LINUX-SAME: comdat
// ITANIUM: ret ptr @_ZN5templIdE3fooEi.arch_sandybridge
// ITANIUM: ret ptr @_ZN5templIdE3fooEi.arch_ivybridge
// ITANIUM: ret ptr @_ZN5templIdE3fooEi.sse4.2
// ITANIUM: ret ptr @_ZN5templIdE3fooEi

// WINDOWS: define weak_odr dso_local i32 @"?foo@?$templ@N@@QEAAHH@Z.resolver"(ptr %0, i32 %1) comdat
// WINDOWS: call i32 @"?foo@?$templ@N@@QEAAHH@Z.arch_sandybridge"
// WINDOWS: call i32 @"?foo@?$templ@N@@QEAAHH@Z.arch_ivybridge"
// WINDOWS: call i32 @"?foo@?$templ@N@@QEAAHH@Z.sse4.2"
// WINDOWS: call i32 @"?foo@?$templ@N@@QEAAHH@Z"

// ITANIUM: define linkonce_odr noundef i32 @_ZN1S3fooEi.sse4.2(ptr {{[^,]*}} %this, i32 noundef %0)
// ITANIUM: ret i32 0

// WINDOWS: define linkonce_odr dso_local noundef i32 @"?foo@S@@QEAAHH@Z.sse4.2"(ptr {{[^,]*}} %this, i32 noundef %0)
// WINDOWS: ret i32 0

// ITANIUM: declare noundef i32 @_ZN1S3fooEi.arch_sandybridge(ptr {{[^,]*}}, i32 noundef)

// WINDOWS: declare dso_local noundef i32 @"?foo@S@@QEAAHH@Z.arch_sandybridge"(ptr {{[^,]*}}, i32 noundef)

// ITANIUM: define linkonce_odr noundef i32 @_ZN1S3fooEi.arch_ivybridge(ptr {{[^,]*}} %this, i32 noundef %0)
// ITANIUM: ret i32 1

// WINDOWS: define linkonce_odr dso_local noundef i32 @"?foo@S@@QEAAHH@Z.arch_ivybridge"(ptr {{[^,]*}} %this, i32 noundef %0)
// WINDOWS: ret i32 1

// ITANIUM: define linkonce_odr noundef i32 @_ZN1S3fooEi(ptr {{[^,]*}} %this, i32 noundef %0)
// ITANIUM: ret i32 2

// WINDOWS: define linkonce_odr dso_local noundef i32 @"?foo@S@@QEAAHH@Z"(ptr {{[^,]*}} %this, i32 noundef %0)
// WINDOWS: ret i32 2

// ITANIUM: define linkonce_odr noundef i32 @_ZN5templIiE3fooEi.sse4.2
// ITANIUM: declare noundef i32 @_ZN5templIiE3fooEi.arch_sandybridge
// ITANIUM: define linkonce_odr noundef i32 @_ZN5templIiE3fooEi.arch_ivybridge
// ITANIUM: define linkonce_odr noundef i32 @_ZN5templIiE3fooEi

// WINDOWS: define linkonce_odr dso_local noundef i32 @"?foo@?$templ@H@@QEAAHH@Z.sse4.2"
// WINDOWS: declare dso_local noundef i32 @"?foo@?$templ@H@@QEAAHH@Z.arch_sandybridge"
// WINDOWS: define linkonce_odr dso_local noundef i32 @"?foo@?$templ@H@@QEAAHH@Z.arch_ivybridge"
// WINDOWS: define linkonce_odr dso_local noundef i32 @"?foo@?$templ@H@@QEAAHH@Z"

// ITANIUM: define linkonce_odr noundef i32 @_ZN5templIdE3fooEi.sse4.2
// ITANIUM: declare noundef i32 @_ZN5templIdE3fooEi.arch_sandybridge
// ITANIUM: define linkonce_odr noundef i32 @_ZN5templIdE3fooEi.arch_ivybridge
// ITANIUM: define linkonce_odr noundef i32 @_ZN5templIdE3fooEi

// WINDOWS: define linkonce_odr dso_local noundef i32 @"?foo@?$templ@N@@QEAAHH@Z.sse4.2"
// WINDOWS: declare dso_local noundef i32 @"?foo@?$templ@N@@QEAAHH@Z.arch_sandybridge"
// WINDOWS: define linkonce_odr dso_local noundef i32 @"?foo@?$templ@N@@QEAAHH@Z.arch_ivybridge"
// WINDOWS: define linkonce_odr dso_local noundef i32 @"?foo@?$templ@N@@QEAAHH@Z"
