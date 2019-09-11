// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LINUX
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

// LINUX: @_ZN1SaSERKS_.ifunc = weak_odr ifunc %struct.S* (%struct.S*, %struct.S*), %struct.S* (%struct.S*, %struct.S*)* ()* @_ZN1SaSERKS_.resolver
// LINUX: @_ZNK9ConvertTocv1SEv.ifunc = weak_odr ifunc void (%struct.ConvertTo*), void (%struct.ConvertTo*)* ()* @_ZNK9ConvertTocv1SEv.resolver
// LINUX: @_ZN1S3fooEi.ifunc = weak_odr ifunc i32 (%struct.S*, i32), i32 (%struct.S*, i32)* ()* @_ZN1S3fooEi.resolver
// LINUX: @_ZN2S23fooEi.ifunc = weak_odr ifunc i32 (%struct.S2*, i32), i32 (%struct.S2*, i32)* ()* @_ZN2S23fooEi.resolver
// Templates:
// LINUX: @_ZN5templIiE3fooEi.ifunc = weak_odr ifunc i32 (%struct.templ*, i32), i32 (%struct.templ*, i32)* ()* @_ZN5templIiE3fooEi.resolver
// LINUX: @_ZN5templIdE3fooEi.ifunc = weak_odr ifunc i32 (%struct.templ.0*, i32), i32 (%struct.templ.0*, i32)* ()* @_ZN5templIdE3fooEi.resolver

// LINUX: define i32 @_Z3barv()
// LINUX: %s = alloca %struct.S, align 1
// LINUX: %s2 = alloca %struct.S, align 1
// LINUX: %C = alloca %struct.ConvertTo, align 1
// LINUX: call dereferenceable(1) %struct.S* @_ZN1SaSERKS_.ifunc(%struct.S* %s2
// LINUX: call void @_ZNK9ConvertTocv1SEv.ifunc(%struct.ConvertTo* %C)
// LINUX: call dereferenceable(1) %struct.S* @_ZN1SaSERKS_.ifunc(%struct.S* %s2
// LINUX: call i32 @_ZN1S3fooEi.ifunc(%struct.S* %s, i32 0)

// WINDOWS: define dso_local i32 @"?bar@@YAHXZ"()
// WINDOWS: %s = alloca %struct.S, align 1
// WINDOWS: %s2 = alloca %struct.S, align 1
// WINDOWS: %C = alloca %struct.ConvertTo, align 1
// WINDOWS: call dereferenceable(1) %struct.S* @"??4S@@QEAAAEAU0@AEBU0@@Z.resolver"(%struct.S* %s2
// WINDOWS: call void @"??BConvertTo@@QEBA?AUS@@XZ.resolver"(%struct.ConvertTo* %C
// WINDOWS: call dereferenceable(1) %struct.S* @"??4S@@QEAAAEAU0@AEBU0@@Z.resolver"(%struct.S* %s2
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.resolver"(%struct.S* %s, i32 0)

// LINUX: define weak_odr %struct.S* (%struct.S*, %struct.S*)* @_ZN1SaSERKS_.resolver() comdat
// LINUX: ret %struct.S* (%struct.S*, %struct.S*)* @_ZN1SaSERKS_.arch_ivybridge
// LINUX: ret %struct.S* (%struct.S*, %struct.S*)* @_ZN1SaSERKS_

// WINDOWS: define weak_odr dso_local %struct.S* @"??4S@@QEAAAEAU0@AEBU0@@Z.resolver"(%struct.S* %0, %struct.S* %1)
// WINDOWS: call %struct.S* @"??4S@@QEAAAEAU0@AEBU0@@Z.arch_ivybridge"
// WINDOWS: call %struct.S* @"??4S@@QEAAAEAU0@AEBU0@@Z"

// LINUX: define weak_odr void (%struct.ConvertTo*)* @_ZNK9ConvertTocv1SEv.resolver() comdat
// LINUX: ret void (%struct.ConvertTo*)* @_ZNK9ConvertTocv1SEv.arch_ivybridge
// LINUX: ret void (%struct.ConvertTo*)* @_ZNK9ConvertTocv1SEv

// WINDOWS: define weak_odr dso_local void @"??BConvertTo@@QEBA?AUS@@XZ.resolver"(%struct.ConvertTo* %0, %struct.S* %1)
// WINDOWS: call void @"??BConvertTo@@QEBA?AUS@@XZ.arch_ivybridge"
// WINDOWS: call void @"??BConvertTo@@QEBA?AUS@@XZ"

// LINUX: define weak_odr i32 (%struct.S*, i32)* @_ZN1S3fooEi.resolver() comdat
// LINUX: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi.arch_sandybridge
// LINUX: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi.arch_ivybridge
// LINUX: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi.sse4.2
// LINUX: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi

// WINDOWS: define weak_odr dso_local i32 @"?foo@S@@QEAAHH@Z.resolver"(%struct.S* %0, i32 %1)
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.arch_sandybridge"
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.arch_ivybridge"
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z.sse4.2"
// WINDOWS: call i32 @"?foo@S@@QEAAHH@Z"

// LINUX: define i32 @_Z4bar2v()
// LINUX: call i32 @_ZN2S23fooEi.ifunc

// WINDOWS: define dso_local i32 @"?bar2@@YAHXZ"()
// WINDOWS: call i32 @"?foo@S2@@QEAAHH@Z.resolver"

// LINUX: define weak_odr i32 (%struct.S2*, i32)* @_ZN2S23fooEi.resolver() comdat
// LINUX: ret i32 (%struct.S2*, i32)* @_ZN2S23fooEi.arch_sandybridge
// LINUX: ret i32 (%struct.S2*, i32)* @_ZN2S23fooEi.arch_ivybridge
// LINUX: ret i32 (%struct.S2*, i32)* @_ZN2S23fooEi.sse4.2
// LINUX: ret i32 (%struct.S2*, i32)* @_ZN2S23fooEi

// WINDOWS: define weak_odr dso_local i32 @"?foo@S2@@QEAAHH@Z.resolver"(%struct.S2* %0, i32 %1)
// WINDOWS: call i32 @"?foo@S2@@QEAAHH@Z.arch_sandybridge"
// WINDOWS: call i32 @"?foo@S2@@QEAAHH@Z.arch_ivybridge"
// WINDOWS: call i32 @"?foo@S2@@QEAAHH@Z.sse4.2"
// WINDOWS: call i32 @"?foo@S2@@QEAAHH@Z"

// LINUX: define i32 @_ZN2S23fooEi.sse4.2(%struct.S2* %this, i32 %0)
// LINUX: define i32 @_ZN2S23fooEi.arch_ivybridge(%struct.S2* %this, i32 %0)
// LINUX: define i32 @_ZN2S23fooEi(%struct.S2* %this, i32 %0)

// WINDOWS: define dso_local i32 @"?foo@S2@@QEAAHH@Z.sse4.2"(%struct.S2* %this, i32 %0)
// WINDOWS: define dso_local i32 @"?foo@S2@@QEAAHH@Z.arch_ivybridge"(%struct.S2* %this, i32 %0)
// WINDOWS: define dso_local i32 @"?foo@S2@@QEAAHH@Z"(%struct.S2* %this, i32 %0)

// LINUX: define i32 @_Z9templ_usev()
// LINUX: call i32 @_ZN5templIiE3fooEi.ifunc
// LINUX: call i32 @_ZN5templIdE3fooEi.ifunc

// WINDOWS: define dso_local i32 @"?templ_use@@YAHXZ"()
// WINDOWS: call i32 @"?foo@?$templ@H@@QEAAHH@Z.resolver"
// WINDOWS: call i32 @"?foo@?$templ@N@@QEAAHH@Z.resolver"

// LINUX: define weak_odr i32 (%struct.templ*, i32)* @_ZN5templIiE3fooEi.resolver() comdat
// LINUX: ret i32 (%struct.templ*, i32)* @_ZN5templIiE3fooEi.arch_sandybridge
// LINUX: ret i32 (%struct.templ*, i32)* @_ZN5templIiE3fooEi.arch_ivybridge
// LINUX: ret i32 (%struct.templ*, i32)* @_ZN5templIiE3fooEi.sse4.2
// LINUX: ret i32 (%struct.templ*, i32)* @_ZN5templIiE3fooEi

// WINDOWS: define weak_odr dso_local i32 @"?foo@?$templ@H@@QEAAHH@Z.resolver"(%struct.templ* %0, i32 %1)
// WINDOWS: call i32 @"?foo@?$templ@H@@QEAAHH@Z.arch_sandybridge"
// WINDOWS: call i32 @"?foo@?$templ@H@@QEAAHH@Z.arch_ivybridge"
// WINDOWS: call i32 @"?foo@?$templ@H@@QEAAHH@Z.sse4.2"
// WINDOWS: call i32 @"?foo@?$templ@H@@QEAAHH@Z"

// LINUX: define weak_odr i32 (%struct.templ.0*, i32)* @_ZN5templIdE3fooEi.resolver() comdat
// LINUX: ret i32 (%struct.templ.0*, i32)* @_ZN5templIdE3fooEi.arch_sandybridge
// LINUX: ret i32 (%struct.templ.0*, i32)* @_ZN5templIdE3fooEi.arch_ivybridge
// LINUX: ret i32 (%struct.templ.0*, i32)* @_ZN5templIdE3fooEi.sse4.2
// LINUX: ret i32 (%struct.templ.0*, i32)* @_ZN5templIdE3fooEi

// WINDOWS: define weak_odr dso_local i32 @"?foo@?$templ@N@@QEAAHH@Z.resolver"(%struct.templ.0* %0, i32 %1) comdat
// WINDOWS: call i32 @"?foo@?$templ@N@@QEAAHH@Z.arch_sandybridge"
// WINDOWS: call i32 @"?foo@?$templ@N@@QEAAHH@Z.arch_ivybridge"
// WINDOWS: call i32 @"?foo@?$templ@N@@QEAAHH@Z.sse4.2"
// WINDOWS: call i32 @"?foo@?$templ@N@@QEAAHH@Z"

// LINUX: define linkonce_odr i32 @_ZN1S3fooEi.sse4.2(%struct.S* %this, i32 %0)
// LINUX: ret i32 0

// WINDOWS: define linkonce_odr dso_local i32 @"?foo@S@@QEAAHH@Z.sse4.2"(%struct.S* %this, i32 %0)
// WINDOWS: ret i32 0

// LINUX: declare i32 @_ZN1S3fooEi.arch_sandybridge(%struct.S*, i32)

// WINDOWS: declare dso_local i32 @"?foo@S@@QEAAHH@Z.arch_sandybridge"(%struct.S*, i32)

// LINUX: define linkonce_odr i32 @_ZN1S3fooEi.arch_ivybridge(%struct.S* %this, i32 %0)
// LINUX: ret i32 1

// WINDOWS: define linkonce_odr dso_local i32 @"?foo@S@@QEAAHH@Z.arch_ivybridge"(%struct.S* %this, i32 %0)
// WINDOWS: ret i32 1

// LINUX: define linkonce_odr i32 @_ZN1S3fooEi(%struct.S* %this, i32 %0)
// LINUX: ret i32 2

// WINDOWS: define linkonce_odr dso_local i32 @"?foo@S@@QEAAHH@Z"(%struct.S* %this, i32 %0)
// WINDOWS: ret i32 2

// LINUX: define linkonce_odr i32 @_ZN5templIiE3fooEi.sse4.2
// LINUX: declare i32 @_ZN5templIiE3fooEi.arch_sandybridge
// LINUX: define linkonce_odr i32 @_ZN5templIiE3fooEi.arch_ivybridge
// LINUX: define linkonce_odr i32 @_ZN5templIiE3fooEi

// WINDOWS: define linkonce_odr dso_local i32 @"?foo@?$templ@H@@QEAAHH@Z.sse4.2"
// WINDOWS: declare dso_local i32 @"?foo@?$templ@H@@QEAAHH@Z.arch_sandybridge"
// WINDOWS: define linkonce_odr dso_local i32 @"?foo@?$templ@H@@QEAAHH@Z.arch_ivybridge"
// WINDOWS: define linkonce_odr dso_local i32 @"?foo@?$templ@H@@QEAAHH@Z"

// LINUX: define linkonce_odr i32 @_ZN5templIdE3fooEi.sse4.2
// LINUX: declare i32 @_ZN5templIdE3fooEi.arch_sandybridge
// LINUX: define linkonce_odr i32 @_ZN5templIdE3fooEi.arch_ivybridge
// LINUX: define linkonce_odr i32 @_ZN5templIdE3fooEi

// WINDOWS: define linkonce_odr dso_local i32 @"?foo@?$templ@N@@QEAAHH@Z.sse4.2"
// WINDOWS: declare dso_local i32 @"?foo@?$templ@N@@QEAAHH@Z.arch_sandybridge"
// WINDOWS: define linkonce_odr dso_local i32 @"?foo@?$templ@N@@QEAAHH@Z.arch_ivybridge"
// WINDOWS: define linkonce_odr dso_local i32 @"?foo@?$templ@N@@QEAAHH@Z"
