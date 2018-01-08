// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

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

// CHECK: @_ZN1SaSERKS_.ifunc = ifunc %struct.S* (%struct.S*, %struct.S*), %struct.S* (%struct.S*, %struct.S*)* ()* @_ZN1SaSERKS_.resolver
// CHECK: @_ZNK9ConvertTocv1SEv.ifunc = ifunc void (%struct.ConvertTo*), void (%struct.ConvertTo*)* ()* @_ZNK9ConvertTocv1SEv.resolver
// CHECK: @_ZN1S3fooEi.ifunc = ifunc i32 (%struct.S*, i32), i32 (%struct.S*, i32)* ()* @_ZN1S3fooEi.resolver
// CHECK: @_ZN2S23fooEi.ifunc = ifunc i32 (%struct.S2*, i32), i32 (%struct.S2*, i32)* ()* @_ZN2S23fooEi.resolver
// Templates:
// CHECK: @_ZN5templIiE3fooEi.ifunc = ifunc i32 (%struct.templ*, i32), i32 (%struct.templ*, i32)* ()* @_ZN5templIiE3fooEi.resolver
// CHECK: @_ZN5templIdE3fooEi.ifunc = ifunc i32 (%struct.templ.0*, i32), i32 (%struct.templ.0*, i32)* ()* @_ZN5templIdE3fooEi.resolver

// CHECK: define i32 @_Z3barv()
// CHECK: %s = alloca %struct.S, align 1
// CHECK: %s2 = alloca %struct.S, align 1
// CHECK: %C = alloca %struct.ConvertTo, align 1
// CHECK: call dereferenceable(1) %struct.S* @_ZN1SaSERKS_.ifunc(%struct.S* %s2
// CHECK: call void @_ZNK9ConvertTocv1SEv.ifunc(%struct.ConvertTo* %C)
// CHECK: call dereferenceable(1) %struct.S* @_ZN1SaSERKS_.ifunc(%struct.S* %s2
// CHECK: call i32 @_ZN1S3fooEi.ifunc(%struct.S* %s, i32 0)

// CHECK: define %struct.S* (%struct.S*, %struct.S*)* @_ZN1SaSERKS_.resolver()
// CHECK: ret %struct.S* (%struct.S*, %struct.S*)* @_ZN1SaSERKS_.arch_ivybridge
// CHECK: ret %struct.S* (%struct.S*, %struct.S*)* @_ZN1SaSERKS_

// CHECK: define void (%struct.ConvertTo*)* @_ZNK9ConvertTocv1SEv.resolver()
// CHECK: ret void (%struct.ConvertTo*)* @_ZNK9ConvertTocv1SEv.arch_ivybridge
// CHECK: ret void (%struct.ConvertTo*)* @_ZNK9ConvertTocv1SEv

// CHECK: define i32 (%struct.S*, i32)* @_ZN1S3fooEi.resolver()
// CHECK: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi.arch_sandybridge
// CHECK: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi.arch_ivybridge
// CHECK: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi.sse4.2
// CHECK: ret i32 (%struct.S*, i32)* @_ZN1S3fooEi

// CHECK: define i32 @_Z4bar2v()
// CHECK:call i32 @_ZN2S23fooEi.ifunc
// define i32 (%struct.S2*, i32)* @_ZN2S23fooEi.resolver()
// CHECK: ret i32 (%struct.S2*, i32)* @_ZN2S23fooEi.arch_sandybridge
// CHECK: ret i32 (%struct.S2*, i32)* @_ZN2S23fooEi.arch_ivybridge
// CHECK: ret i32 (%struct.S2*, i32)* @_ZN2S23fooEi.sse4.2
// CHECK: ret i32 (%struct.S2*, i32)* @_ZN2S23fooEi

// CHECK: define i32 @_ZN2S23fooEi.sse4.2(%struct.S2* %this, i32)
// CHECK: define i32 @_ZN2S23fooEi.arch_ivybridge(%struct.S2* %this, i32)
// CHECK: define i32 @_ZN2S23fooEi(%struct.S2* %this, i32)

// CHECK: define i32 @_Z9templ_usev()
// CHECK:  call i32 @_ZN5templIiE3fooEi.ifunc
// CHECK:  call i32 @_ZN5templIdE3fooEi.ifunc


// CHECK: define i32 (%struct.templ*, i32)* @_ZN5templIiE3fooEi.resolver()
// CHECK: ret i32 (%struct.templ*, i32)* @_ZN5templIiE3fooEi.arch_sandybridge
// CHECK: ret i32 (%struct.templ*, i32)* @_ZN5templIiE3fooEi.arch_ivybridge
// CHECK: ret i32 (%struct.templ*, i32)* @_ZN5templIiE3fooEi.sse4.2
// CHECK: ret i32 (%struct.templ*, i32)* @_ZN5templIiE3fooEi
//
// CHECK: define i32 (%struct.templ.0*, i32)* @_ZN5templIdE3fooEi.resolver()
// CHECK: ret i32 (%struct.templ.0*, i32)* @_ZN5templIdE3fooEi.arch_sandybridge
// CHECK: ret i32 (%struct.templ.0*, i32)* @_ZN5templIdE3fooEi.arch_ivybridge
// CHECK: ret i32 (%struct.templ.0*, i32)* @_ZN5templIdE3fooEi.sse4.2
// CHECK: ret i32 (%struct.templ.0*, i32)* @_ZN5templIdE3fooEi

// CHECK: define linkonce_odr i32 @_ZN1S3fooEi.sse4.2(%struct.S* %this, i32)
// CHECK: ret i32 0

// CHECK: declare i32 @_ZN1S3fooEi.arch_sandybridge(%struct.S*, i32)

// CHECK: define linkonce_odr i32 @_ZN1S3fooEi.arch_ivybridge(%struct.S* %this, i32)
// CHECK: ret i32 1

// CHECK: define linkonce_odr i32 @_ZN1S3fooEi(%struct.S* %this, i32)
// CHECK: ret i32 2

