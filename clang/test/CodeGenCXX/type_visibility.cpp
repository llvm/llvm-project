// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-apple-darwin10 -emit-llvm -o %t.ll
// RUN: FileCheck %s -check-prefix=FUNS < %t.ll
// RUN: FileCheck %s -check-prefix=VARS < %t.ll
// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-apple-darwin10 -fvisibility=hidden -emit-llvm -o %t.ll
// RUN: FileCheck %s -check-prefix=FUNS-HIDDEN < %t.ll
// RUN: FileCheck %s -check-prefix=VARS-HIDDEN < %t.ll

#define HIDDEN __attribute__((visibility("hidden")))
#define PROTECTED __attribute__((visibility("protected")))
#define DEFAULT __attribute__((visibility("default")))
#define TYPE_HIDDEN __attribute__((type_visibility("hidden")))
#define TYPE_PROTECTED __attribute__((type_visibility("protected")))
#define TYPE_DEFAULT __attribute__((type_visibility("default")))

#if !__has_attribute(type_visibility)
#error No type_visibility attribute!
#endif

// The template tests come first because IR-gen reorders RTTI wierdly.
namespace temp0 {
  struct A;
  template <class T> struct TYPE_DEFAULT B {
    virtual void foo() {}
  };

  template struct B<A>;
  // FUNS-LABEL:        define weak_odr void @_ZN5temp01BINS_1AEE3fooEv(
  // VARS:        @_ZTVN5temp01BINS_1AEEE = weak_odr unnamed_addr constant
  // VARS:        @_ZTSN5temp01BINS_1AEEE = weak_odr constant
  // VARS:        @_ZTIN5temp01BINS_1AEEE = weak_odr constant
  // FUNS-HIDDEN-LABEL: define weak_odr hidden void @_ZN5temp01BINS_1AEE3fooEv(
  // VARS-HIDDEN: @_ZTVN5temp01BINS_1AEEE = weak_odr hidden unnamed_addr constant
  // VARS-HIDDEN: @_ZTSN5temp01BINS_1AEEE = weak_odr hidden constant
  // VARS-HIDDEN: @_ZTIN5temp01BINS_1AEEE = weak_odr hidden constant
}

namespace temp1 {
  struct TYPE_DEFAULT A;
  template <class T> struct TYPE_DEFAULT B {
    virtual void foo() {}
  };

  template struct B<A>;
  // FUNS-LABEL:        define weak_odr void @_ZN5temp11BINS_1AEE3fooEv(
  // VARS:        @_ZTVN5temp11BINS_1AEEE = weak_odr unnamed_addr constant
  // VARS:        @_ZTSN5temp11BINS_1AEEE = weak_odr constant
  // VARS:        @_ZTIN5temp11BINS_1AEEE = weak_odr constant
  // FUNS-HIDDEN-LABEL: define weak_odr hidden void @_ZN5temp11BINS_1AEE3fooEv(
  // VARS-HIDDEN: @_ZTVN5temp11BINS_1AEEE = weak_odr unnamed_addr constant
  // VARS-HIDDEN: @_ZTSN5temp11BINS_1AEEE = weak_odr constant
  // VARS-HIDDEN: @_ZTIN5temp11BINS_1AEEE = weak_odr constant
}

namespace temp2 {
  struct TYPE_DEFAULT A;
  template <class T> struct B {
    virtual void foo() {}
  };

  template struct B<A>;
  // FUNS-LABEL:        define weak_odr void @_ZN5temp21BINS_1AEE3fooEv(
  // VARS:        @_ZTVN5temp21BINS_1AEEE = weak_odr unnamed_addr constant
  // VARS:        @_ZTSN5temp21BINS_1AEEE = weak_odr constant
  // VARS:        @_ZTIN5temp21BINS_1AEEE = weak_odr constant
  // FUNS-HIDDEN-LABEL: define weak_odr hidden void @_ZN5temp21BINS_1AEE3fooEv(
  // VARS-HIDDEN: @_ZTVN5temp21BINS_1AEEE = weak_odr hidden unnamed_addr constant
  // VARS-HIDDEN: @_ZTSN5temp21BINS_1AEEE = weak_odr hidden constant
  // VARS-HIDDEN: @_ZTIN5temp21BINS_1AEEE = weak_odr hidden constant
}

namespace temp3 {
  struct TYPE_HIDDEN A;
  template <class T> struct TYPE_DEFAULT B {
    virtual void foo() {}
  };

  template struct B<A>;
  // FUNS-LABEL:        define weak_odr hidden void @_ZN5temp31BINS_1AEE3fooEv(
  // VARS:        @_ZTVN5temp31BINS_1AEEE = weak_odr hidden unnamed_addr constant
  // VARS:        @_ZTSN5temp31BINS_1AEEE = weak_odr hidden constant
  // VARS:        @_ZTIN5temp31BINS_1AEEE = weak_odr hidden constant
  // FUNS-HIDDEN-LABEL: define weak_odr hidden void @_ZN5temp31BINS_1AEE3fooEv(
  // VARS-HIDDEN: @_ZTVN5temp31BINS_1AEEE = weak_odr hidden unnamed_addr constant
  // VARS-HIDDEN: @_ZTSN5temp31BINS_1AEEE = weak_odr hidden constant
  // VARS-HIDDEN: @_ZTIN5temp31BINS_1AEEE = weak_odr hidden constant
}

namespace temp4 {
  struct TYPE_DEFAULT A;
  template <class T> struct TYPE_HIDDEN B {
    virtual void foo() {}
  };

  template struct B<A>;
  // FUNS-LABEL:        define weak_odr void @_ZN5temp41BINS_1AEE3fooEv(
  // VARS:        @_ZTVN5temp41BINS_1AEEE = weak_odr hidden unnamed_addr constant
  // VARS:        @_ZTSN5temp41BINS_1AEEE = weak_odr hidden constant
  // VARS:        @_ZTIN5temp41BINS_1AEEE = weak_odr hidden constant
  // FUNS-HIDDEN-LABEL: define weak_odr hidden void @_ZN5temp41BINS_1AEE3fooEv(
  // VARS-HIDDEN: @_ZTVN5temp41BINS_1AEEE = weak_odr hidden unnamed_addr constant
  // VARS-HIDDEN: @_ZTSN5temp41BINS_1AEEE = weak_odr hidden constant
  // VARS-HIDDEN: @_ZTIN5temp41BINS_1AEEE = weak_odr hidden constant
}

namespace type0 {
  struct TYPE_DEFAULT A {
    virtual void foo();
  };

  void A::foo() {}
  // FUNS-LABEL:        define void @_ZN5type01A3fooEv(
  // VARS:        @_ZTVN5type01AE = unnamed_addr constant
  // VARS:        @_ZTSN5type01AE = constant
  // VARS:        @_ZTIN5type01AE = constant
  // FUNS-HIDDEN-LABEL: define hidden void @_ZN5type01A3fooEv(
  // VARS-HIDDEN: @_ZTVN5type01AE = unnamed_addr constant
  // VARS-HIDDEN: @_ZTSN5type01AE = constant
  // VARS-HIDDEN: @_ZTIN5type01AE = constant
}

namespace type1 {
  struct HIDDEN TYPE_DEFAULT A {
    virtual void foo();
  };

  void A::foo() {}
  // FUNS-LABEL:        define hidden void @_ZN5type11A3fooEv(
  // VARS:        @_ZTVN5type11AE = unnamed_addr constant
  // VARS:        @_ZTSN5type11AE = constant
  // VARS:        @_ZTIN5type11AE = constant
  // FUNS-HIDDEN-LABEL: define hidden void @_ZN5type11A3fooEv(
  // VARS-HIDDEN: @_ZTVN5type11AE = unnamed_addr constant
  // VARS-HIDDEN: @_ZTSN5type11AE = constant
  // VARS-HIDDEN: @_ZTIN5type11AE = constant
}

namespace type2 {
  struct TYPE_HIDDEN A {
    virtual void foo();
  };

  void A::foo() {}
  // FUNS-LABEL:        define void @_ZN5type21A3fooEv(
  // VARS:        @_ZTVN5type21AE = hidden unnamed_addr constant
  // VARS:        @_ZTSN5type21AE = hidden constant
  // VARS:        @_ZTIN5type21AE = hidden constant
  // FUNS-HIDDEN-LABEL: define hidden void @_ZN5type21A3fooEv(
  // VARS-HIDDEN: @_ZTVN5type21AE = hidden unnamed_addr constant
  // VARS-HIDDEN: @_ZTSN5type21AE = hidden constant
  // VARS-HIDDEN: @_ZTIN5type21AE = hidden constant
}

namespace type3 {
  struct DEFAULT TYPE_HIDDEN A {
    virtual void foo();
  };

  void A::foo() {}
  // FUNS-LABEL:        define void @_ZN5type31A3fooEv(
  // VARS:        @_ZTVN5type31AE = hidden unnamed_addr constant
  // VARS:        @_ZTSN5type31AE = hidden constant
  // VARS:        @_ZTIN5type31AE = hidden constant
  // FUNS-HIDDEN-LABEL: define void @_ZN5type31A3fooEv(
  // VARS-HIDDEN: @_ZTVN5type31AE = hidden unnamed_addr constant
  // VARS-HIDDEN: @_ZTSN5type31AE = hidden constant
  // VARS-HIDDEN: @_ZTIN5type31AE = hidden constant
}

