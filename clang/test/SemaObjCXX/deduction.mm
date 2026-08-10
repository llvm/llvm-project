// RUN: %clang_cc1 -fsyntax-only -verify %s

@class NSString;

// Reduced from WebKit.
namespace test0 {
  template <typename T> struct RemovePointer {
    typedef T Type;
  };

  template <typename T> struct RemovePointer<T*> {
    typedef T Type;
  };

  template <typename T> struct RetainPtr {
    typedef typename RemovePointer<T>::Type ValueType;
    typedef ValueType* PtrType;
    RetainPtr(PtrType ptr);
  };
 
  void test(NSString *S) {
    RetainPtr<NSString*> ptr(S);
  }

  void test(id S) {
    RetainPtr<id> ptr(S);
  }
}

@class Test1Class;
@protocol Test1Protocol;
namespace test1 {
  template <typename T> struct RemovePointer {
    typedef T type;
  };
  template <typename T> struct RemovePointer<T*> {
    typedef T type;
  };
  template <typename A, typename B> struct is_same {};
  template <typename A> struct is_same<A,A> {
    static void foo();
  };
  template <typename T> struct tester {
    void test() {
      is_same<T, typename RemovePointer<T>::type*>::foo(); // expected-error 2 {{no member named 'foo'}}
    }
  };

  template struct tester<id>;
  template struct tester<id<Test1Protocol> >;
  template struct tester<Class>;
  template struct tester<Class<Test1Protocol> >;
  template struct tester<Test1Class*>;
  template struct tester<Test1Class<Test1Protocol>*>;

  template struct tester<Test1Class>; // expected-note {{in instantiation}}
  template struct tester<Test1Class<Test1Protocol> >; // expected-note {{in instantiation}}
}

namespace test2 {
  template <typename T> void foo(const T* t) {}
  void test(id x) {
    foo(x);
  }
}
