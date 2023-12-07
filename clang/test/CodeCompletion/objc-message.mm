// Note: the run lines follow their respective tests, since line/column
// matter in this test.

@protocol FooTestProtocol
+ protocolClassMethod;
- protocolInstanceMethod;
@end
@interface Foo <FooTestProtocol> {
  void *isa;
}
+ (int)classMethod1:a withKeyword:b;
+ (void)classMethod2;
+ new;
- instanceMethod1;
@end

@interface Foo (FooTestCategory)
+ categoryClassMethod;
- categoryInstanceMethod;
@end

template<typename T> struct RetainPtr {
  template <typename U> struct RemovePointer { typedef U Type; };
  template <typename U> struct RemovePointer<U*> { typedef U Type; };
    
  typedef typename RemovePointer<T>::Type* PtrType;

  explicit operator PtrType() const;
};

void func(const RetainPtr<Foo>& ptr)
{
  [ptr instanceMethod1];
}

void func(const RetainPtr<id <FooTestProtocol>>& ptr)
{
  [ptr instanceMethod1];
}

// RUN: %clang_cc1 -fsyntax-only -std=c++11 -code-completion-at=%s:%(line-8):8 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: categoryInstanceMethod : [#id#]categoryInstanceMethod
// CHECK-CC1: instanceMethod1 : [#id#]instanceMethod1
// CHECK-CC1: protocolInstanceMethod (InBase) : [#id#]protocolInstanceMethod
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -code-completion-at=%s:%(line-7):8 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: protocolInstanceMethod : [#id#]protocolInstanceMethod
