// RUN: %check_clang_tidy -check-suffix=0-0 %s modernize-use-auto %t  -- -config="{CheckOptions: [{key: modernize-use-auto.RemoveStars, value: false}, {key: modernize-use-auto.MinTypeNameLength, value: 0}]}" -- -frtti
// RUN: %check_clang_tidy -check-suffix=0-8 %s modernize-use-auto %t  -- -config="{CheckOptions: [{key: modernize-use-auto.RemoveStars, value: false}, {key: modernize-use-auto.MinTypeNameLength, value: 8}]}" -- -frtti
// RUN: %check_clang_tidy -check-suffix=1-0 %s modernize-use-auto %t  -- -config="{CheckOptions: [{key: modernize-use-auto.RemoveStars, value: true}, {key: modernize-use-auto.MinTypeNameLength, value: 0}]}" -- -frtti
// RUN: %check_clang_tidy -check-suffix=1-8 %s modernize-use-auto %t  -- -config="{CheckOptions: [{key: modernize-use-auto.RemoveStars, value: true}, {key: modernize-use-auto.MinTypeNameLength, value: 8}]}" -- -frtti

template <class T> extern T foo();
template <class T> struct P {  explicit P(T t) : t_(t) {}  T t_;};
template <class T> P<T> *foo_ptr();
template <class T> P<T> &foo_ref();

int bar() {
  {
    // Lenth(long) = 4
    long i = static_cast<long>(foo<long>());
    // CHECK-FIXES-0-0: auto i = {{.*}}
    // CHECK-FIXES-0-8: long i = {{.*}}
    // CHECK-FIXES-1-0: auto  i = {{.*}}
    // CHECK-FIXES-1-8: long i = {{.*}}
    const long ci = static_cast<long>(foo<const long>());
    // CHECK-FIXES-0-0: auto ci = {{.*}}
    // CHECK-FIXES-0-8: long ci = {{.*}}
    // CHECK-FIXES-1-0: auto  ci = {{.*}}
    // CHECK-FIXES-1-8: long ci = {{.*}}
    long *pi = static_cast<long *>(foo<long *>());
    // CHECK-FIXES-0-0: auto *pi = {{.*}}
    // CHECK-FIXES-0-8: long *pi = {{.*}}
    // CHECK-FIXES-1-0: auto pi = {{.*}}
    // CHECK-FIXES-1-8: long *pi = {{.*}}

    // Length(long       *) is still 5
    long      *     pi2 = static_cast<long *>(foo<long *>());
    // CHECK-FIXES-0-0: auto      *     pi2 = {{.*}}
    // CHECK-FIXES-0-8: long      *     pi2 = {{.*}}
    // CHECK-FIXES-1-0: auto      pi2 = {{.*}}
    // CHECK-FIXES-1-8: long      *     pi2 = {{.*}}

    // Length(long **) = 6
    long **ppi = static_cast<long **>(foo<long **>());
    // CHECK-FIXES-0-0: auto **ppi = {{.*}}
    // CHECK-FIXES-0-8: long **ppi = {{.*}}
    // CHECK-FIXES-1-0: auto ppi = {{.*}}
    // CHECK-FIXES-1-8: long **ppi = {{.*}}
  }

  {
    // Lenth(long int) = 4 + 1 + 3 = 8
    // Lenth(long        int) is still 8
    long int i = static_cast<long int>(foo<long int>());
    // CHECK-FIXES-0-0: auto i = {{.*}}
    // CHECK-FIXES-0-8: auto i = {{.*}}
    // CHECK-FIXES-1-0: auto  i = {{.*}}
    // CHECK-FIXES-1-8: auto  i = {{.*}}

    long int *pi = static_cast<long int *>(foo<long int *>());
    // CHECK-FIXES-0-0: auto *pi = {{.*}}
    // CHECK-FIXES-0-8: auto *pi = {{.*}}
    // CHECK-FIXES-1-0: auto pi = {{.*}}
    // CHECK-FIXES-1-8: auto pi = {{.*}}
  }

  // Templates
  {
    // Length(P<long>) = 7
    P<long>& i = static_cast<P<long>&>(foo_ref<long>());
    // CHECK-FIXES-0-0: auto& i = {{.*}}
    // CHECK-FIXES-0-8: P<long>& i = {{.*}}
    // CHECK-FIXES-1-0: auto & i = {{.*}}
    // CHECK-FIXES-1-8: P<long>& i = {{.*}}

    // Length(P<long*>) = 8
    P<long*>& pi = static_cast<P<long*> &>(foo_ref<long*>());
    // CHECK-FIXES-0-0: auto& pi = {{.*}}
    // CHECK-FIXES-0-8: auto& pi = {{.*}}
    // CHECK-FIXES-1-0: auto & pi = {{.*}}
    // CHECK-FIXES-1-8: auto & pi = {{.*}}

    P<long>* pi2 = static_cast<P<long>*>(foo_ptr<long>());
    // CHECK-FIXES-0-0: auto* pi2 = {{.*}}
    // CHECK-FIXES-0-8: P<long>* pi2 = {{.*}}
    // CHECK-FIXES-1-0: auto  pi2 = {{.*}}
    // CHECK-FIXES-1-8: auto  pi2 = {{.*}}
  }

  return 1;
}
