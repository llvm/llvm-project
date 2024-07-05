// RUN: %clang_cc1 -std=c++11 -triple x86_64 -emit-llvm -o - %s | FileCheck %s

namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation with __size_ replaced by __end_
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    const _E* __end_;

    initializer_list(const _E* __b, const _E* __e)
      : __begin_(__b),
        __end_(__e)
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    initializer_list() : __begin_(nullptr), __end_(nullptr) {}

    size_t    size()  const {return __end_ - __begin_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __end_;}
  };
}

// CHECK: @_ZGR15globalInitList1_ = internal constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK: @globalInitList1 ={{.*}} global {{[^ ]+}} { ptr @_ZGR15globalInitList1_, ptr
std::initializer_list<int> globalInitList1 = {1, 2, 3};

void fn1(int i) {
  // CHECK-LABEL: define{{.*}} void @_Z3fn1i
  // temporary array
  // CHECK: [[array:%[^ ]+]] = alloca [3 x i32]
  // CHECK:      store i32 1, ptr
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: store
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: load
  // CHECK-NEXT: store
  // init the list
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: store ptr
  // CHECK-NEXT: getelementptr
  // CHECK-NEXT: getelementptr inbounds [3 x i32], ptr [[array]], i{{32|64}} 0, i{{32|64}} 3
  // CHECK-NEXT: store ptr
  std::initializer_list<int> intlist{1, 2, i};
}

struct destroyme1 {
  ~destroyme1();
};
struct destroyme2 {
  ~destroyme2();
};


void fn2() {
  // CHECK-LABEL: define{{.*}} void @_Z3fn2v
  void target(std::initializer_list<destroyme1>);
  // objects should be destroyed before dm2, after call returns
  target({ destroyme1(), destroyme1() });
  // CHECK: call void @_ZN10destroyme1D1Ev
  destroyme2 dm2;
  // CHECK: call void @_ZN10destroyme2D1Ev
}

void fn3() {
  // CHECK-LABEL: define{{.*}} void @_Z3fn3v
  // objects should be destroyed after dm2
  auto list = { destroyme1(), destroyme1() };
  destroyme2 dm2;
  // CHECK: call void @_ZN10destroyme2D1Ev
  // CHECK: call void @_ZN10destroyme1D1Ev
}
