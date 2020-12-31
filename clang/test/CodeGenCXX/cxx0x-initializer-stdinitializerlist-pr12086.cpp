// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-STATIC-BL
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -emit-llvm -o - %s -Dconstexpr= | FileCheck %s --check-prefix=CHECK-DYNAMIC-BL
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -emit-llvm -o - %s -DUSE_END | FileCheck %s --check-prefix=CHECK-STATIC-BE
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++11 -emit-llvm -o - %s -DUSE_END -Dconstexpr= | FileCheck %s --check-prefix=CHECK-DYNAMIC-BE

namespace std {
  typedef decltype(sizeof(int)) size_t;

  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
#ifdef USE_END
    const _E* __end_;
#else
    size_t    __size_;
#endif

    constexpr initializer_list(const _E* __b, size_t __s)
      : __begin_(__b),
#ifdef USE_END
        __end_(__b + __s)
#else
        __size_(__s)
#endif
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

#ifdef USE_END
    constexpr initializer_list() : __begin_(nullptr), __end_(nullptr) {}

    size_t    size()  const {return __end_ - __begin_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __end_;}
#else
    constexpr initializer_list() : __begin_(nullptr), __size_(0) {}

    size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
#endif
  };
}

constexpr int a = 2, b = 4, c = 6;
std::initializer_list<std::initializer_list<int>> nested = {
  {1, a}, {3, b}, {5, c}
};

// CHECK-STATIC-BL: @_ZGR6nested0_ = internal constant [2 x i32] [i32 1, i32 2], align 4
// CHECK-STATIC-BL: @_ZGR6nested1_ = internal constant [2 x i32] [i32 3, i32 4], align 4
// CHECK-STATIC-BL: @_ZGR6nested2_ = internal constant [2 x i32] [i32 5, i32 6], align 4
// CHECK-STATIC-BL: @_ZGR6nested_ = internal constant [3 x {{.*}}] [
// CHECK-STATIC-BL:   {{.*}} { i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested0_, i32 0, i32 0), i64 2 },
// CHECK-STATIC-BL:   {{.*}} { i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested1_, i32 0, i32 0), i64 2 },
// CHECK-STATIC-BL:   {{.*}} { i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested2_, i32 0, i32 0), i64 2 }
// CHECK-STATIC-BL: ], align 8
// CHECK-STATIC-BL: @nested ={{.*}} global {{.*}} { {{.*}} getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i32 0, i32 0), i64 3 }, align 8

// CHECK-DYNAMIC-BL: @nested ={{.*}} global
// CHECK-DYNAMIC-BL: @_ZGR6nested_ = internal global [3 x
// CHECK-DYNAMIC-BL: @_ZGR6nested0_ = internal global [2 x i32] zeroinitializer
// CHECK-DYNAMIC-BL: @_ZGR6nested1_ = internal global [2 x i32] zeroinitializer
// CHECK-DYNAMIC-BL: @_ZGR6nested2_ = internal global [2 x i32] zeroinitializer
// CHECK-DYNAMIC-BL: store i32 1, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested0_, i64 0, i64 0)
// CHECK-DYNAMIC-BL: store i32 {{.*}}, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested0_, i64 0, i64 1)
// CHECK-DYNAMIC-BL: store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested0_, i64 0, i64 0),
// CHECK-DYNAMIC-BL:       i32** getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 0, i32 0), align 8
// CHECK-DYNAMIC-BL: store i64 2, i64* getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 0, i32 1), align 8
// CHECK-DYNAMIC-BL: store i32 3, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested1_, i64 0, i64 0)
// CHECK-DYNAMIC-BL: store i32 {{.*}}, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested1_, i64 0, i64 1)
// CHECK-DYNAMIC-BL: store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested1_, i64 0, i64 0),
// CHECK-DYNAMIC-BL:       i32** getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 1, i32 0), align 8
// CHECK-DYNAMIC-BL: store i64 2, i64* getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 1, i32 1), align 8
// CHECK-DYNAMIC-BL: store i32 5, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested2_, i64 0, i64 0)
// CHECK-DYNAMIC-BL: store i32 {{.*}}, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested2_, i64 0, i64 1)
// CHECK-DYNAMIC-BL: store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested2_, i64 0, i64 0),
// CHECK-DYNAMIC-BL:       i32** getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 2, i32 0), align 8
// CHECK-DYNAMIC-BL: store i64 2, i64* getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 2, i32 1), align 8
// CHECK-DYNAMIC-BL: store {{.*}}* getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 0),
// CHECK-DYNAMIC-BL:       {{.*}}** getelementptr inbounds ({{.*}}, {{.*}}* @nested, i32 0, i32 0), align 8
// CHECK-DYNAMIC-BL: store i64 3, i64* getelementptr inbounds ({{.*}}, {{.*}}* @nested, i32 0, i32 1), align 8

// CHECK-STATIC-BE: @_ZGR6nested0_ = internal constant [2 x i32] [i32 1, i32 2], align 4
// CHECK-STATIC-BE: @_ZGR6nested1_ = internal constant [2 x i32] [i32 3, i32 4], align 4
// CHECK-STATIC-BE: @_ZGR6nested2_ = internal constant [2 x i32] [i32 5, i32 6], align 4
// CHECK-STATIC-BE: @_ZGR6nested_ = internal constant [3 x {{.*}}] [
// CHECK-STATIC-BE:   {{.*}} { i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested0_, i32 0, i32 0),
// CHECK-STATIC-BE:            i32* bitcast (i8* getelementptr (i8, i8* bitcast ([2 x i32]* @_ZGR6nested0_ to i8*), i64 8) to i32*) }
// CHECK-STATIC-BE:   {{.*}} { i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested1_, i32 0, i32 0),
// CHECK-STATIC-BE:            i32* bitcast (i8* getelementptr (i8, i8* bitcast ([2 x i32]* @_ZGR6nested1_ to i8*), i64 8) to i32*) }
// CHECK-STATIC-BE:   {{.*}} { i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested2_, i32 0, i32 0),
// CHECK-STATIC-BE:            i32* bitcast (i8* getelementptr (i8, i8* bitcast ([2 x i32]* @_ZGR6nested2_ to i8*), i64 8) to i32*) }
// CHECK-STATIC-BE: ], align 8
// CHECK-STATIC-BE: @nested ={{.*}} global {{.*}} { {{.*}} getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i32 0, i32 0),
// CHECK-STATIC-BE:                           {{.*}} bitcast ({{.*}}* getelementptr (i8, i8* bitcast ([3 x {{.*}}]* @_ZGR6nested_ to i8*), i64 48) to {{.*}}*) }

// CHECK-DYNAMIC-BE: @nested ={{.*}} global
// CHECK-DYNAMIC-BE: @_ZGR6nested_ = internal global [3 x
// CHECK-DYNAMIC-BE: @_ZGR6nested0_ = internal global [2 x i32] zeroinitializer
// CHECK-DYNAMIC-BE: @_ZGR6nested1_ = internal global [2 x i32] zeroinitializer
// CHECK-DYNAMIC-BE: @_ZGR6nested2_ = internal global [2 x i32] zeroinitializer
// CHECK-DYNAMIC-BE: store i32 1, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested0_, i64 0, i64 0)
// CHECK-DYNAMIC-BE: store i32 {{.*}}, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested0_, i64 0, i64 1)
// CHECK-DYNAMIC-BE: store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested0_, i64 0, i64 0),
// CHECK-DYNAMIC-BE:       i32** getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 0, i32 0), align 8
// CHECK-DYNAMIC-BE: store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested0_, i64 1, i64 0),
// CHECK-DYNAMIC-BE:       i32** getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 0, i32 1), align 8
// CHECK-DYNAMIC-BE: store i32 3, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested1_, i64 0, i64 0)
// CHECK-DYNAMIC-BE: store i32 {{.*}}, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested1_, i64 0, i64 1)
// CHECK-DYNAMIC-BE: store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested1_, i64 0, i64 0),
// CHECK-DYNAMIC-BE:       i32** getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 1, i32 0), align 8
// CHECK-DYNAMIC-BE: store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested1_, i64 1, i64 0),
// CHECK-DYNAMIC-BE:       i32** getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 1, i32 1), align 8
// CHECK-DYNAMIC-BE: store i32 5, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested2_, i64 0, i64 0)
// CHECK-DYNAMIC-BE: store i32 {{.*}}, i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested2_, i64 0, i64 1)
// CHECK-DYNAMIC-BE: store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested2_, i64 0, i64 0),
// CHECK-DYNAMIC-BE:       i32** getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 2, i32 0), align 8
// CHECK-DYNAMIC-BE: store i32* getelementptr inbounds ([2 x i32], [2 x i32]* @_ZGR6nested2_, i64 1, i64 0),
// CHECK-DYNAMIC-BE:       i32** getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 2, i32 1), align 8
// CHECK-DYNAMIC-BE: store {{.*}}* getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 0, i64 0),
// CHECK-DYNAMIC-BE:       {{.*}}** getelementptr inbounds ({{.*}}, {{.*}}* @nested, i32 0, i32 0), align 8
// CHECK-DYNAMIC-BE: store {{.*}}* getelementptr inbounds ([3 x {{.*}}], [3 x {{.*}}]* @_ZGR6nested_, i64 1, i64 0),
// CHECK-DYNAMIC-BE:       {{.*}}** getelementptr inbounds ({{.*}}, {{.*}}* @nested, i32 0, i32 1), align 8
