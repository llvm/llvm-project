// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

template <typename T, __SIZE_TYPE__ size>
struct Array {
  T data[size]{};
  constexpr const T* begin() const { return data; }
  constexpr const T* end() const { return data + size; }
};

int f1() {
  static constexpr Array<int, 3> integers{1, 2, 3};
  int sum = 0;
  template for (auto x : integers) sum += x;
  return sum;
}

int f2() {
  static constexpr Array<int, 3> integers{1, 2, 3};
  int sum = 0;
  template for (constexpr auto x : integers) sum += x;
  return sum;
}

int f3() {
  static constexpr Array<int, 0> integers{};
  int sum = 0;
  template for (constexpr auto x : integers) {
    static_assert(false, "not expanded");
    sum += x;
  }
  return sum;
}

int f4() {
  static constexpr Array<int, 2> a{1, 2};
  static constexpr Array<int, 2> b{3, 4};
  int sum = 0;

  template for (auto x : a)
    template for (auto y : b)
      sum += x + y;

  template for (constexpr auto x : a)
    template for (constexpr auto y : b)
      sum += x + y;

  return sum;
}

struct Private {
  static constexpr Array<int, 3> integers{1, 2, 3};
  friend constexpr int friend_func();

private:
  constexpr const int* begin() const { return integers.begin(); }
  constexpr const int* end() const { return integers.end(); }

public:
  static int member_func();
};

int Private::member_func() {
  int sum = 0;
  static constexpr Private p1;
  template for (auto x : p1) sum += x;
  return sum;
}

struct CustomIterator {
  struct iterator {
    int n;

    constexpr iterator operator+(int m) const {
      return {n + m};
    }

    constexpr int operator*() const {
      return n;
    }

    // FIXME: Should be '!=' once we support that properly.
    friend constexpr __PTRDIFF_TYPE__ operator-(iterator a, iterator b) {
      return a.n - b.n;
    }
  };

   constexpr iterator begin() const { return iterator(1); }
   constexpr iterator end() const { return iterator(5); }
};

int custom_iterator() {
  static constexpr CustomIterator c;
  int sum = 0;
  template for (auto x : c) sum += x;
  template for (constexpr auto x : c) sum += x;
  return sum;
}

// CHECK: @_ZZ2f1vE8integers = internal constant %struct.Array { [3 x i32] [i32 1, i32 2, i32 3] }, align 4
// CHECK: @_ZZ2f2vE8integers = internal constant %struct.Array { [3 x i32] [i32 1, i32 2, i32 3] }, align 4
// CHECK: @_ZZ2f3vE8integers = internal constant %struct.Array.0 zeroinitializer, align 4
// CHECK: @_ZZ2f4vE1a = internal constant %struct.Array.1 { [2 x i32] [i32 1, i32 2] }, align 4
// CHECK: @_ZZ2f4vE1b = internal constant %struct.Array.1 { [2 x i32] [i32 3, i32 4] }, align 4
// CHECK: @_ZZN7Private11member_funcEvE2p1 = internal constant %struct.Private zeroinitializer, align 1
// CHECK: @_ZN7Private8integersE = {{.*}} constant %struct.Array { [3 x i32] [i32 1, i32 2, i32 3] }, comdat, align 4
// CHECK: @_ZZ15custom_iteratorvE1c = internal constant %struct.CustomIterator zeroinitializer, align 1
// CHECK: @__const._Z15custom_iteratorv.__begin1 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 1 }, align 4
// CHECK: @__const._Z15custom_iteratorv.__end1 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 5 }, align 4
// CHECK: @__const._Z15custom_iteratorv.__begin1.1 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 1 }, align 4
// CHECK: @__const._Z15custom_iteratorv.__end1.2 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 5 }, align 4

// CHECK-LABEL: define {{.*}} i32 @_Z2f1v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK-NEXT:   %__begin1 = alloca ptr, align 8
// CHECK-NEXT:   %__end1 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZ2f1vE8integers, ptr %__range1, align 8
// CHECK-NEXT:   store ptr @_ZZ2f1vE8integers, ptr %__begin1, align 8
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f1vE8integers, i64 12), ptr %__end1, align 8
// CHECK-NEXT:   %0 = load i32, ptr @_ZZ2f1vE8integers, align 4
// CHECK-NEXT:   store i32 %0, ptr %x, align 4
// CHECK-NEXT:   %1 = load i32, ptr %x, align 4
// CHECK-NEXT:   %2 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %2, %1
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %3 = load i32, ptr getelementptr inbounds (i32, ptr @_ZZ2f1vE8integers, i64 1), align 4
// CHECK-NEXT:   store i32 %3, ptr %x1, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x1, align 4
// CHECK-NEXT:   %5 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add2 = add nsw i32 %5, %4
// CHECK-NEXT:   store i32 %add2, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   %6 = load i32, ptr getelementptr inbounds (i32, ptr @_ZZ2f1vE8integers, i64 2), align 4
// CHECK-NEXT:   store i32 %6, ptr %x4, align 4
// CHECK-NEXT:   %7 = load i32, ptr %x4, align 4
// CHECK-NEXT:   %8 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add5 = add nsw i32 %8, %7
// CHECK-NEXT:   store i32 %add5, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   %9 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %9


// CHECK-LABEL: define {{.*}} i32 @_Z2f2v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK-NEXT:   %__begin1 = alloca ptr, align 8
// CHECK-NEXT:   %__end1 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZ2f2vE8integers, ptr %__range1, align 8
// CHECK-NEXT:   store ptr @_ZZ2f2vE8integers, ptr %__begin1, align 8
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f2vE8integers, i64 12), ptr %__end1, align 8
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   %0 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %0, 1
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store i32 2, ptr %x1, align 4
// CHECK-NEXT:   %1 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add2 = add nsw i32 %1, 2
// CHECK-NEXT:   store i32 %add2, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   store i32 3, ptr %x4, align 4
// CHECK-NEXT:   %2 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add5 = add nsw i32 %2, 3
// CHECK-NEXT:   store i32 %add5, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   %3 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %3


// CHECK-LABEL: define {{.*}} i32 @_Z2f3v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK-NEXT:   %__begin1 = alloca ptr, align 8
// CHECK-NEXT:   %__end1 = alloca ptr, align 8
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZ2f3vE8integers, ptr %__range1, align 8
// CHECK-NEXT:   store ptr @_ZZ2f3vE8integers, ptr %__begin1, align 8
// CHECK-NEXT:   store ptr @_ZZ2f3vE8integers, ptr %__end1, align 8
// CHECK-NEXT:   %0 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %0


// CHECK-LABEL: define {{.*}} i32 @_Z2f4v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK-NEXT:   %__begin1 = alloca ptr, align 8
// CHECK-NEXT:   %__end1 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %__range2 = alloca ptr, align 8
// CHECK-NEXT:   %__begin2 = alloca ptr, align 8
// CHECK-NEXT:   %__end2 = alloca ptr, align 8
// CHECK-NEXT:   %y = alloca i32, align 4
// CHECK-NEXT:   %y2 = alloca i32, align 4
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   %__range27 = alloca ptr, align 8
// CHECK-NEXT:   %__begin28 = alloca ptr, align 8
// CHECK-NEXT:   %__end29 = alloca ptr, align 8
// CHECK-NEXT:   %y10 = alloca i32, align 4
// CHECK-NEXT:   %y14 = alloca i32, align 4
// CHECK-NEXT:   %__range119 = alloca ptr, align 8
// CHECK-NEXT:   %__begin120 = alloca ptr, align 8
// CHECK-NEXT:   %__end121 = alloca ptr, align 8
// CHECK-NEXT:   %x22 = alloca i32, align 4
// CHECK-NEXT:   %__range223 = alloca ptr, align 8
// CHECK-NEXT:   %__begin224 = alloca ptr, align 8
// CHECK-NEXT:   %__end225 = alloca ptr, align 8
// CHECK-NEXT:   %y26 = alloca i32, align 4
// CHECK-NEXT:   %y29 = alloca i32, align 4
// CHECK-NEXT:   %x33 = alloca i32, align 4
// CHECK-NEXT:   %__range234 = alloca ptr, align 8
// CHECK-NEXT:   %__begin235 = alloca ptr, align 8
// CHECK-NEXT:   %__end236 = alloca ptr, align 8
// CHECK-NEXT:   %y37 = alloca i32, align 4
// CHECK-NEXT:   %y40 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1a, ptr %__range1, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1a, ptr %__begin1, align 8
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f4vE1a, i64 8), ptr %__end1, align 8
// CHECK-NEXT:   %0 = load i32, ptr @_ZZ2f4vE1a, align 4
// CHECK-NEXT:   store i32 %0, ptr %x, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__range2, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__begin2, align 8
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f4vE1b, i64 8), ptr %__end2, align 8
// CHECK-NEXT:   %1 = load i32, ptr @_ZZ2f4vE1b, align 4
// CHECK-NEXT:   store i32 %1, ptr %y, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x, align 4
// CHECK-NEXT:   %3 = load i32, ptr %y, align 4
// CHECK-NEXT:   %add = add nsw i32 %2, %3
// CHECK-NEXT:   %4 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add1 = add nsw i32 %4, %add
// CHECK-NEXT:   store i32 %add1, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %5 = load i32, ptr getelementptr inbounds (i32, ptr @_ZZ2f4vE1b, i64 1), align 4
// CHECK-NEXT:   store i32 %5, ptr %y2, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x, align 4
// CHECK-NEXT:   %7 = load i32, ptr %y2, align 4
// CHECK-NEXT:   %add3 = add nsw i32 %6, %7
// CHECK-NEXT:   %8 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add4 = add nsw i32 %8, %add3
// CHECK-NEXT:   store i32 %add4, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   br label %expand.next5
// CHECK: expand.next5:
// CHECK-NEXT:   %9 = load i32, ptr getelementptr inbounds (i32, ptr @_ZZ2f4vE1a, i64 1), align 4
// CHECK-NEXT:   store i32 %9, ptr %x6, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__range27, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__begin28, align 8
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f4vE1b, i64 8), ptr %__end29, align 8
// CHECK-NEXT:   %10 = load i32, ptr @_ZZ2f4vE1b, align 4
// CHECK-NEXT:   store i32 %10, ptr %y10, align 4
// CHECK-NEXT:   %11 = load i32, ptr %x6, align 4
// CHECK-NEXT:   %12 = load i32, ptr %y10, align 4
// CHECK-NEXT:   %add11 = add nsw i32 %11, %12
// CHECK-NEXT:   %13 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add12 = add nsw i32 %13, %add11
// CHECK-NEXT:   store i32 %add12, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next13
// CHECK: expand.next13:
// CHECK-NEXT:   %14 = load i32, ptr getelementptr inbounds (i32, ptr @_ZZ2f4vE1b, i64 1), align 4
// CHECK-NEXT:   store i32 %14, ptr %y14, align 4
// CHECK-NEXT:   %15 = load i32, ptr %x6, align 4
// CHECK-NEXT:   %16 = load i32, ptr %y14, align 4
// CHECK-NEXT:   %add15 = add nsw i32 %15, %16
// CHECK-NEXT:   %17 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add16 = add nsw i32 %17, %add15
// CHECK-NEXT:   store i32 %add16, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end17
// CHECK: expand.end17:
// CHECK-NEXT:   br label %expand.end18
// CHECK: expand.end18:
// CHECK-NEXT:   store ptr @_ZZ2f4vE1a, ptr %__range119, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1a, ptr %__begin120, align 8
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f4vE1a, i64 8), ptr %__end121, align 8
// CHECK-NEXT:   store i32 1, ptr %x22, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__range223, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__begin224, align 8
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f4vE1b, i64 8), ptr %__end225, align 8
// CHECK-NEXT:   store i32 3, ptr %y26, align 4
// CHECK-NEXT:   %18 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add27 = add nsw i32 %18, 4
// CHECK-NEXT:   store i32 %add27, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next28
// CHECK: expand.next28:
// CHECK-NEXT:   store i32 4, ptr %y29, align 4
// CHECK-NEXT:   %19 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add30 = add nsw i32 %19, 5
// CHECK-NEXT:   store i32 %add30, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end31
// CHECK: expand.end31:
// CHECK-NEXT:   br label %expand.next32
// CHECK: expand.next32:
// CHECK-NEXT:   store i32 2, ptr %x33, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__range234, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__begin235, align 8
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f4vE1b, i64 8), ptr %__end236, align 8
// CHECK-NEXT:   store i32 3, ptr %y37, align 4
// CHECK-NEXT:   %20 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add38 = add nsw i32 %20, 5
// CHECK-NEXT:   store i32 %add38, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next39
// CHECK: expand.next39:
// CHECK-NEXT:   store i32 4, ptr %y40, align 4
// CHECK-NEXT:   %21 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add41 = add nsw i32 %21, 6
// CHECK-NEXT:   store i32 %add41, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end42
// CHECK: expand.end42:
// CHECK-NEXT:   br label %expand.end43
// CHECK: expand.end43:
// CHECK-NEXT:   %22 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %22


// CHECK-LABEL: define {{.*}} i32 @_ZN7Private11member_funcEv()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK-NEXT:   %__begin1 = alloca ptr, align 8
// CHECK-NEXT:   %__end1 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZN7Private11member_funcEvE2p1, ptr %__range1, align 8
// CHECK-NEXT:   store ptr @_ZN7Private8integersE, ptr %__begin1, align 8
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZN7Private8integersE, i64 12), ptr %__end1, align 8
// CHECK-NEXT:   %0 = load i32, ptr @_ZN7Private8integersE, align 4
// CHECK-NEXT:   store i32 %0, ptr %x, align 4
// CHECK-NEXT:   %1 = load i32, ptr %x, align 4
// CHECK-NEXT:   %2 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %2, %1
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %3 = load i32, ptr getelementptr inbounds (i32, ptr @_ZN7Private8integersE, i64 1), align 4
// CHECK-NEXT:   store i32 %3, ptr %x1, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x1, align 4
// CHECK-NEXT:   %5 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add2 = add nsw i32 %5, %4
// CHECK-NEXT:   store i32 %add2, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next3
// CHECK: expand.next3:
// CHECK-NEXT:   %6 = load i32, ptr getelementptr inbounds (i32, ptr @_ZN7Private8integersE, i64 2), align 4
// CHECK-NEXT:   store i32 %6, ptr %x4, align 4
// CHECK-NEXT:   %7 = load i32, ptr %x4, align 4
// CHECK-NEXT:   %8 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add5 = add nsw i32 %8, %7
// CHECK-NEXT:   store i32 %add5, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   %9 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %9


// CHECK-LABEL: define {{.*}} i32 @_Z15custom_iteratorv()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK: %__begin1 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK: %__end1 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK: %ref.tmp = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x2 = alloca i32, align 4
// CHECK: %ref.tmp3 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x9 = alloca i32, align 4
// CHECK: %ref.tmp10 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x16 = alloca i32, align 4
// CHECK: %ref.tmp17 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %__range122 = alloca ptr, align 8
// CHECK: %__begin123 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK: %__end124 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x25 = alloca i32, align 4
// CHECK-NEXT:   %x28 = alloca i32, align 4
// CHECK-NEXT:   %x31 = alloca i32, align 4
// CHECK-NEXT:   %x34 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZ15custom_iteratorvE1c, ptr %__range1, align 8
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %__begin1, ptr align 4 @__const._Z15custom_iteratorv.__begin1, i64 4, i1 false)
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %__end1, ptr align 4 @__const._Z15custom_iteratorv.__end1, i64 4, i1 false)
// CHECK-NEXT:   %call = call i32 @_ZNK14CustomIterator8iteratorplEi(ptr {{.*}} %__begin1, i32 {{.*}} 0)
// CHECK: %coerce.dive = getelementptr inbounds nuw %"struct.CustomIterator::iterator", ptr %ref.tmp, i32 0, i32 0
// CHECK-NEXT:   store i32 %call, ptr %coerce.dive, align 4
// CHECK-NEXT:   %call1 = call {{.*}} i32 @_ZNK14CustomIterator8iteratordeEv(ptr {{.*}} %ref.tmp)
// CHECK-NEXT:   store i32 %call1, ptr %x, align 4
// CHECK-NEXT:   %0 = load i32, ptr %x, align 4
// CHECK-NEXT:   %1 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %1, %0
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %call4 = call i32 @_ZNK14CustomIterator8iteratorplEi(ptr {{.*}} %__begin1, i32 {{.*}} 1)
// CHECK: %coerce.dive5 = getelementptr inbounds nuw %"struct.CustomIterator::iterator", ptr %ref.tmp3, i32 0, i32 0
// CHECK-NEXT:   store i32 %call4, ptr %coerce.dive5, align 4
// CHECK-NEXT:   %call6 = call {{.*}} i32 @_ZNK14CustomIterator8iteratordeEv(ptr {{.*}} %ref.tmp3)
// CHECK-NEXT:   store i32 %call6, ptr %x2, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x2, align 4
// CHECK-NEXT:   %3 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add7 = add nsw i32 %3, %2
// CHECK-NEXT:   store i32 %add7, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next8
// CHECK: expand.next8:
// CHECK-NEXT:   %call11 = call i32 @_ZNK14CustomIterator8iteratorplEi(ptr {{.*}} %__begin1, i32 {{.*}} 2)
// CHECK: %coerce.dive12 = getelementptr inbounds nuw %"struct.CustomIterator::iterator", ptr %ref.tmp10, i32 0, i32 0
// CHECK-NEXT:   store i32 %call11, ptr %coerce.dive12, align 4
// CHECK-NEXT:   %call13 = call {{.*}} i32 @_ZNK14CustomIterator8iteratordeEv(ptr {{.*}} %ref.tmp10)
// CHECK-NEXT:   store i32 %call13, ptr %x9, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x9, align 4
// CHECK-NEXT:   %5 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add14 = add nsw i32 %5, %4
// CHECK-NEXT:   store i32 %add14, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next15
// CHECK: expand.next15:
// CHECK-NEXT:   %call18 = call i32 @_ZNK14CustomIterator8iteratorplEi(ptr {{.*}} %__begin1, i32 {{.*}} 3)
// CHECK: %coerce.dive19 = getelementptr inbounds nuw %"struct.CustomIterator::iterator", ptr %ref.tmp17, i32 0, i32 0
// CHECK-NEXT:   store i32 %call18, ptr %coerce.dive19, align 4
// CHECK-NEXT:   %call20 = call {{.*}} i32 @_ZNK14CustomIterator8iteratordeEv(ptr {{.*}} %ref.tmp17)
// CHECK-NEXT:   store i32 %call20, ptr %x16, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x16, align 4
// CHECK-NEXT:   %7 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add21 = add nsw i32 %7, %6
// CHECK-NEXT:   store i32 %add21, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store ptr @_ZZ15custom_iteratorvE1c, ptr %__range122, align 8
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %__begin123, ptr align 4 @__const._Z15custom_iteratorv.__begin1.1, i64 4, i1 false)
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %__end124, ptr align 4 @__const._Z15custom_iteratorv.__end1.2, i64 4, i1 false)
// CHECK-NEXT:   store i32 1, ptr %x25, align 4
// CHECK-NEXT:   %8 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add26 = add nsw i32 %8, 1
// CHECK-NEXT:   store i32 %add26, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next27
// CHECK: expand.next27:
// CHECK-NEXT:   store i32 2, ptr %x28, align 4
// CHECK-NEXT:   %9 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add29 = add nsw i32 %9, 2
// CHECK-NEXT:   store i32 %add29, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next30
// CHECK: expand.next30:
// CHECK-NEXT:   store i32 3, ptr %x31, align 4
// CHECK-NEXT:   %10 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add32 = add nsw i32 %10, 3
// CHECK-NEXT:   store i32 %add32, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next33
// CHECK: expand.next33:
// CHECK-NEXT:   store i32 4, ptr %x34, align 4
// CHECK-NEXT:   %11 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add35 = add nsw i32 %11, 4
// CHECK-NEXT:   store i32 %add35, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end36
// CHECK: expand.end36:
// CHECK-NEXT:   %12 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %12
