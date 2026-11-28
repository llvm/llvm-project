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

    constexpr void operator++() { ++n; }

    constexpr int operator*() const {
      return n;
    }

    friend constexpr bool operator!=(iterator a, iterator b) {
      return a.n != b.n;
    }

    friend constexpr int operator-(iterator a, iterator b) {
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
// CHECK: @_ZZ15custom_iteratorvE1c = internal constant %struct.CustomIterator zeroinitializer, align 1
// CHECK: @__const._Z15custom_iteratorv.__begin1 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 1 }, align 4
// CHECK: @__const._Z15custom_iteratorv.__begin1.1 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 1 }, align 4
// CHECK: @__const._Z15custom_iteratorv.__iter1 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 1 }, align 4
// CHECK: @__const._Z15custom_iteratorv.__iter1.2 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 2 }, align 4
// CHECK: @__const._Z15custom_iteratorv.__iter1.3 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 3 }, align 4
// CHECK: @__const._Z15custom_iteratorv.__iter1.4 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 4 }, align 4
// CHECK: @_ZN7Private8integersE = {{.*}} constant %struct.Array { [3 x i32] [i32 1, i32 2, i32 3] }, comdat, align 4

// CHECK-LABEL: define {{.*}} i32 @_Z2f1v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK-NEXT:   %__begin1 = alloca ptr, align 8
// CHECK-NEXT:   %__iter1 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %__iter11 = alloca ptr, align 8
// CHECK-NEXT:   %x3 = alloca i32, align 4
// CHECK-NEXT:   %__iter16 = alloca ptr, align 8
// CHECK-NEXT:   %x8 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZ2f1vE8integers, ptr %__range1, align 8
// CHECK-NEXT:   %call = call {{.*}} ptr @_ZNK5ArrayIiLm3EE5beginEv(ptr {{.*}} @_ZZ2f1vE8integers)
// CHECK-NEXT:   store ptr %call, ptr %__begin1, align 8
// CHECK-NEXT:   %0 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr = getelementptr inbounds i32, ptr %0, i64 0
// CHECK-NEXT:   store ptr %add.ptr, ptr %__iter1, align 8
// CHECK-NEXT:   %1 = load ptr, ptr %__iter1, align 8
// CHECK-NEXT:   %2 = load i32, ptr %1, align 4
// CHECK-NEXT:   store i32 %2, ptr %x, align 4
// CHECK-NEXT:   %3 = load i32, ptr %x, align 4
// CHECK-NEXT:   %4 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %4, %3
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %5 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr2 = getelementptr inbounds i32, ptr %5, i64 1
// CHECK-NEXT:   store ptr %add.ptr2, ptr %__iter11, align 8
// CHECK-NEXT:   %6 = load ptr, ptr %__iter11, align 8
// CHECK-NEXT:   %7 = load i32, ptr %6, align 4
// CHECK-NEXT:   store i32 %7, ptr %x3, align 4
// CHECK-NEXT:   %8 = load i32, ptr %x3, align 4
// CHECK-NEXT:   %9 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add4 = add nsw i32 %9, %8
// CHECK-NEXT:   store i32 %add4, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next5
// CHECK: expand.next5:
// CHECK-NEXT:   %10 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr7 = getelementptr inbounds i32, ptr %10, i64 2
// CHECK-NEXT:   store ptr %add.ptr7, ptr %__iter16, align 8
// CHECK-NEXT:   %11 = load ptr, ptr %__iter16, align 8
// CHECK-NEXT:   %12 = load i32, ptr %11, align 4
// CHECK-NEXT:   store i32 %12, ptr %x8, align 4
// CHECK-NEXT:   %13 = load i32, ptr %x8, align 4
// CHECK-NEXT:   %14 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add9 = add nsw i32 %14, %13
// CHECK-NEXT:   store i32 %add9, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   %15 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %15


// CHECK-LABEL: define {{.*}} i32 @_Z2f2v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK-NEXT:   %__begin1 = alloca ptr, align 8
// CHECK-NEXT:   %__iter1 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %__iter11 = alloca ptr, align 8
// CHECK-NEXT:   %x2 = alloca i32, align 4
// CHECK-NEXT:   %__iter15 = alloca ptr, align 8
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZ2f2vE8integers, ptr %__range1, align 8
// CHECK-NEXT:   store ptr @_ZZ2f2vE8integers, ptr %__begin1, align 8
// CHECK-NEXT:   store ptr @_ZZ2f2vE8integers, ptr %__iter1, align 8
// CHECK-NEXT:   store i32 1, ptr %x, align 4
// CHECK-NEXT:   %0 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %0, 1
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f2vE8integers, i64 4), ptr %__iter11, align 8
// CHECK-NEXT:   store i32 2, ptr %x2, align 4
// CHECK-NEXT:   %1 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add3 = add nsw i32 %1, 2
// CHECK-NEXT:   store i32 %add3, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next4
// CHECK: expand.next4:
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f2vE8integers, i64 8), ptr %__iter15, align 8
// CHECK-NEXT:   store i32 3, ptr %x6, align 4
// CHECK-NEXT:   %2 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add7 = add nsw i32 %2, 3
// CHECK-NEXT:   store i32 %add7, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   %3 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %3


// CHECK-LABEL: define {{.*}} i32 @_Z2f3v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK-NEXT:   %__begin1 = alloca ptr, align 8
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZ2f3vE8integers, ptr %__range1, align 8
// CHECK-NEXT:   store ptr @_ZZ2f3vE8integers, ptr %__begin1, align 8
// CHECK-NEXT:   %0 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %0


// CHECK-LABEL: define {{.*}} i32 @_Z2f4v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK-NEXT:   %__begin1 = alloca ptr, align 8
// CHECK-NEXT:   %__iter1 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %__range2 = alloca ptr, align 8
// CHECK-NEXT:   %__begin2 = alloca ptr, align 8
// CHECK-NEXT:   %__iter2 = alloca ptr, align 8
// CHECK-NEXT:   %y = alloca i32, align 4
// CHECK-NEXT:   %__iter24 = alloca ptr, align 8
// CHECK-NEXT:   %y6 = alloca i32, align 4
// CHECK-NEXT:   %__iter110 = alloca ptr, align 8
// CHECK-NEXT:   %x12 = alloca i32, align 4
// CHECK-NEXT:   %__range213 = alloca ptr, align 8
// CHECK-NEXT:   %__begin214 = alloca ptr, align 8
// CHECK-NEXT:   %__iter216 = alloca ptr, align 8
// CHECK-NEXT:   %y18 = alloca i32, align 4
// CHECK-NEXT:   %__iter222 = alloca ptr, align 8
// CHECK-NEXT:   %y24 = alloca i32, align 4
// CHECK-NEXT:   %__range129 = alloca ptr, align 8
// CHECK-NEXT:   %__begin130 = alloca ptr, align 8
// CHECK-NEXT:   %__iter131 = alloca ptr, align 8
// CHECK-NEXT:   %x32 = alloca i32, align 4
// CHECK-NEXT:   %__range233 = alloca ptr, align 8
// CHECK-NEXT:   %__begin234 = alloca ptr, align 8
// CHECK-NEXT:   %__iter235 = alloca ptr, align 8
// CHECK-NEXT:   %y36 = alloca i32, align 4
// CHECK-NEXT:   %__iter239 = alloca ptr, align 8
// CHECK-NEXT:   %y40 = alloca i32, align 4
// CHECK-NEXT:   %__iter144 = alloca ptr, align 8
// CHECK-NEXT:   %x45 = alloca i32, align 4
// CHECK-NEXT:   %__range246 = alloca ptr, align 8
// CHECK-NEXT:   %__begin247 = alloca ptr, align 8
// CHECK-NEXT:   %__iter248 = alloca ptr, align 8
// CHECK-NEXT:   %y49 = alloca i32, align 4
// CHECK-NEXT:   %__iter252 = alloca ptr, align 8
// CHECK-NEXT:   %y53 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1a, ptr %__range1, align 8
// CHECK-NEXT:   %call = call {{.*}} ptr @_ZNK5ArrayIiLm2EE5beginEv(ptr {{.*}} @_ZZ2f4vE1a)
// CHECK-NEXT:   store ptr %call, ptr %__begin1, align 8
// CHECK-NEXT:   %0 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr = getelementptr inbounds i32, ptr %0, i64 0
// CHECK-NEXT:   store ptr %add.ptr, ptr %__iter1, align 8
// CHECK-NEXT:   %1 = load ptr, ptr %__iter1, align 8
// CHECK-NEXT:   %2 = load i32, ptr %1, align 4
// CHECK-NEXT:   store i32 %2, ptr %x, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__range2, align 8
// CHECK-NEXT:   %call1 = call {{.*}} ptr @_ZNK5ArrayIiLm2EE5beginEv(ptr {{.*}} @_ZZ2f4vE1b)
// CHECK-NEXT:   store ptr %call1, ptr %__begin2, align 8
// CHECK-NEXT:   %3 = load ptr, ptr %__begin2, align 8
// CHECK-NEXT:   %add.ptr2 = getelementptr inbounds i32, ptr %3, i64 0
// CHECK-NEXT:   store ptr %add.ptr2, ptr %__iter2, align 8
// CHECK-NEXT:   %4 = load ptr, ptr %__iter2, align 8
// CHECK-NEXT:   %5 = load i32, ptr %4, align 4
// CHECK-NEXT:   store i32 %5, ptr %y, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x, align 4
// CHECK-NEXT:   %7 = load i32, ptr %y, align 4
// CHECK-NEXT:   %add = add nsw i32 %6, %7
// CHECK-NEXT:   %8 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add3 = add nsw i32 %8, %add
// CHECK-NEXT:   store i32 %add3, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %9 = load ptr, ptr %__begin2, align 8
// CHECK-NEXT:   %add.ptr5 = getelementptr inbounds i32, ptr %9, i64 1
// CHECK-NEXT:   store ptr %add.ptr5, ptr %__iter24, align 8
// CHECK-NEXT:   %10 = load ptr, ptr %__iter24, align 8
// CHECK-NEXT:   %11 = load i32, ptr %10, align 4
// CHECK-NEXT:   store i32 %11, ptr %y6, align 4
// CHECK-NEXT:   %12 = load i32, ptr %x, align 4
// CHECK-NEXT:   %13 = load i32, ptr %y6, align 4
// CHECK-NEXT:   %add7 = add nsw i32 %12, %13
// CHECK-NEXT:   %14 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add8 = add nsw i32 %14, %add7
// CHECK-NEXT:   store i32 %add8, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   br label %expand.next9
// CHECK: expand.next9:
// CHECK-NEXT:   %15 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr11 = getelementptr inbounds i32, ptr %15, i64 1
// CHECK-NEXT:   store ptr %add.ptr11, ptr %__iter110, align 8
// CHECK-NEXT:   %16 = load ptr, ptr %__iter110, align 8
// CHECK-NEXT:   %17 = load i32, ptr %16, align 4
// CHECK-NEXT:   store i32 %17, ptr %x12, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__range213, align 8
// CHECK-NEXT:   %call15 = call {{.*}} ptr @_ZNK5ArrayIiLm2EE5beginEv(ptr {{.*}} @_ZZ2f4vE1b)
// CHECK-NEXT:   store ptr %call15, ptr %__begin214, align 8
// CHECK-NEXT:   %18 = load ptr, ptr %__begin214, align 8
// CHECK-NEXT:   %add.ptr17 = getelementptr inbounds i32, ptr %18, i64 0
// CHECK-NEXT:   store ptr %add.ptr17, ptr %__iter216, align 8
// CHECK-NEXT:   %19 = load ptr, ptr %__iter216, align 8
// CHECK-NEXT:   %20 = load i32, ptr %19, align 4
// CHECK-NEXT:   store i32 %20, ptr %y18, align 4
// CHECK-NEXT:   %21 = load i32, ptr %x12, align 4
// CHECK-NEXT:   %22 = load i32, ptr %y18, align 4
// CHECK-NEXT:   %add19 = add nsw i32 %21, %22
// CHECK-NEXT:   %23 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add20 = add nsw i32 %23, %add19
// CHECK-NEXT:   store i32 %add20, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next21
// CHECK: expand.next21:
// CHECK-NEXT:   %24 = load ptr, ptr %__begin214, align 8
// CHECK-NEXT:   %add.ptr23 = getelementptr inbounds i32, ptr %24, i64 1
// CHECK-NEXT:   store ptr %add.ptr23, ptr %__iter222, align 8
// CHECK-NEXT:   %25 = load ptr, ptr %__iter222, align 8
// CHECK-NEXT:   %26 = load i32, ptr %25, align 4
// CHECK-NEXT:   store i32 %26, ptr %y24, align 4
// CHECK-NEXT:   %27 = load i32, ptr %x12, align 4
// CHECK-NEXT:   %28 = load i32, ptr %y24, align 4
// CHECK-NEXT:   %add25 = add nsw i32 %27, %28
// CHECK-NEXT:   %29 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add26 = add nsw i32 %29, %add25
// CHECK-NEXT:   store i32 %add26, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end27
// CHECK: expand.end27:
// CHECK-NEXT:   br label %expand.end28
// CHECK: expand.end28:
// CHECK-NEXT:   store ptr @_ZZ2f4vE1a, ptr %__range129, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1a, ptr %__begin130, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1a, ptr %__iter131, align 8
// CHECK-NEXT:   store i32 1, ptr %x32, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__range233, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__begin234, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__iter235, align 8
// CHECK-NEXT:   store i32 3, ptr %y36, align 4
// CHECK-NEXT:   %30 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add37 = add nsw i32 %30, 4
// CHECK-NEXT:   store i32 %add37, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next38
// CHECK: expand.next38:
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f4vE1b, i64 4), ptr %__iter239, align 8
// CHECK-NEXT:   store i32 4, ptr %y40, align 4
// CHECK-NEXT:   %31 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add41 = add nsw i32 %31, 5
// CHECK-NEXT:   store i32 %add41, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end42
// CHECK: expand.end42:
// CHECK-NEXT:   br label %expand.next43
// CHECK: expand.next43:
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f4vE1a, i64 4), ptr %__iter144, align 8
// CHECK-NEXT:   store i32 2, ptr %x45, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__range246, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__begin247, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__iter248, align 8
// CHECK-NEXT:   store i32 3, ptr %y49, align 4
// CHECK-NEXT:   %32 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add50 = add nsw i32 %32, 5
// CHECK-NEXT:   store i32 %add50, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next51
// CHECK: expand.next51:
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f4vE1b, i64 4), ptr %__iter252, align 8
// CHECK-NEXT:   store i32 4, ptr %y53, align 4
// CHECK-NEXT:   %33 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add54 = add nsw i32 %33, 6
// CHECK-NEXT:   store i32 %add54, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end55
// CHECK: expand.end55:
// CHECK-NEXT:   br label %expand.end56
// CHECK: expand.end56:
// CHECK-NEXT:   %34 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %34


// CHECK-LABEL: define {{.*}} i32 @_ZN7Private11member_funcEv()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK-NEXT:   %__begin1 = alloca ptr, align 8
// CHECK-NEXT:   %__iter1 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %__iter11 = alloca ptr, align 8
// CHECK-NEXT:   %x3 = alloca i32, align 4
// CHECK-NEXT:   %__iter16 = alloca ptr, align 8
// CHECK-NEXT:   %x8 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZN7Private11member_funcEvE2p1, ptr %__range1, align 8
// CHECK-NEXT:   %call = call {{.*}} ptr @_ZNK7Private5beginEv(ptr {{.*}} @_ZZN7Private11member_funcEvE2p1)
// CHECK-NEXT:   store ptr %call, ptr %__begin1, align 8
// CHECK-NEXT:   %0 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr = getelementptr inbounds i32, ptr %0, i64 0
// CHECK-NEXT:   store ptr %add.ptr, ptr %__iter1, align 8
// CHECK-NEXT:   %1 = load ptr, ptr %__iter1, align 8
// CHECK-NEXT:   %2 = load i32, ptr %1, align 4
// CHECK-NEXT:   store i32 %2, ptr %x, align 4
// CHECK-NEXT:   %3 = load i32, ptr %x, align 4
// CHECK-NEXT:   %4 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %4, %3
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %5 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr2 = getelementptr inbounds i32, ptr %5, i64 1
// CHECK-NEXT:   store ptr %add.ptr2, ptr %__iter11, align 8
// CHECK-NEXT:   %6 = load ptr, ptr %__iter11, align 8
// CHECK-NEXT:   %7 = load i32, ptr %6, align 4
// CHECK-NEXT:   store i32 %7, ptr %x3, align 4
// CHECK-NEXT:   %8 = load i32, ptr %x3, align 4
// CHECK-NEXT:   %9 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add4 = add nsw i32 %9, %8
// CHECK-NEXT:   store i32 %add4, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next5
// CHECK: expand.next5:
// CHECK-NEXT:   %10 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr7 = getelementptr inbounds i32, ptr %10, i64 2
// CHECK-NEXT:   store ptr %add.ptr7, ptr %__iter16, align 8
// CHECK-NEXT:   %11 = load ptr, ptr %__iter16, align 8
// CHECK-NEXT:   %12 = load i32, ptr %11, align 4
// CHECK-NEXT:   store i32 %12, ptr %x8, align 4
// CHECK-NEXT:   %13 = load i32, ptr %x8, align 4
// CHECK-NEXT:   %14 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add9 = add nsw i32 %14, %13
// CHECK-NEXT:   store i32 %add9, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   %15 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %15


// CHECK-LABEL: define {{.*}} i32 @_Z15custom_iteratorv()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK: %__begin1 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK: %__iter1 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK: %__iter12 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x5 = alloca i32, align 4
// CHECK: %__iter19 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x12 = alloca i32, align 4
// CHECK: %__iter116 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x19 = alloca i32, align 4
// CHECK-NEXT:   %__range122 = alloca ptr, align 8
// CHECK: %__begin123 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK: %__iter124 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x25 = alloca i32, align 4
// CHECK: %__iter128 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x29 = alloca i32, align 4
// CHECK: %__iter132 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x33 = alloca i32, align 4
// CHECK: %__iter136 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x37 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZ15custom_iteratorvE1c, ptr %__range1, align 8
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %__begin1, ptr align 4 @__const._Z15custom_iteratorv.__begin1, i64 4, i1 false)
// CHECK-NEXT:   %call = call i32 @_ZNK14CustomIterator8iteratorplEi(ptr {{.*}} %__begin1, i32 {{.*}} 0)
// CHECK: %coerce.dive = getelementptr inbounds nuw %"struct.CustomIterator::iterator", ptr %__iter1, i32 0, i32 0
// CHECK-NEXT:   store i32 %call, ptr %coerce.dive, align 4
// CHECK-NEXT:   %call1 = call {{.*}} i32 @_ZNK14CustomIterator8iteratordeEv(ptr {{.*}} %__iter1)
// CHECK-NEXT:   store i32 %call1, ptr %x, align 4
// CHECK-NEXT:   %0 = load i32, ptr %x, align 4
// CHECK-NEXT:   %1 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %1, %0
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %call3 = call i32 @_ZNK14CustomIterator8iteratorplEi(ptr {{.*}} %__begin1, i32 {{.*}} 1)
// CHECK: %coerce.dive4 = getelementptr inbounds nuw %"struct.CustomIterator::iterator", ptr %__iter12, i32 0, i32 0
// CHECK-NEXT:   store i32 %call3, ptr %coerce.dive4, align 4
// CHECK-NEXT:   %call6 = call {{.*}} i32 @_ZNK14CustomIterator8iteratordeEv(ptr {{.*}} %__iter12)
// CHECK-NEXT:   store i32 %call6, ptr %x5, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x5, align 4
// CHECK-NEXT:   %3 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add7 = add nsw i32 %3, %2
// CHECK-NEXT:   store i32 %add7, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next8
// CHECK: expand.next8:
// CHECK-NEXT:   %call10 = call i32 @_ZNK14CustomIterator8iteratorplEi(ptr {{.*}} %__begin1, i32 {{.*}} 2)
// CHECK: %coerce.dive11 = getelementptr inbounds nuw %"struct.CustomIterator::iterator", ptr %__iter19, i32 0, i32 0
// CHECK-NEXT:   store i32 %call10, ptr %coerce.dive11, align 4
// CHECK-NEXT:   %call13 = call {{.*}} i32 @_ZNK14CustomIterator8iteratordeEv(ptr {{.*}} %__iter19)
// CHECK-NEXT:   store i32 %call13, ptr %x12, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x12, align 4
// CHECK-NEXT:   %5 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add14 = add nsw i32 %5, %4
// CHECK-NEXT:   store i32 %add14, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next15
// CHECK: expand.next15:
// CHECK-NEXT:   %call17 = call i32 @_ZNK14CustomIterator8iteratorplEi(ptr {{.*}} %__begin1, i32 {{.*}} 3)
// CHECK: %coerce.dive18 = getelementptr inbounds nuw %"struct.CustomIterator::iterator", ptr %__iter116, i32 0, i32 0
// CHECK-NEXT:   store i32 %call17, ptr %coerce.dive18, align 4
// CHECK-NEXT:   %call20 = call {{.*}} i32 @_ZNK14CustomIterator8iteratordeEv(ptr {{.*}} %__iter116)
// CHECK-NEXT:   store i32 %call20, ptr %x19, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x19, align 4
// CHECK-NEXT:   %7 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add21 = add nsw i32 %7, %6
// CHECK-NEXT:   store i32 %add21, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   store ptr @_ZZ15custom_iteratorvE1c, ptr %__range122, align 8
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %__begin123, ptr align 4 @__const._Z15custom_iteratorv.__begin1.1, i64 4, i1 false)
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %__iter124, ptr align 4 @__const._Z15custom_iteratorv.__iter1, i64 4, i1 false)
// CHECK-NEXT:   store i32 1, ptr %x25, align 4
// CHECK-NEXT:   %8 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add26 = add nsw i32 %8, 1
// CHECK-NEXT:   store i32 %add26, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next27
// CHECK: expand.next27:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %__iter128, ptr align 4 @__const._Z15custom_iteratorv.__iter1.2, i64 4, i1 false)
// CHECK-NEXT:   store i32 2, ptr %x29, align 4
// CHECK-NEXT:   %9 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add30 = add nsw i32 %9, 2
// CHECK-NEXT:   store i32 %add30, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next31
// CHECK: expand.next31:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %__iter132, ptr align 4 @__const._Z15custom_iteratorv.__iter1.3, i64 4, i1 false)
// CHECK-NEXT:   store i32 3, ptr %x33, align 4
// CHECK-NEXT:   %10 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add34 = add nsw i32 %10, 3
// CHECK-NEXT:   store i32 %add34, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next35
// CHECK: expand.next35:
// CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %__iter136, ptr align 4 @__const._Z15custom_iteratorv.__iter1.4, i64 4, i1 false)
// CHECK-NEXT:   store i32 4, ptr %x37, align 4
// CHECK-NEXT:   %11 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add38 = add nsw i32 %11, 4
// CHECK-NEXT:   store i32 %add38, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end39
// CHECK: expand.end39:
// CHECK-NEXT:   %12 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %12
