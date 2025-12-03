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
// CHECK: @__const._Z15custom_iteratorv.__end1 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 5 }, align 4
// CHECK: @__const._Z15custom_iteratorv.__begin1.1 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 1 }, align 4
// CHECK: @__const._Z15custom_iteratorv.__end1.2 = private {{.*}} constant %"struct.CustomIterator::iterator" { i32 5 }, align 4
// CHECK: @_ZN7Private8integersE = {{.*}} constant %struct.Array { [3 x i32] [i32 1, i32 2, i32 3] }, comdat, align 4

// CHECK-LABEL: define {{.*}} i32 @_Z2f1v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK-NEXT:   %__begin1 = alloca ptr, align 8
// CHECK-NEXT:   %__end1 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x2 = alloca i32, align 4
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZ2f1vE8integers, ptr %__range1, align 8
// CHECK-NEXT:   %call = call {{.*}} ptr @_ZNK5ArrayIiLm3EE5beginEv(ptr {{.*}} @_ZZ2f1vE8integers)
// CHECK-NEXT:   store ptr %call, ptr %__begin1, align 8
// CHECK-NEXT:   %call1 = call {{.*}} ptr @_ZNK5ArrayIiLm3EE3endEv(ptr {{.*}} @_ZZ2f1vE8integers)
// CHECK-NEXT:   store ptr %call1, ptr %__end1, align 8
// CHECK-NEXT:   %0 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr = getelementptr inbounds i32, ptr %0, i64 0
// CHECK-NEXT:   %1 = load i32, ptr %add.ptr, align 4
// CHECK-NEXT:   store i32 %1, ptr %x, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x, align 4
// CHECK-NEXT:   %3 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %3, %2
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %4 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr3 = getelementptr inbounds i32, ptr %4, i64 1
// CHECK-NEXT:   %5 = load i32, ptr %add.ptr3, align 4
// CHECK-NEXT:   store i32 %5, ptr %x2, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x2, align 4
// CHECK-NEXT:   %7 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add4 = add nsw i32 %7, %6
// CHECK-NEXT:   store i32 %add4, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next5
// CHECK: expand.next5:
// CHECK-NEXT:   %8 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr7 = getelementptr inbounds i32, ptr %8, i64 2
// CHECK-NEXT:   %9 = load i32, ptr %add.ptr7, align 4
// CHECK-NEXT:   store i32 %9, ptr %x6, align 4
// CHECK-NEXT:   %10 = load i32, ptr %x6, align 4
// CHECK-NEXT:   %11 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add8 = add nsw i32 %11, %10
// CHECK-NEXT:   store i32 %add8, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   %12 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %12


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
// CHECK-NEXT:   %y6 = alloca i32, align 4
// CHECK-NEXT:   %x11 = alloca i32, align 4
// CHECK-NEXT:   %__range213 = alloca ptr, align 8
// CHECK-NEXT:   %__begin214 = alloca ptr, align 8
// CHECK-NEXT:   %__end216 = alloca ptr, align 8
// CHECK-NEXT:   %y18 = alloca i32, align 4
// CHECK-NEXT:   %y23 = alloca i32, align 4
// CHECK-NEXT:   %__range129 = alloca ptr, align 8
// CHECK-NEXT:   %__begin130 = alloca ptr, align 8
// CHECK-NEXT:   %__end131 = alloca ptr, align 8
// CHECK-NEXT:   %x32 = alloca i32, align 4
// CHECK-NEXT:   %__range233 = alloca ptr, align 8
// CHECK-NEXT:   %__begin234 = alloca ptr, align 8
// CHECK-NEXT:   %__end235 = alloca ptr, align 8
// CHECK-NEXT:   %y36 = alloca i32, align 4
// CHECK-NEXT:   %y39 = alloca i32, align 4
// CHECK-NEXT:   %x43 = alloca i32, align 4
// CHECK-NEXT:   %__range244 = alloca ptr, align 8
// CHECK-NEXT:   %__begin245 = alloca ptr, align 8
// CHECK-NEXT:   %__end246 = alloca ptr, align 8
// CHECK-NEXT:   %y47 = alloca i32, align 4
// CHECK-NEXT:   %y50 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1a, ptr %__range1, align 8
// CHECK-NEXT:   %call = call {{.*}} ptr @_ZNK5ArrayIiLm2EE5beginEv(ptr {{.*}} @_ZZ2f4vE1a)
// CHECK-NEXT:   store ptr %call, ptr %__begin1, align 8
// CHECK-NEXT:   %call1 = call {{.*}} ptr @_ZNK5ArrayIiLm2EE3endEv(ptr {{.*}} @_ZZ2f4vE1a)
// CHECK-NEXT:   store ptr %call1, ptr %__end1, align 8
// CHECK-NEXT:   %0 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr = getelementptr inbounds i32, ptr %0, i64 0
// CHECK-NEXT:   %1 = load i32, ptr %add.ptr, align 4
// CHECK-NEXT:   store i32 %1, ptr %x, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__range2, align 8
// CHECK-NEXT:   %call2 = call {{.*}} ptr @_ZNK5ArrayIiLm2EE5beginEv(ptr {{.*}} @_ZZ2f4vE1b)
// CHECK-NEXT:   store ptr %call2, ptr %__begin2, align 8
// CHECK-NEXT:   %call3 = call {{.*}} ptr @_ZNK5ArrayIiLm2EE3endEv(ptr {{.*}} @_ZZ2f4vE1b)
// CHECK-NEXT:   store ptr %call3, ptr %__end2, align 8
// CHECK-NEXT:   %2 = load ptr, ptr %__begin2, align 8
// CHECK-NEXT:   %add.ptr4 = getelementptr inbounds i32, ptr %2, i64 0
// CHECK-NEXT:   %3 = load i32, ptr %add.ptr4, align 4
// CHECK-NEXT:   store i32 %3, ptr %y, align 4
// CHECK-NEXT:   %4 = load i32, ptr %x, align 4
// CHECK-NEXT:   %5 = load i32, ptr %y, align 4
// CHECK-NEXT:   %add = add nsw i32 %4, %5
// CHECK-NEXT:   %6 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add5 = add nsw i32 %6, %add
// CHECK-NEXT:   store i32 %add5, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %7 = load ptr, ptr %__begin2, align 8
// CHECK-NEXT:   %add.ptr7 = getelementptr inbounds i32, ptr %7, i64 1
// CHECK-NEXT:   %8 = load i32, ptr %add.ptr7, align 4
// CHECK-NEXT:   store i32 %8, ptr %y6, align 4
// CHECK-NEXT:   %9 = load i32, ptr %x, align 4
// CHECK-NEXT:   %10 = load i32, ptr %y6, align 4
// CHECK-NEXT:   %add8 = add nsw i32 %9, %10
// CHECK-NEXT:   %11 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add9 = add nsw i32 %11, %add8
// CHECK-NEXT:   store i32 %add9, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   br label %expand.next10
// CHECK: expand.next10:
// CHECK-NEXT:   %12 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr12 = getelementptr inbounds i32, ptr %12, i64 1
// CHECK-NEXT:   %13 = load i32, ptr %add.ptr12, align 4
// CHECK-NEXT:   store i32 %13, ptr %x11, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__range213, align 8
// CHECK-NEXT:   %call15 = call {{.*}} ptr @_ZNK5ArrayIiLm2EE5beginEv(ptr {{.*}} @_ZZ2f4vE1b)
// CHECK-NEXT:   store ptr %call15, ptr %__begin214, align 8
// CHECK-NEXT:   %call17 = call {{.*}} ptr @_ZNK5ArrayIiLm2EE3endEv(ptr {{.*}} @_ZZ2f4vE1b)
// CHECK-NEXT:   store ptr %call17, ptr %__end216, align 8
// CHECK-NEXT:   %14 = load ptr, ptr %__begin214, align 8
// CHECK-NEXT:   %add.ptr19 = getelementptr inbounds i32, ptr %14, i64 0
// CHECK-NEXT:   %15 = load i32, ptr %add.ptr19, align 4
// CHECK-NEXT:   store i32 %15, ptr %y18, align 4
// CHECK-NEXT:   %16 = load i32, ptr %x11, align 4
// CHECK-NEXT:   %17 = load i32, ptr %y18, align 4
// CHECK-NEXT:   %add20 = add nsw i32 %16, %17
// CHECK-NEXT:   %18 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add21 = add nsw i32 %18, %add20
// CHECK-NEXT:   store i32 %add21, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next22
// CHECK: expand.next22:
// CHECK-NEXT:   %19 = load ptr, ptr %__begin214, align 8
// CHECK-NEXT:   %add.ptr24 = getelementptr inbounds i32, ptr %19, i64 1
// CHECK-NEXT:   %20 = load i32, ptr %add.ptr24, align 4
// CHECK-NEXT:   store i32 %20, ptr %y23, align 4
// CHECK-NEXT:   %21 = load i32, ptr %x11, align 4
// CHECK-NEXT:   %22 = load i32, ptr %y23, align 4
// CHECK-NEXT:   %add25 = add nsw i32 %21, %22
// CHECK-NEXT:   %23 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add26 = add nsw i32 %23, %add25
// CHECK-NEXT:   store i32 %add26, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end27
// CHECK: expand.end27:
// CHECK-NEXT:   br label %expand.end28
// CHECK: expand.end28:
// CHECK-NEXT:   store ptr @_ZZ2f4vE1a, ptr %__range129, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1a, ptr %__begin130, align 8
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f4vE1a, i64 8), ptr %__end131, align 8
// CHECK-NEXT:   store i32 1, ptr %x32, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__range233, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__begin234, align 8
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f4vE1b, i64 8), ptr %__end235, align 8
// CHECK-NEXT:   store i32 3, ptr %y36, align 4
// CHECK-NEXT:   %24 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add37 = add nsw i32 %24, 4
// CHECK-NEXT:   store i32 %add37, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next38
// CHECK: expand.next38:
// CHECK-NEXT:   store i32 4, ptr %y39, align 4
// CHECK-NEXT:   %25 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add40 = add nsw i32 %25, 5
// CHECK-NEXT:   store i32 %add40, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end41
// CHECK: expand.end41:
// CHECK-NEXT:   br label %expand.next42
// CHECK: expand.next42:
// CHECK-NEXT:   store i32 2, ptr %x43, align 4
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__range244, align 8
// CHECK-NEXT:   store ptr @_ZZ2f4vE1b, ptr %__begin245, align 8
// CHECK-NEXT:   store ptr getelementptr (i8, ptr @_ZZ2f4vE1b, i64 8), ptr %__end246, align 8
// CHECK-NEXT:   store i32 3, ptr %y47, align 4
// CHECK-NEXT:   %26 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add48 = add nsw i32 %26, 5
// CHECK-NEXT:   store i32 %add48, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next49
// CHECK: expand.next49:
// CHECK-NEXT:   store i32 4, ptr %y50, align 4
// CHECK-NEXT:   %27 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add51 = add nsw i32 %27, 6
// CHECK-NEXT:   store i32 %add51, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end52
// CHECK: expand.end52:
// CHECK-NEXT:   br label %expand.end53
// CHECK: expand.end53:
// CHECK-NEXT:   %28 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %28


// CHECK-LABEL: define {{.*}} i32 @_ZN7Private11member_funcEv()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %__range1 = alloca ptr, align 8
// CHECK-NEXT:   %__begin1 = alloca ptr, align 8
// CHECK-NEXT:   %__end1 = alloca ptr, align 8
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x2 = alloca i32, align 4
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   store ptr @_ZZN7Private11member_funcEvE2p1, ptr %__range1, align 8
// CHECK-NEXT:   %call = call {{.*}} ptr @_ZNK7Private5beginEv(ptr {{.*}} @_ZZN7Private11member_funcEvE2p1)
// CHECK-NEXT:   store ptr %call, ptr %__begin1, align 8
// CHECK-NEXT:   %call1 = call {{.*}} ptr @_ZNK7Private3endEv(ptr {{.*}} @_ZZN7Private11member_funcEvE2p1)
// CHECK-NEXT:   store ptr %call1, ptr %__end1, align 8
// CHECK-NEXT:   %0 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr = getelementptr inbounds i32, ptr %0, i64 0
// CHECK-NEXT:   %1 = load i32, ptr %add.ptr, align 4
// CHECK-NEXT:   store i32 %1, ptr %x, align 4
// CHECK-NEXT:   %2 = load i32, ptr %x, align 4
// CHECK-NEXT:   %3 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add = add nsw i32 %3, %2
// CHECK-NEXT:   store i32 %add, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next
// CHECK: expand.next:
// CHECK-NEXT:   %4 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr3 = getelementptr inbounds i32, ptr %4, i64 1
// CHECK-NEXT:   %5 = load i32, ptr %add.ptr3, align 4
// CHECK-NEXT:   store i32 %5, ptr %x2, align 4
// CHECK-NEXT:   %6 = load i32, ptr %x2, align 4
// CHECK-NEXT:   %7 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add4 = add nsw i32 %7, %6
// CHECK-NEXT:   store i32 %add4, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next5
// CHECK: expand.next5:
// CHECK-NEXT:   %8 = load ptr, ptr %__begin1, align 8
// CHECK-NEXT:   %add.ptr7 = getelementptr inbounds i32, ptr %8, i64 2
// CHECK-NEXT:   %9 = load i32, ptr %add.ptr7, align 4
// CHECK-NEXT:   store i32 %9, ptr %x6, align 4
// CHECK-NEXT:   %10 = load i32, ptr %x6, align 4
// CHECK-NEXT:   %11 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add8 = add nsw i32 %11, %10
// CHECK-NEXT:   store i32 %add8, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end
// CHECK: expand.end:
// CHECK-NEXT:   %12 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %12


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
