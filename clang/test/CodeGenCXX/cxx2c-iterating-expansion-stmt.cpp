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
// CHECK: @_ZZ2f1vE8__range1 = internal constant ptr @_ZZ2f1vE8integers, align 8
// CHECK: @_ZZ2f1vE8__begin1 = internal constant ptr @_ZZ2f1vE8integers, align 8
// CHECK: @_ZZ2f1vE6__end1 = internal constant ptr getelementptr (i8, ptr @_ZZ2f1vE8integers, i64 12), align 8
// CHECK: @_ZZ2f2vE8integers = internal constant %struct.Array { [3 x i32] [i32 1, i32 2, i32 3] }, align 4
// CHECK: @_ZZ2f2vE8__range1 = internal constant ptr @_ZZ2f2vE8integers, align 8
// CHECK: @_ZZ2f2vE8__begin1 = internal constant ptr @_ZZ2f2vE8integers, align 8
// CHECK: @_ZZ2f2vE6__end1 = internal constant ptr getelementptr (i8, ptr @_ZZ2f2vE8integers, i64 12), align 8
// CHECK: @_ZZ2f3vE8integers = internal constant %struct.Array.0 zeroinitializer, align 4
// CHECK: @_ZZ2f3vE8__range1 = internal constant ptr @_ZZ2f3vE8integers, align 8
// CHECK: @_ZZ2f3vE8__begin1 = internal constant ptr @_ZZ2f3vE8integers, align 8
// CHECK: @_ZZ2f3vE6__end1 = internal constant ptr @_ZZ2f3vE8integers, align 8
// CHECK: @_ZZ2f4vE1a = internal constant %struct.Array.1 { [2 x i32] [i32 1, i32 2] }, align 4
// CHECK: @_ZZ2f4vE1b = internal constant %struct.Array.1 { [2 x i32] [i32 3, i32 4] }, align 4
// CHECK: @_ZZ2f4vE8__range1 = internal constant ptr @_ZZ2f4vE1a, align 8
// CHECK: @_ZZ2f4vE8__begin1 = internal constant ptr @_ZZ2f4vE1a, align 8
// CHECK: @_ZZ2f4vE6__end1 = internal constant ptr getelementptr (i8, ptr @_ZZ2f4vE1a, i64 8), align 8
// CHECK: @_ZZ2f4vE8__range2 = internal constant ptr @_ZZ2f4vE1b, align 8
// CHECK: @_ZZ2f4vE8__begin2 = internal constant ptr @_ZZ2f4vE1b, align 8
// CHECK: @_ZZ2f4vE6__end2 = internal constant ptr getelementptr (i8, ptr @_ZZ2f4vE1b, i64 8), align 8
// CHECK: @_ZZ2f4vE8__range2_0 = internal constant ptr @_ZZ2f4vE1b, align 8
// CHECK: @_ZZ2f4vE8__begin2_0 = internal constant ptr @_ZZ2f4vE1b, align 8
// CHECK: @_ZZ2f4vE6__end2_0 = internal constant ptr getelementptr (i8, ptr @_ZZ2f4vE1b, i64 8), align 8
// CHECK: @_ZZ2f4vE8__range1_0 = internal constant ptr @_ZZ2f4vE1a, align 8
// CHECK: @_ZZ2f4vE8__begin1_0 = internal constant ptr @_ZZ2f4vE1a, align 8
// CHECK: @_ZZ2f4vE6__end1_0 = internal constant ptr getelementptr (i8, ptr @_ZZ2f4vE1a, i64 8), align 8
// CHECK: @_ZZ2f4vE8__range2_1 = internal constant ptr @_ZZ2f4vE1b, align 8
// CHECK: @_ZZ2f4vE8__begin2_1 = internal constant ptr @_ZZ2f4vE1b, align 8
// CHECK: @_ZZ2f4vE6__end2_1 = internal constant ptr getelementptr (i8, ptr @_ZZ2f4vE1b, i64 8), align 8
// CHECK: @_ZZ2f4vE8__range2_2 = internal constant ptr @_ZZ2f4vE1b, align 8
// CHECK: @_ZZ2f4vE8__begin2_2 = internal constant ptr @_ZZ2f4vE1b, align 8
// CHECK: @_ZZ2f4vE6__end2_2 = internal constant ptr getelementptr (i8, ptr @_ZZ2f4vE1b, i64 8), align 8
// CHECK: @_ZZN7Private11member_funcEvE2p1 = internal constant %struct.Private zeroinitializer, align 1
// CHECK: @_ZZN7Private11member_funcEvE8__range1 = internal constant ptr @_ZZN7Private11member_funcEvE2p1, align 8
// CHECK: @_ZZN7Private11member_funcEvE8__begin1 = internal constant ptr @_ZN7Private8integersE, align 8
// CHECK: @_ZN7Private8integersE = {{.*}} constant %struct.Array { [3 x i32] [i32 1, i32 2, i32 3] }, comdat, align 4
// CHECK: @_ZZN7Private11member_funcEvE6__end1 = internal constant ptr getelementptr (i8, ptr @_ZN7Private8integersE, i64 12), align 8
// CHECK: @_ZZ15custom_iteratorvE1c = internal constant %struct.CustomIterator zeroinitializer, align 1
// CHECK: @_ZZ15custom_iteratorvE8__range1 = internal constant ptr @_ZZ15custom_iteratorvE1c, align 8
// CHECK: @_ZZ15custom_iteratorvE8__begin1 = internal constant %"struct.CustomIterator::iterator" { i32 1 }, align 4
// CHECK: @_ZZ15custom_iteratorvE6__end1 = internal constant %"struct.CustomIterator::iterator" { i32 5 }, align 4
// CHECK: @_ZZ15custom_iteratorvE8__range1_0 = internal constant ptr @_ZZ15custom_iteratorvE1c, align 8
// CHECK: @_ZZ15custom_iteratorvE8__begin1_0 = internal constant %"struct.CustomIterator::iterator" { i32 1 }, align 4
// CHECK: @_ZZ15custom_iteratorvE6__end1_0 = internal constant %"struct.CustomIterator::iterator" { i32 5 }, align 4


// CHECK-LABEL: define {{.*}} i32 @_Z2f1v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
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
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
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
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   %0 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %0


// CHECK-LABEL: define {{.*}} i32 @_Z2f4v()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %y = alloca i32, align 4
// CHECK-NEXT:   %y2 = alloca i32, align 4
// CHECK-NEXT:   %x6 = alloca i32, align 4
// CHECK-NEXT:   %y7 = alloca i32, align 4
// CHECK-NEXT:   %y11 = alloca i32, align 4
// CHECK-NEXT:   %x16 = alloca i32, align 4
// CHECK-NEXT:   %y17 = alloca i32, align 4
// CHECK-NEXT:   %y20 = alloca i32, align 4
// CHECK-NEXT:   %x24 = alloca i32, align 4
// CHECK-NEXT:   %y25 = alloca i32, align 4
// CHECK-NEXT:   %y28 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   %0 = load i32, ptr @_ZZ2f4vE1a, align 4
// CHECK-NEXT:   store i32 %0, ptr %x, align 4
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
// CHECK-NEXT:   %10 = load i32, ptr @_ZZ2f4vE1b, align 4
// CHECK-NEXT:   store i32 %10, ptr %y7, align 4
// CHECK-NEXT:   %11 = load i32, ptr %x6, align 4
// CHECK-NEXT:   %12 = load i32, ptr %y7, align 4
// CHECK-NEXT:   %add8 = add nsw i32 %11, %12
// CHECK-NEXT:   %13 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add9 = add nsw i32 %13, %add8
// CHECK-NEXT:   store i32 %add9, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next10
// CHECK: expand.next10:
// CHECK-NEXT:   %14 = load i32, ptr getelementptr inbounds (i32, ptr @_ZZ2f4vE1b, i64 1), align 4
// CHECK-NEXT:   store i32 %14, ptr %y11, align 4
// CHECK-NEXT:   %15 = load i32, ptr %x6, align 4
// CHECK-NEXT:   %16 = load i32, ptr %y11, align 4
// CHECK-NEXT:   %add12 = add nsw i32 %15, %16
// CHECK-NEXT:   %17 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add13 = add nsw i32 %17, %add12
// CHECK-NEXT:   store i32 %add13, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end14
// CHECK: expand.end14:
// CHECK-NEXT:   br label %expand.end15
// CHECK: expand.end15:
// CHECK-NEXT:   store i32 1, ptr %x16, align 4
// CHECK-NEXT:   store i32 3, ptr %y17, align 4
// CHECK-NEXT:   %18 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add18 = add nsw i32 %18, 4
// CHECK-NEXT:   store i32 %add18, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next19
// CHECK: expand.next19:
// CHECK-NEXT:   store i32 4, ptr %y20, align 4
// CHECK-NEXT:   %19 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add21 = add nsw i32 %19, 5
// CHECK-NEXT:   store i32 %add21, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end22
// CHECK: expand.end22:
// CHECK-NEXT:   br label %expand.next23
// CHECK: expand.next23:
// CHECK-NEXT:   store i32 2, ptr %x24, align 4
// CHECK-NEXT:   store i32 3, ptr %y25, align 4
// CHECK-NEXT:   %20 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add26 = add nsw i32 %20, 5
// CHECK-NEXT:   store i32 %add26, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next27
// CHECK: expand.next27:
// CHECK-NEXT:   store i32 4, ptr %y28, align 4
// CHECK-NEXT:   %21 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add29 = add nsw i32 %21, 6
// CHECK-NEXT:   store i32 %add29, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end30
// CHECK: expand.end30:
// CHECK-NEXT:   br label %expand.end31
// CHECK: expand.end31:
// CHECK-NEXT:   %22 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %22


// CHECK-LABEL: define {{.*}} i32 @_ZN7Private11member_funcEv()
// CHECK: entry:
// CHECK-NEXT:   %sum = alloca i32, align 4
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK-NEXT:   %x1 = alloca i32, align 4
// CHECK-NEXT:   %x4 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
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
// CHECK-NEXT:   %x = alloca i32, align 4
// CHECK: %ref.tmp = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x2 = alloca i32, align 4
// CHECK: %ref.tmp3 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x9 = alloca i32, align 4
// CHECK: %ref.tmp10 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x16 = alloca i32, align 4
// CHECK: %ref.tmp17 = alloca %"struct.CustomIterator::iterator", align 4
// CHECK-NEXT:   %x22 = alloca i32, align 4
// CHECK-NEXT:   %x25 = alloca i32, align 4
// CHECK-NEXT:   %x28 = alloca i32, align 4
// CHECK-NEXT:   %x31 = alloca i32, align 4
// CHECK-NEXT:   store i32 0, ptr %sum, align 4
// CHECK-NEXT:   %call = call i32 @_ZNK14CustomIterator8iteratorplEi(ptr {{.*}} @_ZZ15custom_iteratorvE8__begin1, i32 {{.*}} 0)
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
// CHECK-NEXT:   %call4 = call i32 @_ZNK14CustomIterator8iteratorplEi(ptr {{.*}} @_ZZ15custom_iteratorvE8__begin1, i32 {{.*}} 1)
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
// CHECK-NEXT:   %call11 = call i32 @_ZNK14CustomIterator8iteratorplEi(ptr {{.*}} @_ZZ15custom_iteratorvE8__begin1, i32 {{.*}} 2)
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
// CHECK-NEXT:   %call18 = call i32 @_ZNK14CustomIterator8iteratorplEi(ptr {{.*}} @_ZZ15custom_iteratorvE8__begin1, i32 {{.*}} 3)
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
// CHECK-NEXT:   store i32 1, ptr %x22, align 4
// CHECK-NEXT:   %8 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add23 = add nsw i32 %8, 1
// CHECK-NEXT:   store i32 %add23, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next24
// CHECK: expand.next24:
// CHECK-NEXT:   store i32 2, ptr %x25, align 4
// CHECK-NEXT:   %9 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add26 = add nsw i32 %9, 2
// CHECK-NEXT:   store i32 %add26, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next27
// CHECK: expand.next27:
// CHECK-NEXT:   store i32 3, ptr %x28, align 4
// CHECK-NEXT:   %10 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add29 = add nsw i32 %10, 3
// CHECK-NEXT:   store i32 %add29, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.next30
// CHECK: expand.next30:
// CHECK-NEXT:   store i32 4, ptr %x31, align 4
// CHECK-NEXT:   %11 = load i32, ptr %sum, align 4
// CHECK-NEXT:   %add32 = add nsw i32 %11, 4
// CHECK-NEXT:   store i32 %add32, ptr %sum, align 4
// CHECK-NEXT:   br label %expand.end33
// CHECK: expand.end33:
// CHECK-NEXT:   %12 = load i32, ptr %sum, align 4
// CHECK-NEXT:   ret i32 %12
