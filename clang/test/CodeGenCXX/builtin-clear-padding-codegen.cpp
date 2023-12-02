// RUN: %clang_cc1 -triple=x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

struct alignas(4) Foo {
  char a;
  alignas(2) char b;
};

struct alignas(4) Bar {
  char c;
  alignas(2) char d;
};

struct alignas(4) Baz : Foo {
  char e;
  Bar f;
};

// Baz structure:
// "a", PAD_1, "b", PAD_2, "c", PAD_3, PAD_4, PAD_5, "c", PAD_6, "d", PAD_7
// %struct.Baz = type { %struct.Foo, i8, [3 x i8], %struct.Bar }
// %struct.Foo = type { i8, i8, i8, i8 }
// %struct.Bar = type { i8, i8, i8, i8 }

// CHECK-LABEL: define void @_Z7testBazP3Baz(%struct.Baz* %baz)
// CHECK: [[ADDR:%.*]] = alloca %struct.Baz*
// CHECK: store %struct.Baz* %baz, %struct.Baz** [[ADDR]]
// CHECK: [[BAZ:%.*]] = load %struct.Baz*, %struct.Baz** [[ADDR]]
// CHECK: [[BAZ_RAW_PTR:%.*]] = bitcast %struct.Baz* [[BAZ]] to i8*

// CHECK: [[FOO_BASE:%.*]] = getelementptr inbounds %struct.Baz, %struct.Baz* [[BAZ]], i32 0, i32 0
// CHECK: [[FOO_RAW_PTR:%.*]] = bitcast %struct.Foo* [[FOO_BASE]] to i8*
// CHECK: [[PAD_1:%.*]] = getelementptr i8, i8* [[FOO_RAW_PTR]], i32 1
// CHECK: store i8 0, i8* [[PAD_1]]
// CHECK: [[PAD_2:%.*]] = getelementptr i8, i8* [[FOO_RAW_PTR]], i32 3
// CHECK: store i8 0, i8* [[PAD_2]]

// CHECK: [[PAD_3:%.*]] = getelementptr i8, i8* [[BAZ_RAW_PTR]], i32 5
// CHECK: store i8 0, i8* [[PAD_3]]
// CHECK: [[PAD_4:%.*]] = getelementptr i8, i8* [[BAZ_RAW_PTR]], i32 6
// CHECK: store i8 0, i8* [[PAD_4]]
// CHECK: [[PAD_5:%.*]] = getelementptr i8, i8* [[BAZ_RAW_PTR]], i32 7
// CHECK: store i8 0, i8* [[PAD_5]]

// CHECK: [[BAR_MEMBER:%.*]] = getelementptr inbounds %struct.Baz, %struct.Baz* [[BAZ]], i32 0, i32 3
// CHECK: [[BAR_RAW_PTR:%.*]] = bitcast %struct.Bar* [[BAR_MEMBER]] to i8*
// CHECK: [[PAD_6:%.*]] = getelementptr i8, i8* [[BAR_RAW_PTR]], i32 1
// CHECK: store i8 0, i8* [[PAD_6]]
// CHECK: [[PAD_7:%.*]] = getelementptr i8, i8* [[BAR_RAW_PTR]], i32 3
// CHECK: store i8 0, i8* [[PAD_7]]
// CHECK: ret void
void testBaz(Baz *baz) {
  __builtin_clear_padding(baz);
}

struct UnsizedTail {
  int size;
  alignas(8) char buf[];

  UnsizedTail(int size) : size(size) {}
};

// UnsizedTail structure:
// "size", PAD_1, PAD_2, PAD_3, PAD_4
// %struct.UnsizedTail = type { i32, [4 x i8], [0 x i8] }

// CHECK-LABEL: define void @_Z15testUnsizedTailP11UnsizedTail(%struct.UnsizedTail* %u)
// CHECK: [[U_ADDR:%.*]] = alloca %struct.UnsizedTail*
// CHECK: store %struct.UnsizedTail* %u, %struct.UnsizedTail** [[U_ADDR]]
// CHECK: [[U:%.*]] = load %struct.UnsizedTail*, %struct.UnsizedTail** [[U_ADDR]]
// CHECK: [[U_RAW_PTR:%.*]] = bitcast %struct.UnsizedTail* [[U]] to i8*
// CHECK: [[PAD_1:%.*]] = getelementptr i8, i8* [[U_RAW_PTR]], i32 4
// CHECK: store i8 0, i8* [[PAD_1]]
// CHECK: [[PAD_2:%.*]] = getelementptr i8, i8* [[U_RAW_PTR]], i32 5
// CHECK: store i8 0, i8* [[PAD_2]]
// CHECK: [[PAD_3:%.*]] = getelementptr i8, i8* [[U_RAW_PTR]], i32 6
// CHECK: store i8 0, i8* [[PAD_3]]
// CHECK: [[PAD_4:%.*]] = getelementptr i8, i8* [[U_RAW_PTR]], i32 7
// CHECK: store i8 0, i8* [[PAD_4]]
// CHECK: ret void
void testUnsizedTail(UnsizedTail *u) {
  __builtin_clear_padding(u);
}

struct ArrOfStructsWithPadding {
  Bar bars[2];
};

// ArrOfStructsWithPadding structure:
// "c" (1), PAD_1, "d" (1), PAD_2, "c" (2), PAD_3, "d" (2), PAD_4
// %struct.ArrOfStructsWithPadding = type { [2 x %struct.Bar] }

// CHECK-LABEL: define void @_Z27testArrOfStructsWithPaddingP23ArrOfStructsWithPadding(%struct.ArrOfStructsWithPadding* %arr)
// CHECK: [[ARR_ADDR:%.*]] = alloca %struct.ArrOfStructsWithPadding*
// CHECK: store %struct.ArrOfStructsWithPadding* %arr, %struct.ArrOfStructsWithPadding** [[ARR_ADDR]]
// CHECK: [[ARR:%.*]] = load %struct.ArrOfStructsWithPadding*, %struct.ArrOfStructsWithPadding** [[ARR_ADDR]]
// CHECK: [[BARS:%.*]] = getelementptr inbounds %struct.ArrOfStructsWithPadding, %struct.ArrOfStructsWithPadding* [[ARR]], i32 0, i32 0
// CHECK: [[FIRST:%.*]] = getelementptr inbounds [2 x %struct.Bar], [2 x %struct.Bar]* [[BARS]], i64 0, i64 0
// CHECK: [[FIRST_RAW_PTR:%.*]] = bitcast %struct.Bar* [[FIRST]] to i8*
// CHECK: [[PAD_1:%.*]] = getelementptr i8, i8* [[FIRST_RAW_PTR]], i32 1
// CHECK: store i8 0, i8* [[PAD_1]]
// CHECK: [[PAD_2:%.*]] = getelementptr i8, i8* %4, i32 3
// CHECK: store i8 0, i8* [[PAD_2]]
// CHECK: [[SECOND:%.*]] = getelementptr inbounds [2 x %struct.Bar], [2 x %struct.Bar]* [[BARS]], i64 0, i64 1
// CHECK: [[SECOND_RAW_PTR:%.*]] = bitcast %struct.Bar* [[SECOND]] to i8*
// CHECK: [[PAD_3:%.*]] = getelementptr i8, i8* [[SECOND_RAW_PTR]], i32 1
// CHECK: store i8 0, i8* [[PAD_3]]
// CHECK: [[PAD_4:%.*]] = getelementptr i8, i8* [[SECOND_RAW_PTR]], i32 3
// CHECK: store i8 0, i8* [[PAD_4]]
// CHECK: ret void
void testArrOfStructsWithPadding(ArrOfStructsWithPadding *arr) {
  __builtin_clear_padding(arr);
}
