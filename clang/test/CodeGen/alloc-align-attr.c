// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s

__INT32_TYPE__*m1(__INT32_TYPE__ i) __attribute__((alloc_align(1)));

// Condition where parameter to m1 is not size_t.
__INT32_TYPE__ test1(__INT32_TYPE__ a) {
// CHECK: define i32 @test1
  return *m1(a);
// CHECK: call i32* @m1(i32 [[PARAM1:%[^\)]+]])
// CHECK: [[ALIGNCAST1:%.+]] = zext i32 [[PARAM1]] to i64
// CHECK: [[MASK1:%.+]] = sub i64 [[ALIGNCAST1]], 1
// CHECK: [[PTRINT1:%.+]] = ptrtoint
// CHECK: [[MASKEDPTR1:%.+]] = and i64 [[PTRINT1]], [[MASK1]]
// CHECK: [[MASKCOND1:%.+]] = icmp eq i64 [[MASKEDPTR1]], 0
// CHECK: call void @llvm.assume(i1 [[MASKCOND1]])
}
// Condition where test2 param needs casting.
__INT32_TYPE__ test2(__SIZE_TYPE__ a) {
// CHECK: define i32 @test2
  return *m1(a);
// CHECK: [[CONV2:%.+]] = trunc i64 %{{.+}} to i32
// CHECK: call i32* @m1(i32 [[CONV2]])
// CHECK: [[ALIGNCAST2:%.+]] = zext i32 [[CONV2]] to i64
// CHECK: [[MASK2:%.+]] = sub i64 [[ALIGNCAST2]], 1
// CHECK: [[PTRINT2:%.+]] = ptrtoint
// CHECK: [[MASKEDPTR2:%.+]] = and i64 [[PTRINT2]], [[MASK2]]
// CHECK: [[MASKCOND2:%.+]] = icmp eq i64 [[MASKEDPTR2]], 0
// CHECK: call void @llvm.assume(i1 [[MASKCOND2]])
}
__INT32_TYPE__ *m2(__SIZE_TYPE__ i) __attribute__((alloc_align(1)));

// test3 param needs casting, but 'm2' is correct.
__INT32_TYPE__ test3(__INT32_TYPE__ a) {
// CHECK: define i32 @test3
  return *m2(a);
// CHECK: [[CONV3:%.+]] = sext i32 %{{.+}} to i64
// CHECK: call i32* @m2(i64 [[CONV3]])
// CHECK: [[MASK3:%.+]] = sub i64 [[CONV3]], 1
// CHECK: [[PTRINT3:%.+]] = ptrtoint
// CHECK: [[MASKEDPTR3:%.+]] = and i64 [[PTRINT3]], [[MASK3]]
// CHECK: [[MASKCOND3:%.+]] = icmp eq i64 [[MASKEDPTR3]], 0
// CHECK: call void @llvm.assume(i1 [[MASKCOND3]])
}

// Every type matches, canonical example.
__INT32_TYPE__ test4(__SIZE_TYPE__ a) {
// CHECK: define i32 @test4
  return *m2(a);
// CHECK: call i32* @m2(i64 [[PARAM4:%[^\)]+]])
// CHECK: [[MASK4:%.+]] = sub i64 [[PARAM4]], 1
// CHECK: [[PTRINT4:%.+]] = ptrtoint
// CHECK: [[MASKEDPTR4:%.+]] = and i64 [[PTRINT4]], [[MASK4]]
// CHECK: [[MASKCOND4:%.+]] = icmp eq i64 [[MASKEDPTR4]], 0
// CHECK: call void @llvm.assume(i1 [[MASKCOND4]])
}


struct Empty {};
struct MultiArgs { __INT64_TYPE__ a, b;};
// Struct parameter doesn't take up an IR parameter, 'i' takes up 2.
// Truncation to i64 is permissible, since alignments of greater than 2^64 are insane.
__INT32_TYPE__ *m3(struct Empty s, __int128_t i) __attribute__((alloc_align(2)));
__INT32_TYPE__ test5(__int128_t a) {
// CHECK: define i32 @test5
  struct Empty e;
  return *m3(e, a);
// CHECK: call i32* @m3(i64 %{{.*}}, i64 %{{.*}})
// CHECK: [[ALIGNCAST5:%.+]] = trunc i128 %{{.*}} to i64
// CHECK: [[MASK5:%.+]] = sub i64 [[ALIGNCAST5]], 1
// CHECK: [[PTRINT5:%.+]] = ptrtoint
// CHECK: [[MASKEDPTR5:%.+]] = and i64 [[PTRINT5]], [[MASK5]]
// CHECK: [[MASKCOND5:%.+]] = icmp eq i64 [[MASKEDPTR5]], 0
// CHECK: call void @llvm.assume(i1 [[MASKCOND5]])
}
// Struct parameter takes up 2 parameters, 'i' takes up 2.
__INT32_TYPE__ *m4(struct MultiArgs s, __int128_t i) __attribute__((alloc_align(2)));
__INT32_TYPE__ test6(__int128_t a) {
// CHECK: define i32 @test6
  struct MultiArgs e;
  return *m4(e, a);
// CHECK: call i32* @m4(i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK: [[ALIGNCAST6:%.+]] = trunc i128 %{{.*}} to i64
// CHECK: [[MASK6:%.+]] = sub i64 [[ALIGNCAST6]], 1
// CHECK: [[PTRINT6:%.+]] = ptrtoint
// CHECK: [[MASKEDPTR6:%.+]] = and i64 [[PTRINT6]], [[MASK6]]
// CHECK: [[MASKCOND6:%.+]] = icmp eq i64 [[MASKEDPTR6]], 0
// CHECK: call void @llvm.assume(i1 [[MASKCOND6]])
}

