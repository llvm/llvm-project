// RUN: %clang_cc1 -no-opaque-pointers %s -triple x86_64-pc-win32 -emit-llvm -o - | FileCheck %s

typedef __SIZE_TYPE__ size_t;
typedef __WCHAR_TYPE__ wchar_t;

int wmemcmp_test(const wchar_t *s1, const wchar_t *s2, size_t n) {
  // CHECK: [[S1:%.*]] = load
  // CHECK: [[S2:%.*]] = load
  // CHECK: [[N:%.*]] = load
  // CHECK: [[N0:%.*]] = icmp eq i64 [[N]], 0
  // CHECK: br i1 [[N0]], label %[[EXIT:.*]], label %[[GT:.*]]

  // CHECK: [[GT]]:
  // CHECK: [[S1P:%.*]] = phi i16* [ [[S1]], %[[ENTRY:.*]] ], [ [[S1N:.*]], %[[NEXT:.*]] ]
  // CHECK: [[S2P:%.*]] = phi i16* [ [[S2]], %[[ENTRY]] ], [ [[S2N:.*]], %[[NEXT]] ]
  // CHECK: [[NP:%.*]] = phi i64 [ [[N]], %[[ENTRY]] ], [ [[NN:.*]], %[[NEXT]] ]
  // CHECK: [[S1L:%.*]] = load i16, i16* [[S1P]], align 2
  // CHECK: [[S2L:%.*]] = load i16, i16* [[S2P]], align 2
  // CHECK: [[CMPGT:%.*]] = icmp ugt i16 [[S1L]], [[S2L]]
  // CHECK: br i1 [[CMPGT]], label %[[EXIT]], label %[[LT:.*]]

  // CHECK: [[LT]]:
  // CHECK: [[CMPLT:%.*]] = icmp ult i16 [[S1L]], [[S2L]]
  // CHECK: br i1 [[CMPLT]], label %[[EXIT]], label %[[NEXT:.*]]

  // CHECK: [[NEXT]]:
  // CHECK: [[S1N]] = getelementptr inbounds i16, i16* [[S1P]], i32 1
  // CHECK: [[S2N]] = getelementptr inbounds i16, i16* [[S2P]], i32 1
  // CHECK: [[NN]] = sub i64 [[NP]], 1
  // CHECK: [[NN0:%.*]] = icmp eq i64 [[NN]], 0
  // CHECK: br i1 [[NN0]], label %[[EXIT]], label %[[GT]]
  //
  // CHECK: [[EXIT]]:
  // CHECK: [[RV:%.*]] = phi i32 [ 0, %[[ENTRY]] ], [ 1, %[[GT]] ], [ -1, %[[LT]] ], [ 0, %[[NEXT]] ]
  // CHECK: ret i32 [[RV]]
  return __builtin_wmemcmp(s1, s2, n);
}
