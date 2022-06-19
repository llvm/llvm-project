// RUN: %clang_cc1 -no-opaque-pointers %s -triple x86_64-pc-win32 -emit-llvm -o - | FileCheck %s

typedef __SIZE_TYPE__ size_t;
typedef __WCHAR_TYPE__ wchar_t;

const wchar_t *wmemchr_test(const wchar_t *s, const wchar_t c, size_t n) {
  // CHECK-LABEL: define dso_local i16* @wmemchr_test
  // CHECK: [[S:%.*]] = load
  // CHECK: [[C:%.*]] = load
  // CHECK: [[N:%.*]] = load
  // CHECK: [[N0:%.*]] = icmp eq i64 [[N]], 0
  // CHECK: br i1 [[N0]], label %[[EXIT:.*]], label %[[EQ:.*]]

  // CHECK: [[EQ]]:
  // CHECK: [[SP:%.*]] = phi i16* [ [[S]], %[[ENTRY:.*]] ], [ [[SN:.*]], %[[NEXT:.*]] ]
  // CHECK: [[NP:%.*]] = phi i64 [ [[N]], %[[ENTRY]] ], [ [[NN:.*]], %[[NEXT]] ]
  // CHECK: [[SL:%.*]] = load i16, i16* [[SP]], align 2
  // CHECK: [[RES:%.*]] = getelementptr inbounds i16, i16* [[SP]], i32 0
  // CHECK: [[CMPEQ:%.*]] = icmp eq i16 [[SL]], [[C]]
  // CHECK: br i1 [[CMPEQ]], label %[[EXIT]], label %[[LT:.*]]

  // CHECK: [[NEXT]]:
  // CHECK: [[SN]] = getelementptr inbounds i16, i16* [[SP]], i32 1
  // CHECK: [[NN]] = sub i64 [[NP]], 1
  // CHECK: [[NN0:%.*]] = icmp eq i64 [[NN]], 0
  // CHECK: br i1 [[NN0]], label %[[EXIT]], label %[[EQ]]
  //
  // CHECK: [[EXIT]]:
  // CHECK: [[RV:%.*]] = phi i16* [ null, %[[ENTRY]] ], [ null, %[[NEXT]] ], [ [[RES]], %[[EQ]] ] 
  // CHECK: ret i16* [[RV]]
  return __builtin_wmemchr(s, c, n);
}
