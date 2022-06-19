// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-unknown-linux-gnu -fcxx-exceptions -fexceptions -O0 %s -emit-llvm -o - | FileCheck %s

extern "C" int printf(const char *, ...);

static int N;
struct S {
  S()
  __attribute__((nothrow)) { printf("%d: S()\n", ++N); }
  ~S() __attribute__((nothrow)) { printf("%d: ~S()\n", N--); }
  int n[17];
};
// CHECK: [[struct_S:%.+]] = type { [17 x i32] }
void print(int n, int a, int b, int c, int d) {
  printf("n=%d\n,sizeof(S)=%d\nsizeof(array_t[0][0])=%d\nsizeof(array_t[0])=%d\nsizeof(array_t)=%d\n",
         n, a, b, c, d);
  if (n == 2)
    throw(n);
}

void test(int n) {
  // CHECK: define{{.*}} void {{.*test.*}}(i32 noundef [[n:%.+]]) #
  // CHECK: [[n_addr:%.+]] = alloca
  // CHECK-NEXT: [[saved_stack:%.+]] = alloca
  // CHECK-NEXT: [[vla_expr:%.+]] = alloca i64, align 8
  // CHECK-NEXT: [[vla_expr1:%.+]] = alloca i64, align 8
  // CHECK-NEXT: [[sizeof_S:%.+]] = alloca
  // CHECK-NEXT: [[sizeof_array_t_0_0:%.+]] = alloca
  // CHECK-NEXT: [[sizeof_array_t_0:%.+]] = alloca
  // CHECK-NEXT: [[sizeof_array_t:%.+]] = alloca
  // CHECK-NEXT: [[exn_slot:%.+]] = alloca i8*
  // CHECK-NEXT: [[ehselector_slot:%.+]] = alloca i32
  // CHECK-NEXT: store i32 [[n]], i32* [[n_addr]]
  // CHECK-NEXT: [[t0:%.+]] = load i32, i32* [[n_addr]]
  // CHECK-NEXT: [[t1:%.+]] = zext i32 [[t0]] to i64
  // CHECK-NEXT: [[t2:%.+]] = load i32, i32* [[n_addr]]
  // CHECK-NEXT: [[add:%.+]] = add nsw i32 [[t2]], 1
  // CHECK-NEXT: [[t3:%.+]] = zext i32 [[add]] to i64
  // CHECK-NEXT: [[t4:%.+]] = call i8* @llvm.stacksave()
  // CHECK-NEXT: store i8* [[t4]], i8** [[saved_stack]]
  // CHECK-NEXT: [[t5:%.+]] = mul nuw i64 [[t1]], [[t3]]
  // CHECK-NEXT: [[vla:%.+]] = alloca [[struct_S]], i64 [[t5]]
  // CHECK-NEXT: store i64 [[t1]], i64* [[vla_expr]]
  // CHECK-NEXT: store i64 [[t3]], i64* [[vla_expr1]]
  // CHECK-NEXT: [[t6:%.+]] = mul nuw i64 [[t1]], [[t3]]
  // CHECK-NEXT: [[isempty:%.+]] = icmp eq i64 [[t6]], 0
  // CHECK-NEXT: br i1 [[isempty]], label %[[arrayctor_cont:.+]], label %[[new_ctorloop:.+]]

  S array_t[n][n + 1];

  // CHECK: [[new_ctorloop]]
  // CHECK-NEXT: [[arrayctor_end:%.+]] = getelementptr inbounds [[struct_S]], [[struct_S]]* [[vla]], i64 [[t6]]
  // CHECK-NEXT: br label %[[arrayctor_loop:.+]]

  // CHECK: [[arrayctor_loop]]
  // CHECK-NEXT: [[arrayctor_cur:%.+]] = phi [[struct_S]]* [ [[vla]], %[[new_ctorloop]] ], [ [[arrayctor_next:%.+]], %[[arrayctor_loop]] ]
  // CHECK-NEXT: call void [[ctor:@.+]]([[struct_S]]* {{[^,]*}} [[arrayctor_cur]])
  // CHECK-NEXT: [[arrayctor_next]] = getelementptr inbounds [[struct_S]], [[struct_S]]* [[arrayctor_cur]], i64 1
  // CHECK-NEXT: [[arrayctor_done:%.+]] = icmp eq [[struct_S]]* [[arrayctor_next]], [[arrayctor_end]]
  // CHECK-NEXT: br i1 [[arrayctor_done]], label %[[arrayctor_cont]], label %[[arrayctor_loop]]

  int sizeof_S = sizeof(S);
  int sizeof_array_t_0_0 = sizeof(array_t[0][0]);
  int sizeof_array_t_0 = sizeof(array_t[0]);
  int sizeof_array_t = sizeof(array_t);
  print(n, sizeof_S, sizeof_array_t_0_0, sizeof_array_t_0, sizeof_array_t);

  //  CHECK: [[arrayctor_cont]]
  //  CHECK-NEXT: store i32 68, i32* [[sizeof_S]]
  //  CHECK-NEXT: store i32 68, i32* [[sizeof_array_t_0_0]]
  //  CHECK: [[t8:%.+]] = mul nuw i64 68, [[t3]]
  //  CHECK-NEXT: [[conv:%.+]] = trunc i64 [[t8]] to i32
  //  CHECK-NEXT: store i32 [[conv]], i32* [[sizeof_array_t_0]]
  //  CHECK-NEXT: [[t9:%.+]] = mul nuw i64 [[t1]], [[t3]]
  //  CHECK-NEXT: [[t10:%.+]] = mul nuw i64 68, [[t9]]
  //  CHECK-NEXT: [[conv1:%.+]] = trunc i64 [[t10]] to i32
  //  CHECK-NEXT: store i32 [[conv1]], i32* [[sizeof_array_t]]
  //  CHECK-NEXT: [[t11:%.+]] = load i32, i32* [[n_addr:%.+]]
  //  CHECK-NEXT: [[t12:%.+]] = load i32, i32* [[sizeof_S]]
  //  CHECK-NEXT: [[t13:%.+]] = load i32, i32* [[sizeof_array_t_0_0]]
  //  CHECK-NEXT: [[t14:%.+]] = load i32, i32* [[sizeof_array_t_0]]
  //  CHECK-NEXT: [[t15:%.+]] = load i32, i32* [[sizeof_array_t]]
  //  CHECK-NEXT: invoke void @{{.*print.*}}(i32 noundef [[t11]], i32 noundef [[t12]], i32 noundef [[t13]], i32 noundef [[t14]], i32 noundef [[t15]])
  //  CHECK-NEXT: to label %[[invoke_cont:.+]] unwind label %[[lpad:.+]]

  //  CHECK: [[invoke_cont]]
  //  CHECK-NEXT: [[t16:%.+]] = mul nuw i64 [[t1]], [[t3]]
  //  CHECK-NEXT: [[t17:%.+]] = getelementptr inbounds [[struct_S]], [[struct_S]]* [[vla]], i64 [[t16]]
  //  CHECK-NEXT: [[arraydestroy_isempty:%.+]] = icmp eq [[struct_S]]* [[vla]], [[t17]]
  //  CHECK-NEXT: br i1 [[arraydestroy_isempty]], label %[[arraydestroy_done2:.+]], label %[[arraydestroy_body:.+]]

  //  CHECK: [[arraydestroy_body]]
  //  CHECK-NEXT: [[arraydestroy_elementPast:%.+]] = phi [[struct_S]]* [ [[t17]], %[[invoke_cont]] ], [ [[arraydestroy_element:%.+]], %[[arraydestroy_body]] ]
  //  CHECK-NEXT: [[arraydestroy_element]] = getelementptr inbounds [[struct_S]], [[struct_S]]* [[arraydestroy_elementPast]]
  //  CHECK-NEXT: call void @[[dtor:.+]]([[struct_S]]* {{[^,]*}} [[arraydestroy_element]])
  //  CHECK-NEXT: [[arraydestroy_done:%.+]] = icmp eq [[struct_S]]* [[arraydestroy_element]], [[vla]]
  //  CHECK-NEXT: br i1 [[arraydestroy_done]], label %[[arraydestroy_done2]], label %[[arraydestroy_body]]

  //  CHECK: [[arraydestroy_done2]]
  //  CHECK-NEXT: [[t17:%.+]] = load i8*, i8** [[saved_stack]]
  //  CHECK-NEXT: call void @llvm.stackrestore(i8* [[t17]])
  //  CHECK: ret void

  //  CHECK: [[lpad]]
  //  CHECK-NEXT: [[t19:%.+]] = landingpad { i8*, i32 }
  //  CHECK: [[t20:%.+]] = extractvalue { i8*, i32 } [[t19]], 0
  //  CHECK-NEXT: store i8* [[t20]], i8** [[exn_slot]]
  //  CHECK-NEXT: [[t21:%.+]] = extractvalue { i8*, i32 } [[t19]], 1
  //  CHECK-NEXT: store i32 [[t21]], i32* [[ehselector_slot]]
  //  CHECK-NEXT: [[t22:%.+]] = mul nuw i64 [[t1]], [[t3]]
  //  CHECK-NEXT: [[t23:%.+]] = getelementptr inbounds [[struct_S]], [[struct_S]]* [[vla]], i64 [[t22]]
  //  CHECK-NEXT: [[arraydestroy_isempty3:%.+]] = icmp eq [[struct_S]]* [[vla]], [[t23]]
  //  CHECK-NEXT: br i1 [[arraydestroy_isempty3]], label %[[arraydestroy_done8:.+]], label %[[arraydestroy_body4:.+]]

  //  CHECK: [[arraydestroy_body4]]
  //  CHECK: [[arraydestroy_elementPast5:%.+]] = phi [[struct_S]]* [ [[t23]], %[[lpad]] ], [ [[arraydestroy_element6:.+]], %[[arraydestroy_body4]] ]
  //  CHECK-NEXT: [[arraydestroy_element6]] = getelementptr inbounds [[struct_S]], [[struct_S]]* [[arraydestroy_elementPast5]], i64 -1
  //  CHECK-NEXT: call void @[[dtor]]([[struct_S]]* {{[^,]*}} [[arraydestroy_element6]])
  //  CHECK-NEXT: [[arraydestroy_done7:%.+]] = icmp eq [[struct_S]]* [[arraydestroy_element6]], [[vla]]
  //  CHECK-NEXT: br i1 [[arraydestroy_done7]], label %[[arraydestroy_done8]], label %[[arraydestroy_body4]]

  //  CHECK: [[arraydestroy_done8]]
  //  CHECK-NEXT: br label %[[eh_resume:.+]]

  //  CHECK: [[eh_resume]]
  //  CHECK-NEXT: [[exn:%.+]] = load i8*, i8** [[exn_slot]]
  //  CHECK-NEXT: [[sel:%.+]] = load i32, i32* [[ehselector_slot]]
  //  CHECK-NEXT: [[lpad_val:%.+]] = insertvalue { i8*, i32 } undef, i8* [[exn]], 0
  //  CHECK-NEXT: [[lpad_val9:%.+]] = insertvalue { i8*, i32 } [[lpad_val]], i32 [[sel]], 1
  //  CHECK-NEXT: resume { i8*, i32 } [[lpad_val9]]
}

int main() {
  try {
    test(2);
  } catch (int e) {
    printf("expeption %d\n", e);
  }
  try {
    test(3);
  } catch (int e) {
    printf("expeption %d", e);
  }
}
