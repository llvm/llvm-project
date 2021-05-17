// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// CHECK:    @_ZZ4testvE3foo = internal addrspace(1) constant i32 66, align 4
// CHECK: @[[STR:[.a-zA-Z0-9_]+]] = private unnamed_addr addrspace(1) constant [14 x i8] c"Hello, world!\00", align 1

// CHECK-LABEL: @_Z4testv
void test() {
  static const int foo = 0x42;

  // CHECK: %i.ascast = addrspacecast i32* %i to i32 addrspace(4)*
  // CHECK: %[[ARR:[a-zA-Z0-9]+]] = alloca [42 x i32]
  // CHECK: %[[ARR]].ascast = addrspacecast [42 x i32]* %[[ARR]] to [42 x i32] addrspace(4)*

  int i = 0;
  int *pptr = &i;
  // CHECK: store i32 addrspace(4)* %i.ascast, i32 addrspace(4)* addrspace(4)* %pptr.ascast
  bool is_i_ptr = (pptr == &i);
  // CHECK: %[[VALPPTR:[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %pptr.ascast
  // CHECK: %cmp{{[0-9]*}} = icmp eq i32 addrspace(4)* %[[VALPPTR]], %i.ascast
  *pptr = foo;

  int var23 = 23;
  char *cp = (char *)&var23;
  *cp = 41;
  // CHECK: store i32 23, i32 addrspace(4)* %[[VAR:[a-zA-Z0-9.]+]]
  // CHECK: [[VARCAST:%.*]] = bitcast i32 addrspace(4)* %[[VAR]] to i8 addrspace(4)*
  // CHECK: store i8 addrspace(4)* [[VARCAST]], i8 addrspace(4)* addrspace(4)* %{{.*}}

  int arr[42];
  char *cpp = (char *)arr;
  *cpp = 43;
  // CHECK: [[ARRDECAY:%.*]] = getelementptr inbounds [42 x i32], [42 x i32] addrspace(4)* %[[ARR]].ascast, i64 0, i64 0
  // CHECK: [[ARRCAST:%.*]] = bitcast i32 addrspace(4)* [[ARRDECAY]] to i8 addrspace(4)*
  // CHECK: store i8 addrspace(4)* [[ARRCAST]], i8 addrspace(4)* addrspace(4)* %{{.*}}

  int *aptr = arr + 10;
  if (aptr < arr + sizeof(arr))
    *aptr = 44;
  // CHECK: %[[VALAPTR:.*]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %aptr.ascast
  // CHECK: %[[ARRDCY2:.*]] = getelementptr inbounds [42 x i32], [42 x i32] addrspace(4)* %[[ARR]].ascast, i64 0, i64 0
  // CHECK: %[[ADDPTR:.*]] = getelementptr inbounds i32, i32  addrspace(4)* %[[ARRDCY2]], i64 168
  // CHECK: %cmp{{[0-9]+}} = icmp ult i32 addrspace(4)* %[[VALAPTR]], %[[ADDPTR]]

  const char *str = "Hello, world!";
  // CHECK: store i8 addrspace(4)* getelementptr inbounds ([14 x i8], [14 x i8] addrspace(4)* addrspacecast ([14 x i8] addrspace(1)* @[[STR]] to [14 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* addrspace(4)* %[[STRVAL:[a-zA-Z0-9]+]].ascast, align 8

  i = str[0];

  const char *phi_str = i > 2 ? str : "Another hello world!";
  (void)phi_str;
  // CHECK: %[[COND:[a-zA-Z0-9]+]] = icmp sgt i32 %{{.*}}, 2
  // CHECK: br i1 %[[COND]], label %[[CONDTRUE:[.a-zA-Z0-9]+]], label %[[CONDFALSE:[.a-zA-Z0-9]+]]

  // CHECK: [[CONDTRUE]]:
  // CHECK-NEXT: %[[VALTRUE:[a-zA-Z0-9]+]] = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %str.ascast
  // CHECK-NEXT: br label %[[CONDEND:[.a-zA-Z0-9]+]]

  // CHECK: [[CONDFALSE]]:

  // CHECK: [[CONDEND]]:
  // CHECK-NEXT: phi i8 addrspace(4)* [ %[[VALTRUE]], %[[CONDTRUE]] ], [ getelementptr inbounds ([21 x i8], [21 x i8] addrspace(4)* addrspacecast ([21 x i8] addrspace(1)* @{{.*}} to [21 x i8] addrspace(4)*), i64 0, i64 0), %[[CONDFALSE]] ]

  const char *select_null = i > 2 ? "Yet another Hello world" : nullptr;
  (void)select_null;
  // CHECK: select i1 %{{.*}}, i8 addrspace(4)* getelementptr inbounds ([24 x i8], [24 x i8] addrspace(4)* addrspacecast ([24 x i8] addrspace(1)* @{{.*}} to [24 x i8] addrspace(4)*), i64 0, i64 0)

  const char *select_str_trivial1 = true ? str : "Another hello world!";
  (void)select_str_trivial1;
  // CHECK: %[[TRIVIALTRUE:[a-zA-Z0-9]+]] = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %[[STRVAL]]
  // CHECK: store i8 addrspace(4)* %[[TRIVIALTRUE]], i8 addrspace(4)* addrspace(4)* %{{.*}}, align 8

  const char *select_str_trivial2 = false ? str : "Another hello world!";
  (void)select_str_trivial2;
}
