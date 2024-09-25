// RUN: %clang_cc1 -fblocks -emit-llvm %s -o %t
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK1
// RUN: FileCheck %s -input-file=%t -check-prefix=CHECK2

struct A {
  int a;
  float b;
  char c;
};

void test_nest_captured_stmt(int param, int size, int param_arr[size]) {
  int w;
  int arr[param][size];
  // CHECK1: %struct.anon{{.*}} = type { ptr, ptr, [[SIZE_TYPE:i.+]], ptr, ptr, [[SIZE_TYPE]], [[SIZE_TYPE]], ptr }
  // CHECK1: %struct.anon{{.*}} = type { ptr, ptr, ptr, ptr, [[SIZE_TYPE]], ptr, ptr, [[SIZE_TYPE]], [[SIZE_TYPE]], ptr }
  // CHECK1: [[T:%struct.anon.*]] = type { ptr, ptr, ptr, ptr, ptr, [[SIZE_TYPE]], ptr, ptr, [[SIZE_TYPE]], [[SIZE_TYPE]], ptr }
  #pragma clang __debug captured
  {
    int x;
    int *y = &w;
    #pragma clang __debug captured
    {
      struct A z;
      #pragma clang __debug captured
      {
        w = x = z.a = 1;
        *y = param;
        z.b = 0.1f;
        z.c = 'c';
        param_arr[size - 1] = 2;
        arr[10][z.a] = 12;

        // CHECK1: define internal {{.*}}void @__captured_stmt.2(ptr
        // CHECK1: [[PARAM_ARR_SIZE_REF:%.+]] = getelementptr inbounds nuw [[T]], ptr {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 5
        // CHECK1: [[PARAM_ARR_SIZE:%.+]] = load [[SIZE_TYPE]], ptr [[PARAM_ARR_SIZE_REF]]
        // CHECK1: [[ARR_SIZE1_REF:%.+]] = getelementptr inbounds nuw [[T]], ptr {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 8
        // CHECK1: [[ARR_SIZE1:%.+]] = load [[SIZE_TYPE]], ptr [[ARR_SIZE1_REF]]
        // CHECK1: [[ARR_SIZE2_REF:%.+]] = getelementptr inbounds nuw [[T]], ptr {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 9
        // CHECK1: [[ARR_SIZE2:%.+]] = load [[SIZE_TYPE]], ptr [[ARR_SIZE2_REF]]
        //
        // CHECK1: getelementptr inbounds nuw [[T]], ptr {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
        // CHECK1-NEXT: load ptr, ptr
        // CHECK1-NEXT: getelementptr inbounds nuw %struct.A, ptr
        // CHECK1-NEXT: store i{{.+}} 1
        //
        // CHECK1: getelementptr inbounds nuw [[T]], ptr {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
        // CHECK1-NEXT: load ptr, ptr
        // CHECK1-NEXT: store i{{[0-9]+}} 1
        //
        // CHECK1: getelementptr inbounds nuw [[T]], ptr {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
        // CHECK1-NEXT: load ptr, ptr
        // CHECK1-NEXT: store i{{[0-9]+}} 1
        //
        // CHECK1: getelementptr inbounds nuw [[T]], ptr {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 4
        // CHECK1-NEXT: load ptr, ptr
        // CHECK1-NEXT: load i{{[0-9]+}}, ptr
        // CHECK1-NEXT: getelementptr inbounds nuw [[T]], ptr {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 3
        // CHECK1-NEXT: load ptr, ptr
        // CHECK1-NEXT: load ptr, ptr
        // CHECK1-NEXT: store i{{[0-9]+}}
        //
        // CHECK1: getelementptr inbounds nuw [[T]], ptr {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
        // CHECK1-NEXT: load ptr, ptr
        // CHECK1-NEXT: getelementptr inbounds nuw %struct.A, ptr
        // CHECK1-NEXT: store float
        //
        // CHECK1: getelementptr inbounds nuw [[T]], ptr {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
        // CHECK1-NEXT: load ptr, ptr
        // CHECK1-NEXT: getelementptr inbounds nuw %struct.A, ptr
        // CHECK1-NEXT: store i8 99
        //
        // CHECK1-DAG: [[SIZE_ADDR_REF:%.*]] = getelementptr inbounds nuw [[T]], ptr {{.*}}, i{{.+}} 0, i{{.+}} 7
        // CHECK1-DAG: [[SIZE_ADDR:%.*]] = load ptr, ptr [[SIZE_ADDR_REF]]
        // CHECK1-DAG: [[SIZE:%.*]] = load i{{.+}}, ptr [[SIZE_ADDR]]
        // CHECK1-DAG: [[PARAM_ARR_IDX:%.*]] = sub nsw i{{.+}} [[SIZE]], 1
        // CHECK1-DAG: [[PARAM_ARR_ADDR_REF:%.*]] = getelementptr inbounds nuw [[T]], ptr {{.*}}, i{{.+}} 0, i{{.+}} 6
        // CHECK1-DAG: [[PARAM_ARR_ADDR:%.*]] = load ptr, ptr [[PARAM_ARR_ADDR_REF]]
        // CHECK1-DAG: [[PARAM_ARR:%.*]] = load ptr, ptr [[PARAM_ARR_ADDR]]
        // CHECK1-DAG: [[PARAM_ARR_SIZE_MINUS_1_ADDR:%.*]] = getelementptr inbounds i{{.+}}, ptr [[PARAM_ARR]], i{{.*}}
        // CHECK1: store i{{.+}} 2, ptr [[PARAM_ARR_SIZE_MINUS_1_ADDR]]
        //
        // CHECK1-DAG: [[Z_ADDR_REF:%.*]] = getelementptr inbounds nuw [[T]], ptr {{.*}}, i{{.+}} 0, i{{.+}} 2
        // CHECK1-DAG: [[Z_ADDR:%.*]] = load ptr, ptr [[Z_ADDR_REF]]
        // CHECK1-DAG: [[Z_A_ADDR:%.*]] = getelementptr inbounds nuw %struct.A, ptr [[Z_ADDR]], i{{.+}} 0, i{{.+}} 0
        // CHECK1-DAG: [[ARR_IDX_2:%.*]] = load i{{.+}}, ptr [[Z_A_ADDR]]
        // CHECK1-DAG: [[ARR_ADDR_REF:%.*]] = getelementptr inbounds nuw [[T]], ptr {{.*}}, i{{.+}} 0, i{{.+}} 10
        // CHECK1-DAG: [[ARR_ADDR:%.*]] = load ptr, ptr [[ARR_ADDR_REF]]
        // CHECK1-DAG: [[ARR_IDX_1:%.*]] = mul {{.*}} 10
        // CHECK1-DAG: [[ARR_10_ADDR:%.*]] = getelementptr inbounds i{{.+}}, ptr [[ARR_ADDR]], i{{.*}} [[ARR_IDX_1]]
        // CHECK1-DAG: [[ARR_10_Z_A_ADDR:%.*]] = getelementptr inbounds i{{.+}}, ptr [[ARR_10_ADDR]], i{{.*}}
        // CHECK1: store i{{.+}} 12, ptr [[ARR_10_Z_A_ADDR]]
      }
    }
  }
}

void test_nest_block(void) {
  __block int x;
  int y;
  ^{
    int z;
    x = z;
    #pragma clang __debug captured
    {
      z = y; // OK
    }
  }();

  // CHECK2: define internal {{.*}}void @{{.*}}test_nest_block_block_invoke
  //
  // CHECK2: [[Z:%[0-9a-z_]*]] = alloca i{{[0-9]+}},
  // CHECK2: alloca %struct.anon{{.*}}
  //
  // CHECK2: store i{{[0-9]+}}
  // CHECK2: store ptr [[Z]]
  //
  // CHECK2: getelementptr inbounds nuw %struct.anon
  // CHECK2-NEXT: getelementptr inbounds
  // CHECK2-NEXT: store ptr
  //
  // CHECK2: call void @__captured_stmt

  int a;
  #pragma clang __debug captured
  {
    __block int b;
    int c;
    __block int d;
    ^{
      b = a;
      b = c;
      b = d;
    }();
  }

  // CHECK2: alloca %struct.__block_byref_b
  // CHECK2-NEXT: [[C:%[0-9a-z_]*]] = alloca i{{[0-9]+}}
  // CHECK2-NEXT: alloca %struct.__block_byref_d
  //
  // CHECK2: store ptr
  //
  // CHECK2: [[CapA:%[0-9a-z_.]*]] = getelementptr inbounds {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 7
  //
  // CHECK2: getelementptr inbounds nuw %struct.anon{{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK2: load ptr, ptr
  // CHECK2: load i{{[0-9]+}}, ptr
  // CHECK2: store i{{[0-9]+}} {{.*}}, ptr [[CapA]]
  //
  // CHECK2: [[CapC:%[0-9a-z_.]*]] = getelementptr inbounds {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 8
  // CHECK2-NEXT: [[Val:%[0-9a-z_]*]] = load i{{[0-9]+}}, ptr [[C]]
  // CHECK2-NEXT: store i{{[0-9]+}} [[Val]], ptr [[CapC]]
  //
  // CHECK2: store ptr
}
