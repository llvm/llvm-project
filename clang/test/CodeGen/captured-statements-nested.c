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
  // CHECK1: %struct.anon{{.*}} = type { [[INT:i.+]]*, [[INT]]*, [[SIZE_TYPE:i.+]], [[INT]]**, [[INT]]*, [[SIZE_TYPE]], [[SIZE_TYPE]], [[INT]]* }
  // CHECK1: %struct.anon{{.*}} = type { [[INT]]*, [[INT]]*, [[INT]]**, [[INT]]*, [[SIZE_TYPE]], [[INT]]**, [[INT]]*, [[SIZE_TYPE]], [[SIZE_TYPE]], [[INT]]* }
  // CHECK1: [[T:%struct.anon.*]] = type { [[INT]]*, [[INT]]*, %struct.A*, [[INT]]**, [[INT]]*, [[SIZE_TYPE]], [[INT]]**, [[INT]]*, [[SIZE_TYPE]], [[SIZE_TYPE]], [[INT]]* }
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

        // CHECK1: define internal {{.*}}void @__captured_stmt{{.*}}([[T]]
        // CHECK1: [[PARAM_ARR_SIZE_REF:%.+]] = getelementptr inbounds [[T]], [[T]]* {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 5
        // CHECK1: [[PARAM_ARR_SIZE:%.+]] = load [[SIZE_TYPE]], [[SIZE_TYPE]]* [[PARAM_ARR_SIZE_REF]]
        // CHECK1: [[ARR_SIZE1_REF:%.+]] = getelementptr inbounds [[T]], [[T]]* {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 8
        // CHECK1: [[ARR_SIZE1:%.+]] = load [[SIZE_TYPE]], [[SIZE_TYPE]]* [[ARR_SIZE1_REF]]
        // CHECK1: [[ARR_SIZE2_REF:%.+]] = getelementptr inbounds [[T]], [[T]]* {{.+}}, i{{[0-9]+}} 0, i{{[0-9]+}} 9
        // CHECK1: [[ARR_SIZE2:%.+]] = load [[SIZE_TYPE]], [[SIZE_TYPE]]* [[ARR_SIZE2_REF]]
        //
        // CHECK1: getelementptr inbounds [[T]], [[T]]* {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
        // CHECK1-NEXT: load %struct.A*, %struct.A**
        // CHECK1-NEXT: getelementptr inbounds %struct.A, %struct.A*
        // CHECK1-NEXT: store i{{.+}} 1
        //
        // CHECK1: getelementptr inbounds [[T]], [[T]]* {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 1
        // CHECK1-NEXT: load i{{[0-9]+}}*, i{{[0-9]+}}**
        // CHECK1-NEXT: store i{{[0-9]+}} 1
        //
        // CHECK1: getelementptr inbounds [[T]], [[T]]* {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
        // CHECK1-NEXT: load i{{[0-9]+}}*, i{{[0-9]+}}**
        // CHECK1-NEXT: store i{{[0-9]+}} 1
        //
        // CHECK1: getelementptr inbounds [[T]], [[T]]* {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 4
        // CHECK1-NEXT: load i{{[0-9]+}}*, i{{[0-9]+}}**
        // CHECK1-NEXT: load i{{[0-9]+}}, i{{[0-9]+}}*
        // CHECK1-NEXT: getelementptr inbounds [[T]], [[T]]* {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 3
        // CHECK1-NEXT: load i{{[0-9]+}}**, i{{[0-9]+}}***
        // CHECK1-NEXT: load i{{[0-9]+}}*, i{{[0-9]+}}**
        // CHECK1-NEXT: store i{{[0-9]+}}
        //
        // CHECK1: getelementptr inbounds [[T]], [[T]]* {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
        // CHECK1-NEXT: load %struct.A*, %struct.A**
        // CHECK1-NEXT: getelementptr inbounds %struct.A, %struct.A*
        // CHECK1-NEXT: store float
        //
        // CHECK1: getelementptr inbounds [[T]], [[T]]* {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 2
        // CHECK1-NEXT: load %struct.A*, %struct.A**
        // CHECK1-NEXT: getelementptr inbounds %struct.A, %struct.A*
        // CHECK1-NEXT: store i8 99
        //
        // CHECK1-DAG: [[SIZE_ADDR_REF:%.*]] = getelementptr inbounds [[T]], [[T]]* {{.*}}, i{{.+}} 0, i{{.+}} 7
        // CHECK1-DAG: [[SIZE_ADDR:%.*]] = load i{{.+}}*, i{{.+}}** [[SIZE_ADDR_REF]]
        // CHECK1-DAG: [[SIZE:%.*]] = load i{{.+}}, i{{.+}}* [[SIZE_ADDR]]
        // CHECK1-DAG: [[PARAM_ARR_IDX:%.*]] = sub nsw i{{.+}} [[SIZE]], 1
        // CHECK1-DAG: [[PARAM_ARR_ADDR_REF:%.*]] = getelementptr inbounds [[T]], [[T]]* {{.*}}, i{{.+}} 0, i{{.+}} 6
        // CHECK1-DAG: [[PARAM_ARR_ADDR:%.*]] = load i{{.+}}**, i{{.+}}*** [[PARAM_ARR_ADDR_REF]]
        // CHECK1-DAG: [[PARAM_ARR:%.*]] = load i{{.+}}*, i{{.+}}** [[PARAM_ARR_ADDR]]
        // CHECK1-DAG: [[PARAM_ARR_SIZE_MINUS_1_ADDR:%.*]] = getelementptr inbounds i{{.+}}, i{{.+}}* [[PARAM_ARR]], i{{.*}}
        // CHECK1: store i{{.+}} 2, i{{.+}}* [[PARAM_ARR_SIZE_MINUS_1_ADDR]]
        //
        // CHECK1-DAG: [[Z_ADDR_REF:%.*]] = getelementptr inbounds [[T]], [[T]]* {{.*}}, i{{.+}} 0, i{{.+}} 2
        // CHECK1-DAG: [[Z_ADDR:%.*]] = load %struct.A*, %struct.A** [[Z_ADDR_REF]]
        // CHECK1-DAG: [[Z_A_ADDR:%.*]] = getelementptr inbounds %struct.A, %struct.A* [[Z_ADDR]], i{{.+}} 0, i{{.+}} 0
        // CHECK1-DAG: [[ARR_IDX_2:%.*]] = load i{{.+}}, i{{.+}}* [[Z_A_ADDR]]
        // CHECK1-DAG: [[ARR_ADDR_REF:%.*]] = getelementptr inbounds [[T]], [[T]]* {{.*}}, i{{.+}} 0, i{{.+}} 10
        // CHECK1-DAG: [[ARR_ADDR:%.*]] = load i{{.+}}*, i{{.+}}** [[ARR_ADDR_REF]]
        // CHECK1-DAG: [[ARR_IDX_1:%.*]] = mul {{.*}} 10
        // CHECK1-DAG: [[ARR_10_ADDR:%.*]] = getelementptr inbounds i{{.+}}, i{{.+}}* [[ARR_ADDR]], i{{.*}} [[ARR_IDX_1]]
        // CHECK1-DAG: [[ARR_10_Z_A_ADDR:%.*]] = getelementptr inbounds i{{.+}}, i{{.+}}* [[ARR_10_ADDR]], i{{.*}}
        // CHECK1: store i{{.+}} 12, i{{.+}}* [[ARR_10_Z_A_ADDR]]
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
  // CHECK2: store i{{[0-9]+}}* [[Z]]
  //
  // CHECK2: getelementptr inbounds %struct.anon
  // CHECK2-NEXT: getelementptr inbounds
  // CHECK2-NEXT: store i{{[0-9]+}}*
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
  // CHECK2: bitcast %struct.__block_byref_b*
  // CHECK2-NEXT: store i8*
  //
  // CHECK2: [[CapA:%[0-9a-z_.]*]] = getelementptr inbounds {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 7
  //
  // CHECK2: getelementptr inbounds %struct.anon{{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK2: load i{{[0-9]+}}*, i{{[0-9]+}}**
  // CHECK2: load i{{[0-9]+}}, i{{[0-9]+}}*
  // CHECK2: store i{{[0-9]+}} {{.*}}, i{{[0-9]+}}* [[CapA]]
  //
  // CHECK2: [[CapC:%[0-9a-z_.]*]] = getelementptr inbounds {{.*}}, i{{[0-9]+}} 0, i{{[0-9]+}} 8
  // CHECK2-NEXT: [[Val:%[0-9a-z_]*]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[C]]
  // CHECK2-NEXT: store i{{[0-9]+}} [[Val]], i{{[0-9]+}}* [[CapC]]
  //
  // CHECK2: bitcast %struct.__block_byref_d*
  // CHECK2-NEXT: store i8*
}
