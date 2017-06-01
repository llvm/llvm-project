// RUN: %clang_cc1 %s -std=c++11 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -std=c++11 -emit-pch -o %t
// RUN: %clang_cc1 %s -std=c++11 -include-pch %t -emit-llvm -o - | FileCheck %s

#ifndef HEADER
#define HEADER

typedef __INTPTR_TYPE__ intptr_t;

// CHECK-DAG:   [[CAP_TYPE1:%.+]] = type { [[INTPTR_T:i.+]], [[INTPTR_T]]*, [[INTPTR_T]]* }
// CHECK-DAG:   [[CAP_TYPE2:%.+]] = type { [[INTPTR_T]], [[INTPTR_T]]* }
// CHECK-DAG:   [[CAP_TYPE3:%.+]] = type { [[INTPTR_T]]*, [[INTPTR_T]], [[INTPTR_T]], [[INTPTR_T]]*, [[INTPTR_T]]* }
// CHECK-DAG:   [[CAP_TYPE4:%.+]] = type { [[INTPTR_T]]*, [[INTPTR_T]], [[INTPTR_T]]*, [[INTPTR_T]], [[INTPTR_T]]* }

// CHECK:       define {{.*}}void [[G:@.+]](
// CHECK:       [[N_ADDR:%.+]] = alloca [[INTPTR_T]]
// CHECK:       store [[INTPTR_T]] %{{.+}}, [[INTPTR_T]]* [[N_ADDR]]
// CHECK:       [[N_VAL:%.+]] = load [[INTPTR_T]], [[INTPTR_T]]* [[N_ADDR]]
// CHECK:       [[CAP_EXPR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE1]], [[CAP_TYPE1]]* [[CAP_ARG:%.+]], i{{.+}} 0, i{{.+}} 0
// CHECK:       store [[INTPTR_T]] [[N_VAL]], [[INTPTR_T]]* [[CAP_EXPR_REF]]
// CHECK:       [[CAP_BUFFER_ADDR:%.+]] = getelementptr inbounds [[CAP_TYPE1]], [[CAP_TYPE1]]* [[CAP_ARG]], i{{.+}} 0, i{{.+}} 1
// CHECK:       store [[INTPTR_T]]* %{{.+}}, [[INTPTR_T]]** [[CAP_BUFFER_ADDR]]
// CHECK:       [[CAP_N_REF:%.+]] = getelementptr inbounds [[CAP_TYPE1]], [[CAP_TYPE1]]* [[CAP_ARG:%.+]], i{{.+}} 0, i{{.+}} 2
// CHECK:       store [[INTPTR_T]]* [[N_ADDR]], [[INTPTR_T]]** [[CAP_N_REF]]
// CHECK:       call{{.*}} void [[G_LAMBDA:@.+]]([[CAP_TYPE1]]* [[CAP_ARG]])
// CHECK:       ret void
void g(intptr_t n) {
  intptr_t buffer[n];
  [&buffer, &n]() {
    __typeof(buffer) x;
  }();
}

// CHECK: void [[G_LAMBDA]]([[CAP_TYPE1]]*
// CHECK: [[THIS:%.+]] = load [[CAP_TYPE1]]*, [[CAP_TYPE1]]**
// CHECK: [[N_ADDR:%.+]] = getelementptr inbounds [[CAP_TYPE1]], [[CAP_TYPE1]]* [[THIS]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[N:%.+]] = load [[INTPTR_T]], [[INTPTR_T]]* [[N_ADDR]]
// CHECK: [[BUFFER_ADDR:%.+]] = getelementptr inbounds [[CAP_TYPE1]], [[CAP_TYPE1]]* [[THIS]], i{{.+}} 0, i{{.+}} 1
// CHECK: [[BUFFER:%.+]] = load [[INTPTR_T]]*, [[INTPTR_T]]** [[BUFFER_ADDR]]
// CHECK: call i{{.+}}* @llvm.stacksave()
// CHECK: alloca [[INTPTR_T]], [[INTPTR_T]] [[N]]
// CHECK: call void @llvm.stackrestore(
// CHECK: ret void

template <typename T>
void f(T n, T m) {
  intptr_t buffer[n + m];
  [&buffer]() {
    __typeof(buffer) x;
  }();
}

template <typename T>
intptr_t getSize(T);

template <typename T>
void b(intptr_t n, T arg) {
  typedef intptr_t ArrTy[getSize(arg)];
  ArrTy buffer2;
  ArrTy buffer1[n + arg];
  intptr_t a;
  [&]() {
    n = sizeof(buffer1[n]);
    [&](){
      n = sizeof(buffer2);
      n = sizeof(buffer1);
    }();
  }();
}

// CHECK-LABEL: @main
int main() {
  // CHECK:       call {{.*}}void [[G]]([[INTPTR_T]] [[INTPTR_T_ATTR:(signext )?]]1)
  g((intptr_t)1);
  // CHECK:       call {{.*}}void [[F_INT:@.+]]([[INTPTR_T]] [[INTPTR_T_ATTR]]1, [[INTPTR_T]] [[INTPTR_T_ATTR]]2)
  f((intptr_t)1, (intptr_t)2);
  // CHECK:       call {{.*}}void [[B_INT:@.+]]([[INTPTR_T]] [[INTPTR_T_ATTR]]12, [[INTPTR_T]] [[INTPTR_T_ATTR]]13)
  b((intptr_t)12, (intptr_t)13);
  // CHECK:       ret i32 0
  return 0;
}

// CHECK: define linkonce_odr {{.*}}void [[F_INT]]([[INTPTR_T]]
// CHECK: [[SIZE:%.+]] = add
// CHECK: call i{{.+}}* @llvm.stacksave()
// CHECK: [[BUFFER_ADDR:%.+]] = alloca [[INTPTR_T]], [[INTPTR_T]] [[SIZE]]
// CHECK: [[CAP_SIZE_REF:%.+]] = getelementptr inbounds [[CAP_TYPE2]], [[CAP_TYPE2]]* [[CAP_ARG:%.+]], i{{.+}} 0, i{{.+}} 0
// CHECK: store [[INTPTR_T]] [[SIZE]], [[INTPTR_T]]* [[CAP_SIZE_REF]]
// CHECK: [[CAP_BUFFER_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE2]], [[CAP_TYPE2]]* [[CAP_ARG]], i{{.+}} 0, i{{.+}} 1
// CHECK: store [[INTPTR_T]]* [[BUFFER_ADDR]], [[INTPTR_T]]** [[CAP_BUFFER_ADDR_REF]]
// CHECK: call{{.*}} void [[F_INT_LAMBDA:@.+]]([[CAP_TYPE2]]* [[CAP_ARG]])
// CHECK: call void @llvm.stackrestore(
// CHECK: ret void
// CHECK: void [[B_INT]]([[INTPTR_T]]
// CHECK: [[SIZE1:%.+]] = call {{.*}}[[INTPTR_T]]
// CHECK: call i{{.+}}* @llvm.stacksave()
// CHECK: [[BUFFER2_ADDR:%.+]] = alloca [[INTPTR_T]], [[INTPTR_T]] [[SIZE1]]
// CHECK: [[SIZE2:%.+]] = add
// CHECK: [[BUFFER1_ADDR:%.+]] = alloca [[INTPTR_T]], [[INTPTR_T]]
// CHECK: [[CAP_N_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[CAP_ARG:%.+]], i{{.+}} 0, i{{.+}} 0
// CHECK: store [[INTPTR_T]]* {{%.+}}, [[INTPTR_T]]** [[CAP_N_ADDR_REF]]
// CHECK: [[CAP_SIZE2_REF:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[CAP_ARG]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i{{[0-9]+}} [[SIZE2]], i{{[0-9]+}}* [[CAP_SIZE2_REF]]
// CHECK: [[CAP_SIZE1_REF:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[CAP_ARG]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i{{[0-9]+}} [[SIZE1]], i{{[0-9]+}}* [[CAP_SIZE1_REF]]
// CHECK: [[CAP_BUFFER1_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[CAP_ARG]], i{{.+}} 0, i{{.+}} 3
// CHECK: store [[INTPTR_T]]* [[BUFFER1_ADDR]], [[INTPTR_T]]** [[CAP_BUFFER1_ADDR_REF]]
// CHECK: [[CAP_BUFFER2_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[CAP_ARG]], i{{.+}} 0, i{{.+}} 4
// CHECK: store [[INTPTR_T]]* [[BUFFER2_ADDR]], [[INTPTR_T]]** [[CAP_BUFFER2_ADDR_REF]]
// CHECK: call{{.*}} void [[B_INT_LAMBDA:@.+]]([[CAP_TYPE3]]* [[CAP_ARG]])
// CHECK: call void @llvm.stackrestore(
// CHECK: ret void

// CHECK: define linkonce_odr{{.*}} void [[F_INT_LAMBDA]]([[CAP_TYPE2]]*
// CHECK: [[THIS:%.+]] = load [[CAP_TYPE2]]*, [[CAP_TYPE2]]**
// CHECK: [[SIZE_REF:%.+]] = getelementptr inbounds [[CAP_TYPE2]], [[CAP_TYPE2]]* [[THIS]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[SIZE:%.+]] = load [[INTPTR_T]], [[INTPTR_T]]* [[SIZE_REF]]
// CHECK: call i{{.+}}* @llvm.stacksave()
// CHECK: alloca [[INTPTR_T]], [[INTPTR_T]] [[SIZE]]
// CHECK: call void @llvm.stackrestore(
// CHECK: ret void

// CHECK: define linkonce_odr{{.*}} void [[B_INT_LAMBDA]]([[CAP_TYPE3]]*
// CHECK: [[SIZE2_REF:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[THIS:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[SIZE2:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SIZE2_REF]]
// CHECK: [[SIZE1_REF:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[SIZE1:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SIZE1_REF]]
// CHECK: [[BUFFER1_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: [[BUFFER1_ADDR:%.+]] = load [[INTPTR_T]]*, [[INTPTR_T]]** [[BUFFER1_ADDR_REF]]
// CHECK: [[N_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[N_ADDR:%.+]] = load [[INTPTR_T]]*, [[INTPTR_T]]** [[N_ADDR_REF]]
// CHECK: [[N:%.+]] = load [[INTPTR_T]], [[INTPTR_T]]* [[N_ADDR]]
// CHECK: [[ELEM_OFFSET:%.+]] = mul {{.*}} i{{[0-9]+}} [[N]], [[SIZE1]]
// CHECK: [[ELEM_ADDR:%.+]] = getelementptr inbounds [[INTPTR_T]], [[INTPTR_T]]* [[BUFFER1_ADDR]], i{{[0-9]+}} [[ELEM_OFFSET]]
// CHECK: [[SIZEOF:%.+]] = mul {{.*}} i{{[0-9]+}} {{[0-9]+}}, [[SIZE1]]
// CHECK: [[N_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[N_ADDR:%.+]] = load [[INTPTR_T]]*, [[INTPTR_T]]** [[N_ADDR_REF]]
// CHECK: store [[INTPTR_T]] {{%.+}}, [[INTPTR_T]]* [[N_ADDR]]
// CHECK: [[N_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE4]], [[CAP_TYPE4]]* [[CAP:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[N_ADDR_REF_ORIG:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[N_ADDR_ORIG:%.+]] = load [[INTPTR_T]]*, [[INTPTR_T]]** [[N_ADDR_REF_ORIG]]
// CHECK: store [[INTPTR_T]]* [[N_ADDR_ORIG]], [[INTPTR_T]]** [[N_ADDR_REF]]
// CHECK: [[SIZE1_REF:%.+]] = getelementptr inbounds [[CAP_TYPE4]], [[CAP_TYPE4]]* [[CAP]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: store i{{[0-9]+}} [[SIZE1]], i{{[0-9]+}}* [[SIZE1_REF]]
// CHECK: [[BUFFER2_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE4]], [[CAP_TYPE4]]* [[CAP]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[BUFFER2_ADDR_REF_ORIG:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
// CHECK: [[BUFFER2_ADDR_ORIG:%.+]] = load [[INTPTR_T]]*, [[INTPTR_T]]** [[BUFFER2_ADDR_REF_ORIG]]
// CHECK: store [[INTPTR_T]]* [[BUFFER2_ADDR_ORIG]], [[INTPTR_T]]** [[BUFFER2_ADDR_REF]]
// CHECK: [[SIZE2_REF:%.+]] = getelementptr inbounds [[CAP_TYPE4]], [[CAP_TYPE4]]* [[CAP]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: store i{{[0-9]+}} [[SIZE2]], i{{[0-9]+}}* [[SIZE2_REF]]
// CHECK: [[BUFFER1_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE4]], [[CAP_TYPE4]]* [[CAP]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
// CHECK: [[BUFFER1_ADDR_REF_ORIG:%.+]] = getelementptr inbounds [[CAP_TYPE3]], [[CAP_TYPE3]]* [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: [[BUFFER1_ADDR_ORIG:%.+]] = load [[INTPTR_T]]*, [[INTPTR_T]]** [[BUFFER1_ADDR_REF_ORIG]]
// CHECK: store [[INTPTR_T]]* [[BUFFER1_ADDR_ORIG]], [[INTPTR_T]]** [[BUFFER1_ADDR_REF]]
// CHECK: call{{.*}} void [[B_INT_LAMBDA_LAMBDA:@.+]]([[CAP_TYPE4]]* [[CAP]])
// CHECK: ret void

// CHECK: define linkonce_odr{{.*}} void [[B_INT_LAMBDA_LAMBDA]]([[CAP_TYPE4]]*
// CHECK: [[SIZE1_REF:%.+]] = getelementptr inbounds [[CAP_TYPE4]], [[CAP_TYPE4]]* [[THIS:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[SIZE1:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SIZE1_REF]]
// CHECK: [[SIZE2_REF:%.+]] = getelementptr inbounds [[CAP_TYPE4]], [[CAP_TYPE4]]* [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: [[SIZE2:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[SIZE2_REF]]
// CHECK: [[BUFFER2_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE4]], [[CAP_TYPE4]]* [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[BUFFER2_ADDR:%.+]] = load [[INTPTR_T]]*, [[INTPTR_T]]** [[BUFFER2_ADDR_REF]]
// CHECK: [[SIZEOF_BUFFER2:%.+]] = mul {{.*}} i{{[0-9]+}} {{[0-9]+}}, [[SIZE1]]
// CHECK: [[BUFFER1_ADDR_REF:%.+]] = getelementptr inbounds [[CAP_TYPE4]], [[CAP_TYPE4]]* [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
// CHECK: [[BUFFER1_ADDR:%.+]] = load [[INTPTR_T]]*, [[INTPTR_T]]** [[BUFFER1_ADDR_REF]]
// CHECK: [[MUL:%.+]] = mul {{.*}} i{{[0-9]+}} [[SIZE2]], [[SIZE1]]
// CHECK: mul {{.*}} i{{[0-9]+}} {{[0-9]+}}, [[MUL]]
// CHECK: ret void
#endif
