// RUN: %clang_cc1 %s -std=c++11 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -std=c++11 -emit-pch -o %t
// RUN: %clang_cc1 %s -std=c++11 -include-pch %t -emit-llvm -o - | FileCheck %s

#ifndef HEADER
#define HEADER

typedef __INTPTR_TYPE__ intptr_t;

// CHECK-DAG:   [[CAP_TYPE1:%.+]] = type { [[INTPTR_T:i.+]], ptr, ptr }
// CHECK-DAG:   [[CAP_TYPE2:%.+]] = type { [[INTPTR_T]], ptr }
// CHECK-DAG:   [[CAP_TYPE3:%.+]] = type { ptr, [[INTPTR_T]], [[INTPTR_T]], ptr, ptr }
// CHECK-DAG:   [[CAP_TYPE4:%.+]] = type { ptr, [[INTPTR_T]], ptr, [[INTPTR_T]], ptr }

// CHECK:       define {{.*}}void [[G:@.+]](
// CHECK:       [[N_ADDR:%.+]] = alloca [[INTPTR_T]]
// CHECK:       store [[INTPTR_T]] %{{.+}}, ptr [[N_ADDR]]
// CHECK:       [[N_VAL:%.+]] = load [[INTPTR_T]], ptr [[N_ADDR]]
// CHECK:       [[CAP_EXPR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE1]], ptr [[CAP_ARG:%.+]], i{{.+}} 0, i{{.+}} 0
// CHECK:       store [[INTPTR_T]] [[N_VAL]], ptr [[CAP_EXPR_REF]]
// CHECK:       [[CAP_BUFFER_ADDR:%.+]] = getelementptr inbounds nuw [[CAP_TYPE1]], ptr [[CAP_ARG]], i{{.+}} 0, i{{.+}} 1
// CHECK:       store ptr %{{.+}}, ptr [[CAP_BUFFER_ADDR]]
// CHECK:       [[CAP_N_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE1]], ptr [[CAP_ARG:%.+]], i{{.+}} 0, i{{.+}} 2
// CHECK:       store ptr [[N_ADDR]], ptr [[CAP_N_REF]]
// CHECK:       call{{.*}} void [[G_LAMBDA:@.+]](ptr {{[^,]*}} [[CAP_ARG]])
// CHECK:       ret void
void g(intptr_t n) {
  intptr_t buffer[n];
  [&buffer, &n]() {
    __typeof(buffer) x;
  }();
}

// CHECK: void [[G_LAMBDA]](ptr
// CHECK: [[THIS:%.+]] = load ptr, ptr
// CHECK: [[N_ADDR:%.+]] = getelementptr inbounds nuw [[CAP_TYPE1]], ptr [[THIS]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[N:%.+]] = load [[INTPTR_T]], ptr [[N_ADDR]]
// CHECK: [[BUFFER_ADDR:%.+]] = getelementptr inbounds nuw [[CAP_TYPE1]], ptr [[THIS]], i{{.+}} 0, i{{.+}} 1
// CHECK: [[BUFFER:%.+]] = load ptr, ptr [[BUFFER_ADDR]]
// CHECK: call ptr @llvm.stacksave.p0()
// CHECK: alloca [[INTPTR_T]], [[INTPTR_T]] [[N]]
// CHECK: call void @llvm.stackrestore.p0(
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
  // CHECK:       call {{.*}}void [[G]]([[INTPTR_T]] noundef [[INTPTR_T_ATTR:(signext )?]]1)
  g((intptr_t)1);
  // CHECK:       call {{.*}}void [[F_INT:@.+]]([[INTPTR_T]] noundef [[INTPTR_T_ATTR]]1, [[INTPTR_T]] noundef [[INTPTR_T_ATTR]]2)
  f((intptr_t)1, (intptr_t)2);
  // CHECK:       call {{.*}}void [[B_INT:@.+]]([[INTPTR_T]] noundef [[INTPTR_T_ATTR]]12, [[INTPTR_T]] noundef [[INTPTR_T_ATTR]]13)
  b((intptr_t)12, (intptr_t)13);
  // CHECK:       ret i32 0
  return 0;
}

// CHECK: define linkonce_odr {{.*}}void [[F_INT]]([[INTPTR_T]]
// CHECK: [[SIZE:%.+]] = add
// CHECK: call ptr @llvm.stacksave.p0()
// CHECK: [[BUFFER_ADDR:%.+]] = alloca [[INTPTR_T]], [[INTPTR_T]] [[SIZE]]
// CHECK: [[CAP_SIZE_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE2]], ptr [[CAP_ARG:%.+]], i{{.+}} 0, i{{.+}} 0
// CHECK: store [[INTPTR_T]] [[SIZE]], ptr [[CAP_SIZE_REF]]
// CHECK: [[CAP_BUFFER_ADDR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE2]], ptr [[CAP_ARG]], i{{.+}} 0, i{{.+}} 1
// CHECK: store ptr [[BUFFER_ADDR]], ptr [[CAP_BUFFER_ADDR_REF]]
// CHECK: call{{.*}} void [[F_INT_LAMBDA:@.+]](ptr {{[^,]*}} [[CAP_ARG]])
// CHECK: call void @llvm.stackrestore.p0(
// CHECK: ret void
// CHECK: void [[B_INT]]([[INTPTR_T]]
// CHECK: [[SIZE1:%.+]] = call {{.*}}[[INTPTR_T]]
// CHECK: call ptr @llvm.stacksave.p0()
// CHECK: [[BUFFER2_ADDR:%.+]] = alloca [[INTPTR_T]], [[INTPTR_T]] [[SIZE1]]
// CHECK: [[SIZE2:%.+]] = add
// CHECK: [[BUFFER1_ADDR:%.+]] = alloca [[INTPTR_T]], [[INTPTR_T]]
// CHECK: [[CAP_N_ADDR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[CAP_ARG:%.+]], i{{.+}} 0, i{{.+}} 0
// CHECK: store ptr {{%.+}}, ptr [[CAP_N_ADDR_REF]]
// CHECK: [[CAP_SIZE2_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[CAP_ARG]], i{{.+}} 0, i{{.+}} 1
// CHECK: store i{{[0-9]+}} [[SIZE2]], ptr [[CAP_SIZE2_REF]]
// CHECK: [[CAP_SIZE1_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[CAP_ARG]], i{{.+}} 0, i{{.+}} 2
// CHECK: store i{{[0-9]+}} [[SIZE1]], ptr [[CAP_SIZE1_REF]]
// CHECK: [[CAP_BUFFER1_ADDR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[CAP_ARG]], i{{.+}} 0, i{{.+}} 3
// CHECK: store ptr [[BUFFER1_ADDR]], ptr [[CAP_BUFFER1_ADDR_REF]]
// CHECK: [[CAP_BUFFER2_ADDR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[CAP_ARG]], i{{.+}} 0, i{{.+}} 4
// CHECK: store ptr [[BUFFER2_ADDR]], ptr [[CAP_BUFFER2_ADDR_REF]]
// CHECK: call{{.*}} void [[B_INT_LAMBDA:@.+]](ptr {{[^,]*}} [[CAP_ARG]])
// CHECK: call void @llvm.stackrestore.p0(
// CHECK: ret void

// CHECK: define linkonce_odr{{.*}} void [[F_INT_LAMBDA]](ptr
// CHECK: [[THIS:%.+]] = load ptr, ptr
// CHECK: [[SIZE_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE2]], ptr [[THIS]], i{{.+}} 0, i{{.+}} 0
// CHECK: [[SIZE:%.+]] = load [[INTPTR_T]], ptr [[SIZE_REF]]
// CHECK: call ptr @llvm.stacksave.p0()
// CHECK: alloca [[INTPTR_T]], [[INTPTR_T]] [[SIZE]]
// CHECK: call void @llvm.stackrestore.p0(
// CHECK: ret void

// CHECK: define linkonce_odr{{.*}} void [[B_INT_LAMBDA]](ptr
// CHECK: [[SIZE2_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[THIS:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[SIZE2:%.+]] = load i{{[0-9]+}}, ptr [[SIZE2_REF]]
// CHECK: [[SIZE1_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[SIZE1:%.+]] = load i{{[0-9]+}}, ptr [[SIZE1_REF]]
// CHECK: [[BUFFER1_ADDR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: [[BUFFER1_ADDR:%.+]] = load ptr, ptr [[BUFFER1_ADDR_REF]]
// CHECK: [[N_ADDR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[N_ADDR:%.+]] = load ptr, ptr [[N_ADDR_REF]]
// CHECK: [[N:%.+]] = load [[INTPTR_T]], ptr [[N_ADDR]]
// CHECK: [[ELEM_OFFSET:%.+]] = mul {{.*}} i{{[0-9]+}} [[N]], [[SIZE1]]
// CHECK: [[ELEM_ADDR:%.+]] = getelementptr inbounds [[INTPTR_T]], ptr [[BUFFER1_ADDR]], i{{[0-9]+}} [[ELEM_OFFSET]]
// CHECK: [[SIZEOF:%.+]] = mul {{.*}} i{{[0-9]+}} {{[0-9]+}}, [[SIZE1]]
// CHECK: [[N_ADDR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[N_ADDR:%.+]] = load ptr, ptr [[N_ADDR_REF]]
// CHECK: store [[INTPTR_T]] {{%.+}}, ptr [[N_ADDR]]
// CHECK: [[N_ADDR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE4]], ptr [[CAP:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[N_ADDR_REF_ORIG:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK: [[N_ADDR_ORIG:%.+]] = load ptr, ptr [[N_ADDR_REF_ORIG]]
// CHECK: store ptr [[N_ADDR_ORIG]], ptr [[N_ADDR_REF]]
// CHECK: [[SIZE1_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE4]], ptr [[CAP]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: store i{{[0-9]+}} [[SIZE1]], ptr [[SIZE1_REF]]
// CHECK: [[BUFFER2_ADDR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE4]], ptr [[CAP]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[BUFFER2_ADDR_REF_ORIG:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
// CHECK: [[BUFFER2_ADDR_ORIG:%.+]] = load ptr, ptr [[BUFFER2_ADDR_REF_ORIG]]
// CHECK: store ptr [[BUFFER2_ADDR_ORIG]], ptr [[BUFFER2_ADDR_REF]]
// CHECK: [[SIZE2_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE4]], ptr [[CAP]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: store i{{[0-9]+}} [[SIZE2]], ptr [[SIZE2_REF]]
// CHECK: [[BUFFER1_ADDR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE4]], ptr [[CAP]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
// CHECK: [[BUFFER1_ADDR_REF_ORIG:%.+]] = getelementptr inbounds nuw [[CAP_TYPE3]], ptr [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: [[BUFFER1_ADDR_ORIG:%.+]] = load ptr, ptr [[BUFFER1_ADDR_REF_ORIG]]
// CHECK: store ptr [[BUFFER1_ADDR_ORIG]], ptr [[BUFFER1_ADDR_REF]]
// CHECK: call{{.*}} void [[B_INT_LAMBDA_LAMBDA:@.+]](ptr {{[^,]*}} [[CAP]])
// CHECK: ret void

// CHECK: define linkonce_odr{{.*}} void [[B_INT_LAMBDA_LAMBDA]](ptr
// CHECK: [[SIZE1_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE4]], ptr [[THIS:%.+]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK: [[SIZE1:%.+]] = load i{{[0-9]+}}, ptr [[SIZE1_REF]]
// CHECK: [[SIZE2_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE4]], ptr [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
// CHECK: [[SIZE2:%.+]] = load i{{[0-9]+}}, ptr [[SIZE2_REF]]
// CHECK: [[BUFFER2_ADDR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE4]], ptr [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
// CHECK: [[BUFFER2_ADDR:%.+]] = load ptr, ptr [[BUFFER2_ADDR_REF]]
// CHECK: [[SIZEOF_BUFFER2:%.+]] = mul {{.*}} i{{[0-9]+}} {{[0-9]+}}, [[SIZE1]]
// CHECK: [[BUFFER1_ADDR_REF:%.+]] = getelementptr inbounds nuw [[CAP_TYPE4]], ptr [[THIS]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
// CHECK: [[BUFFER1_ADDR:%.+]] = load ptr, ptr [[BUFFER1_ADDR_REF]]
// CHECK: [[MUL:%.+]] = mul {{.*}} i{{[0-9]+}} [[SIZE2]], [[SIZE1]]
// CHECK: mul {{.*}} i{{[0-9]+}} {{[0-9]+}}, [[MUL]]
// CHECK: ret void
#endif
