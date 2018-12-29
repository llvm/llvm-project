// RUN: %clang -cc1 -triple spir64-unknown-linux-sycldevice  -std=c++11 -fsycl-is-device -emit-llvm -x c++ %s -o - | FileCheck %s
void bar(int & Data) {}
// CHECK: define spir_func void @[[RAW_REF:[a-zA-Z0-9_]+]](i32* dereferenceable(4) %
void bar2(int & Data) {}
// CHECK: define spir_func void @[[RAW_REF2:[a-zA-Z0-9_]+]](i32* dereferenceable(4) %
void bar(__local int &Data) {}
// CHECK: define spir_func void [[LOC_REF:@[a-zA-Z0-9_]+]](i32 addrspace(3)* dereferenceable(4) %
void foo(int * Data) {}
// CHECK: define spir_func void @[[RAW_PTR:[a-zA-Z0-9_]+]](i32* %
void foo2(int * Data) {}
// CHECK: define spir_func void @[[RAW_PTR2:[a-zA-Z0-9_]+]](i32* %
void foo(__attribute__((address_space(3))) int * Data) {}
// CHECK: define spir_func void [[LOC_PTR:@[a-zA-Z0-9_]+]](i32 addrspace(3)* %

template<typename T>
void tmpl(T t){}
// See Check Lines below.

void usages() {
  // CHECK: [[GLOB:%[a-zA-Z0-9]+]] = alloca i32 addrspace(1)*
  __attribute__((address_space(1))) int *GLOB;
  // CHECK: [[LOC:%[a-zA-Z0-9]+]] = alloca i32 addrspace(3)*
  __local int *LOC;
  // CHECK: [[NoAS:%[a-zA-Z0-9]+]] = alloca i32*
  int *NoAS;

  bar(*GLOB);
  // CHECK: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB]]
  // CHECK: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[GLOB_CAST]])
  bar2(*GLOB);
  // CHECK: [[GLOB_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB]]
  // CHECK: [[GLOB_CAST2:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD2]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_REF2]](i32 addrspace(4)* [[GLOB_CAST2]])

  bar(*LOC);
  // CHECK: [[LOC_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOC]]
  // CHECK: call spir_func void [[LOC_REF]](i32 addrspace(3)* dereferenceable(4) [[LOC_LOAD]])
  bar2(*LOC);
  // CHECK: [[LOC_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOC]]
  // CHECK: [[LOC_CAST2:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOC_LOAD2]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_REF2]](i32 addrspace(4)* [[LOC_CAST2]])

  bar(*NoAS);
  // CHECK: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load i32*, i32** [[NoAS]]
  // CHECK: call spir_func void @[[RAW_REF]](i32* dereferenceable(4) [[NoAS_LOAD]])
  bar2(*NoAS);
  // CHECK: [[NoAS_LOAD2:%[a-zA-Z0-9]+]] = load i32*, i32** [[NoAS]]
  // CHECK: call spir_func void @[[RAW_REF2]](i32* dereferenceable(4) [[NoAS_LOAD2]])

  foo(GLOB);
  // CHECK: [[GLOB_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB]]
  // CHECK: [[GLOB_CAST3:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD3]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_PTR]](i32 addrspace(4)* [[GLOB_CAST3]])
  foo2(GLOB);
  // CHECK: [[GLOB_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB]]
  // CHECK: [[GLOB_CAST4:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD4]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_PTR2]](i32 addrspace(4)* [[GLOB_CAST4]])
  foo(LOC);
  // CHECK: [[LOC_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOC]]
  // CHECK: call spir_func void [[LOC_PTR]](i32 addrspace(3)* [[LOC_LOAD3]])
  foo2(LOC);
  // CHECK: [[LOC_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOC]]
  // CHECK: [[LOC_CAST4:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOC_LOAD4]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_PTR2]](i32 addrspace(4)* [[LOC_CAST4]])
  foo(NoAS);
  // CHECK: [[NoAS_LOAD3:%[a-zA-Z0-9]+]] = load i32*, i32** [[NoAS]]
  // CHECK: call spir_func void @[[RAW_PTR]](i32* [[NoAS_LOAD3]])
  foo2(NoAS);
  // CHECK: [[NoAS_LOAD4:%[a-zA-Z0-9]+]] = load i32*, i32** [[NoAS]]
  // CHECK: call spir_func void @[[RAW_PTR2]](i32* [[NoAS_LOAD4]])

  // Ensure that we still get 3 different template instantiations.
  tmpl(GLOB);
  // CHECK: [[GLOB_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB]]
  // CHECK: call spir_func void [[GLOB_TMPL:@[a-zA-Z0-9_]+]](i32 addrspace(1)* [[GLOB_LOAD4]])
  tmpl(LOC);
  // CHECK: [[LOC_LOAD5:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOC]]
  // CHECK: call spir_func void [[LOC_TMPL:@[a-zA-Z0-9_]+]](i32 addrspace(3)* [[LOC_LOAD5]])
  tmpl(NoAS);
  // CHECK: [[NoAS_LOAD5:%[a-zA-Z0-9]+]] = load i32*, i32** [[NoAS]]
  // CHECK: call spir_func void [[AS0_TMPL:@[a-zA-Z0-9_]+]](i32* [[NoAS_LOAD5]])
}

// CHECK: define linkonce_odr spir_func void [[GLOB_TMPL]](i32 addrspace(1)* %
// CHECK: define linkonce_odr spir_func void [[LOC_TMPL]](i32 addrspace(3)* %
// CHECK: define linkonce_odr spir_func void [[AS0_TMPL]](i32* %

void usages2() {
  __attribute__((address_space(0))) int *PRIV_NUM;
  // CHECK: [[PRIV_NUM:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(5)*
  __attribute__((address_space(5))) int *PRIV_NUM2;
  // CHECK: [[PRIV_NUM2:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(5)*
  __private int *PRIV;
  // CHECK: [[PRIV:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(5)*
  __attribute__((address_space(1))) int *GLOB_NUM;
  // CHECK: [[GLOB_NUM:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(1)*
  __global int *GLOB;
  // CHECK: [[GLOB:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(1)*
  __attribute__((address_space(2))) int *CONST_NUM;
  // CHECK: [[CONST_NUM:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(2)*
  __constant int *CONST;
  // CHECK: [[CONST:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(2)*
  __attribute__((address_space(3))) int *LOCAL_NUM;
  // CHECK: [[LOCAL_NUM:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(3)*
  __local int *LOCAL;
  // CHECK: [[LOCAL:%[a-zA-Z0-9_]+]] = alloca i32 addrspace(3)*

  bar(*PRIV_NUM);
  // CHECK: [[PRIV_NUM_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)** [[PRIV_NUM]]
  // CHECK: [[PRIV_NUM_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(5)* [[PRIV_NUM_LOAD]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[PRIV_NUM_CAST]])
  bar(*PRIV_NUM2);
  // CHECK: [[PRIV_NUM2_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)** [[PRIV_NUM2]]
  // CHECK: [[PRIV_NUM2_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(5)* [[PRIV_NUM2_LOAD]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[PRIV_NUM2_CAST]])
  bar(*PRIV);
  // CHECK: [[PRIV_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)** [[PRIV]]
  // CHECK: [[PRIV_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(5)* [[PRIV_LOAD]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[PRIV_CAST]])
  bar(*GLOB_NUM);
  // CHECK: [[GLOB_NUM_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB_NUM]]
  // CHECK: [[GLOB_NUM_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_NUM_LOAD]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[GLOB_NUM_CAST]])
  bar(*GLOB);
  // CHECK: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)** [[GLOB]]
  // CHECK: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[GLOB_CAST]])
  bar(*CONST_NUM);
  // CHECK: [[CONST_NUM_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(2)*, i32 addrspace(2)** [[CONST_NUM]]
  // CHECK: [[CONST_NUM_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(2)* [[CONST_NUM_LOAD]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[CONST_NUM_CAST]])
  bar(*CONST);
  // CHECK: [[CONST_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(2)*, i32 addrspace(2)** [[CONST]]
  // CHECK: [[CONST_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(2)* [[CONST_LOAD]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_REF]](i32 addrspace(4)* [[CONST_CAST]])
  bar2(*LOCAL_NUM);
  // CHECK: [[LOCAL_NUM_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOCAL_NUM]]
  // CHECK: [[LOCAL_NUM_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOCAL_NUM_LOAD]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_REF2]](i32 addrspace(4)* [[LOCAL_NUM_CAST]])
  bar2(*LOCAL);
  // CHECK: [[LOCAL_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)** [[LOCAL]]
  // CHECK: [[LOCAL_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOCAL_LOAD]] to i32 addrspace(4)*
  // CHECK: call spir_func void @new.[[RAW_REF2]](i32 addrspace(4)* [[LOCAL_CAST]])
}

// CHECK: define spir_func void @new.[[RAW_REF]](i32 addrspace(4)* dereferenceable(4)

// CHECK: define spir_func void @new.[[RAW_REF2]](i32 addrspace(4)* dereferenceable(4)

// CHECK: define spir_func void @new.[[RAW_PTR2]](i32 addrspace(4)*

// CHECK: define spir_func void @new.[[RAW_PTR]](i32 addrspace(4)*

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}
int main() {
  kernel_single_task<class fake_kernel>([]() { usages();usages2(); });
  return 0;
}

