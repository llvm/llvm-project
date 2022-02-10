// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
void bar(int &Data) {}
// CHECK-DAG: define{{.*}} spir_func void @[[RAW_REF:[a-zA-Z0-9_]+]](i32 addrspace(4)* noundef align 4 dereferenceable(4) %
void bar2(int &Data) {}
// CHECK-DAG: define{{.*}} spir_func void @[[RAW_REF2:[a-zA-Z0-9_]+]](i32 addrspace(4)* noundef align 4 dereferenceable(4) %
void bar(__attribute__((opencl_local)) int &Data) {}
// CHECK-DAG: define{{.*}} spir_func void [[LOC_REF:@[a-zA-Z0-9_]+]](i32 addrspace(3)* noundef align 4 dereferenceable(4) %
void foo(int *Data) {}
// CHECK-DAG: define{{.*}} spir_func void @[[RAW_PTR:[a-zA-Z0-9_]+]](i32 addrspace(4)* noundef %
void foo2(int *Data) {}
// CHECK-DAG: define{{.*}} spir_func void @[[RAW_PTR2:[a-zA-Z0-9_]+]](i32 addrspace(4)* noundef %
void foo(__attribute__((opencl_local)) int *Data) {}
// CHECK-DAG: define{{.*}} spir_func void [[LOC_PTR:@[a-zA-Z0-9_]+]](i32 addrspace(3)* noundef %

template <typename T>
void tmpl(T t) {}
// See Check Lines below.

void usages() {
  // CHECK-DAG: [[GLOB:%[a-zA-Z0-9]+]] = alloca i32 addrspace(1)*
  // CHECK-DAG: [[GLOB]].ascast = addrspacecast i32 addrspace(1)** [[GLOB]] to i32 addrspace(1)* addrspace(4)*
  __attribute__((opencl_global)) int *GLOB;
  // CHECK-DAG: [[LOC:%[a-zA-Z0-9]+]] = alloca i32 addrspace(3)*
  // CHECK-DAG: [[LOC]].ascast = addrspacecast i32 addrspace(3)** [[LOC]] to i32 addrspace(3)* addrspace(4)*
  __attribute__((opencl_local)) int *LOC;
  // CHECK-DAG: [[NoAS:%[a-zA-Z0-9]+]] = alloca i32 addrspace(4)*
  // CHECK-DAG: [[NoAS]].ascast = addrspacecast i32 addrspace(4)** [[NoAS]] to i32 addrspace(4)* addrspace(4)*
  int *NoAS;
  // CHECK-DAG: [[PRIV:%[a-zA-Z0-9]+]] = alloca i32*
  // CHECK-DAG: [[PRIV]].ascast = addrspacecast i32** [[PRIV]] to i32* addrspace(4)*
  __attribute__((opencl_private)) int *PRIV;
  // CHECK-DAG: [[GLOB_DEVICE:%[a-zA-Z0-9]+]] = alloca i32 addrspace(5)*
  __attribute__((opencl_global_device)) int *GLOBDEVICE;
  // CHECK-DAG: [[GLOB_HOST:%[a-zA-Z0-9]+]] = alloca i32 addrspace(6)*
  __attribute__((opencl_global_host)) int *GLOBHOST;

  // Explicit conversions
  // From named address spaces to default address space
  // CHECK-DAG: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[GLOB]].ascast
  // CHECK-DAG: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD]] to i32 addrspace(4)*
  // CHECK-DAG: store i32 addrspace(4)* [[GLOB_CAST]], i32 addrspace(4)* addrspace(4)* [[NoAS]].ascast
  NoAS = (int *)GLOB;
  // CHECK-DAG: [[LOC_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* [[LOC]].ascast
  // CHECK-DAG: [[LOC_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOC_LOAD]] to i32 addrspace(4)*
  // CHECK-DAG: store i32 addrspace(4)* [[LOC_CAST]], i32 addrspace(4)* addrspace(4)* [[NoAS]].ascast
  NoAS = (int *)LOC;
  // CHECK-DAG: [[PRIV_LOAD:%[a-zA-Z0-9]+]] = load i32*, i32* addrspace(4)* [[PRIV]].ascast
  // CHECK-DAG: [[PRIV_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32* [[PRIV_LOAD]] to i32 addrspace(4)*
  // CHECK-DAG: store i32 addrspace(4)* [[PRIV_CAST]], i32 addrspace(4)* addrspace(4)* [[NoAS]].ascast
  NoAS = (int *)PRIV;
  // From default address space to named address space
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]].ascast
  // CHECK-DAG: [[NoAS_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(4)* [[NoAS_LOAD]] to i32 addrspace(1)*
  // CHECK-DAG: store i32 addrspace(1)* [[NoAS_CAST]], i32 addrspace(1)* addrspace(4)* [[GLOB]].ascast
  GLOB = (__attribute__((opencl_global)) int *)NoAS;
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]].ascast
  // CHECK-DAG: [[NoAS_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(4)* [[NoAS_LOAD]] to i32 addrspace(3)*
  // CHECK-DAG: store i32 addrspace(3)* [[NoAS_CAST]], i32 addrspace(3)* addrspace(4)* [[LOC]].ascast
  LOC = (__attribute__((opencl_local)) int *)NoAS;
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]].ascast
  // CHECK-DAG: [[NoAS_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(4)* [[NoAS_LOAD]] to i32*
  // CHECK-DAG: store i32* [[NoAS_CAST]], i32* addrspace(4)* [[PRIV]].ascast
  PRIV = (__attribute__((opencl_private)) int *)NoAS;
  // From opencl_global_[host/device] address spaces to opencl_global
  // CHECK-DAG: [[GLOBDEVICE_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(5)*, i32 addrspace(5)* addrspace(4)* [[GLOB_DEVICE]].ascast
  // CHECK-DAG: [[GLOBDEVICE_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(5)* [[GLOBDEVICE_LOAD]] to i32 addrspace(1)*
  // CHECK-DAG: store i32 addrspace(1)* [[GLOBDEVICE_CAST]], i32 addrspace(1)* addrspace(4)* [[GLOB]].ascast
  GLOB = (__attribute__((opencl_global)) int *)GLOBDEVICE;
  // CHECK-DAG: [[GLOBHOST_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(6)*, i32 addrspace(6)* addrspace(4)* [[GLOB_HOST]].ascast
  // CHECK-DAG: [[GLOBHOST_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(6)* [[GLOBHOST_LOAD]] to i32 addrspace(1)*
  // CHECK-DAG: store i32 addrspace(1)* [[GLOBHOST_CAST]], i32 addrspace(1)* addrspace(4)* [[GLOB]].ascast
  GLOB = (__attribute__((opencl_global)) int *)GLOBHOST;

  bar(*GLOB);
  // CHECK-DAG: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[GLOB]].ascast
  // CHECK-DAG: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* noundef align 4 dereferenceable(4) [[GLOB_CAST]])
  bar2(*GLOB);
  // CHECK-DAG: [[GLOB_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[GLOB]].ascast
  // CHECK-DAG: [[GLOB_CAST2:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD2]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* noundef align 4 dereferenceable(4) [[GLOB_CAST2]])

  bar(*LOC);
  // CHECK-DAG: [[LOC_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* [[LOC]].ascast
  // CHECK-DAG: call spir_func void [[LOC_REF]](i32 addrspace(3)* noundef align 4 dereferenceable(4) [[LOC_LOAD]])
  bar2(*LOC);
  // CHECK-DAG: [[LOC_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* [[LOC]].ascast
  // CHECK-DAG: [[LOC_CAST2:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOC_LOAD2]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* noundef align 4 dereferenceable(4) [[LOC_CAST2]])

  bar(*NoAS);
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]].ascast
  // CHECK-DAG: call spir_func void @[[RAW_REF]](i32 addrspace(4)* noundef align 4 dereferenceable(4) [[NoAS_LOAD]])
  bar2(*NoAS);
  // CHECK-DAG: [[NoAS_LOAD2:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]].ascast
  // CHECK-DAG: call spir_func void @[[RAW_REF2]](i32 addrspace(4)* noundef align 4 dereferenceable(4) [[NoAS_LOAD2]])

  foo(GLOB);
  // CHECK-DAG: [[GLOB_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[GLOB]].ascast
  // CHECK-DAG: [[GLOB_CAST3:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD3]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_PTR]](i32 addrspace(4)* noundef [[GLOB_CAST3]])
  foo2(GLOB);
  // CHECK-DAG: [[GLOB_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[GLOB]].ascast
  // CHECK-DAG: [[GLOB_CAST4:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(1)* [[GLOB_LOAD4]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_PTR2]](i32 addrspace(4)* noundef [[GLOB_CAST4]])
  foo(LOC);
  // CHECK-DAG: [[LOC_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* [[LOC]].ascast
  // CHECK-DAG: call spir_func void [[LOC_PTR]](i32 addrspace(3)* noundef [[LOC_LOAD3]])
  foo2(LOC);
  // CHECK-DAG: [[LOC_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* [[LOC]].ascast
  // CHECK-DAG: [[LOC_CAST4:%[a-zA-Z0-9]+]] = addrspacecast i32 addrspace(3)* [[LOC_LOAD4]] to i32 addrspace(4)*
  // CHECK-DAG: call spir_func void @[[RAW_PTR2]](i32 addrspace(4)* noundef [[LOC_CAST4]])
  foo(NoAS);
  // CHECK-DAG: [[NoAS_LOAD3:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]].ascast
  // CHECK-DAG: call spir_func void @[[RAW_PTR]](i32 addrspace(4)* noundef [[NoAS_LOAD3]])
  foo2(NoAS);
  // CHECK-DAG: [[NoAS_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]].ascast
  // CHECK-DAG: call spir_func void @[[RAW_PTR2]](i32 addrspace(4)* noundef [[NoAS_LOAD4]])

  // Ensure that we still get 3 different template instantiations.
  tmpl(GLOB);
  // CHECK-DAG: [[GLOB_LOAD4:%[a-zA-Z0-9]+]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* [[GLOB]].ascast
  // CHECK-DAG: call spir_func void @_Z4tmplIPU3AS1iEvT_(i32 addrspace(1)* noundef [[GLOB_LOAD4]])
  tmpl(LOC);
  // CHECK-DAG: [[LOC_LOAD5:%[a-zA-Z0-9]+]] = load i32 addrspace(3)*, i32 addrspace(3)* addrspace(4)* [[LOC]].ascast
  // CHECK-DAG: call spir_func void @_Z4tmplIPU3AS3iEvT_(i32 addrspace(3)* noundef [[LOC_LOAD5]])
  tmpl(PRIV);
  // CHECK-DAG: [[PRIV_LOAD5:%[a-zA-Z0-9]+]] = load i32*, i32* addrspace(4)* [[PRIV]].ascast
  // CHECK-DAG: call spir_func void @_Z4tmplIPU3AS0iEvT_(i32* noundef [[PRIV_LOAD5]])
  tmpl(NoAS);
  // CHECK-DAG: [[NoAS_LOAD5:%[a-zA-Z0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* [[NoAS]].ascast
  // CHECK-DAG: call spir_func void @_Z4tmplIPiEvT_(i32 addrspace(4)* noundef [[NoAS_LOAD5]])
}

// CHECK-DAG: define linkonce_odr spir_func void @_Z4tmplIPU3AS1iEvT_(i32 addrspace(1)* noundef %
// CHECK-DAG: define linkonce_odr spir_func void @_Z4tmplIPU3AS3iEvT_(i32 addrspace(3)* noundef %
// CHECK-DAG: define linkonce_odr spir_func void @_Z4tmplIPU3AS0iEvT_(i32* noundef %
// CHECK-DAG: define linkonce_odr spir_func void @_Z4tmplIPiEvT_(i32 addrspace(4)* noundef %
