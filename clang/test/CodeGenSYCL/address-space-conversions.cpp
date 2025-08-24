// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
void bar(int &Data) {}
// CHECK-DAG: define{{.*}} spir_func void @[[RAW_REF:[a-zA-Z0-9_]+]](ptr addrspace(4) noundef align 4 dereferenceable(4) %
void bar2(int &Data) {}
// CHECK-DAG: define{{.*}} spir_func void @[[RAW_REF2:[a-zA-Z0-9_]+]](ptr addrspace(4) noundef align 4 dereferenceable(4) %
void bar(__attribute__((opencl_local)) int &Data) {}
// CHECK-DAG: define{{.*}} spir_func void [[LOC_REF:@[a-zA-Z0-9_]+]](ptr addrspace(3) noundef align 4 dereferenceable(4) %
void foo(int *Data) {}
// CHECK-DAG: define{{.*}} spir_func void @[[RAW_PTR:[a-zA-Z0-9_]+]](ptr addrspace(4) noundef %
void foo2(int *Data) {}
// CHECK-DAG: define{{.*}} spir_func void @[[RAW_PTR2:[a-zA-Z0-9_]+]](ptr addrspace(4) noundef %
void foo(__attribute__((opencl_local)) int *Data) {}
// CHECK-DAG: define{{.*}} spir_func void [[LOC_PTR:@[a-zA-Z0-9_]+]](ptr addrspace(3) noundef %

template <typename T>
void tmpl(T t) {}
// See Check Lines below.

[[clang::sycl_external]] void usages() {
  int *NoAS;
  // CHECK-DAG: [[NoAS:%[a-zA-Z0-9]+]] = alloca ptr addrspace(4)
  __attribute__((opencl_global)) int *GLOB;
  // CHECK-DAG: [[GLOB:%[a-zA-Z0-9]+]] = alloca ptr addrspace(1)
  __attribute__((opencl_local)) int *LOC;
  // CHECK-DAG: [[LOC:%[a-zA-Z0-9]+]] = alloca ptr addrspace(3)
  __attribute__((opencl_private)) int *PRIV;
  // CHECK-DAG: [[PRIV:%[a-zA-Z0-9]+]] = alloca ptr
  __attribute__((opencl_global_device)) int *GLOBDEVICE;
  // CHECK-DAG: [[GLOB_DEVICE:%[a-zA-Z0-9]+]] = alloca ptr addrspace(5)
  __attribute__((opencl_global_host)) int *GLOBHOST;
  // CHECK-DAG: [[GLOB_HOST:%[a-zA-Z0-9]+]] = alloca ptr addrspace(6)

  // CHECK-DAG: [[NoAS]].ascast = addrspacecast ptr [[NoAS]] to ptr addrspace(4)
  // CHECK-DAG: [[GLOB]].ascast = addrspacecast ptr [[GLOB]] to ptr addrspace(4)
  // CHECK-DAG: [[LOC]].ascast = addrspacecast ptr [[LOC]] to ptr addrspace(4)
  // CHECK-DAG: [[PRIV]].ascast = addrspacecast ptr [[PRIV]] to ptr addrspace(4)
  LOC = nullptr;
  // CHECK-DAG: store ptr addrspace(3) null, ptr addrspace(4) [[LOC]].ascast, align 8
  GLOB = nullptr;
  // CHECK-DAG: store ptr addrspace(1) null, ptr addrspace(4) [[GLOB]].ascast, align 8

  // Explicit conversions
  // From named address spaces to default address space
  // CHECK-DAG: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr addrspace(4) [[GLOB]].ascast
  // CHECK-DAG: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(1) [[GLOB_LOAD]] to ptr addrspace(4)
  // CHECK-DAG: store ptr addrspace(4) [[GLOB_CAST]], ptr addrspace(4) [[NoAS]].ascast
  NoAS = (int *)GLOB;
  // CHECK-DAG: [[LOC_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr addrspace(4) [[LOC]].ascast
  // CHECK-DAG: [[LOC_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(3) [[LOC_LOAD]] to ptr addrspace(4)
  // CHECK-DAG: store ptr addrspace(4) [[LOC_CAST]], ptr addrspace(4) [[NoAS]].ascast
  NoAS = (int *)LOC;
  // CHECK-DAG: [[PRIV_LOAD:%[a-zA-Z0-9]+]] = load ptr, ptr addrspace(4) [[PRIV]].ascast
  // CHECK-DAG: [[PRIV_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr [[PRIV_LOAD]] to ptr addrspace(4)
  // CHECK-DAG: store ptr addrspace(4) [[PRIV_CAST]], ptr addrspace(4) [[NoAS]].ascast
  NoAS = (int *)PRIV;
  // From default address space to named address space
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[NoAS]].ascast
  // CHECK-DAG: [[NoAS_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(4) [[NoAS_LOAD]] to ptr addrspace(1)
  // CHECK-DAG: store ptr addrspace(1) [[NoAS_CAST]], ptr addrspace(4) [[GLOB]].ascast
  GLOB = (__attribute__((opencl_global)) int *)NoAS;
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[NoAS]].ascast
  // CHECK-DAG: [[NoAS_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(4) [[NoAS_LOAD]] to ptr addrspace(3)
  // CHECK-DAG: store ptr addrspace(3) [[NoAS_CAST]], ptr addrspace(4) [[LOC]].ascast
  LOC = (__attribute__((opencl_local)) int *)NoAS;
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[NoAS]].ascast
  // CHECK-DAG: [[NoAS_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(4) [[NoAS_LOAD]] to ptr
  // CHECK-DAG: store ptr [[NoAS_CAST]], ptr addrspace(4) [[PRIV]].ascast
  PRIV = (__attribute__((opencl_private)) int *)NoAS;
  // From opencl_global_[host/device] address spaces to opencl_global
  // CHECK-DAG: [[GLOBDEVICE_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(5), ptr addrspace(4) [[GLOB_DEVICE]].ascast
  // CHECK-DAG: [[GLOBDEVICE_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(5) [[GLOBDEVICE_LOAD]] to ptr addrspace(1)
  // CHECK-DAG: store ptr addrspace(1) [[GLOBDEVICE_CAST]], ptr addrspace(4) [[GLOB]].ascast
  GLOB = (__attribute__((opencl_global)) int *)GLOBDEVICE;
  // CHECK-DAG: [[GLOBHOST_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(6), ptr addrspace(4) [[GLOB_HOST]].ascast
  // CHECK-DAG: [[GLOBHOST_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(6) [[GLOBHOST_LOAD]] to ptr addrspace(1)
  // CHECK-DAG: store ptr addrspace(1) [[GLOBHOST_CAST]], ptr addrspace(4) [[GLOB]].ascast
  GLOB = (__attribute__((opencl_global)) int *)GLOBHOST;

  bar(*GLOB);
  // CHECK-DAG: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr addrspace(4) [[GLOB]].ascast
  // CHECK-DAG: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(1) [[GLOB_LOAD]] to ptr addrspace(4)
  // CHECK-DAG: call spir_func void @[[RAW_REF]](ptr addrspace(4) noundef align 4 dereferenceable(4) [[GLOB_CAST]])
  bar2(*GLOB);
  // CHECK-DAG: [[GLOB_LOAD2:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr addrspace(4) [[GLOB]].ascast
  // CHECK-DAG: [[GLOB_CAST2:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(1) [[GLOB_LOAD2]] to ptr addrspace(4)
  // CHECK-DAG: call spir_func void @[[RAW_REF2]](ptr addrspace(4) noundef align 4 dereferenceable(4) [[GLOB_CAST2]])

  bar(*LOC);
  // CHECK-DAG: [[LOC_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr addrspace(4) [[LOC]].ascast
  // CHECK-DAG: call spir_func void [[LOC_REF]](ptr addrspace(3) noundef align 4 dereferenceable(4) [[LOC_LOAD]])
  bar2(*LOC);
  // CHECK-DAG: [[LOC_LOAD2:%[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr addrspace(4) [[LOC]].ascast
  // CHECK-DAG: [[LOC_CAST2:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(3) [[LOC_LOAD2]] to ptr addrspace(4)
  // CHECK-DAG: call spir_func void @[[RAW_REF2]](ptr addrspace(4) noundef align 4 dereferenceable(4) [[LOC_CAST2]])

  bar(*NoAS);
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[NoAS]].ascast
  // CHECK-DAG: call spir_func void @[[RAW_REF]](ptr addrspace(4) noundef align 4 dereferenceable(4) [[NoAS_LOAD]])
  bar2(*NoAS);
  // CHECK-DAG: [[NoAS_LOAD2:%[a-zA-Z0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[NoAS]].ascast
  // CHECK-DAG: call spir_func void @[[RAW_REF2]](ptr addrspace(4) noundef align 4 dereferenceable(4) [[NoAS_LOAD2]])

  foo(GLOB);
  // CHECK-DAG: [[GLOB_LOAD3:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr addrspace(4) [[GLOB]].ascast
  // CHECK-DAG: [[GLOB_CAST3:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(1) [[GLOB_LOAD3]] to ptr addrspace(4)
  // CHECK-DAG: call spir_func void @[[RAW_PTR]](ptr addrspace(4) noundef [[GLOB_CAST3]])
  foo2(GLOB);
  // CHECK-DAG: [[GLOB_LOAD4:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr addrspace(4) [[GLOB]].ascast
  // CHECK-DAG: [[GLOB_CAST4:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(1) [[GLOB_LOAD4]] to ptr addrspace(4)
  // CHECK-DAG: call spir_func void @[[RAW_PTR2]](ptr addrspace(4) noundef [[GLOB_CAST4]])
  foo(LOC);
  // CHECK-DAG: [[LOC_LOAD3:%[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr addrspace(4) [[LOC]].ascast
  // CHECK-DAG: call spir_func void [[LOC_PTR]](ptr addrspace(3) noundef [[LOC_LOAD3]])
  foo2(LOC);
  // CHECK-DAG: [[LOC_LOAD4:%[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr addrspace(4) [[LOC]].ascast
  // CHECK-DAG: [[LOC_CAST4:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(3) [[LOC_LOAD4]] to ptr addrspace(4)
  // CHECK-DAG: call spir_func void @[[RAW_PTR2]](ptr addrspace(4) noundef [[LOC_CAST4]])
  foo(NoAS);
  // CHECK-DAG: [[NoAS_LOAD3:%[a-zA-Z0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[NoAS]].ascast
  // CHECK-DAG: call spir_func void @[[RAW_PTR]](ptr addrspace(4) noundef [[NoAS_LOAD3]])
  foo2(NoAS);
  // CHECK-DAG: [[NoAS_LOAD4:%[a-zA-Z0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[NoAS]].ascast
  // CHECK-DAG: call spir_func void @[[RAW_PTR2]](ptr addrspace(4) noundef [[NoAS_LOAD4]])

  // Ensure that we still get 3 different template instantiations.
  tmpl(GLOB);
  // CHECK-DAG: [[GLOB_LOAD4:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr addrspace(4) [[GLOB]].ascast
  // CHECK-DAG: call spir_func void @_Z4tmplIPU3AS1iEvT_(ptr addrspace(1) noundef [[GLOB_LOAD4]])
  tmpl(LOC);
  // CHECK-DAG: [[LOC_LOAD5:%[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr addrspace(4) [[LOC]].ascast
  // CHECK-DAG: call spir_func void @_Z4tmplIPU3AS3iEvT_(ptr addrspace(3) noundef [[LOC_LOAD5]])
  tmpl(PRIV);
  // CHECK-DAG: [[PRIV_LOAD5:%[a-zA-Z0-9]+]] = load ptr, ptr addrspace(4) [[PRIV]].ascast
  // CHECK-DAG: call spir_func void @_Z4tmplIPU3AS0iEvT_(ptr noundef [[PRIV_LOAD5]])
  tmpl(NoAS);
  // CHECK-DAG: [[NoAS_LOAD5:%[a-zA-Z0-9]+]] = load ptr addrspace(4), ptr addrspace(4) [[NoAS]].ascast
  // CHECK-DAG: call spir_func void @_Z4tmplIPiEvT_(ptr addrspace(4) noundef [[NoAS_LOAD5]])
}

// CHECK-DAG: define linkonce_odr spir_func void @_Z4tmplIPU3AS1iEvT_(ptr addrspace(1) noundef %
// CHECK-DAG: define linkonce_odr spir_func void @_Z4tmplIPU3AS3iEvT_(ptr addrspace(3) noundef %
// CHECK-DAG: define linkonce_odr spir_func void @_Z4tmplIPU3AS0iEvT_(ptr noundef %
// CHECK-DAG: define linkonce_odr spir_func void @_Z4tmplIPiEvT_(ptr addrspace(4) noundef %
