// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
void bar(int &Data) {}
// CHECK-DAG: define {{.*}} void @[[RAW_REF:[a-zA-Z0-9_]+]](ptr noundef nonnull align 4 dereferenceable(4) %
void bar2(int &Data) {}
// CHECK-DAG: define {{.*}} void @[[RAW_REF2:[a-zA-Z0-9_]+]](ptr noundef nonnull align 4 dereferenceable(4) %
void bar(__attribute__((opencl_local)) int &Data) {}
// CHECK-DAG: define {{.*}} void @[[LOCAL_REF:[a-zA-Z0-9_]+]](ptr addrspace(3) noundef align 4 dereferenceable(4) %
void foo(int *Data) {}
// CHECK-DAG: define {{.*}} void @[[RAW_PTR:[a-zA-Z0-9_]+]](ptr noundef %
void foo2(int *Data) {}
// CHECK-DAG: define {{.*}} void @[[RAW_PTR2:[a-zA-Z0-9_]+]](ptr noundef %
void foo(__attribute__((opencl_local)) int *Data) {}
// CHECK-DAG: define {{.*}} void @[[LOC_PTR:[a-zA-Z0-9_]+]](ptr addrspace(3) noundef %

template <typename T>
void tmpl(T t) {}
// See Check Lines below.

[[clang::sycl_external]] void usages() {
  int *NoAS;
  // CHECK-DAG: [[NoAS:%[a-zA-Z0-9]+]] = alloca ptr, align 8, addrspace(5)
  __attribute__((opencl_global)) int *GLOB;
  // CHECK-DAG: [[GLOB:%[a-zA-Z0-9]+]] = alloca ptr addrspace(1), align 8, addrspace(5)
  __attribute__((opencl_local)) int *LOC;
  // CHECK-DAG: [[LOC:%[a-zA-Z0-9]+]] = alloca ptr addrspace(3), align 4, addrspace(5)
  __attribute__((opencl_private)) int *PRIV;
  // CHECK-DAG: [[PRIV:%[a-zA-Z0-9]+]] = alloca ptr addrspace(5), align 4, addrspace(5)
  __attribute__((opencl_global_device)) int *GLOBDEVICE;
  // CHECK-DAG: [[GLOB_DEVICE:%[a-zA-Z0-9]+]] = alloca ptr addrspace(1), align 8, addrspace(5)
  __attribute__((opencl_global_host)) int *GLOBHOST;
  // CHECK-DAG: [[GLOB_HOST:%[a-zA-Z0-9]+]] = alloca ptr addrspace(1), align 8, addrspace(5)
  LOC = nullptr;
  // CHECK-DAG: store ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr [[LOC]].ascast, align 4
  GLOB = nullptr;
  // CHECK-DAG: store ptr addrspace(1) null, ptr [[GLOB]].ascast, align 8
  NoAS = (int *)GLOB;
  // CHECK-DAG: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr [[GLOB]].ascast, align 8
  // CHECK-DAG: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(1) [[GLOB_LOAD]] to ptr
  // CHECK-DAG: store ptr [[GLOB_CAST]], ptr [[NoAS]].ascast, align 8
  NoAS = (int *)LOC;
  // CHECK-DAG: [[LOC_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr [[LOC]].ascast, align 4
  // CHECK-DAG: [[LOC_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(3) [[LOC_LOAD]] to ptr
  // CHECK-DAG: store ptr [[LOC_CAST]], ptr [[NoAS]].ascast, align 8
  NoAS = (int *)PRIV;
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(5), ptr [[PRIV]].ascast, align 4
  // CHECK-DAG: [[NoAS_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(5) [[NoAS_LOAD]] to ptr
  // CHECK-DAG: store ptr %5, ptr [[NoAS]].ascast, align 8
  GLOB = (__attribute__((opencl_global)) int *)NoAS;
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load ptr, ptr [[NoAS]].ascast, align 8
  // CHECK-DAG: [[NoAS_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr %6 to ptr addrspace(1)
  // CHECK-DAG: store ptr addrspace(1) %7, ptr [[GLOB]].ascast, align 8
  LOC = (__attribute__((opencl_local)) int *)NoAS;
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load ptr, ptr [[NoAS]].ascast, align 8
  // CHECK-DAG: [[NoAS_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr [[NoAS_LOAD]] to ptr addrspace(3)
  // CHECK-DAG: store ptr addrspace(3) %9, ptr [[LOC]].ascast, align 4
  PRIV = (__attribute__((opencl_private)) int *)NoAS;
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load ptr, ptr [[NoAS]].ascast, align 8
  // CHECK-DAG: [[NoAS_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr [[NoAS_LOAD]] to ptr addrspace(5)
  // CHECK-DAG: store ptr addrspace(5) [[NoAS_CAST]], ptr [[PRIV]].ascast, align 4
  GLOB = (__attribute__((opencl_global)) int *)GLOBDEVICE;
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr [[GLOB]]DEVICE.ascast, align 8
  // CHECK-DAG: store ptr addrspace(1) [[NoAS_LOAD]], ptr [[GLOB]].ascast, align 8
  GLOB = (__attribute__((opencl_global)) int *)GLOBHOST;
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr [[GLOB]]HOST.ascast, align 8
  // CHECK-DAG: tore ptr addrspace(1) [[NoAS_LOAD]], ptr [[GLOB]].ascast, align 8
  bar(*GLOB);
  // CHECK-DAG: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr [[GLOB]].ascast, align 8
  // CHECK-DAG: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(1) [[GLOB_LOAD]] to ptr
  // CHECK-DAG: call void @[[RAW_REF]](ptr noundef nonnull align 4 dereferenceable(4) [[GLOB_CAST]])
  bar2(*GLOB);
  // CHECK-DAG: [[GLOB_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr [[GLOB]].ascast, align 8
  // CHECK-DAG: [[GLOB_CAST:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(1) [[GLOB_LOAD]] to ptr
  // CHECK-DAG: call void @[[RAW_REF2]](ptr noundef nonnull align 4 dereferenceable(4) [[GLOB_CAST]])
  bar(*LOC);
  // CHECK-DAG: [[LOC_LOAD:%[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr [[LOC]].ascast, align 4
  // CHECK-DAG: call void @_Z3barRU3AS3i(ptr addrspace(3) noundef align 4 dereferenceable(4) [[LOC_LOAD]])
  bar2(*LOC);
  // CHECK-DAG: [[LOC_LOAD2:%[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr [[LOC]].ascast, align 4
  // CHECK-DAG: [[LOC_CAST2:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(3) [[LOC_LOAD2]] to ptr
  // CHECK-DAG: call void @_Z4bar2Ri(ptr noundef nonnull align 4 dereferenceable(4) [[LOC_CAST2]])
  bar(*NoAS);
  // CHECK-DAG: [[NoAS_LOAD:%[a-zA-Z0-9]+]] = load ptr, ptr [[NoAS]].ascast, align 8
  // CHECK-DAG: call void @_Z3barRi(ptr noundef nonnull align 4 dereferenceable(4) [[NoAS_LOAD]])
  bar2(*NoAS);
  // CHECK-DAG: [[NoAS_LOAD2:%[a-zA-Z0-9]+]] = load ptr, ptr [[NoAS]].ascast, align 8
  // CHECK-DAG: call void @_Z4bar2Ri(ptr noundef nonnull align 4 dereferenceable(4) [[NoAS_LOAD2]])
  foo(GLOB);
  // CHECK-DAG: [[GLOB_LOAD3:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr [[GLOB]].ascast, align 8
  // CHECK-DAG: [[GLOB_CAST3:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(1) [[GLOB_LOAD3]] to ptr
  // CHECK-DAG: call void @[[RAW_PTR]](ptr noundef [[GLOB_CAST3]])
   foo2(GLOB);
  // CHECK-DAG: [[GLOB_LOAD4:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr [[GLOB]].ascast, align 8
  // CHECK-DAG: [[GLOB_CAST4:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(1) [[GLOB_LOAD4]] to ptr
  // CHECK-DAG: call void @[[RAW_PTR2]](ptr noundef [[GLOB_CAST4]])
  foo(LOC);
  // CHECK-DAG: [[LOC_LOAD3:%[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr [[LOC]].ascast, align 4
  // CHECK-DAG: call void @[[LOC_PTR]](ptr addrspace(3) noundef [[LOC_LOAD3]])
  foo2(LOC);
  // CHECK-DAG: [[LOC_LOAD4:%[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr [[LOC]].ascast, align 4
  // CHECK-DAG: [[LOC_CAST4:%[a-zA-Z0-9]+]] = addrspacecast ptr addrspace(3) [[LOC_LOAD4]] to ptr
  // CHECK-DAG: call void @[[RAW_PTR2]](ptr noundef [[LOC_CAST4]])
  foo(NoAS);
  // CHECK-DAG: [[NoAS_LOAD3:%[a-zA-Z0-9]+]] = load ptr, ptr [[NoAS]].ascast, align 8
  // CHECK-DAG: call void @[[RAW_PTR]](ptr noundef [[NoAS_LOAD3]])
  foo2(NoAS);
  // CHECK-DAG: [[NoAS_LOAD4:%[a-zA-Z0-9]+]] = load ptr, ptr [[NoAS]].ascast, align 8
  // CHECK-DAG: call void @[[RAW_PTR2]](ptr noundef [[NoAS_LOAD4]])

  // Ensure that we still get 3 different template instantiations.
  tmpl(GLOB);
  // CHECK-DAG: [[GLOB_LOAD4:%[a-zA-Z0-9]+]] = load ptr addrspace(1), ptr [[GLOB]].ascast, align 8
  // CHECK-DAG: call void @_Z4tmplIPU3AS1iEvT_(ptr addrspace(1) noundef [[GLOB_LOAD4]])
  tmpl(LOC);
  // CHECK-DAG: [[LOC_LOAD5:%[a-zA-Z0-9]+]] = load ptr addrspace(3), ptr [[LOC]].ascast, align 4
  // CHECK-DAG: call void @_Z4tmplIPU3AS3iEvT_(ptr addrspace(3) noundef [[LOC_LOAD5]])
  tmpl(PRIV);
  // CHECK-DAG: [[PRIV_LOAD5:%[a-zA-Z0-9]+]] = load ptr addrspace(5), ptr [[PRIV]].ascast, align 4
  // CHECK-DAG: call void @_Z4tmplIPU3AS5iEvT_(ptr addrspace(5) noundef [[PRIV_LOAD5]])
  tmpl(NoAS);
  // CHECK-DAG: [[NoAS_LOAD5:%[a-zA-Z0-9]+]] = load ptr, ptr [[NoAS]].ascast, align 8
  // CHECK-DAG: call void @_Z4tmplIPiEvT_(ptr noundef [[NoAS_LOAD5]])
}

// CHECK-DAG: define linkonce_odr void @_Z4tmplIPU3AS1iEvT_(ptr addrspace(1) noundef %
// CHECK-DAG: define linkonce_odr void @_Z4tmplIPU3AS3iEvT_(ptr addrspace(3) noundef %
// CHECK-DAG: define linkonce_odr void @_Z4tmplIPU3AS5iEvT_(ptr addrspace(5) noundef %
// CHECK-DAG: define linkonce_odr void @_Z4tmplIPiEvT_(ptr noundef %

