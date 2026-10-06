// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -Wno-deprecated-attributes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -Wno-deprecated-attributes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -Wno-deprecated-attributes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// Port of clang/test/CodeGenSYCL/address-space-conversions.cpp. Local
// variables and parameters are allocated in the private address space and
// accessed through the generic address space.
//
// TODO(cir): Calls are missing the spir_func calling convention and null
// pointers to named address spaces are not emitted as an addrspacecast of the
// generic null pointer.

void bar(int &Data) {}
void bar2(int &Data) {}
void bar(int [[clang::sycl_local]] &Data) {}
void foo(int *Data) {}
void foo2(int *Data) {}
void foo(int [[clang::sycl_local]] *Data) {}

template <typename T>
void tmpl(T t) {}

[[clang::sycl_external]] void usages() {
  int *NoAS;
  int [[clang::sycl_global]] *GLOB;
  int [[clang::sycl_local]] *LOC;
  int [[clang::sycl_private]] *PRIV;
  int __attribute__((opencl_global_device)) *GLOBDEVICE;
  int __attribute__((opencl_global_host)) *GLOBHOST;

  LOC = nullptr;
  GLOB = nullptr;

  // Explicit conversions
  // From named address spaces to default address space
  NoAS = (int *)GLOB;
  NoAS = (int *)LOC;
  NoAS = (int *)PRIV;
  // From default address space to named address space
  GLOB = (int [[clang::sycl_global]] *)NoAS;
  LOC = (int [[clang::sycl_local]] *)NoAS;
  PRIV = (int [[clang::sycl_private]] *)NoAS;
  // From opencl_global_[host/device] address spaces to sycl_global
  GLOB = (int [[clang::sycl_global]] *)GLOBDEVICE;
  GLOB = (int [[clang::sycl_global]] *)GLOBHOST;

  bar(*GLOB);
  bar2(*GLOB);

  bar(*LOC);
  bar2(*LOC);

  bar(*NoAS);
  bar2(*NoAS);

  foo(GLOB);
  foo2(GLOB);
  foo(LOC);
  foo2(LOC);
  foo(NoAS);
  foo2(NoAS);

  // Ensure that we still get 3 different template instantiations.
  tmpl(GLOB);
  tmpl(LOC);
  tmpl(PRIV);
  tmpl(NoAS);
}


// CIR-LABEL: cir.func {{.*}}@_Z6usagesv(
// CIR-NEXT: %[[NOAS:.*]] = cir.alloca "NoAS" align(8) : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>>
// CIR-NEXT: %[[GLOB:.*]] = cir.alloca "GLOB" align(8) : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>
// CIR-NEXT: %[[LOC:.*]] = cir.alloca "LOC" align(8) : !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>>
// CIR-NEXT: %[[PRIV:.*]] = cir.alloca "PRIV" align(8) : !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT: %[[GLOBDEVICE:.*]] = cir.alloca "GLOBDEVICE" align(8) : !cir.ptr<!cir.ptr<!s32i, target_address_space(5)>>
// CIR-NEXT: %[[GLOBHOST:.*]] = cir.alloca "GLOBHOST" align(8) : !cir.ptr<!cir.ptr<!s32i, target_address_space(6)>>
// CIR-NEXT: %[[GLOBHOST_ASCAST:.*]] = cir.cast address_space %[[GLOBHOST]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(6)>> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(6)>, target_address_space(4)>
// CIR-NEXT: %[[GLOBDEVICE_ASCAST:.*]] = cir.cast address_space %[[GLOBDEVICE]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(5)>> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(5)>, target_address_space(4)>
// CIR-NEXT: %[[PRIV_ASCAST:.*]] = cir.cast address_space %[[PRIV]] : !cir.ptr<!cir.ptr<!s32i>> -> !cir.ptr<!cir.ptr<!s32i>, target_address_space(4)>
// CIR-NEXT: %[[LOC_ASCAST:.*]] = cir.cast address_space %[[LOC]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>, target_address_space(4)>
// CIR-NEXT: %[[GLOB_ASCAST:.*]] = cir.cast address_space %[[GLOB]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>, target_address_space(4)>
// CIR-NEXT: %[[NOAS_ASCAST:.*]] = cir.cast address_space %[[NOAS]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>

// CIR-LABEL: cir.func {{.*}}@_Z3barRi(
// CIR-NEXT: %[[DATA:.*]] = cir.alloca "Data" align(8) init const : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>>
// CIR-NEXT: %[[DATA_ASCAST:.*]] = cir.cast address_space %[[DATA]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>
// CIR-NEXT: cir.store %arg0, %[[DATA_ASCAST]] : !cir.ptr<!s32i, target_address_space(4)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>
// CIR-NEXT: cir.return

// CIR-LABEL: cir.func {{.*}}@_Z3barRU3AS3i(
// CIR-NEXT: %[[DATA:.*]] = cir.alloca "Data" align(8) init const : !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>>
// CIR-NEXT: %[[DATA_ASCAST:.*]] = cir.cast address_space %[[DATA]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>, target_address_space(4)>
// CIR-NEXT: cir.store %arg0, %[[DATA_ASCAST]] : !cir.ptr<!s32i, target_address_space(3)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>, target_address_space(4)>
// CIR-NEXT: cir.return

// CIR-LABEL: cir.func {{.*}}@_Z3fooPi(
// CIR-NEXT: %[[DATA:.*]] = cir.alloca "Data" align(8) init : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>>
// CIR-NEXT: %[[DATA_ASCAST:.*]] = cir.cast address_space %[[DATA]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>
// CIR-NEXT: cir.store %arg0, %[[DATA_ASCAST]] : !cir.ptr<!s32i, target_address_space(4)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(4)>, target_address_space(4)>
// CIR-NEXT: cir.return

// CIR-LABEL: cir.func {{.*}}@_Z3fooPU3AS3i(
// CIR-NEXT: %[[DATA:.*]] = cir.alloca "Data" align(8) init : !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>>
// CIR-NEXT: %[[DATA_ASCAST:.*]] = cir.cast address_space %[[DATA]] : !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>> -> !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>, target_address_space(4)>
// CIR-NEXT: cir.store %arg0, %[[DATA_ASCAST]] : !cir.ptr<!s32i, target_address_space(3)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>, target_address_space(4)>
// CIR-NEXT: cir.return

// LLVM-CIR-LABEL: define {{.*}}spir_func void @_Z6usagesv(
// LLVM-CIR: %[[NOAS:.*]] = alloca ptr addrspace(4), align 8
// LLVM-CIR-NEXT: %[[GLOB:.*]] = alloca ptr addrspace(1), align 8
// LLVM-CIR-NEXT: %[[LOC:.*]] = alloca ptr addrspace(3), align 8
// LLVM-CIR-NEXT: %[[PRIV:.*]] = alloca ptr, align 8
// LLVM-CIR-NEXT: %[[GLOBDEVICE:.*]] = alloca ptr addrspace(5), align 8
// LLVM-CIR-NEXT: %[[GLOBHOST:.*]] = alloca ptr addrspace(6), align 8
// LLVM-CIR-NEXT: %[[GLOBHOST_ASCAST:.*]] = addrspacecast ptr %[[GLOBHOST]] to ptr addrspace(4)
// LLVM-CIR-NEXT: %[[GLOBDEVICE_ASCAST:.*]] = addrspacecast ptr %[[GLOBDEVICE]] to ptr addrspace(4)
// LLVM-CIR-NEXT: %[[PRIV_ASCAST:.*]] = addrspacecast ptr %[[PRIV]] to ptr addrspace(4)
// LLVM-CIR-NEXT: %[[LOC_ASCAST:.*]] = addrspacecast ptr %[[LOC]] to ptr addrspace(4)
// LLVM-CIR-NEXT: %[[GLOB_ASCAST:.*]] = addrspacecast ptr %[[GLOB]] to ptr addrspace(4)
// LLVM-CIR-NEXT: %[[NOAS_ASCAST:.*]] = addrspacecast ptr %[[NOAS]] to ptr addrspace(4)
// LLVM-CIR-NEXT: store ptr addrspace(3) null, ptr addrspace(4) %[[LOC_ASCAST]], align 8
// LLVM-CIR-NEXT: store ptr addrspace(1) null, ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP0:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP1:.*]] = addrspacecast ptr addrspace(1) %[[TMP0]] to ptr addrspace(4)
// LLVM-CIR-NEXT: store ptr addrspace(4) %[[TMP1]], ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP2:.*]] = load ptr addrspace(3), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP3:.*]] = addrspacecast ptr addrspace(3) %[[TMP2]] to ptr addrspace(4)
// LLVM-CIR-NEXT: store ptr addrspace(4) %[[TMP3]], ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP4:.*]] = load ptr, ptr addrspace(4) %[[PRIV_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP5:.*]] = addrspacecast ptr %[[TMP4]] to ptr addrspace(4)
// LLVM-CIR-NEXT: store ptr addrspace(4) %[[TMP5]], ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP6:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP7:.*]] = addrspacecast ptr addrspace(4) %[[TMP6]] to ptr addrspace(1)
// LLVM-CIR-NEXT: store ptr addrspace(1) %[[TMP7]], ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP8:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP9:.*]] = addrspacecast ptr addrspace(4) %[[TMP8]] to ptr addrspace(3)
// LLVM-CIR-NEXT: store ptr addrspace(3) %[[TMP9]], ptr addrspace(4) %[[LOC_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP10:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP11:.*]] = addrspacecast ptr addrspace(4) %[[TMP10]] to ptr
// LLVM-CIR-NEXT: store ptr %[[TMP11]], ptr addrspace(4) %[[PRIV_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP12:.*]] = load ptr addrspace(5), ptr addrspace(4) %[[GLOBDEVICE_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP13:.*]] = addrspacecast ptr addrspace(5) %[[TMP12]] to ptr addrspace(1)
// LLVM-CIR-NEXT: store ptr addrspace(1) %[[TMP13]], ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP14:.*]] = load ptr addrspace(6), ptr addrspace(4) %[[GLOBHOST_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP15:.*]] = addrspacecast ptr addrspace(6) %[[TMP14]] to ptr addrspace(1)
// LLVM-CIR-NEXT: store ptr addrspace(1) %[[TMP15]], ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP16:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP17:.*]] = addrspacecast ptr addrspace(1) %[[TMP16]] to ptr addrspace(4)
// LLVM-CIR-NEXT: call void @_Z3barRi(ptr addrspace(4) noundef align 4 dereferenceable(4) %[[TMP17]])
// LLVM-CIR-NEXT: %[[TMP18:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP19:.*]] = addrspacecast ptr addrspace(1) %[[TMP18]] to ptr addrspace(4)
// LLVM-CIR-NEXT: call void @_Z4bar2Ri(ptr addrspace(4) noundef align 4 dereferenceable(4) %[[TMP19]])
// LLVM-CIR-NEXT: %[[TMP20:.*]] = load ptr addrspace(3), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// LLVM-CIR-NEXT: call void @_Z3barRU3AS3i(ptr addrspace(3) noundef align 4 dereferenceable(4) %[[TMP20]])
// LLVM-CIR-NEXT: %[[TMP21:.*]] = load ptr addrspace(3), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP22:.*]] = addrspacecast ptr addrspace(3) %[[TMP21]] to ptr addrspace(4)
// LLVM-CIR-NEXT: call void @_Z4bar2Ri(ptr addrspace(4) noundef align 4 dereferenceable(4) %[[TMP22]])
// LLVM-CIR-NEXT: %[[TMP23:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// LLVM-CIR-NEXT: call void @_Z3barRi(ptr addrspace(4) noundef align 4 dereferenceable(4) %[[TMP23]])
// LLVM-CIR-NEXT: %[[TMP24:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// LLVM-CIR-NEXT: call void @_Z4bar2Ri(ptr addrspace(4) noundef align 4 dereferenceable(4) %[[TMP24]])
// LLVM-CIR-NEXT: %[[TMP25:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP26:.*]] = addrspacecast ptr addrspace(1) %[[TMP25]] to ptr addrspace(4)
// LLVM-CIR-NEXT: call void @_Z3fooPi(ptr addrspace(4) noundef %[[TMP26]])
// LLVM-CIR-NEXT: %[[TMP27:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP28:.*]] = addrspacecast ptr addrspace(1) %[[TMP27]] to ptr addrspace(4)
// LLVM-CIR-NEXT: call void @_Z4foo2Pi(ptr addrspace(4) noundef %[[TMP28]])
// LLVM-CIR-NEXT: %[[TMP29:.*]] = load ptr addrspace(3), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// LLVM-CIR-NEXT: call void @_Z3fooPU3AS3i(ptr addrspace(3) noundef %[[TMP29]])
// LLVM-CIR-NEXT: %[[TMP30:.*]] = load ptr addrspace(3), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// LLVM-CIR-NEXT: %[[TMP31:.*]] = addrspacecast ptr addrspace(3) %[[TMP30]] to ptr addrspace(4)
// LLVM-CIR-NEXT: call void @_Z4foo2Pi(ptr addrspace(4) noundef %[[TMP31]])
// LLVM-CIR-NEXT: %[[TMP32:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// LLVM-CIR-NEXT: call void @_Z3fooPi(ptr addrspace(4) noundef %[[TMP32]])
// LLVM-CIR-NEXT: %[[TMP33:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// LLVM-CIR-NEXT: call void @_Z4foo2Pi(ptr addrspace(4) noundef %[[TMP33]])
// LLVM-CIR-NEXT: %[[TMP34:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// LLVM-CIR-NEXT: call void @_Z4tmplIPU3AS1iEvT_(ptr addrspace(1) noundef %[[TMP34]])
// LLVM-CIR-NEXT: %[[TMP35:.*]] = load ptr addrspace(3), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// LLVM-CIR-NEXT: call void @_Z4tmplIPU3AS3iEvT_(ptr addrspace(3) noundef %[[TMP35]])
// LLVM-CIR-NEXT: %[[TMP36:.*]] = load ptr, ptr addrspace(4) %[[PRIV_ASCAST]], align 8
// LLVM-CIR-NEXT: call void @_Z4tmplIPU3AS0iEvT_(ptr noundef %[[TMP36]])
// LLVM-CIR-NEXT: %[[TMP37:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// LLVM-CIR-NEXT: call void @_Z4tmplIPiEvT_(ptr addrspace(4) noundef %[[TMP37]])
// LLVM-CIR-NEXT: ret void

// OGCG-LABEL: define {{.*}}spir_func void @_Z6usagesv(
// OGCG: %[[NOAS:.*]] = alloca ptr addrspace(4), align 8
// OGCG-NEXT: %[[GLOB:.*]] = alloca ptr addrspace(1), align 8
// OGCG-NEXT: %[[LOC:.*]] = alloca ptr addrspace(3), align 8
// OGCG-NEXT: %[[PRIV:.*]] = alloca ptr, align 8
// OGCG-NEXT: %[[GLOBDEVICE:.*]] = alloca ptr addrspace(5), align 8
// OGCG-NEXT: %[[GLOBHOST:.*]] = alloca ptr addrspace(6), align 8
// OGCG-NEXT: %[[NOAS_ASCAST:.*]] = addrspacecast ptr %[[NOAS]] to ptr addrspace(4)
// OGCG-NEXT: %[[GLOB_ASCAST:.*]] = addrspacecast ptr %[[GLOB]] to ptr addrspace(4)
// OGCG-NEXT: %[[LOC_ASCAST:.*]] = addrspacecast ptr %[[LOC]] to ptr addrspace(4)
// OGCG-NEXT: %[[PRIV_ASCAST:.*]] = addrspacecast ptr %[[PRIV]] to ptr addrspace(4)
// OGCG-NEXT: %[[GLOBDEVICE_ASCAST:.*]] = addrspacecast ptr %[[GLOBDEVICE]] to ptr addrspace(4)
// OGCG-NEXT: %[[GLOBHOST_ASCAST:.*]] = addrspacecast ptr %[[GLOBHOST]] to ptr addrspace(4)
// OGCG-NEXT: store ptr addrspace(3) addrspacecast (ptr addrspace(4) null to ptr addrspace(3)), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// OGCG-NEXT: store ptr addrspace(1) addrspacecast (ptr addrspace(4) null to ptr addrspace(1)), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// OGCG-NEXT: %[[TMP0:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// OGCG-NEXT: %[[TMP1:.*]] = addrspacecast ptr addrspace(1) %[[TMP0]] to ptr addrspace(4)
// OGCG-NEXT: store ptr addrspace(4) %[[TMP1]], ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// OGCG-NEXT: %[[TMP2:.*]] = load ptr addrspace(3), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// OGCG-NEXT: %[[TMP3:.*]] = addrspacecast ptr addrspace(3) %[[TMP2]] to ptr addrspace(4)
// OGCG-NEXT: store ptr addrspace(4) %[[TMP3]], ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// OGCG-NEXT: %[[TMP4:.*]] = load ptr, ptr addrspace(4) %[[PRIV_ASCAST]], align 8
// OGCG-NEXT: %[[TMP5:.*]] = addrspacecast ptr %[[TMP4]] to ptr addrspace(4)
// OGCG-NEXT: store ptr addrspace(4) %[[TMP5]], ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// OGCG-NEXT: %[[TMP6:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// OGCG-NEXT: %[[TMP7:.*]] = addrspacecast ptr addrspace(4) %[[TMP6]] to ptr addrspace(1)
// OGCG-NEXT: store ptr addrspace(1) %[[TMP7]], ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// OGCG-NEXT: %[[TMP8:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// OGCG-NEXT: %[[TMP9:.*]] = addrspacecast ptr addrspace(4) %[[TMP8]] to ptr addrspace(3)
// OGCG-NEXT: store ptr addrspace(3) %[[TMP9]], ptr addrspace(4) %[[LOC_ASCAST]], align 8
// OGCG-NEXT: %[[TMP10:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// OGCG-NEXT: %[[TMP11:.*]] = addrspacecast ptr addrspace(4) %[[TMP10]] to ptr
// OGCG-NEXT: store ptr %[[TMP11]], ptr addrspace(4) %[[PRIV_ASCAST]], align 8
// OGCG-NEXT: %[[TMP12:.*]] = load ptr addrspace(5), ptr addrspace(4) %[[GLOBDEVICE_ASCAST]], align 8
// OGCG-NEXT: %[[TMP13:.*]] = addrspacecast ptr addrspace(5) %[[TMP12]] to ptr addrspace(1)
// OGCG-NEXT: store ptr addrspace(1) %[[TMP13]], ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// OGCG-NEXT: %[[TMP14:.*]] = load ptr addrspace(6), ptr addrspace(4) %[[GLOBHOST_ASCAST]], align 8
// OGCG-NEXT: %[[TMP15:.*]] = addrspacecast ptr addrspace(6) %[[TMP14]] to ptr addrspace(1)
// OGCG-NEXT: store ptr addrspace(1) %[[TMP15]], ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// OGCG-NEXT: %[[TMP16:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// OGCG-NEXT: %[[TMP17:.*]] = addrspacecast ptr addrspace(1) %[[TMP16]] to ptr addrspace(4)
// OGCG-NEXT: call spir_func void @_Z3barRi(ptr addrspace(4) noundef align 4 dereferenceable(4) %[[TMP17]])
// OGCG-NEXT: %[[TMP18:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// OGCG-NEXT: %[[TMP19:.*]] = addrspacecast ptr addrspace(1) %[[TMP18]] to ptr addrspace(4)
// OGCG-NEXT: call spir_func void @_Z4bar2Ri(ptr addrspace(4) noundef align 4 dereferenceable(4) %[[TMP19]])
// OGCG-NEXT: %[[TMP20:.*]] = load ptr addrspace(3), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// OGCG-NEXT: call spir_func void @_Z3barRU3AS3i(ptr addrspace(3) noundef align 4 dereferenceable(4) %[[TMP20]])
// OGCG-NEXT: %[[TMP21:.*]] = load ptr addrspace(3), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// OGCG-NEXT: %[[TMP22:.*]] = addrspacecast ptr addrspace(3) %[[TMP21]] to ptr addrspace(4)
// OGCG-NEXT: call spir_func void @_Z4bar2Ri(ptr addrspace(4) noundef align 4 dereferenceable(4) %[[TMP22]])
// OGCG-NEXT: %[[TMP23:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// OGCG-NEXT: call spir_func void @_Z3barRi(ptr addrspace(4) noundef align 4 dereferenceable(4) %[[TMP23]])
// OGCG-NEXT: %[[TMP24:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// OGCG-NEXT: call spir_func void @_Z4bar2Ri(ptr addrspace(4) noundef align 4 dereferenceable(4) %[[TMP24]])
// OGCG-NEXT: %[[TMP25:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// OGCG-NEXT: %[[TMP26:.*]] = addrspacecast ptr addrspace(1) %[[TMP25]] to ptr addrspace(4)
// OGCG-NEXT: call spir_func void @_Z3fooPi(ptr addrspace(4) noundef %[[TMP26]])
// OGCG-NEXT: %[[TMP27:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// OGCG-NEXT: %[[TMP28:.*]] = addrspacecast ptr addrspace(1) %[[TMP27]] to ptr addrspace(4)
// OGCG-NEXT: call spir_func void @_Z4foo2Pi(ptr addrspace(4) noundef %[[TMP28]])
// OGCG-NEXT: %[[TMP29:.*]] = load ptr addrspace(3), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// OGCG-NEXT: call spir_func void @_Z3fooPU3AS3i(ptr addrspace(3) noundef %[[TMP29]])
// OGCG-NEXT: %[[TMP30:.*]] = load ptr addrspace(3), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// OGCG-NEXT: %[[TMP31:.*]] = addrspacecast ptr addrspace(3) %[[TMP30]] to ptr addrspace(4)
// OGCG-NEXT: call spir_func void @_Z4foo2Pi(ptr addrspace(4) noundef %[[TMP31]])
// OGCG-NEXT: %[[TMP32:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// OGCG-NEXT: call spir_func void @_Z3fooPi(ptr addrspace(4) noundef %[[TMP32]])
// OGCG-NEXT: %[[TMP33:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// OGCG-NEXT: call spir_func void @_Z4foo2Pi(ptr addrspace(4) noundef %[[TMP33]])
// OGCG-NEXT: %[[TMP34:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[GLOB_ASCAST]], align 8
// OGCG-NEXT: call spir_func void @_Z4tmplIPU3AS1iEvT_(ptr addrspace(1) noundef %[[TMP34]])
// OGCG-NEXT: %[[TMP35:.*]] = load ptr addrspace(3), ptr addrspace(4) %[[LOC_ASCAST]], align 8
// OGCG-NEXT: call spir_func void @_Z4tmplIPU3AS3iEvT_(ptr addrspace(3) noundef %[[TMP35]])
// OGCG-NEXT: %[[TMP36:.*]] = load ptr, ptr addrspace(4) %[[PRIV_ASCAST]], align 8
// OGCG-NEXT: call spir_func void @_Z4tmplIPU3AS0iEvT_(ptr noundef %[[TMP36]])
// OGCG-NEXT: %[[TMP37:.*]] = load ptr addrspace(4), ptr addrspace(4) %[[NOAS_ASCAST]], align 8
// OGCG-NEXT: call spir_func void @_Z4tmplIPiEvT_(ptr addrspace(4) noundef %[[TMP37]])
// OGCG-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z3barRi(
// LLVM-SAME: ptr addrspace(4) noundef align 4 dereferenceable(4) %[[DATA:.*]])
// LLVM: %[[DATA_ADDR:.*]] = alloca ptr addrspace(4), align 8
// LLVM-NEXT: %[[DATA_ADDR_ASCAST:.*]] = addrspacecast ptr %[[DATA_ADDR]] to ptr addrspace(4)
// LLVM-NEXT: store ptr addrspace(4) %[[DATA]], ptr addrspace(4) %[[DATA_ADDR_ASCAST]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z4bar2Ri(
// LLVM-SAME: ptr addrspace(4) noundef align 4 dereferenceable(4) %[[DATA:.*]])
// LLVM: %[[DATA_ADDR:.*]] = alloca ptr addrspace(4), align 8
// LLVM-NEXT: %[[DATA_ADDR_ASCAST:.*]] = addrspacecast ptr %[[DATA_ADDR]] to ptr addrspace(4)
// LLVM-NEXT: store ptr addrspace(4) %[[DATA]], ptr addrspace(4) %[[DATA_ADDR_ASCAST]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z3barRU3AS3i(
// LLVM-SAME: ptr addrspace(3) noundef align 4 dereferenceable(4) %[[DATA:.*]])
// LLVM: %[[DATA_ADDR:.*]] = alloca ptr addrspace(3), align 8
// LLVM-NEXT: %[[DATA_ADDR_ASCAST:.*]] = addrspacecast ptr %[[DATA_ADDR]] to ptr addrspace(4)
// LLVM-NEXT: store ptr addrspace(3) %[[DATA]], ptr addrspace(4) %[[DATA_ADDR_ASCAST]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z3fooPi(
// LLVM-SAME: ptr addrspace(4) noundef %[[DATA:.*]])
// LLVM: %[[DATA_ADDR:.*]] = alloca ptr addrspace(4), align 8
// LLVM-NEXT: %[[DATA_ADDR_ASCAST:.*]] = addrspacecast ptr %[[DATA_ADDR]] to ptr addrspace(4)
// LLVM-NEXT: store ptr addrspace(4) %[[DATA]], ptr addrspace(4) %[[DATA_ADDR_ASCAST]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z4foo2Pi(
// LLVM-SAME: ptr addrspace(4) noundef %[[DATA:.*]])
// LLVM: %[[DATA_ADDR:.*]] = alloca ptr addrspace(4), align 8
// LLVM-NEXT: %[[DATA_ADDR_ASCAST:.*]] = addrspacecast ptr %[[DATA_ADDR]] to ptr addrspace(4)
// LLVM-NEXT: store ptr addrspace(4) %[[DATA]], ptr addrspace(4) %[[DATA_ADDR_ASCAST]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z3fooPU3AS3i(
// LLVM-SAME: ptr addrspace(3) noundef %[[DATA:.*]])
// LLVM: %[[DATA_ADDR:.*]] = alloca ptr addrspace(3), align 8
// LLVM-NEXT: %[[DATA_ADDR_ASCAST:.*]] = addrspacecast ptr %[[DATA_ADDR]] to ptr addrspace(4)
// LLVM-NEXT: store ptr addrspace(3) %[[DATA]], ptr addrspace(4) %[[DATA_ADDR_ASCAST]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z4tmplIPU3AS1iEvT_(
// LLVM-SAME: ptr addrspace(1) noundef %[[T:.*]])
// LLVM: %[[T_ADDR:.*]] = alloca ptr addrspace(1), align 8
// LLVM-NEXT: %[[T_ADDR_ASCAST:.*]] = addrspacecast ptr %[[T_ADDR]] to ptr addrspace(4)
// LLVM-NEXT: store ptr addrspace(1) %[[T]], ptr addrspace(4) %[[T_ADDR_ASCAST]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z4tmplIPU3AS3iEvT_(
// LLVM-SAME: ptr addrspace(3) noundef %[[T:.*]])
// LLVM: %[[T_ADDR:.*]] = alloca ptr addrspace(3), align 8
// LLVM-NEXT: %[[T_ADDR_ASCAST:.*]] = addrspacecast ptr %[[T_ADDR]] to ptr addrspace(4)
// LLVM-NEXT: store ptr addrspace(3) %[[T]], ptr addrspace(4) %[[T_ADDR_ASCAST]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z4tmplIPU3AS0iEvT_(
// LLVM-SAME: ptr noundef %[[T:.*]])
// LLVM: %[[T_ADDR:.*]] = alloca ptr, align 8
// LLVM-NEXT: %[[T_ADDR_ASCAST:.*]] = addrspacecast ptr %[[T_ADDR]] to ptr addrspace(4)
// LLVM-NEXT: store ptr %[[T]], ptr addrspace(4) %[[T_ADDR_ASCAST]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}spir_func void @_Z4tmplIPiEvT_(
// LLVM-SAME: ptr addrspace(4) noundef %[[T:.*]])
// LLVM: %[[T_ADDR:.*]] = alloca ptr addrspace(4), align 8
// LLVM-NEXT: %[[T_ADDR_ASCAST:.*]] = addrspacecast ptr %[[T_ADDR]] to ptr addrspace(4)
// LLVM-NEXT: store ptr addrspace(4) %[[T]], ptr addrspace(4) %[[T_ADDR_ASCAST]], align 8
// LLVM-NEXT: ret void
