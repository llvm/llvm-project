// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsycl-is-device -disable-llvm-passes -Wno-deprecated-attributes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsycl-is-device -disable-llvm-passes -Wno-deprecated-attributes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVM-CIR
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsycl-is-device -disable-llvm-passes -Wno-deprecated-attributes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// Port of clang/test/CodeGenSYCL/cuda-address-space-conversions.cpp.

void bar(int &Data) {}
void bar2(int &Data) {}
void bar(int [[clang::sycl_local]] &Data) {}
void foo(int *Data) {}
void foo2(int *Data) {}
void foo(int [[clang::sycl_local]] *Data) {}

template <typename T>
void tmpl(T t);

[[clang::sycl_external]] void usages() {
  int *NoAS;
  int [[clang::sycl_global]] *GLOB;
  int [[clang::sycl_local]] *LOC;
  int [[clang::sycl_private]] *PRIV;
  int __attribute__((opencl_global_device)) *GLOBDEVICE;
  int __attribute__((opencl_global_host)) *GLOBHOST;
  LOC = nullptr;
  GLOB = nullptr;
  NoAS = (int *)GLOB;
  NoAS = (int *)LOC;
  NoAS = (int *)PRIV;
  GLOB = (int [[clang::sycl_global]] *)NoAS;
  LOC = (int [[clang::sycl_local]] *)NoAS;
  PRIV = (int [[clang::sycl_private]] *)NoAS;
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
  tmpl(GLOB);
  tmpl(LOC);
  tmpl(PRIV);
  tmpl(NoAS);
}

// CIR-LABEL: cir.func {{.*}}@_Z6usagesv(
// CIR-NEXT: %[[NOAS:.*]] = cir.alloca "NoAS" align(8) : !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT: %[[GLOB:.*]] = cir.alloca "GLOB" align(8) : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>
// CIR-NEXT: %[[LOC:.*]] = cir.alloca "LOC" align(8) : !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>>
// CIR-NEXT: %[[PRIV:.*]] = cir.alloca "PRIV" align(8) : !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT: %[[GLOBDEVICE:.*]] = cir.alloca "GLOBDEVICE" align(8) : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>
// CIR-NEXT: %[[GLOBHOST:.*]] = cir.alloca "GLOBHOST" align(8) : !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>
// On NVPTX, a null local pointer is an address space cast of the generic
// null pointer, so that it converts back to a generic null pointer.
// CIR-NEXT: %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i>
// CIR-NEXT: %[[LOC_NULL:.*]] = cir.cast address_space %[[NULL]] : !cir.ptr<!s32i> -> !cir.ptr<!s32i, target_address_space(3)>
// CIR-NEXT: cir.store align(8) %[[LOC_NULL]], %[[LOC]] : !cir.ptr<!s32i, target_address_space(3)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>>
// CIR-NEXT: %[[GLOB_NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!s32i, target_address_space(1)>
// CIR-NEXT: cir.store align(8) %[[GLOB_NULL]], %[[GLOB]] : !cir.ptr<!s32i, target_address_space(1)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(1)>>

// CIR-LABEL: cir.func {{.*}}@_Z3barRi(
// CIR-NEXT: %[[DATA:.*]] = cir.alloca "Data" align(8) init const : !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT: cir.store %arg0, %[[DATA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT: cir.return

// CIR-LABEL: cir.func {{.*}}@_Z3barRU3AS3i(
// CIR-NEXT: %[[DATA:.*]] = cir.alloca "Data" align(8) init const : !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>>
// CIR-NEXT: cir.store %arg0, %[[DATA]] : !cir.ptr<!s32i, target_address_space(3)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>>
// CIR-NEXT: cir.return

// CIR-LABEL: cir.func {{.*}}@_Z3fooPi(
// CIR-NEXT: %[[DATA:.*]] = cir.alloca "Data" align(8) init : !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT: cir.store %arg0, %[[DATA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
// CIR-NEXT: cir.return

// CIR-LABEL: cir.func {{.*}}@_Z3fooPU3AS3i(
// CIR-NEXT: %[[DATA:.*]] = cir.alloca "Data" align(8) init : !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>>
// CIR-NEXT: cir.store %arg0, %[[DATA]] : !cir.ptr<!s32i, target_address_space(3)>, !cir.ptr<!cir.ptr<!s32i, target_address_space(3)>>
// CIR-NEXT: cir.return

// LLVM-CIR-LABEL: define {{.*}}void @_Z6usagesv(
// LLVM-CIR: %[[NOAS:.*]] = alloca ptr, align 8
// LLVM-CIR-NEXT: %[[GLOB:.*]] = alloca ptr addrspace(1), align 8
// LLVM-CIR-NEXT: %[[LOC:.*]] = alloca ptr addrspace(3), align 8
// LLVM-CIR-NEXT: %[[PRIV:.*]] = alloca ptr, align 8
// LLVM-CIR-NEXT: %[[GLOBDEVICE:.*]] = alloca ptr addrspace(1), align 8
// LLVM-CIR-NEXT: %[[GLOBHOST:.*]] = alloca ptr addrspace(1), align 8
// LLVM-CIR-NEXT: store ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr %[[LOC]], align 8
// LLVM-CIR-NEXT: store ptr addrspace(1) null, ptr %[[GLOB]], align 8
// LLVM-CIR-NEXT: %[[TMP0:.*]] = load ptr addrspace(1), ptr %[[GLOB]], align 8
// LLVM-CIR-NEXT: %[[TMP1:.*]] = addrspacecast ptr addrspace(1) %[[TMP0]] to ptr
// LLVM-CIR-NEXT: store ptr %[[TMP1]], ptr %[[NOAS]], align 8
// LLVM-CIR-NEXT: %[[TMP2:.*]] = load ptr addrspace(3), ptr %[[LOC]], align 8
// LLVM-CIR-NEXT: %[[TMP3:.*]] = addrspacecast ptr addrspace(3) %[[TMP2]] to ptr
// LLVM-CIR-NEXT: store ptr %[[TMP3]], ptr %[[NOAS]], align 8
// LLVM-CIR-NEXT: %[[TMP4:.*]] = load ptr, ptr %[[PRIV]], align 8
// LLVM-CIR-NEXT: store ptr %[[TMP4]], ptr %[[NOAS]], align 8
// LLVM-CIR-NEXT: %[[TMP5:.*]] = load ptr, ptr %[[NOAS]], align 8
// LLVM-CIR-NEXT: %[[TMP6:.*]] = addrspacecast ptr %[[TMP5]] to ptr addrspace(1)
// LLVM-CIR-NEXT: store ptr addrspace(1) %[[TMP6]], ptr %[[GLOB]], align 8
// LLVM-CIR-NEXT: %[[TMP7:.*]] = load ptr, ptr %[[NOAS]], align 8
// LLVM-CIR-NEXT: %[[TMP8:.*]] = addrspacecast ptr %[[TMP7]] to ptr addrspace(3)
// LLVM-CIR-NEXT: store ptr addrspace(3) %[[TMP8]], ptr %[[LOC]], align 8
// LLVM-CIR-NEXT: %[[TMP9:.*]] = load ptr, ptr %[[NOAS]], align 8
// LLVM-CIR-NEXT: store ptr %[[TMP9]], ptr %[[PRIV]], align 8
// LLVM-CIR-NEXT: %[[TMP10:.*]] = load ptr addrspace(1), ptr %[[GLOBDEVICE]], align 8
// LLVM-CIR-NEXT: store ptr addrspace(1) %[[TMP10]], ptr %[[GLOB]], align 8
// LLVM-CIR-NEXT: %[[TMP11:.*]] = load ptr addrspace(1), ptr %[[GLOBHOST]], align 8
// LLVM-CIR-NEXT: store ptr addrspace(1) %[[TMP11]], ptr %[[GLOB]], align 8
// LLVM-CIR-NEXT: %[[TMP12:.*]] = load ptr addrspace(1), ptr %[[GLOB]], align 8
// LLVM-CIR-NEXT: %[[TMP13:.*]] = addrspacecast ptr addrspace(1) %[[TMP12]] to ptr
// LLVM-CIR-NEXT: call void @_Z3barRi(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP13]])
// LLVM-CIR-NEXT: %[[TMP14:.*]] = load ptr addrspace(1), ptr %[[GLOB]], align 8
// LLVM-CIR-NEXT: %[[TMP15:.*]] = addrspacecast ptr addrspace(1) %[[TMP14]] to ptr
// LLVM-CIR-NEXT: call void @_Z4bar2Ri(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP15]])
// LLVM-CIR-NEXT: %[[TMP16:.*]] = load ptr addrspace(3), ptr %[[LOC]], align 8
// LLVM-CIR-NEXT: call void @_Z3barRU3AS3i(ptr addrspace(3) noundef align 4 dereferenceable(4) %[[TMP16]])
// LLVM-CIR-NEXT: %[[TMP17:.*]] = load ptr addrspace(3), ptr %[[LOC]], align 8
// LLVM-CIR-NEXT: %[[TMP18:.*]] = addrspacecast ptr addrspace(3) %[[TMP17]] to ptr
// LLVM-CIR-NEXT: call void @_Z4bar2Ri(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP18]])
// LLVM-CIR-NEXT: %[[TMP19:.*]] = load ptr, ptr %[[NOAS]], align 8
// LLVM-CIR-NEXT: call void @_Z3barRi(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP19]])
// LLVM-CIR-NEXT: %[[TMP20:.*]] = load ptr, ptr %[[NOAS]], align 8
// LLVM-CIR-NEXT: call void @_Z4bar2Ri(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP20]])
// LLVM-CIR-NEXT: %[[TMP21:.*]] = load ptr addrspace(1), ptr %[[GLOB]], align 8
// LLVM-CIR-NEXT: %[[TMP22:.*]] = addrspacecast ptr addrspace(1) %[[TMP21]] to ptr
// LLVM-CIR-NEXT: call void @_Z3fooPi(ptr noundef %[[TMP22]])
// LLVM-CIR-NEXT: %[[TMP23:.*]] = load ptr addrspace(1), ptr %[[GLOB]], align 8
// LLVM-CIR-NEXT: %[[TMP24:.*]] = addrspacecast ptr addrspace(1) %[[TMP23]] to ptr
// LLVM-CIR-NEXT: call void @_Z4foo2Pi(ptr noundef %[[TMP24]])
// LLVM-CIR-NEXT: %[[TMP25:.*]] = load ptr addrspace(3), ptr %[[LOC]], align 8
// LLVM-CIR-NEXT: call void @_Z3fooPU3AS3i(ptr addrspace(3) noundef %[[TMP25]])
// LLVM-CIR-NEXT: %[[TMP26:.*]] = load ptr addrspace(3), ptr %[[LOC]], align 8
// LLVM-CIR-NEXT: %[[TMP27:.*]] = addrspacecast ptr addrspace(3) %[[TMP26]] to ptr
// LLVM-CIR-NEXT: call void @_Z4foo2Pi(ptr noundef %[[TMP27]])
// LLVM-CIR-NEXT: %[[TMP28:.*]] = load ptr, ptr %[[NOAS]], align 8
// LLVM-CIR-NEXT: call void @_Z3fooPi(ptr noundef %[[TMP28]])
// LLVM-CIR-NEXT: %[[TMP29:.*]] = load ptr, ptr %[[NOAS]], align 8
// LLVM-CIR-NEXT: call void @_Z4foo2Pi(ptr noundef %[[TMP29]])
// LLVM-CIR-NEXT: %[[TMP30:.*]] = load ptr addrspace(1), ptr %[[GLOB]], align 8
// LLVM-CIR-NEXT: call void @_Z4tmplIPU3AS1iEvT_(ptr addrspace(1) noundef %[[TMP30]])
// LLVM-CIR-NEXT: %[[TMP31:.*]] = load ptr addrspace(3), ptr %[[LOC]], align 8
// LLVM-CIR-NEXT: call void @_Z4tmplIPU3AS3iEvT_(ptr addrspace(3) noundef %[[TMP31]])
// LLVM-CIR-NEXT: %[[TMP32:.*]] = load ptr, ptr %[[PRIV]], align 8
// LLVM-CIR-NEXT: call void @_Z4tmplIPiEvT_(ptr noundef %[[TMP32]])
// LLVM-CIR-NEXT: %[[TMP33:.*]] = load ptr, ptr %[[NOAS]], align 8
// LLVM-CIR-NEXT: call void @_Z4tmplIPiEvT_(ptr noundef %[[TMP33]])
// LLVM-CIR-NEXT: ret void

// OGCG-LABEL: define {{.*}}void @_Z6usagesv(
// OGCG: %[[NOAS:.*]] = alloca ptr, align 8
// OGCG-NEXT: %[[GLOB:.*]] = alloca ptr addrspace(1), align 8
// OGCG-NEXT: %[[LOC:.*]] = alloca ptr addrspace(3), align 8
// OGCG-NEXT: %[[PRIV:.*]] = alloca ptr, align 8
// OGCG-NEXT: %[[GLOBDEVICE:.*]] = alloca ptr addrspace(1), align 8
// OGCG-NEXT: %[[GLOBHOST:.*]] = alloca ptr addrspace(1), align 8
// OGCG-NEXT: store ptr addrspace(3) addrspacecast (ptr null to ptr addrspace(3)), ptr %[[LOC]], align 8
// OGCG-NEXT: store ptr addrspace(1) null, ptr %[[GLOB]], align 8
// OGCG-NEXT: %[[TMP0:.*]] = load ptr addrspace(1), ptr %[[GLOB]], align 8
// OGCG-NEXT: %[[TMP1:.*]] = addrspacecast ptr addrspace(1) %[[TMP0]] to ptr
// OGCG-NEXT: store ptr %[[TMP1]], ptr %[[NOAS]], align 8
// OGCG-NEXT: %[[TMP2:.*]] = load ptr addrspace(3), ptr %[[LOC]], align 8
// OGCG-NEXT: %[[TMP3:.*]] = addrspacecast ptr addrspace(3) %[[TMP2]] to ptr
// OGCG-NEXT: store ptr %[[TMP3]], ptr %[[NOAS]], align 8
// OGCG-NEXT: %[[TMP4:.*]] = load ptr, ptr %[[PRIV]], align 8
// OGCG-NEXT: store ptr %[[TMP4]], ptr %[[NOAS]], align 8
// OGCG-NEXT: %[[TMP5:.*]] = load ptr, ptr %[[NOAS]], align 8
// OGCG-NEXT: %[[TMP6:.*]] = addrspacecast ptr %[[TMP5]] to ptr addrspace(1)
// OGCG-NEXT: store ptr addrspace(1) %[[TMP6]], ptr %[[GLOB]], align 8
// OGCG-NEXT: %[[TMP7:.*]] = load ptr, ptr %[[NOAS]], align 8
// OGCG-NEXT: %[[TMP8:.*]] = addrspacecast ptr %[[TMP7]] to ptr addrspace(3)
// OGCG-NEXT: store ptr addrspace(3) %[[TMP8]], ptr %[[LOC]], align 8
// OGCG-NEXT: %[[TMP9:.*]] = load ptr, ptr %[[NOAS]], align 8
// OGCG-NEXT: store ptr %[[TMP9]], ptr %[[PRIV]], align 8
// OGCG-NEXT: %[[TMP10:.*]] = load ptr addrspace(1), ptr %[[GLOBDEVICE]], align 8
// OGCG-NEXT: store ptr addrspace(1) %[[TMP10]], ptr %[[GLOB]], align 8
// OGCG-NEXT: %[[TMP11:.*]] = load ptr addrspace(1), ptr %[[GLOBHOST]], align 8
// OGCG-NEXT: store ptr addrspace(1) %[[TMP11]], ptr %[[GLOB]], align 8
// OGCG-NEXT: %[[TMP12:.*]] = load ptr addrspace(1), ptr %[[GLOB]], align 8
// OGCG-NEXT: %[[TMP13:.*]] = addrspacecast ptr addrspace(1) %[[TMP12]] to ptr
// OGCG-NEXT: call void @_Z3barRi(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP13]])
// OGCG-NEXT: %[[TMP14:.*]] = load ptr addrspace(1), ptr %[[GLOB]], align 8
// OGCG-NEXT: %[[TMP15:.*]] = addrspacecast ptr addrspace(1) %[[TMP14]] to ptr
// OGCG-NEXT: call void @_Z4bar2Ri(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP15]])
// OGCG-NEXT: %[[TMP16:.*]] = load ptr addrspace(3), ptr %[[LOC]], align 8
// OGCG-NEXT: call void @_Z3barRU3AS3i(ptr addrspace(3) noundef align 4 dereferenceable(4) %[[TMP16]])
// OGCG-NEXT: %[[TMP17:.*]] = load ptr addrspace(3), ptr %[[LOC]], align 8
// OGCG-NEXT: %[[TMP18:.*]] = addrspacecast ptr addrspace(3) %[[TMP17]] to ptr
// OGCG-NEXT: call void @_Z4bar2Ri(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP18]])
// OGCG-NEXT: %[[TMP19:.*]] = load ptr, ptr %[[NOAS]], align 8
// OGCG-NEXT: call void @_Z3barRi(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP19]])
// OGCG-NEXT: %[[TMP20:.*]] = load ptr, ptr %[[NOAS]], align 8
// OGCG-NEXT: call void @_Z4bar2Ri(ptr noundef nonnull align 4 dereferenceable(4) %[[TMP20]])
// OGCG-NEXT: %[[TMP21:.*]] = load ptr addrspace(1), ptr %[[GLOB]], align 8
// OGCG-NEXT: %[[TMP22:.*]] = addrspacecast ptr addrspace(1) %[[TMP21]] to ptr
// OGCG-NEXT: call void @_Z3fooPi(ptr noundef %[[TMP22]])
// OGCG-NEXT: %[[TMP23:.*]] = load ptr addrspace(1), ptr %[[GLOB]], align 8
// OGCG-NEXT: %[[TMP24:.*]] = addrspacecast ptr addrspace(1) %[[TMP23]] to ptr
// OGCG-NEXT: call void @_Z4foo2Pi(ptr noundef %[[TMP24]])
// OGCG-NEXT: %[[TMP25:.*]] = load ptr addrspace(3), ptr %[[LOC]], align 8
// OGCG-NEXT: call void @_Z3fooPU3AS3i(ptr addrspace(3) noundef %[[TMP25]])
// OGCG-NEXT: %[[TMP26:.*]] = load ptr addrspace(3), ptr %[[LOC]], align 8
// OGCG-NEXT: %[[TMP27:.*]] = addrspacecast ptr addrspace(3) %[[TMP26]] to ptr
// OGCG-NEXT: call void @_Z4foo2Pi(ptr noundef %[[TMP27]])
// OGCG-NEXT: %[[TMP28:.*]] = load ptr, ptr %[[NOAS]], align 8
// OGCG-NEXT: call void @_Z3fooPi(ptr noundef %[[TMP28]])
// OGCG-NEXT: %[[TMP29:.*]] = load ptr, ptr %[[NOAS]], align 8
// OGCG-NEXT: call void @_Z4foo2Pi(ptr noundef %[[TMP29]])
// OGCG-NEXT: %[[TMP30:.*]] = load ptr addrspace(1), ptr %[[GLOB]], align 8
// OGCG-NEXT: call void @_Z4tmplIPU3AS1iEvT_(ptr addrspace(1) noundef %[[TMP30]])
// OGCG-NEXT: %[[TMP31:.*]] = load ptr addrspace(3), ptr %[[LOC]], align 8
// OGCG-NEXT: call void @_Z4tmplIPU3AS3iEvT_(ptr addrspace(3) noundef %[[TMP31]])
// OGCG-NEXT: %[[TMP32:.*]] = load ptr, ptr %[[PRIV]], align 8
// OGCG-NEXT: call void @_Z4tmplIPiEvT_(ptr noundef %[[TMP32]])
// OGCG-NEXT: %[[TMP33:.*]] = load ptr, ptr %[[NOAS]], align 8
// OGCG-NEXT: call void @_Z4tmplIPiEvT_(ptr noundef %[[TMP33]])
// OGCG-NEXT: ret void

// LLVM-LABEL: define {{.*}}void @_Z3barRi(
// LLVM-SAME: ptr noundef nonnull align 4 dereferenceable(4) %[[DATA:.*]])
// LLVM: %[[DATA_ADDR:.*]] = alloca ptr, align 8
// LLVM-NEXT: store ptr %[[DATA]], ptr %[[DATA_ADDR]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}void @_Z4bar2Ri(
// LLVM-SAME: ptr noundef nonnull align 4 dereferenceable(4) %[[DATA:.*]])
// LLVM: %[[DATA_ADDR:.*]] = alloca ptr, align 8
// LLVM-NEXT: store ptr %[[DATA]], ptr %[[DATA_ADDR]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}void @_Z3barRU3AS3i(
// LLVM-SAME: ptr addrspace(3) noundef align 4 dereferenceable(4) %[[DATA:.*]])
// LLVM: %[[DATA_ADDR:.*]] = alloca ptr addrspace(3), align 8
// LLVM-NEXT: store ptr addrspace(3) %[[DATA]], ptr %[[DATA_ADDR]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}void @_Z3fooPi(
// LLVM-SAME: ptr noundef %[[DATA:.*]])
// LLVM: %[[DATA_ADDR:.*]] = alloca ptr, align 8
// LLVM-NEXT: store ptr %[[DATA]], ptr %[[DATA_ADDR]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}void @_Z4foo2Pi(
// LLVM-SAME: ptr noundef %[[DATA:.*]])
// LLVM: %[[DATA_ADDR:.*]] = alloca ptr, align 8
// LLVM-NEXT: store ptr %[[DATA]], ptr %[[DATA_ADDR]], align 8
// LLVM-NEXT: ret void

// LLVM-LABEL: define {{.*}}void @_Z3fooPU3AS3i(
// LLVM-SAME: ptr addrspace(3) noundef %[[DATA:.*]])
// LLVM: %[[DATA_ADDR:.*]] = alloca ptr addrspace(3), align 8
// LLVM-NEXT: store ptr addrspace(3) %[[DATA]], ptr %[[DATA_ADDR]], align 8
// LLVM-NEXT: ret void

// LLVM: declare void @_Z4tmplIPU3AS1iEvT_(ptr addrspace(1) noundef)
// LLVM: declare void @_Z4tmplIPU3AS3iEvT_(ptr addrspace(3) noundef)
// LLVM: declare void @_Z4tmplIPiEvT_(ptr noundef)
