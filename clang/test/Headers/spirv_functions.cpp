// RUN: %clang_cc1 -Wno-unused-value -O0 -internal-isystem %S/../../lib/Headers -include __clang_spirv_builtins.h -triple spirv64 -emit-llvm %s -fsycl-is-device -o - | FileCheck %s -check-prefixes=SPV
// RUN: %clang_cc1 -Wno-unused-value -O0 -internal-isystem %S/../../lib/Headers -include __clang_spirv_builtins.h -triple nvptx64 -emit-llvm %s -fsycl-is-device -o - | FileCheck %s -check-prefixes=NV


// SPV: void @_Z9test_castPi
// SPV: call noundef ptr addrspace(1) @llvm.spv.generic.cast.to.ptr.explicit.p1
// SPV: call noundef ptr addrspace(3) @llvm.spv.generic.cast.to.ptr.explicit.p3
// SPV: call noundef ptr @llvm.spv.generic.cast.to.ptr.explicit.p0
// SPV: addrspacecast ptr addrspace(4) %{{.*}} to ptr addrspace(1)
// SPV: addrspacecast ptr addrspace(4) %{{.*}} to ptr addrspace(3)
// SPV: addrspacecast ptr addrspace(4) %{{.*}} to ptr
// NV: void @_Z9test_castPi
// NV: call noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi
// NV: call noundef ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi
// NV: call noundef ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi
// NV: addrspacecast ptr %{{.*}} to ptr addrspace(1)
// NV: addrspacecast ptr %{{.*}} to ptr addrspace(3)
[[clang::sycl_external]] void test_cast(int* p) {
  __spirv_GenericCastToPtrExplicit_ToGlobal(p, 5);
  __spirv_GenericCastToPtrExplicit_ToLocal(p, 4);
  __spirv_GenericCastToPtrExplicit_ToPrivate(p, 7);
  __spirv_GenericCastToPtr_ToGlobal(p, 5);
  __spirv_GenericCastToPtr_ToLocal(p, 4);
  __spirv_GenericCastToPtr_ToPrivate(p, 7);
}
