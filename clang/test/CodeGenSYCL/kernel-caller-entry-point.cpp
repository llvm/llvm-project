// RUN: %clang_cc1 -fsycl-is-host -emit-llvm -triple x86_64-unknown-linux-gnu -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-HOST,CHECK-HOST-LINUX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple amdgcn-amd-amdhsa -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-AMDGCN %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple nvptx-nvidia-cuda -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-NVPTX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple nvptx64-nvidia-cuda -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-NVPTX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple spir-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple spir64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple spirv32-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple spirv64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR %s
// RUN: %clang_cc1 -fsycl-is-host -emit-llvm -triple x86_64-pc-windows-msvc -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-HOST,CHECK-HOST-WINDOWS %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple amdgcn-amd-amdhsa -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-AMDGCN %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple nvptx-nvidia-cuda -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-NVPTX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple nvptx64-nvidia-cuda -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-NVPTX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple spir-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple spir64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple spirv32-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple spirv64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR %s
// RUN: %clang_cc1 -fsycl-is-host -emit-llvm -triple x86_64-uefi -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-HOST,CHECK-HOST-WINDOWS %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple amdgcn-amd-amdhsa -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-AMDGCN %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple nvptx-nvidia-cuda -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-NVPTX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple nvptx64-nvidia-cuda -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-NVPTX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple spir-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple spir64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple spirv32-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple spirv64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR %s

// Test the generation of SYCL kernel caller functions. These functions are
// generated from functions declared with the sycl_kernel_entry_point attribute
// and emited during device compilation. They are not emitted during device
// compilation.

struct single_purpose_kernel_name;
struct single_purpose_kernel {
  void operator()() const {}
};

[[clang::sycl_kernel_entry_point(single_purpose_kernel_name)]]
void single_purpose_kernel_task(single_purpose_kernel kernelFunc) {
  kernelFunc();
}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kernelFunc) {
  kernelFunc(42);
}

int main() {
  single_purpose_kernel obj;
  single_purpose_kernel_task(obj);
  int capture;
  auto lambda = [=](auto) { (void) capture; };
  kernel_single_task<decltype(lambda)>(lambda);
}

// Verify that SYCL kernel caller functions are not emitted during host
// compilation.
//
// CHECK-HOST-NOT: _ZTS26single_purpose_kernel_name
// CHECK-HOST-NOT: _ZTSZ4mainE18lambda_kernel_name

// Verify that sycl_kernel_entry_point attributed functions are not emitted
// during device compilation.
//
// CHECK-DEVICE-NOT: single_purpose_kernel_task
// CHECK-DEVICE-NOT: kernel_single_task

// Verify that no code is generated for the bodies of sycl_kernel_entry_point
// attributed functions during host compilation. ODR-use of these functions may
// require them to be emitted, but they have no effect if called.
//
// CHECK-HOST-LINUX:      define dso_local void @_Z26single_purpose_kernel_task21single_purpose_kernel() #{{[0-9]+}} {
// CHECK-HOST-LINUX-NEXT: entry:
// CHECK-HOST-LINUX-NEXT:   %kernelFunc = alloca %struct.single_purpose_kernel, align 1
// CHECK-HOST-LINUX-NEXT:   ret void
// CHECK-HOST-LINUX-NEXT: }
//
// CHECK-HOST-LINUX:      define internal void @_Z18kernel_single_taskIZ4mainEUlT_E_S1_EvT0_(i32 %kernelFunc.coerce) #{{[0-9]+}} {
// CHECK-HOST-LINUX-NEXT: entry:
// CHECK-HOST-LINUX-NEXT:   %kernelFunc = alloca %class.anon, align 4
// CHECK-HOST-LINUX-NEXT:   %coerce.dive = getelementptr inbounds nuw %class.anon, ptr %kernelFunc, i32 0, i32 0
// CHECK-HOST-LINUX-NEXT:   store i32 %kernelFunc.coerce, ptr %coerce.dive, align 4
// CHECK-HOST-LINUX-NEXT:   ret void
// CHECK-HOST-LINUX-NEXT: }
//
// CHECK-HOST-WINDOWS:      define dso_local void @"?single_purpose_kernel_task@@YAXUsingle_purpose_kernel@@@Z"(i8 %kernelFunc.coerce) #{{[0-9]+}} {
// CHECK-HOST-WINDOWS-NEXT: entry:
// CHECK-HOST-WINDOWS-NEXT:   %kernelFunc = alloca %struct.single_purpose_kernel, align 1
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive = getelementptr inbounds nuw %struct.single_purpose_kernel, ptr %kernelFunc, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   store i8 %kernelFunc.coerce, ptr %coerce.dive, align 1
// CHECK-HOST-WINDOWS-NEXT:   ret void
// CHECK-HOST-WINDOWS-NEXT: }
//
// CHECK-HOST-WINDOWS:      define internal void @"??$kernel_single_task@V<lambda_1>@?0??main@@9@V1?0??2@9@@@YAXV<lambda_1>@?0??main@@9@@Z"(i32 %kernelFunc.coerce) #{{[0-9]+}} {
// CHECK-HOST-WINDOWS-NEXT: entry:
// CHECK-HOST-WINDOWS-NEXT:   %kernelFunc = alloca %class.anon, align 4
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive = getelementptr inbounds nuw %class.anon, ptr %kernelFunc, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   store i32 %kernelFunc.coerce, ptr %coerce.dive, align 4
// CHECK-HOST-WINDOWS-NEXT:   ret void
// CHECK-HOST-WINDOWS-NEXT: }

// Verify that SYCL kernel caller functions are emitted for each device target.
//
// main() shouldn't be emitted in device code.
// CHECK-NOT: @main()

// IR for the SYCL kernel caller function generated for
// single_purpose_kernel_task with single_purpose_kernel_name as the SYCL kernel
// name type.
//
// CHECK-AMDGCN:      Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-AMDGCN-NEXT: define dso_local amdgpu_kernel void @_ZTS26single_purpose_kernel_name
// CHECK-AMDGCN-SAME:   (ptr addrspace(4) noundef byref(%struct.single_purpose_kernel) align 1 %0) #[[AMDGCN_ATTR0:[0-9]+]] {
// CHECK-AMDGCN-NEXT: entry:
// CHECK-AMDGCN-NEXT:   %coerce = alloca %struct.single_purpose_kernel, align 1, addrspace(5)
// CHECK-AMDGCN-NEXT:   %kernelFunc = addrspacecast ptr addrspace(5) %coerce to ptr
// CHECK-AMDGCN-NEXT:   call void @llvm.memcpy.p0.p4.i64(ptr align 1 %kernelFunc, ptr addrspace(4) align 1 %0, i64 1, i1 false)
// CHECK-AMDGCN-NEXT:   call void @_ZNK21single_purpose_kernelclEv
// CHECK-AMDGCN-SAME:     (ptr noundef nonnull align 1 dereferenceable(1) %kernelFunc) #[[AMDGCN_ATTR1:[0-9]+]]
// CHECK-AMDGCN-NEXT:   ret void
// CHECK-AMDGCN-NEXT: }
// CHECK-AMDGCN:      define linkonce_odr void @_ZNK21single_purpose_kernelclEv
//
// CHECK-NVPTX:       Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-NVPTX-NEXT:  define dso_local ptx_kernel void @_ZTS26single_purpose_kernel_name
// CHECK-NVPTX-SAME:    (ptr noundef byval(%struct.single_purpose_kernel) align 1 %kernelFunc) #[[NVPTX_ATTR0:[0-9]+]] {
// CHECK-NVPTX-NEXT:  entry:
// CHECK-NVPTX-NEXT:    call void @_ZNK21single_purpose_kernelclEv
// CHECK-NVPTX-SAME:      (ptr noundef nonnull align 1 dereferenceable(1) %kernelFunc) #[[NVPTX_ATTR1:[0-9]+]]
// CHECK-NVPTX-NEXT:    ret void
// CHECK-NVPTX-NEXT:  }
// CHECK-NVPTX:       define linkonce_odr void @_ZNK21single_purpose_kernelclEv
//
// CHECK-SPIR:        Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-SPIR-NEXT:   define {{[a-z_ ]*}}spir_kernel void @_ZTS26single_purpose_kernel_name
// CHECK-SPIR-SAME:     (ptr noundef byval(%struct.single_purpose_kernel) align 1 %kernelFunc) #[[SPIR_ATTR0:[0-9]+]] {
// CHECK-SPIR-NEXT:   entry:
// CHECK-SPIR-NEXT:     %kernelFunc.ascast = addrspacecast ptr %kernelFunc to ptr addrspace(4)
// CHECK-SPIR-NEXT:     call spir_func void @_ZNK21single_purpose_kernelclEv
// CHECK-SPIR-SAME:       (ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) %kernelFunc.ascast) #[[SPIR_ATTR1:[0-9]+]]
// CHECK-SPIR-NEXT:     ret void
// CHECK-SPIR-NEXT:   }
// CHECK-SPIR:        define linkonce_odr spir_func void @_ZNK21single_purpose_kernelclEv

// IR for the SYCL kernel caller function generated for kernel_single_task with
// lambda_kernel_name as the SYCL kernel name type.
//
// CHECK-AMDGCN:      Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-AMDGCN-NEXT: define dso_local amdgpu_kernel void @_ZTSZ4mainEUlT_E_
// CHECK-AMDGCN-SAME:   (i32 %kernelFunc.coerce) #[[AMDGCN_ATTR0]] {
// CHECK-AMDGCN-NEXT: entry:
// CHECK-AMDGCN-NEXT:   %kernelFunc = alloca %class.anon, align 4, addrspace(5)
// CHECK-AMDGCN-NEXT:   %kernelFunc1 = addrspacecast ptr addrspace(5) %kernelFunc to ptr
// CHECK-AMDGCN-NEXT:   %coerce.dive = getelementptr inbounds nuw %class.anon, ptr %kernelFunc1, i32 0, i32 0
// CHECK-AMDGCN-NEXT:   store i32 %kernelFunc.coerce, ptr %coerce.dive, align 4
// CHECK-AMDGCN-NEXT:   call void @_ZZ4mainENKUlT_E_clIiEEDaS_
// CHECK-AMDGCN-SAME:     (ptr noundef nonnull align 4 dereferenceable(4) %kernelFunc1, i32 noundef 42) #[[AMDGCN_ATTR1]]
// CHECK-AMDGCN-NEXT:   ret void
// CHECK-AMDGCN-NEXT: }
// CHECK-AMDGCN:      define internal void @_ZZ4mainENKUlT_E_clIiEEDaS_
//
// CHECK-NVPTX:       Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-NVPTX-NEXT:  define dso_local ptx_kernel void @_ZTSZ4mainEUlT_E_
// CHECK-NVPTX-SAME:    (ptr noundef byval(%class.anon) align 4 %kernelFunc) #[[NVPTX_ATTR0]] {
// CHECK-NVPTX-NEXT:  entry:
// CHECK-NVPTX-NEXT:    call void @_ZZ4mainENKUlT_E_clIiEEDaS_
// CHECK-NVPTX-SAME:      (ptr noundef nonnull align 4 dereferenceable(4) %kernelFunc, i32 noundef 42) #[[NVPTX_ATTR1]]
// CHECK-NVPTX-NEXT:    ret void
// CHECK-NVPTX-NEXT:  }
// CHECK-NVPTX:       define internal void @_ZZ4mainENKUlT_E_clIiEEDaS_
//
// CHECK-SPIR:        Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-SPIR-NEXT:   define {{[a-z_ ]*}}spir_kernel void @_ZTSZ4mainEUlT_E_
// CHECK-SPIR-SAME:     (ptr noundef byval(%class.anon) align 4 %kernelFunc) #[[SPIR_ATTR0]] {
// CHECK-SPIR-NEXT:   entry:
// CHECK-SPIR-NEXT:     %kernelFunc.ascast = addrspacecast ptr %kernelFunc to ptr addrspace(4)
// CHECK-SPIR-NEXT:     call spir_func void @_ZZ4mainENKUlT_E_clIiEEDaS_
// CHECK-SPIR-SAME:       (ptr addrspace(4) noundef align 4 dereferenceable_or_null(4) %kernelFunc.ascast, i32 noundef 42) #[[SPIR_ATTR1]]
// CHECK-SPIR-NEXT:     ret void
// CHECK-SPIR-NEXT:   }
// CHECK-SPIR:        define internal spir_func void @_ZZ4mainENKUlT_E_clIiEEDaS_

// CHECK-AMDGCN: #[[AMDGCN_ATTR0]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// CHECK-AMDGCN: #[[AMDGCN_ATTR1]] = { convergent nounwind }
//
// CHECK-NVPTX: #[[NVPTX_ATTR0]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+ptx32" }
// CHECK-NVPTX: #[[NVPTX_ATTR1]] = { convergent nounwind }
//
// CHECK-SPIR: #[[SPIR_ATTR0]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// CHECK-SPIR: #[[SPIR_ATTR1]] = { convergent nounwind }
