// RUN: %clang_cc1 -fsycl-is-host -emit-llvm -triple x86_64-unknown-linux-gnu -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-HOST,CHECK-HOST-LINUX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple amdgcn-amd-amdhsa -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-AMDGCN %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple nvptx-nvidia-cuda -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-NVPTX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple nvptx64-nvidia-cuda -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-NVPTX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple spir-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR,CHECK-SPIRNV %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple spir64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR,CHECK-SPIRNV %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple spirv32-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR,CHECK-SPIRV %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-unknown-linux-gnu -triple spirv64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR,CHECK-SPIRV %s
// RUN: %clang_cc1 -fsycl-is-host -emit-llvm -triple x86_64-pc-windows-msvc -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-HOST,CHECK-HOST-WINDOWS %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple amdgcn-amd-amdhsa -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-AMDGCN %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple nvptx-nvidia-cuda -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-NVPTX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple nvptx64-nvidia-cuda -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-NVPTX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple spir-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR,CHECK-SPIRNV %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple spir64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR,CHECK-SPIRNV %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple spirv32-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR,CHECK-SPIRV %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-pc-windows-msvc -triple spirv64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR,CHECK-SPIRV %s
// RUN: %clang_cc1 -fsycl-is-host -emit-llvm -triple x86_64-uefi -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-HOST,CHECK-HOST-WINDOWS %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple amdgcn-amd-amdhsa -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-AMDGCN %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple nvptx-nvidia-cuda -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-NVPTX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple nvptx64-nvidia-cuda -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-NVPTX %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple spir-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR,CHECK-SPIRNV %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple spir64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR,CHECK-SPIRNV %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple spirv32-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR,CHECK-SPIRV %s
// RUN: %clang_cc1 -fsycl-is-device -emit-llvm -aux-triple x86_64-uefi -triple spirv64-unknown-unknown -std=c++17 %s -o - | FileCheck --check-prefixes=CHECK-DEVICE,CHECK-SPIR,CHECK-SPIRV %s

// Test code generation for functions declared with the sycl_kernel_entry_point
// attribute. During host compilation, the bodies of such functions are replaced
// with calls to a function template or variable template (with suitable call
// operator) named sycl_kernel_launch. During device compilation, the bodies of
// these functions are used to generate offload kernel entry points (SYCL kernel
// caller functions).

template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

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

// Exercise code gen with kernel name types named with esoteric characters.
struct \u03b4\u03c4\u03c7; // Delta Tau Chi (δτχ)

class handler {
  template <typename KernelName, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...) {}
public:
  template <typename KernelName, typename KernelType>
  [[clang::sycl_kernel_entry_point(KernelName)]]
  void kernel_entry_point(KernelType k, int a, int b) {
    k(a, b);
  }
};

struct copyable {
  int i;
  ~copyable();
};

int main() {
  single_purpose_kernel obj;
  single_purpose_kernel_task(obj);
  int capture;
  auto lambda = [=](auto) { (void) capture; };
  kernel_single_task<decltype(lambda)>(lambda);
  kernel_single_task<\u03b4\u03c4\u03c7>([](int){});
  handler h;
  copyable c{42};
  h.kernel_entry_point<struct KN>([=] (int a, int b) { return c.i + a + b; }, 1, 2);
}

// Verify that SYCL kernel caller functions are not emitted during host
// compilation.
//
// CHECK-HOST-NOT: define {{.*}} @_ZTS26single_purpose_kernel_name
// CHECK-HOST-NOT: define {{.*}} @_ZTSZ4mainEUlT_E_
// CHECK-HOST-NOT: define {{.*}} @"_ZTS6\CE\B4\CF\84\CF\87"
// CHECK-HOST-NOT: define {{.*}} @_ZTSZ4mainE2KN

// Verify that sycl_kernel_entry_point attributed functions are not emitted
// during device compilation.
//
// CHECK-DEVICE-NOT: single_purpose_kernel_task
// CHECK-DEVICE-NOT: kernel_single_task
// CHECK-DEVICE-NOT: kernel_entry_point

// Verify that kernel launch code is generated for sycl_kernel_entry_point
// attributed functions during host compilation.
//
// CHECK-HOST-LINUX:      @.str = private unnamed_addr constant [33 x i8] c"_ZTS26single_purpose_kernel_name\00", align 1
// CHECK-HOST-LINUX:      @.str.1 = private unnamed_addr constant [18 x i8] c"_ZTSZ4mainEUlT_E_\00", align 1
// CHECK-HOST-LINUX:      @.str.2 = private unnamed_addr constant [12 x i8] c"_ZTS6\CE\B4\CF\84\CF\87\00", align 1
//
// CHECK-HOST-LINUX:      define dso_local void @_Z26single_purpose_kernel_task21single_purpose_kernel() #{{[0-9]+}} {
// CHECK-HOST-LINUX-NEXT: entry:
// CHECK-HOST-LINUX-NEXT:   %kernelFunc = alloca %struct.single_purpose_kernel, align 1
// CHECK-HOST-LINUX-NEXT:   %agg.tmp = alloca %struct.single_purpose_kernel, align 1
// CHECK-HOST-LINUX-NEXT:   call void @_Z18sycl_kernel_launchI26single_purpose_kernel_nameJ21single_purpose_kernelEEvPKcDpT0_(ptr noundef @.str)
// CHECK-HOST-LINUX-NEXT:   ret void
// CHECK-HOST-LINUX-NEXT: }
//
// CHECK-HOST-LINUX:      define internal void @_Z18kernel_single_taskIZ4mainEUlT_E_S1_EvT0_(i32 %kernelFunc.coerce) #{{[0-9]+}} {
// CHECK-HOST-LINUX-NEXT: entry:
// CHECK-HOST-LINUX-NEXT:   %kernelFunc = alloca %class.anon, align 4
// CHECK-HOST-LINUX-NEXT:   %agg.tmp = alloca %class.anon, align 4
// CHECK-HOST-LINUX-NEXT:   %coerce.dive = getelementptr inbounds nuw %class.anon, ptr %kernelFunc, i32 0, i32 0
// CHECK-HOST-LINUX-NEXT:   store i32 %kernelFunc.coerce, ptr %coerce.dive, align 4
// CHECK-HOST-LINUX-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %kernelFunc, i64 4, i1 false)
// CHECK-HOST-LINUX-NEXT:   %coerce.dive1 = getelementptr inbounds nuw %class.anon, ptr %agg.tmp, i32 0, i32 0
// CHECK-HOST-LINUX-NEXT:   %0 = load i32, ptr %coerce.dive1, align 4
// CHECK-HOST-LINUX-NEXT:   call void @_Z18sycl_kernel_launchIZ4mainEUlT_E_JS1_EEvPKcDpT0_(ptr noundef @.str.1, i32 %0)
// CHECK-HOST-LINUX-NEXT:   ret void
// CHECK-HOST-LINUX-NEXT: }
//
// CHECK-HOST-LINUX:      define internal void @"_Z18kernel_single_taskI6\CE\B4\CF\84\CF\87Z4mainEUliE_EvT0_"() #{{[0-9]+}} {
// CHECK-HOST-LINUX-NEXT: entry:
// CHECK-HOST-LINUX-NEXT:   %kernelFunc = alloca %class.anon.0, align 1
// CHECK-HOST-LINUX-NEXT:   %agg.tmp = alloca %class.anon.0, align 1
// CHECK-HOST-LINUX-NEXT:   call void @"_Z18sycl_kernel_launchI6\CE\B4\CF\84\CF\87JZ4mainEUliE_EEvPKcDpT0_"(ptr noundef @.str.2)
// CHECK-HOST-LINUX-NEXT:   ret void
// CHECK-HOST-LINUX-NEXT: }


// CHECK-HOST-LINUX:      define internal void @_ZN7handler18kernel_entry_pointIZ4mainE2KNZ4mainEUliiE_EEvT0_ii(ptr noundef nonnull align 1 dereferenceable(1) %this, ptr noundef %k, i32 noundef %a, i32 noundef %b) #{{[0-9]+}} align 2 {
// CHECK-HOST-LINUX-NEXT: entry:
// CHECK-HOST-LINUX-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-HOST-LINUX-NEXT:   %k.indirect_addr = alloca ptr, align 8
// CHECK-HOST-LINUX-NEXT:   %a.addr = alloca i32, align 4
// CHECK-HOST-LINUX-NEXT:   %b.addr = alloca i32, align 4
// CHECK-HOST-LINUX-NEXT:   %agg.tmp = alloca %class.anon.1, align 4
// CHECK-HOST-LINUX-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-HOST-LINUX-NEXT:   store ptr %k, ptr %k.indirect_addr, align 8
// CHECK-HOST-LINUX-NEXT:   store i32 %a, ptr %a.addr, align 4
// CHECK-HOST-LINUX-NEXT:   store i32 %b, ptr %b.addr, align 4
// CHECK-HOST-LINUX-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-HOST-LINUX-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %k, i64 4, i1 false)
// CHECK-HOST-LINUX-NEXT:   %0 = load i32, ptr %a.addr, align 4
// CHECK-HOST-LINUX-NEXT:   %1 = load i32, ptr %b.addr, align 4
// CHECK-HOST-LINUX-NEXT:   call void @_ZN7handler18sycl_kernel_launchIZ4mainE2KNJZ4mainEUliiE_iiEEEvPKcDpT0_(ptr noundef nonnull align 1 dereferenceable(1) %this1, ptr noundef @.str.3, ptr noundef %agg.tmp, i32 noundef %0, i32 noundef %1)
// CHECK-HOST-LINUX-NEXT:   call void @_ZZ4mainENUliiE_D1Ev(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %agg.tmp) #{{[0-9]+}}
// CHECK-HOST-LINUX-NEXT:   ret void
// CHECK-HOST-LINUX-NEXT: }

// CHECK-HOST-WINDOWS:      define dso_local void @"?single_purpose_kernel_task@@YAXUsingle_purpose_kernel@@@Z"(i8 %kernelFunc.coerce) #{{[0-9]+}} {
// CHECK-HOST-WINDOWS-NEXT: entry:
// CHECK-HOST-WINDOWS-NEXT:   %kernelFunc = alloca %struct.single_purpose_kernel, align 1
// CHECK-HOST-WINDOWS-NEXT:   %agg.tmp = alloca %struct.single_purpose_kernel, align 1
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive = getelementptr inbounds nuw %struct.single_purpose_kernel, ptr %kernelFunc, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   store i8 %kernelFunc.coerce, ptr %coerce.dive, align 1
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive1 = getelementptr inbounds nuw %struct.single_purpose_kernel, ptr %agg.tmp, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   %0 = load i8, ptr %coerce.dive1, align 1
// CHECK-HOST-WINDOWS-NEXT:   call void @"??$sycl_kernel_launch@Usingle_purpose_kernel_name@@Usingle_purpose_kernel@@@@YAXPEBDUsingle_purpose_kernel@@@Z"(ptr noundef @"??_C@_0CB@KFIJOMLB@_ZTS26single_purpose_kernel_name@", i8 %0)
// CHECK-HOST-WINDOWS-NEXT:   ret void
// CHECK-HOST-WINDOWS-NEXT: }
//
// CHECK-HOST-WINDOWS:      define internal void @"??$kernel_single_task@V<lambda_1>@?0??main@@9@V1?0??2@9@@@YAXV<lambda_1>@?0??main@@9@@Z"(i32 %kernelFunc.coerce) #{{[0-9]+}} {
// CHECK-HOST-WINDOWS-NEXT: entry:
// CHECK-HOST-WINDOWS-NEXT:   %kernelFunc = alloca %class.anon, align 4
// CHECK-HOST-WINDOWS-NEXT:   %agg.tmp = alloca %class.anon, align 4
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive = getelementptr inbounds nuw %class.anon, ptr %kernelFunc, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   store i32 %kernelFunc.coerce, ptr %coerce.dive, align 4
// CHECK-HOST-WINDOWS-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %kernelFunc, i64 4, i1 false)
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive1 = getelementptr inbounds nuw %class.anon, ptr %agg.tmp, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   %0 = load i32, ptr %coerce.dive1, align 4
// CHECK-HOST-WINDOWS-NEXT:   call void @"??$sycl_kernel_launch@V<lambda_1>@?0??main@@9@V1?0??2@9@@@YAXPEBDV<lambda_1>@?0??main@@9@@Z"(ptr noundef @"??_C@_0BC@NHCDOLAA@_ZTSZ4mainEUlT_E_?$AA@", i32 %0)
//
// CHECK-HOST-WINDOWS-NEXT:   ret void
// CHECK-HOST-WINDOWS-NEXT: }
//
// CHECK-HOST-WINDOWS:      define internal void @"??$kernel_single_task@U\CE\B4\CF\84\CF\87@@V<lambda_2>@?0??main@@9@@@YAXV<lambda_2>@?0??main@@9@@Z"(i8 %kernelFunc.coerce) #{{[0-9]+}} {
// CHECK-HOST-WINDOWS-NEXT: entry:
// CHECK-HOST-WINDOWS-NEXT:   %kernelFunc = alloca %class.anon.0, align 1
// CHECK-HOST-WINDOWS-NEXT:   %agg.tmp = alloca %class.anon.0, align 1
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive = getelementptr inbounds nuw %class.anon.0, ptr %kernelFunc, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   store i8 %kernelFunc.coerce, ptr %coerce.dive, align 1
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive1 = getelementptr inbounds nuw %class.anon.0, ptr %agg.tmp, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   %0 = load i8, ptr %coerce.dive1, align 1
// CHECK-HOST-WINDOWS-NEXT:   call void @"??$sycl_kernel_launch@U\CE\B4\CF\84\CF\87@@V<lambda_2>@?0??main@@9@@@YAXPEBDV<lambda_2>@?0??main@@9@@Z"(ptr noundef @"??_C@_0M@BCGAEMBE@_ZTS6?N?$LE?O?$IE?O?$IH?$AA@", i8 %0)
// CHECK-HOST-WINDOWS-NEXT:   ret void
// CHECK-HOST-WINDOWS-NEXT: }

// CHECK-HOST-WINDOWS:      define internal void @"??$kernel_entry_point@UKN@?1??main@@9@V<lambda_3>@?0??2@9@@handler@@QEAAXV<lambda_3>@?0??main@@9@HH@Z"(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 %k.coerce, i32 noundef %a, i32 noundef %b) #{{[0-9]+}} align 2
// CHECK-HOST-WINDOWS-NEXT: entry:
// CHECK-HOST-WINDOWS-NEXT:   %k = alloca %class.anon.1, align 4
// CHECK-HOST-WINDOWS-NEXT:   %b.addr = alloca i32, align 4
// CHECK-HOST-WINDOWS-NEXT:   %a.addr = alloca i32, align 4
// CHECK-HOST-WINDOWS-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-HOST-WINDOWS-NEXT:   %agg.tmp = alloca %class.anon.1, align 4
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive = getelementptr inbounds nuw %class.anon.1, ptr %k, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive1 = getelementptr inbounds nuw %struct.copyable, ptr %coerce.dive, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   store i32 %k.coerce, ptr %coerce.dive1, align 4
// CHECK-HOST-WINDOWS-NEXT:   store i32 %b, ptr %b.addr, align 4
// CHECK-HOST-WINDOWS-NEXT:   store i32 %a, ptr %a.addr, align 4
// CHECK-HOST-WINDOWS-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-HOST-WINDOWS-NEXT:   %this2 = load ptr, ptr %this.addr, align 8
// CHECK-HOST-WINDOWS-NEXT:   %0 = load i32, ptr %b.addr, align 4
// CHECK-HOST-WINDOWS-NEXT:   %1 = load i32, ptr %a.addr, align 4
// CHECK-HOST-WINDOWS-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %k, i64 4, i1 false)
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive3 = getelementptr inbounds nuw %class.anon.1, ptr %agg.tmp, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive4 = getelementptr inbounds nuw %struct.copyable, ptr %coerce.dive3, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   %2 = load i32, ptr %coerce.dive4, align 4
// CHECK-HOST-WINDOWS-NEXT:   call void @"??$sycl_kernel_launch@UKN@?1??main@@9@V<lambda_3>@?0??2@9@HH@handler@@AEAAXPEBDV<lambda_3>@?0??main@@9@HH@Z"(ptr noundef nonnull align 1 dereferenceable(1) %this2, ptr noundef @"??_C@_0P@DLGHPODL@_ZTSZ4mainE2KN?$AA@", i32 %2, i32 noundef %1, i32 noundef %0)
// CHECK-HOST-WINDOWS-NEXT:   call void @"??1<lambda_3>@?0??main@@9@QEAA@XZ"(ptr noundef nonnull align 4 dead_on_return(4) dereferenceable(4) %k) #{{[0-9]+}}
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

// IR for the SYCL kernel caller function generated for kernel_single_task with
// the Delta Tau Chi type as the SYCL kernel name type.
//
// CHECK-AMDGCN:      Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-AMDGCN-NEXT: define dso_local amdgpu_kernel void @"_ZTS6\CE\B4\CF\84\CF\87"
// CHECK-AMDGCN-SAME:   (ptr addrspace(4) noundef byref(%class.anon.0) align 1 %0) #[[AMDGCN_ATTR0]] {
// CHECK-AMDGCN-NEXT: entry:
// CHECK-AMDGCN-NEXT:   %coerce = alloca %class.anon.0, align 1, addrspace(5)
// CHECK-AMDGCN-NEXT:   %kernelFunc = addrspacecast ptr addrspace(5) %coerce to ptr
// CHECK-AMDGCN-NEXT:   call void @llvm.memcpy.p0.p4.i64(ptr align 1 %kernelFunc, ptr addrspace(4) align 1 %0, i64 1, i1 false)
// CHECK-AMDGCN-NEXT:   call void @_ZZ4mainENKUliE_clEi
// CHECK-AMDGCN-SAME:     (ptr noundef nonnull align 1 dereferenceable(1) %kernelFunc, i32 noundef 42) #[[AMDGCN_ATTR1:[0-9]+]]
// CHECK-AMDGCN-NEXT:   ret void
// CHECK-AMDGCN-NEXT: }
// CHECK-AMDGCN:      define internal void @_ZZ4mainENKUliE_clEi
//
// CHECK-NVPTX:       Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-NVPTX-NEXT:  define dso_local ptx_kernel void @"_ZTS6\CE\B4\CF\84\CF\87"
// CHECK-NVPTX-SAME:    (ptr noundef byval(%class.anon.0) align 1 %kernelFunc) #[[NVPTX_ATTR0:[0-9]+]] {
// CHECK-NVPTX-NEXT:  entry:
// CHECK-NVPTX-NEXT:    call void @_ZZ4mainENKUliE_clEi
// CHECK-NVPTX-SAME:      (ptr noundef nonnull align 1 dereferenceable(1) %kernelFunc, i32 noundef 42) #[[NVPTX_ATTR1:[0-9]+]]
// CHECK-NVPTX-NEXT:    ret void
// CHECK-NVPTX-NEXT:  }
// CHECK-NVPTX:       define internal void @_ZZ4mainENKUliE_clEi
//
// CHECK-SPIR:        Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-SPIR-NEXT:   define {{[a-z_ ]*}}spir_kernel void @"_ZTS6\CE\B4\CF\84\CF\87"
// CHECK-SPIR-SAME:     (ptr noundef byval(%class.anon.0) align 1 %kernelFunc) #[[SPIR_ATTR0:[0-9]+]] {
// CHECK-SPIR-NEXT:   entry:
// CHECK-SPIR-NEXT:     %kernelFunc.ascast = addrspacecast ptr %kernelFunc to ptr addrspace(4)
// CHECK-SPIR-NEXT:     call spir_func void @_ZZ4mainENKUliE_clEi
// CHECK-SPIR-SAME:       (ptr addrspace(4) noundef align 1 dereferenceable_or_null(1) %kernelFunc.ascast, i32 noundef 42) #[[SPIR_ATTR1:[0-9]+]]
// CHECK-SPIR-NEXT:     ret void
// CHECK-SPIR-NEXT:   }
// CHECK-SPIR:        define internal spir_func void @_ZZ4mainENKUliE_clEi

// IR for the SYCL kernel caller function generated for
// handler::kernel_entry_point with main::KN as the SYCL kernel name type.
//
// CHECK-AMDGCN:      Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-AMDGCN-NEXT: define dso_local amdgpu_kernel void @_ZTSZ4mainE2KN
// CHECK-AMDGCN-SAME:   (i32 %k.coerce, i32 noundef %a, i32 noundef %b) #[[AMDGCN_ATTR0]] {
// CHECK-AMDGCN-NEXT: entry:
// CHECK-AMDGCN-NEXT:   %k = alloca %class.anon.1, align 4, addrspace(5)
// CHECK-AMDGCN-NEXT:   %a.addr = alloca i32, align 4, addrspace(5)
// CHECK-AMDGCN-NEXT:   %b.addr = alloca i32, align 4, addrspace(5)
// CHECK-AMDGCN-NEXT:   %k2 = addrspacecast ptr addrspace(5) %k to ptr
// CHECK-AMDGCN-NEXT:   %a.addr.ascast = addrspacecast ptr addrspace(5) %a.addr to ptr
// CHECK-AMDGCN-NEXT:   %b.addr.ascast = addrspacecast ptr addrspace(5) %b.addr to ptr
// CHECK-AMDGCN-NEXT:   %coerce.dive = getelementptr inbounds nuw %class.anon.1, ptr %k2, i32 0, i32 0
// CHECK-AMDGCN-NEXT:   %coerce.dive1 = getelementptr inbounds nuw %struct.copyable, ptr %coerce.dive, i32 0, i32 0
// CHECK-AMDGCN-NEXT:   store i32 %k.coerce, ptr %coerce.dive1, align 4
// CHECK-AMDGCN-NEXT:   store i32 %a, ptr %a.addr.ascast, align 4
// CHECK-AMDGCN-NEXT:   store i32 %b, ptr %b.addr.ascast, align 4
// CHECK-AMDGCN-NEXT:   %0 = load i32, ptr %a.addr.ascast, align 4
// CHECK-AMDGCN-NEXT:   %1 = load i32, ptr %b.addr.ascast, align 4
// CHECK-AMDGCN-NEXT:   %call = call noundef i32 @_ZZ4mainENKUliiE_clEii
// CHECK-AMDGCN-SAME:     (ptr noundef nonnull align 4 dereferenceable(4) %k2, i32 noundef %0, i32 noundef %1) #[[AMDGCN_ATTR1:[0-9]+]]
// CHECK-AMDGCN-NEXT:   ret void
// CHECK-AMDGCN-NEXT: }
//
// CHECK-NVPTX:       Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-NVPTX-NEXT:  define dso_local ptx_kernel void @_ZTSZ4mainE2KN
// CHECK-NVPTX-SAME:    (ptr noundef byval(%class.anon.1) align 4 %k, i32 noundef %a, i32 noundef %b) #[[NVPTX_ATTR0:[0-9]+]] {
// CHECK-NVPTX-NEXT:  entry:
// CHECK-NVPTX-NEXT:    %a.addr = alloca i32, align 4
// CHECK-NVPTX-NEXT:    %b.addr = alloca i32, align 4
// CHECK-NVPTX-NEXT:    store i32 %a, ptr %a.addr, align 4
// CHECK-NVPTX-NEXT:    store i32 %b, ptr %b.addr, align 4
// CHECK-NVPTX-NEXT:    %0 = load i32, ptr %a.addr, align 4
// CHECK-NVPTX-NEXT:    %1 = load i32, ptr %b.addr, align 4
// CHECK-NVPTX-NEXT:    %call = call noundef i32 @_ZZ4mainENKUliiE_clEii
// CHECK-NVPTX-SAME:      (ptr noundef nonnull align 4 dereferenceable(4) %k, i32 noundef %0, i32 noundef %1) #[[NVPTX_ATTR1:[0-9]+]]
// CHECK-NVPTX-NEXT:    ret void
// CHECK-NVPTX-NEXT:  }
//
// CHECK-SPIRNV:      Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-SPIRNV-NEXT: define dso_local spir_kernel void @_ZTSZ4mainE2KN
// CHECK-SPIRNV-SAME:   (ptr noundef %k, i32 noundef %a, i32 noundef %b) #[[SPIR_ATTR0:[0-9]+]] {
// CHECK-SPIRNV-NEXT: entry:
// CHECK-SPIRNV-NEXT:   %k.indirect_addr = alloca ptr addrspace(4), align {{[48]}}
// CHECK-SPIRNV-NEXT:   %a.addr = alloca i32, align 4
// CHECK-SPIRNV-NEXT:   %b.addr = alloca i32, align 4
// CHECK-SPIRNV-NEXT:   %k.indirect_addr.ascast = addrspacecast ptr %k.indirect_addr to ptr addrspace(4)
// CHECK-SPIRNV-NEXT:   %a.addr.ascast = addrspacecast ptr %a.addr to ptr addrspace(4)
// CHECK-SPIRNV-NEXT:   %b.addr.ascast = addrspacecast ptr %b.addr to ptr addrspace(4)
// CHECK-SPIRNV-NEXT:   store ptr %k, ptr addrspace(4) %k.indirect_addr.ascast, align {{[48]}}
// CHECK-SPIRNV-NEXT:   %k.ascast = addrspacecast ptr %k to ptr addrspace(4)
// CHECK-SPIRNV-NEXT:   store i32 %a, ptr addrspace(4) %a.addr.ascast, align 4
// CHECK-SPIRNV-NEXT:   store i32 %b, ptr addrspace(4) %b.addr.ascast, align 4
// CHECK-SPIRNV-NEXT:   %0 = load i32, ptr addrspace(4) %a.addr.ascast, align 4
// CHECK-SPIRNV-NEXT:   %1 = load i32, ptr addrspace(4) %b.addr.ascast, align 4
// CHECK-SPIRNV-NEXT:   %call = call spir_func noundef i32 @_ZZ4mainENKUliiE_clEii
// CHECK-SPIRNV-SAME:     (ptr addrspace(4) noundef align 4 dereferenceable_or_null(4) %k.ascast, i32 noundef %0, i32 noundef %1) #[[SPIR_ATTR1:[0-9]+]]
// CHECK-SPIRNV-NEXT:   ret void
// CHECK-SPIRNV-NEXT: }
//
// CHECK-SPIRV:       Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
// CHECK-SPIRV-NEXT:  define spir_kernel void @_ZTSZ4mainE2KN
// CHECK-SPIRV-SAME:    (ptr noundef byval(%class.anon.1) align 4 %k, i32 noundef %a, i32 noundef %b) #[[SPIR_ATTR0:[0-9]+]] {
// CHECK-SPIRV-NEXT:  entry:
// CHECK-SPIRV-NEXT:    %a.addr = alloca i32, align 4
// CHECK-SPIRV-NEXT:    %b.addr = alloca i32, align 4
// CHECK-SPIRV-NEXT:    %a.addr.ascast = addrspacecast ptr %a.addr to ptr addrspace(4)
// CHECK-SPIRV-NEXT:    %b.addr.ascast = addrspacecast ptr %b.addr to ptr addrspace(4)
// CHECK-SPIRV-NEXT:    %k.ascast = addrspacecast ptr %k to ptr addrspace(4)
// CHECK-SPIRV-NEXT:    store i32 %a, ptr addrspace(4) %a.addr.ascast, align 4
// CHECK-SPIRV-NEXT:    store i32 %b, ptr addrspace(4) %b.addr.ascast, align 4
// CHECK-SPIRV-NEXT:    %0 = load i32, ptr addrspace(4) %a.addr.ascast, align 4
// CHECK-SPIRV-NEXT:    %1 = load i32, ptr addrspace(4) %b.addr.ascast, align 4
// CHECK-SPIRV-NEXT:    %call = call spir_func noundef i32 @_ZZ4mainENKUliiE_clEii
// CHECK-SPIRV-SAME:      (ptr addrspace(4) noundef align 4 dereferenceable_or_null(4) %k.ascast, i32 noundef %0, i32 noundef %1) #[[SPIR_ATTR1:[0-9]+]]
// CHECK-SPIRV-NEXT:    ret void
// CHECK-SPIRV-NEXT:  }

// CHECK-AMDGCN: #[[AMDGCN_ATTR0]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// CHECK-AMDGCN: #[[AMDGCN_ATTR1]] = { convergent nounwind }
//
// CHECK-NVPTX: #[[NVPTX_ATTR0]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// CHECK-NVPTX: #[[NVPTX_ATTR1]] = { convergent nounwind }
//
// CHECK-SPIR: #[[SPIR_ATTR0]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// CHECK-SPIR: #[[SPIR_ATTR1]] = { convergent nounwind }
