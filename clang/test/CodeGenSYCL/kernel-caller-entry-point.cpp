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
// and emited during device compilation.
// Test the generation of SYCL kernel launch statements during host compilation.
// These statements are calls to sycl_enqueus_kernel_launch functions or class
// members in case skep-attributed functions are also members of the same class.

template <typename KernelName, typename KernelObj>
void sycl_kernel_launch(const char *, KernelObj) {}

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

class Handler {
template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}
public:
template<typename KNT, typename KT>
[[clang::sycl_kernel_entry_point(KNT)]]
void skep(KT k, int a, int b) {
  k(a, b);
}
};

struct auto_name;

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void __kernel_single_task(const KernelType KernelFunc) {
  KernelFunc();
}

template <typename KernelType, typename KernelName = auto_name>
void pf(KernelType K) {
  __kernel_single_task<KernelName>(K);
}
struct DCopyable {
  int i;
  ~DCopyable();
};

int main() {
  single_purpose_kernel obj;
  single_purpose_kernel_task(obj);
  int capture;
  auto lambda = [=](auto) { (void) capture; };
  kernel_single_task<decltype(lambda)>(lambda);
  kernel_single_task<\u03b4\u03c4\u03c7>([](int){});
  Handler H;
  H.skep<class notaverygoodkernelname>([=](int a, int b){return a+b;}, 1, 2);

  DCopyable b;
  pf([b](){});
}

// Verify that SYCL kernel caller functions are not emitted during host
// compilation.
//
// CHECK-HOST-NOT: define {{.*}} @_ZTS26single_purpose_kernel_name
// CHECK-HOST-NOT: define {{.*}} @_ZTSZ4mainEUlT_E_
// CHECK-HOST-NOT: define {{.*}} @"_ZTS6\CE\B4\CF\84\CF\87"

// Verify that sycl_kernel_entry_point attributed functions are not emitted
// during device compilation.
//
// CHECK-DEVICE-NOT: single_purpose_kernel_task
// CHECK-DEVICE-NOT: kernel_single_task

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
// CHECK-HOST-LINUX-NEXT:   call void @_Z18sycl_kernel_launchI26single_purpose_kernel_name21single_purpose_kernelEvPKcT0_(ptr noundef @.str)
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
// CHECK-HOST-LINUX-NEXT:   call void @_Z18sycl_kernel_launchIZ4mainEUlT_E_S1_EvPKcT0_(ptr noundef @.str.1, i32 %0)
// CHECK-HOST-LINUX-NEXT:   ret void
// CHECK-HOST-LINUX-NEXT: }
//
// CHECK-HOST-LINUX:      define internal void @"_Z18kernel_single_taskI6\CE\B4\CF\84\CF\87Z4mainEUliE_EvT0_"() #{{[0-9]+}} {
// CHECK-HOST-LINUX-NEXT: entry:
// CHECK-HOST-LINUX-NEXT:   %kernelFunc = alloca %class.anon.0, align 1
// CHECK-HOST-LINUX-NEXT:   %agg.tmp = alloca %class.anon.0, align 1
// CHECK-HOST-LINUX-NEXT:   call void @"_Z18sycl_kernel_launchI6\CE\B4\CF\84\CF\87Z4mainEUliE_EvPKcT0_"(ptr noundef @.str.2)
// CHECK-HOST-LINUX-NEXT:   ret void
// CHECK-HOST-LINUX-NEXT: }

// CHECK-HOST-LINUX: define internal void @_ZN7Handler4skepIZ4mainE22notaverygoodkernelnameZ4mainEUliiE_EEvT0_ii(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 noundef %a, i32 noundef %b) #0 align 2 {
// CHECK-HOST-LINUX-NEXT: entry:
// CHECK-HOST-LINUX-NEXT:   %k = alloca %class.anon.1, align 1
// CHECK-HOST-LINUX-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-HOST-LINUX-NEXT:   %a.addr = alloca i32, align 4
// CHECK-HOST-LINUX-NEXT:   %b.addr = alloca i32, align 4
// CHECK-HOST-LINUX-NEXT:   %agg.tmp = alloca %class.anon.1, align 1
// CHECK-HOST-LINUX-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-HOST-LINUX-NEXT:   store i32 %a, ptr %a.addr, align 4
// CHECK-HOST-LINUX-NEXT:   store i32 %b, ptr %b.addr, align 4
// CHECK-HOST-LINUX-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-HOST-LINUX-NEXT:   %0 = load i32, ptr %a.addr, align 4
// CHECK-HOST-LINUX-NEXT:   %1 = load i32, ptr %b.addr, align 4
// CHECK-HOST-LINUX-NEXT:   call void @_ZN7Handler18sycl_kernel_launchIZ4mainE22notaverygoodkernelnameJZ4mainEUliiE_iiEEEvPKcDpT0_(ptr noundef nonnull align 1 dereferenceable(1) %this1, ptr noundef @.str.3, i32 noundef %0, i32 noundef %1)
// CHECK-HOST-LINUX-NEXT:   ret void
// CHECK-HOST-LINUX-NEXT: }

// CHECK-HOST-LINUX: define internal void @_Z20__kernel_single_taskI9auto_nameZ4mainEUlvE_EvT0_(ptr noundef %KernelFunc)
// CHECK-HOST-LINUX-NEXT: entry:
// CHECK-HOST-LINUX-NEXT:   %KernelFunc.indirect_addr = alloca ptr, align 8
// CHECK-HOST-LINUX-NEXT:   %agg.tmp = alloca %class.anon.3, align 4
// CHECK-HOST-LINUX-NEXT:   store ptr %KernelFunc, ptr %KernelFunc.indirect_addr, align 8
// CHECK-HOST-LINUX-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %KernelFunc, i64 4, i1 false)
// CHECK-HOST-LINUX-NEXT:   call void @_Z18sycl_kernel_launchI9auto_nameZ4mainEUlvE_EvPKcT0_(ptr noundef @.str.4, ptr noundef %agg.tmp)
// CHECK-HOST-LINUX-NEXT:   call void @_ZZ4mainENUlvE_D1Ev(ptr noundef nonnull align 4 dereferenceable(4) %agg.tmp) #4
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

// CHECK-HOST-WINDOWS: define internal void @"??$skep@Vnotaverygoodkernelname@?1??main@@9@V<lambda_3>@?0??2@9@@Handler@@QEAAXV<lambda_3>@?0??main@@9@HH@Z"(ptr noundef nonnull align 1 dereferenceable(1) %this, i8 %k.coerce, i32 noundef %a, i32 noundef %b) #0 align 2 {
// CHECK-HOST-WINDOWS-NEXT: entry:
// CHECK-HOST-WINDOWS-NEXT:   %k = alloca %class.anon.1, align 1
// CHECK-HOST-WINDOWS-NEXT:   %b.addr = alloca i32, align 4
// CHECK-HOST-WINDOWS-NEXT:   %a.addr = alloca i32, align 4
// CHECK-HOST-WINDOWS-NEXT:   %this.addr = alloca ptr, align 8
// CHECK-HOST-WINDOWS-NEXT:   %agg.tmp = alloca %class.anon.1, align 1
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive = getelementptr inbounds nuw %class.anon.1, ptr %k, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   store i8 %k.coerce, ptr %coerce.dive, align 1
// CHECK-HOST-WINDOWS-NEXT:   store i32 %b, ptr %b.addr, align 4
// CHECK-HOST-WINDOWS-NEXT:   store i32 %a, ptr %a.addr, align 4
// CHECK-HOST-WINDOWS-NEXT:   store ptr %this, ptr %this.addr, align 8
// CHECK-HOST-WINDOWS-NEXT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK-HOST-WINDOWS-NEXT:   %0 = load i32, ptr %b.addr, align 4
// CHECK-HOST-WINDOWS-NEXT:   %1 = load i32, ptr %a.addr, align 4
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive2 = getelementptr inbounds nuw %class.anon.1, ptr %agg.tmp, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   %2 = load i8, ptr %coerce.dive2, align 1
// CHECK-HOST-WINDOWS-NEXT:   call void @"??$sycl_kernel_launch@Vnotaverygoodkernelname@?1??main@@9@V<lambda_3>@?0??2@9@HH@Handler@@AEAAXPEBDV<lambda_3>@?0??main@@9@HH@Z"(ptr noundef nonnull align 1 dereferenceable(1) %this1, ptr noundef @"??_C@_0CE@NJIGCEIA@_ZTSZ4mainE22notaverygoodkerneln@", i8 %2, i32 noundef %1, i32 noundef %0)
// CHECK-HOST-WINDOWS-NEXT:   ret void
// CHECK-HOST-WINDOWS-NEXT: }

// CHECK-HOST-WINDOWS: define internal void @"??$__kernel_single_task@Uauto_name@@V<lambda_4>@?0??main@@9@@@YAXV<lambda_4>@?0??main@@9@@Z"(i32 %KernelFunc.coerce)
// CHECK-HOST-WINDOWS-NEXT: entry:
// CHECK-HOST-WINDOWS-NEXT:   %KernelFunc = alloca %class.anon.3, align 4
// CHECK-HOST-WINDOWS-NEXT:   %agg.tmp = alloca %class.anon.3, align 4
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive = getelementptr inbounds nuw %class.anon.3, ptr %KernelFunc, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive1 = getelementptr inbounds nuw %struct.DCopyable, ptr %coerce.dive, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   store i32 %KernelFunc.coerce, ptr %coerce.dive1, align 4
// CHECK-HOST-WINDOWS-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %agg.tmp, ptr align 4 %KernelFunc, i64 4, i1 false)
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive2 = getelementptr inbounds nuw %class.anon.3, ptr %agg.tmp, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   %coerce.dive3 = getelementptr inbounds nuw %struct.DCopyable, ptr %coerce.dive2, i32 0, i32 0
// CHECK-HOST-WINDOWS-NEXT:   %0 = load i32, ptr %coerce.dive3, align 4
// CHECK-HOST-WINDOWS-NEXT:   call void @"??$sycl_kernel_launch@Uauto_name@@V<lambda_4>@?0??main@@9@@@YAXPEBDV<lambda_4>@?0??main@@9@@Z"(ptr noundef @"??_C@_0P@HMAAEHI@_ZTS9auto_name?$AA@", i32 %0)
// CHECK-HOST-WINDOWS-NEXT:   call void @"??1<lambda_4>@?0??main@@9@QEAA@XZ"(ptr noundef nonnull align 4 dereferenceable(4) %KernelFunc)
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


// CHECK-AMDGCN: #[[AMDGCN_ATTR0]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// CHECK-AMDGCN: #[[AMDGCN_ATTR1]] = { convergent nounwind }
//
// CHECK-NVPTX: #[[NVPTX_ATTR0]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// CHECK-NVPTX: #[[NVPTX_ATTR1]] = { convergent nounwind }
//
// CHECK-SPIR: #[[SPIR_ATTR0]] = { convergent mustprogress noinline norecurse nounwind optnone "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
// CHECK-SPIR: #[[SPIR_ATTR1]] = { convergent nounwind }
