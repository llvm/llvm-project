// Checks how SYCL device code on SPIR-V passes kernel objects and the
// arguments and results of device functions. The default address space is
// Generic.

// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fexperimental-abi-lowering -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"
// RUN: %if cir-enabled %{ %clang_cc1 -std=c++20 -fsycl-is-device -triple spirv64-unknown-unknown -fclangir -emit-llvm %s -o - | FileCheck %s %}

template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel_entry_point(KernelName)]]
void kernel_single_task(KernelType kf) { kf(); }

struct Small { int a, b; };
struct NonCopyable {
  int i;
  NonCopyable();
  NonCopyable(const NonCopyable &);
  ~NonCopyable();
};

// Pointers and references keep their address space outside kernels.
// CHECK: define spir_func void @_Z7dev_ptrPiRi(ptr addrspace(4) noundef %{{[^,)]+}}, ptr addrspace(4) noundef align 4 dereferenceable(4) %{{[^,)]+}})
[[clang::sycl_external]] void dev_ptr(int *p, int &r) {}

// CHECK: define spir_func void @_Z14dev_global_ptrPU3AS1i(ptr addrspace(1) noundef %{{[^,)]+}})
[[clang::sycl_external]] void dev_global_ptr(__attribute__((opencl_global)) int *p) {}

// Sub-word integers and bool are extended.
// CHECK: define spir_func noundef signext i8 @_Z7dev_extabt(i8 noundef signext %{{[^,)]+}}, i1 noundef zeroext %{{[^,)]+}}, i16 noundef zeroext %{{[^,)]+}})
[[clang::sycl_external]] signed char dev_ext(signed char c, bool b, unsigned short s) { return c; }

// Trivially copyable records are copied byval and returned through a private
// sret.
// CHECK: define spir_func void @_Z9dev_small5Small(ptr noundef byval(%struct.Small) align 4 %{{[^,)]+}})
[[clang::sycl_external]] void dev_small(Small s) {}

// CHECK: define spir_func void @_Z13dev_ret_smallv(ptr dead_on_unwind noalias writable sret(%struct.Small) align 4 %{{[^,)]+}})
[[clang::sycl_external]] Small dev_ret_small() { return {}; }

// A record that cannot be copied is built in place through a Generic sret.
// CHECK-LABEL: define spir_func void @_Z8dev_makev(
// CHECK-SAME: ptr addrspace(4) dead_on_unwind noalias writable sret(%struct.NonCopyable) align 4 %[[RET:[^,)]+]])
// CHECK:   call spir_func void @_ZN11NonCopyableC1Ev(ptr addrspace(4) noundef align 4 dereferenceable_or_null(4) %[[RET]])
[[clang::sycl_external]] NonCopyable dev_make() { return NonCopyable(); }

// It is passed as the private address of the caller's copy, which the callee
// uses through a Generic pointer.
// CHECK-LABEL: define spir_func noundef i32 @_Z8dev_take11NonCopyable(
// CHECK-SAME: ptr nofreeobj noundef align 4 dereferenceable(4) %[[N:[^,)]+]])
// CHECK:   addrspacecast ptr %[[N]] to ptr addrspace(4)
[[clang::sycl_external]] int dev_take(NonCopyable n) { return n.i; }

// CHECK-LABEL: define spir_func noundef i32 @_Z7dev_usev()
// CHECK:   call spir_func void @_Z8dev_makev(ptr addrspace(4) dead_on_unwind writable sret(%struct.NonCopyable) align 4 %{{[^,)]+}})
// CHECK:   call spir_func noundef i32 @_Z8dev_take11NonCopyable(ptr nofreeobj noundef align 4 dereferenceable(4) %{{[^,)]+}})
[[clang::sycl_external]] int dev_use() { NonCopyable n = dev_make(); return dev_take(n); }

struct Functor {
  int *p;
  Small s;
  void operator()() const { *p = s.a; }
};

struct KFunctor;
struct KLambda;
struct KEmpty;

// Kernel objects are copied byval: a functor, a lambda with captures, and an
// empty lambda.
// CHECK: define spir_kernel void @_ZTS8KFunctor(ptr noundef byval(%struct.Functor) align 8 %{{[^,)]+}})
// CHECK: define spir_kernel void @_ZTS7KLambda(ptr noundef byval(%class.anon{{[.0-9]*}}) align 8 %{{[^,)]+}})
// CHECK: define spir_kernel void @_ZTS6KEmpty(ptr noundef byval(%class.anon{{[.0-9]*}}) align 1 %{{[^,)]+}})
void test(int *p, Small s) {
  kernel_single_task<KFunctor>(Functor{p, s});
  kernel_single_task<KLambda>([=] { *p = s.a; });
  kernel_single_task<KEmpty>([] {});
}
