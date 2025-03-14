// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s

class kernel_name_1;
class kernel_name_2;
class kernel_name_3;
typedef kernel_name_3 kernel_name_TD;
class kernel_name_4;
struct constexpr_kernel_name;

template<typename KN>
struct kernel_id_1 {
  using type = KN;
};

struct kernel_id_2 {
  using type = kernel_name_2;
};

struct kernel_id_3 {
  using invalid_name = kernel_name_2;
};

template <typename name, typename Func>
__attribute__((sycl_kernel_entry_point(name))) void kernel_single_task(const Func kernelFunc) {
  kernelFunc();
}

struct SYCLKernel {
  int m;
  public:
  void operator()() const {}
};

void testBuiltinBeforeKernelInvocation2();
void testBuiltinBeforeKernelInvocation1() {
  const char* testBefore1 = __builtin_sycl_kernel_name(kernel_id_1<kernel_name_4>()); // Valid
  constexpr const char* testBefore2 = __builtin_sycl_kernel_name(kernel_id_1<constexpr_kernel_name>()); // expected-error {{constexpr variable 'testBefore2' must be initialized by a constant expression}}
}

void testBuiltinBeforeKernelInvocation2() {
  SYCLKernel Obj;
  kernel_single_task<kernel_name_4>(Obj);
  kernel_single_task<constexpr_kernel_name>(Obj);
  testBuiltinBeforeKernelInvocation2();
}

void test() {
  SYCLKernel Obj;
  kernel_single_task<kernel_name_1>(Obj);
  kernel_single_task<kernel_name_2>(Obj);
  kernel_single_task<kernel_name_TD>(Obj);
  const char* test1 = __builtin_sycl_kernel_name(kernel_id_1<kernel_name_1>()); // Valid
  const char* test2 = __builtin_sycl_kernel_name(kernel_id_1<kernel_name_TD>()); // Valid
  const char* test3 = __builtin_sycl_kernel_name(kernel_id_2()); // Valid
  const char* test4 = __builtin_sycl_kernel_name(kernel_id_3()); // expected-error {{invalid argument; expected a class or structure with a member typedef or type alias alias named 'type'}}
  const char* test5 = __builtin_sycl_kernel_name("str"); // expected-error {{invalid argument; expected a class or structure with a member typedef or type alias alias named 'type'}}
  const char* test6 = __builtin_sycl_kernel_name(kernel_id_2(), kernel_id_2()); // expected-error {{builtin takes one argument}}
}

