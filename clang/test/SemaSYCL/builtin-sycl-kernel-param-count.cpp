// RUN: %clang_cc1 -fsycl-is-host -fsyntax-only -verify %s

class kernel_name_1;
class kernel_name_2;
class kernel_name_3;
typedef kernel_name_3 kernel_name_TD;

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

void test() {
  SYCLKernel Obj;
  kernel_single_task<kernel_name_1>(Obj);
  kernel_single_task<kernel_name_2>(Obj);
  kernel_single_task<kernel_name_TD>(Obj);
  int test1 = __builtin_sycl_kernel_param_count(kernel_id_1<kernel_name_1>()); // Valid
  int test2 = __builtin_sycl_kernel_param_count(kernel_id_1<kernel_name_TD>()); // Valid
  int test3 = __builtin_sycl_kernel_param_count(kernel_id_2()); // Valid
  int test4 = __builtin_sycl_kernel_param_count(kernel_id_3()); // expected-error {{invalid argument; expected a class or structure with a member typedef or type alias alias named 'type'}}
  int test5 = __builtin_sycl_kernel_param_count("str"); // expected-error {{invalid argument; expected a class or structure with a member typedef or type alias alias named 'type'}}
  int test6 = __builtin_sycl_kernel_param_count(kernel_id_2(), kernel_id_2()); // expected-error {{builtin takes one argument}}
}

