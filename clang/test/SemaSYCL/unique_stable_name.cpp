// RUN: %clang_cc1 %s -std=c++17 -triple x86_64-pc-windows-msvc -fsycl-is-device -verify -fsyntax-only -Wno-unused
// RUN: %clang_cc1 %s -std=c++17 -triple x86_64-linux-gnu -fsycl-is-device -verify -fsyntax-only -Wno-unused

template <typename KernelName, typename KernelType>
[[clang::sycl_kernel]] void kernel_single_task(KernelType kernelFunc) { // #kernelSingleTask
  kernelFunc();
}

// kernel1 - expect error
// The current function is named with a lambda (i.e., takes a lambda as a
// template parameter. Call the builtin on the current function then it is
// passed to a kernel. Test that passing the given function to the unique
// stable name builtin and then to the kernel throws an error because the
// latter causes its name mangling to change.
template <typename Func>
void kernel1func(const Func &F1) {
  constexpr const char *F1_output = __builtin_sycl_unique_stable_name(Func); // #USN_F1
  // expected-error@#kernelSingleTask{{kernel instantiation changes the result of an evaluated '__builtin_sycl_unique_stable_name'}}
  // expected-note@#kernel1func_call{{in instantiation of function template specialization}}
  // expected-note@#USN_F1{{'__builtin_sycl_unique_stable_name' evaluated here}}
  // expected-note@+1{{in instantiation of function template specialization}}
  kernel_single_task<class kernel1>(F1); // #kernel1_call
}

void callkernel1() {
  kernel1func([]() {}); // #kernel1func_call
}

// kernel2 - expect error
// The current function is named with a lambda (i.e., takes a lambda as a
// template parameter). Call the builtin on the given function,
// then an empty lambda is passed to kernel.
// Test that passing the given function to the unique stable name builtin and
// then passing a different lambda to the kernel still throws an error because
// the calling context is part of naming the kernel. Even though the given
// function (F2) is not passed to the kernel, its mangling changes due to
// kernel call with the unrelated lambda.
template <typename Func>
void kernel2func(const Func &F2) {
  constexpr const char *F2_output = __builtin_sycl_unique_stable_name(Func); // #USN_F2
  // expected-error@#kernelSingleTask{{kernel instantiation changes the result of an evaluated '__builtin_sycl_unique_stable_name'}}
  // expected-note@#kernel2func_call{{in instantiation of function template specialization}}
  // expected-note@#USN_F2{{'__builtin_sycl_unique_stable_name' evaluated here}}
  // expected-note@+1{{in instantiation of function template specialization}}
  kernel_single_task<class kernel2>([]() {});
}

void callkernel2() {
  kernel2func([]() {}); // #kernel2func_call
}

template <template <typename> typename Outer, typename Inner>
struct S {
  void operator()() const;
};

template <typename Ty>
struct Tangerine {};

template <typename Func>
void kernel3_4func(const Func &F) {
  // Test that passing the same lambda to two kernels does not cause an error
  // because the kernel uses do not interfere with each other or invalidate
  // the stable name in any way.
  kernel_single_task<class kernel3>(F);
  kernel_single_task<class kernel4>(F);
  // Using the same functor twice should be fine
}

// kernel3 and kernel4 - expect no errors
void callkernel3_4() {
  kernel3_4func([]() {});
}

template <typename T>
static constexpr const char *output1 = __builtin_sycl_unique_stable_name(T);

#define MACRO()                      \
  auto l14 = []() { return 1; };     \
  constexpr const char *l14_output = \
      __builtin_sycl_unique_stable_name(decltype(l14));

int main() {

  // kernel5 - expect no error
  // Test that passing the lambda to the unique stable name builtin and then
  // using the lambda in a way that does not  contribute to the kernel name
  // does not cause an error because the  stable name is not invalidated in
  // this situation.
  auto l5 = []() {};
  constexpr const char *l5_output =
      __builtin_sycl_unique_stable_name(decltype(l5));
  kernel_single_task<class kernel5>(
      [=]() { l5(); }); // Used in the kernel, but not the kernel name itself

  // kernel6 - expect error
  // Test that passing the lambda to the unique stable name builtin and then
  // using the same lambda in the naming of a kernel causes a diagnostic on the
  // kernel use due to the change in results to the stable name.
  auto l6 = []() { return 1; };
  constexpr const char *l6_output =
      __builtin_sycl_unique_stable_name(decltype(l6)); // #USN_l6
  // expected-error@#kernelSingleTask{{kernel instantiation changes the result of an evaluated '__builtin_sycl_unique_stable_name'}}
  // expected-note@#USN_l6{{'__builtin_sycl_unique_stable_name' evaluated here}}
  // expected-note@+1{{in instantiation of function template specialization}}
  kernel_single_task<class kernel6>(l6); // Used in the kernel name after builtin

  // kernel7 - expect error
  // Same as kernel11 (below) except make the lambda part of naming the kernel.
  // Test that passing a lambda to the unique stable name builtin and then
  // passing a second lambda to the kernel throws an error because the first
  // lambda is included in the signature of the second lambda, hence it changes
  // the mangling of the kernel.
  auto l7 = []() { return 1; };
  auto l8 = [](decltype(l7) *derp = nullptr) { return 2; };
  constexpr const char *l7_output =
      __builtin_sycl_unique_stable_name(decltype(l7)); // #USN_l7
  // expected-error@#kernelSingleTask{{kernel instantiation changes the result of an evaluated '__builtin_sycl_unique_stable_name'}}
  // expected-note@#USN_l7{{'__builtin_sycl_unique_stable_name' evaluated here}}
  // expected-note@+1{{in instantiation of function template specialization}}
  kernel_single_task<class kernel7>(l8);

  // kernel8 and kernel9 - expect error
  // Tests that passing a lambda to the unique stable name builtin and passing it
  // to a kernel called with an if constexpr branch causes a diagnostic on the
  // kernel9 use due to the change in the results to the stable name. This happens
  // even though the use of kernel9 happens in the false branch of a constexpr if
  // because both the true and the false branches cause the instantiation of
  // kernel_single_task.
  auto l9 = []() { return 1; };
  auto l10 = []() { return 2; };
  constexpr const char *l10_output =
      __builtin_sycl_unique_stable_name(decltype(l10)); // #USN_l10
  if constexpr (1) {
    kernel_single_task<class kernel8>(l9);
  } else {
    // expected-error@#kernelSingleTask{{kernel instantiation changes the result of an evaluated '__builtin_sycl_unique_stable_name'}}
    // expected-note@#USN_l10{{'__builtin_sycl_unique_stable_name' evaluated here}}
    // expected-note@+1{{in instantiation of function template specialization}}
    kernel_single_task<class kernel9>(l10);
  }

  // kernel11 - expect no error
  // Test that passing a lambda to the unique stable name builtin and then
  // passing a second lambda capturing the first one to the kernel does not
  // throw an error because the first lambda is not involved in naming the
  // kernel i.e., the mangling does not change.
  auto l11 = []() { return 1; };
  auto l12 = [l11]() { return 2; };
  constexpr const char *l11_output =
      __builtin_sycl_unique_stable_name(decltype(l11));
  kernel_single_task<class kernel11>(l12);

  // kernel12 - expect an error
  // Test that passing a lambda to the unique stable name builtin and then
  // passing it to the kernel as a template template parameter causes a
  // diagnostic on the kernel use due to template template parameter being
  // involved in the mangling of the kernel name.
  auto l13 = []() { return 1; };
  constexpr const char *l13_output =
      __builtin_sycl_unique_stable_name(decltype(l13)); // #USN_l13
  // expected-error@#kernelSingleTask{{kernel instantiation changes the result of an evaluated '__builtin_sycl_unique_stable_name'}}
  // expected-note@#USN_l13{{'__builtin_sycl_unique_stable_name' evaluated here}}
  // expected-note@+1{{in instantiation of function template specialization}}
  kernel_single_task<class kernel12>(S<Tangerine, decltype(l13)>{});

  // kernel13 - expect an error
  // Test that passing a lambda to the unique stable name builtin within a macro
  // and then calling the macro within the kernel causes an error on the kernel
  // and diagnoses in all the expected places despite the use of a macro.
  // expected-error@#kernelSingleTask{{kernel instantiation changes the result of an evaluated '__builtin_sycl_unique_stable_name'}}
  // expected-note@#USN_MACRO{{'__builtin_sycl_unique_stable_name' evaluated here}}
  // expected-note@+1{{in instantiation of function template specialization}}
  kernel_single_task<class kernel13>(
      []() {
        MACRO(); // #USN_MACRO
      });
}

namespace NS {}

void f() {
  // expected-error@+1{{unknown type name 'bad_var'}}
  __builtin_sycl_unique_stable_name(bad_var);
  // expected-error@+1{{use of undeclared identifier 'bad'}}
  __builtin_sycl_unique_stable_name(bad::type);
  // expected-error@+1{{no type named 'still_bad' in namespace 'NS'}}
  __builtin_sycl_unique_stable_name(NS::still_bad);

  // FIXME: warning about side-effects in an unevaluated context expected, but
  // none currently emitted.
  int i = 0;
  __builtin_sycl_unique_stable_name(decltype(i++));

  // Tests that use within a VLA does not diagnose as a side-effecting use in
  // an unevaluated context because the use within a VLA extent forces
  // evaluation.
  int j = 55;
  __builtin_sycl_unique_stable_name(int[++j]); // no warning expected
}

template <typename T>
void f2() {
  // expected-error@+1{{no type named 'bad_val' in 'St'}}
  __builtin_sycl_unique_stable_name(typename T::bad_val);
  // expected-error@+1{{no type named 'bad_type' in 'St'}}
  __builtin_sycl_unique_stable_name(typename T::bad_type);
}

struct St {};

void use() {
  // expected-note@+1{{in instantiation of}}
  f2<St>();
}
