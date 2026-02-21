// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++17 -fcxx-exceptions -fexceptions -fsycl-is-host -emit-llvm -o - %s | FileCheck %s

// Validate generation of exception handling code for functions declared
// with the sycl_kernel_entry_point attribute that implicitly call a
// sycl_kernel_launch function that may throw an exception. Exception
// handling is not relevant for the generated offload kernel entry point
// function, so device compilation is intentionally not exercised.

// A unique kernel name type is required for each declared kernel entry point.
template<int> struct KN;

// A generic kernel object type.
template<int, int = 0>
struct KT {
  void operator()() const;
};


// Validate that exception handling instructions are omitted when a
// potentially throwing sycl_kernel_entry_point attributed function
// calls a potentially throwing sycl_kernel_launch function (a thrown
// exception will propagate with no explicit handling required).
namespace ns1 {
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...);
  [[clang::sycl_kernel_entry_point(KN<1>)]]
  void skep(KT<1> k) {
    k();
  }
}
// CHECK: ; Function Attrs: mustprogress noinline optnone
// CHECK: define dso_local void @_ZN3ns14skepE2KTILi1ELi0EE() #{{[0-9]+}} {
// CHECK:   call void @_ZN3ns118sycl_kernel_launchI2KNILi1EEJ2KTILi1ELi0EEEEEvPKcDpT0_(ptr noundef @.str)
// CHECK:   ret void
// CHECK: }


// Validate that exception handling instructions are emitted when a
// non-throwing sycl_kernel_entry_point attributed function calls
// a potentially throwing sycl_kernel_launch function.
namespace ns2 {
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...);
  [[clang::sycl_kernel_entry_point(KN<2>)]]
  void skep(KT<2> k) noexcept {
    k();
  }
}
// CHECK: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK: define dso_local void @_ZN3ns24skepE2KTILi2ELi0EE() #{{[0-9]+}} personality ptr @__gxx_personality_v0 {
// CHECK:   invoke void @_ZN3ns218sycl_kernel_launchI2KNILi2EEJ2KTILi2ELi0EEEEEvPKcDpT0_(ptr noundef @.str.1)
// CHECK:           to label %invoke.cont unwind label %terminate.lpad
// CHECK: invoke.cont:
// CHECK:   ret void
// CHECK: terminate.lpad:
// CHECK:   call void @__clang_call_terminate(ptr %1) #{{[0-9]+}}
// CHECK:   unreachable
// CHECK: }


// Validate that exception handling instructions are omitted when a
// potentially throwing sycl_kernel_entry_point attributed function
// calls a non-throwing sycl_kernel_launch function (a thrown
// exception will terminate within sycl_kernel_launch).
namespace ns3 {
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...) noexcept;
  [[clang::sycl_kernel_entry_point(KN<3>)]]
  void skep(KT<3> k) {
    k();
  }
}
// CHECK: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK: define dso_local void @_ZN3ns34skepE2KTILi3ELi0EE() #{{[0-9]+}} {
// CHECK:   call void @_ZN3ns318sycl_kernel_launchI2KNILi3EEJ2KTILi3ELi0EEEEEvPKcDpT0_(ptr noundef @.str.2)
// CHECK:   ret void
// CHECK: }


// Validate that exception handling instructions are omitted when a
// non-throwing sycl_kernel_entry_point attributed function calls a
// non-throwing sycl_kernel_launch function.
namespace ns4 {
  template<typename KN, typename... Ts>
  void sycl_kernel_launch(const char *, Ts...) noexcept;
  [[clang::sycl_kernel_entry_point(KN<4>)]]
  void skep(KT<4> k) noexcept {
    k();
  }
}
// CHECK: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK: define dso_local void @_ZN3ns44skepE2KTILi4ELi0EE() #{{[0-9]+}} {
// CHECK:   call void @_ZN3ns418sycl_kernel_launchI2KNILi4EEJ2KTILi4ELi0EEEEEvPKcDpT0_(ptr noundef @.str.3)
// CHECK:   ret void
// CHECK: }
