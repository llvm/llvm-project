// RUN: %clang_cc1 -isystem %S/Inputs/ -fsycl-is-device -triple spirv64 -aux-triple x86_64-pc-windows-msvc -fsyntax-only -verify %s
// RUN: %clang_cc1 -isystem %S/Inputs/ -fsycl-is-device -triple spirv64 -fsyntax-only -verify %s

// Check that there is no error/warning emitted for cdecl functions compiled for
// SYCL device. Make sure variadic calls from within device code are diagnosed.

__inline __cdecl int printf(char const* const _Format, ...) { return 0; }

// FIXME: that should be diagnosed.
[[clang::sycl_external]] int foo(int, ...) { return 0; }

__inline __cdecl int moo() { return 0; }

void bar() {
  printf("hello\n");
}

template<typename KN, typename...Args>
void sycl_kernel_launch(Args ...args) {}

template<typename KN, typename K>
[[clang::sycl_kernel_entry_point(KN)]]
__cdecl void sycl_entry_point(K k) {
  k(); // expected-note {{called by}}
}

int main() {
  //expected-error@+1 {{SYCL device code does not support variadic functions}}
  sycl_entry_point<class kn>([]() { printf("world\n");
     moo();
  //expected-error@+1 {{SYCL device code does not support variadic functions}}
     foo(1,2); });
  bar();
  return 0;
}
