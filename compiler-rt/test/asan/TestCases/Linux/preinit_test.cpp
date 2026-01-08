// RUN: mkdir -p %t.dir && cd %t.dir
// RUN: %clangxx -DFUNC=zzzz %s -shared -o %dynamiclib -fPIC
// RUN: %clangxx_asan -DFUNC=main %s -o %t.dir/EXE %ld_flags_rpath_exe
// RUN: %run %t.dir/EXE

// GNU driver doesn't handle .so files properly.
// REQUIRES: Clang

// This test ensures that we call __asan_init early enough.
// We build a shared library w/o asan instrumentation
// and the binary with asan instrumentation.
// Both files include the same header (emulated by -DFUNC here)
// with C++ template magic which runs global initializer at library load time.
// The function get() is instrumented with asan, but called
// before the usual constructors are run.
// So, we must make sure that __asan_init is executed even earlier.
//
// See http://gcc.gnu.org/bugzilla/show_bug.cgi?id=56393

struct A {
  int foo() const { return 0; }
};
A get () { return A(); }
template <class> struct O {
  static A const e;
};
template <class T> A const O <T>::e = get();
int FUNC() {
  return O<int>::e.foo();
}

