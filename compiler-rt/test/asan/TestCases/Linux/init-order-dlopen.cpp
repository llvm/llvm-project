// Regression test for
// https://code.google.com/p/address-sanitizer/issues/detail?id=178

// RUN: %clangxx_asan -O0 -DSHARED_LIB %s -fPIC -shared -o %t-so.so
// RUN: %clangxx_asan -O0 %s %libdl -Wl,--export-dynamic -o %t
// RUN: %env_asan_opts=strict_init_order=true %run %t 2>&1

// dlopen() can not be intercepted on Android, making strict_init_order nearly
// useless there.
// UNSUPPORTED: android

#if defined(SHARED_LIB)
#include <stdio.h>

struct Bar {
  Bar(int val) : val(val) { printf("Bar::Bar(%d)\n", val); }
  int val;
};

int get_foo_val();
Bar global_bar(get_foo_val());
#else  // SHARED LIB
#include <dlfcn.h>
#include <stdio.h>
#include <string>
struct Foo {
  Foo() : val(42) { printf("Foo::Foo()\n"); }
  int val;
};

Foo global_foo;

int get_foo_val() {
  return global_foo.val;
}

int main(int argc, char *argv[]) {
  std::string path = std::string(argv[0]) + "-so.so";
  void *handle = dlopen(path.c_str(), RTLD_NOW);
  if (!handle) {
    printf("error in dlopen(): %s\n", dlerror());
    return 1;
  }
  printf("%d\n", get_foo_val());
  return 0;
}
#endif  // SHARED_LIB
