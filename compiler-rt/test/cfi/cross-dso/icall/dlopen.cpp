// RUN: %clangxx_cfi_dso -DSHARED_LIB %s -fPIC -shared -o %t1-so.so
// RUN: %clangxx_cfi_dso %s -o %t1
// RUN: %expect_crash %t1 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %t1 cast 2>&1 | FileCheck --check-prefix=CFI-CAST %s
// RUN: %expect_crash %t1 dlclose 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi_dso -DB32 -DSHARED_LIB %s -fPIC -shared -o %t2-so.so
// RUN: %clangxx_cfi_dso -DB32 %s -o %t2
// RUN: %expect_crash %t2 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %t2 cast 2>&1 | FileCheck --check-prefix=CFI-CAST %s
// RUN: %expect_crash %t2 dlclose 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi_dso -DB64 -DSHARED_LIB %s -fPIC -shared -o %t3-so.so
// RUN: %clangxx_cfi_dso -DB64 %s -o %t3
// RUN: %expect_crash %t3 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %t3 cast 2>&1 | FileCheck --check-prefix=CFI-CAST %s
// RUN: %expect_crash %t3 dlclose 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx_cfi_dso -DBM -DSHARED_LIB %s -fPIC -shared -o %t4-so.so
// RUN: %clangxx_cfi_dso -DBM %s -o %t4
// RUN: %expect_crash %t4 2>&1 | FileCheck --check-prefix=CFI %s
// RUN: %expect_crash %t4 cast 2>&1 | FileCheck --check-prefix=CFI-CAST %s
// RUN: %expect_crash %t4 dlclose 2>&1 | FileCheck --check-prefix=CFI %s

// RUN: %clangxx -g -DBM -DSHARED_LIB -DNOCFI %s -fPIC -shared -o %t5-so.so
// RUN: %clangxx -g -DBM -DNOCFI %s -ldl -o %t5
// RUN: %t5 2>&1 | FileCheck --check-prefix=NCFI %s
// RUN: %t5 cast 2>&1 | FileCheck --check-prefix=NCFI %s
// RUN: %t5 dlclose 2>&1 | FileCheck --check-prefix=NCFI %s

// Test that calls to uninstrumented library are unchecked.
// RUN: %clangxx -DBM -DSHARED_LIB %s -fPIC -shared -o %t6-so.so
// RUN: %clangxx_cfi_dso -DBM %s -o %t6
// RUN: %t6 2>&1 | FileCheck --check-prefix=NCFI %s
// RUN: %t6 cast 2>&1 | FileCheck --check-prefix=NCFI %s

// Call-after-dlclose is checked on the caller side.
// RUN: %expect_crash %t6 dlclose 2>&1 | FileCheck --check-prefix=CFI %s

// Tests calls into dlopen-ed library.
// REQUIRES: cxxabi

#include <assert.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>

#include <string>

struct A {
  virtual void f();
};

#ifdef SHARED_LIB

#include "../../utils.h"
struct B {
  virtual void f();
};
void B::f() {}

extern "C" void *create_B() {
  create_derivers<B>();
  return (void *)(new B());
}

extern "C" __attribute__((aligned(4096))) void do_nothing() {}

#else

void A::f() {}

static const int kCodeAlign = 4096;
static const int kCodeSize = 4096;
static char saved_code[kCodeSize];
static char *real_start;

static void save_code(char *p) {
  real_start = (char *)(((uintptr_t)p) & ~(kCodeAlign - 1));
  memcpy(saved_code, real_start, kCodeSize);
}

static void restore_code() {
  char *code =
      (char *)mmap(real_start, kCodeSize, PROT_READ | PROT_WRITE | PROT_EXEC,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, 0, 0);
  assert(code == real_start);
  memcpy(code, saved_code, kCodeSize);
  __clear_cache(code, code + kCodeSize);
}

int main(int argc, char *argv[]) {
  const bool test_cast = argc > 1 && strcmp(argv[1], "cast") == 0;
  const bool test_dlclose = argc > 1 && strcmp(argv[1], "dlclose") == 0;

  std::string name = std::string(argv[0]) + "-so.so";
  void *handle = dlopen(name.c_str(), RTLD_NOW);
  assert(handle);
  void *(*create_B)() = (void *(*)())dlsym(handle, "create_B");
  assert(create_B);

  void *p = create_B();
  A *a;

  // CFI: =0=
  // CFI-CAST: =0=
  // NCFI: =0=
  fprintf(stderr, "=0=\n");

  if (test_cast) {
    // Test cast. BOOM.
    a = (A*)p;
  } else {
    // Invisible to CFI. Test virtual call later.
    memcpy(&a, &p, sizeof(a));
  }

  // CFI: =1=
  // CFI-CAST-NOT: =1=
  // NCFI: =1=
  fprintf(stderr, "=1=\n");

  if (test_dlclose) {
    // Imitate an attacker sneaking in an executable page where a dlclose()d
    // library was loaded. This needs to pass w/o CFI, so for the testing
    // purpose, we just copy the bytes of a "void f() {}" function back and
    // forth.
    void (*do_nothing)() = (void (*)())dlsym(handle, "do_nothing");
    assert(do_nothing);
    save_code((char *)do_nothing);

    int res = dlclose(handle);
    assert(res == 0);

    restore_code();

    do_nothing(); // UB here
  } else {
    a->f(); // UB here
  }

  // CFI-NOT: =2=
  // CFI-CAST-NOT: =2=
  // NCFI: =2=
  fprintf(stderr, "=2=\n");
}
#endif
