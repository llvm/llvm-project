// REQUIRES: leak-detection, glibc, asan-64-bits
// UNSUPPORTED: android, aarch64-target-arch, target=powerpc64{{.*}}
//
// Regression test for LeakSanitizer crashing at exit after dlclose() unmaps a
// module's dynamic TLS (DTLS) block, and for tracking new DTLS allocations
// when a dlclose()d TLS module ID (dso_id) is reused by a subsequent dlopen().
//
// RUN: %clangxx_asan -DDSO1 %s -shared -fPIC -o %t-dso1a.so
// RUN: %clangxx_asan -DDSO1 %s -shared -fPIC -o %t-dso1b.so
// RUN: %clangxx_asan -DDSO2 %s -shared -fPIC -o %t-dso2.so
// RUN: %clangxx_asan -DDSO3 %s -shared -fPIC -o %t-dso3.so
// RUN: %clangxx_asan -DDSO4 %s -shared -fPIC -o %t-dso4.so
// RUN: %clangxx_asan %s -ldl -pthread -o %t
// RUN: %env_asan_opts=detect_leaks=1:quarantine_size_mb=0 \
// RUN:   LSAN_OPTIONS=log_threads=1:leak_check_at_exit=1 \
// RUN:   %run %t %t-dso1a.so %t-dso1b.so %t-dso2.so %t-dso3.so %t-dso4.so \
// RUN:   2>&1 | FileCheck %s

#if defined(DSO1)
// Exceeds glibc static TLS surplus so dynamic TLS is used; 16 MiB also forces
// ASan's secondary (mmap-backed) allocator so freeing it with
// quarantine_size_mb=0 immediately munmaps the pages.
__thread char huge_tls_buf[16 * 1024 * 1024];

extern "C" void touch_dso1() {
  huge_tls_buf[0] = 1;
  huge_tls_buf[sizeof(huge_tls_buf) - 1] = 2;
}
#elif defined(DSO2)
// Also exceeds glibc static TLS surplus so dynamic TLS is guaranteed when this
// module reuses a dlclose()d TLS module ID.
__thread void *reused_tls_buf[(2 * 1024 * 1024) / sizeof(void *)];

extern "C" void set_dso2_tls(void *p) { reused_tls_buf[0] = p; }
extern "C" void *get_dso2_tls() { return reused_tls_buf[0]; }
extern "C" void *get_dso2_tls_addr() { return &reused_tls_buf[0]; }
#elif defined(DSO3)
__thread char trigger_tls_buf[16 * 1024 * 1024];

extern "C" void touch_dso3() {
  trigger_tls_buf[0] = 3;
  trigger_tls_buf[sizeof(trigger_tls_buf) - 1] = 4;
}
#elif defined(DSO4)
// Anchor module kept loaded at slot N+2 so closing dso3 at slot N+1 does not
// shrink glibc's dl_tls_max_dtv_idx below N+1.
__thread char anchor_tls_buf[1024 * 1024];

extern "C" void touch_dso4() { anchor_tls_buf[0] = 5; }
#else
#  include <assert.h>
#  include <dlfcn.h>
#  include <stdio.h>
#  include <stdlib.h>

typedef void (*touch_fn)();
typedef void (*set_fn)(void *);
typedef void *(*get_fn)();

int main(int argc, char **argv) {
  assert(argc == 6);
  const char *dso1a_path = argv[1];
  const char *dso1b_path = argv[2];
  const char *dso2_path = argv[3];
  const char *dso3_path = argv[4];
  const char *dso4_path = argv[5];

  // Open four modules concurrently so they get consecutive TLS modids:
  //   dso1a -> slot N
  //   dso3  -> slot N+1
  //   dso4  -> slot N+2 (anchor kept open)
  //   dso1b -> slot N+3
  // Touch all four so their DTLS blocks are allocated and recorded in LSan's
  // DTV.
  void *h1a = dlopen(dso1a_path, RTLD_LAZY | RTLD_LOCAL);
  void *h3 = dlopen(dso3_path, RTLD_LAZY | RTLD_LOCAL);
  void *h4 = dlopen(dso4_path, RTLD_LAZY | RTLD_LOCAL);
  void *h1b = dlopen(dso1b_path, RTLD_LAZY | RTLD_LOCAL);
  assert(h1a && h3 && h4 && h1b);
  auto touch1a = reinterpret_cast<touch_fn>(dlsym(h1a, "touch_dso1"));
  auto touch3 = reinterpret_cast<touch_fn>(dlsym(h3, "touch_dso3"));
  auto touch4 = reinterpret_cast<touch_fn>(dlsym(h4, "touch_dso4"));
  auto touch1b = reinterpret_cast<touch_fn>(dlsym(h1b, "touch_dso1"));
  assert(touch1a && touch3 && touch4 && touch1b);
  touch1a();
  touch3();
  touch4();
  touch1b();

  // Close dso1a (slot N), dso3 (slot N+1), and dso1b (slot N+3), leaving dso4
  // (slot N+2) open:
  //   - Closing dso1b (the highest modid, N+3) shrinks dl_tls_max_dtv_idx down
  //     to N+2 (dso4) on all glibc versions.
  //   - On newer glibc (where dlclose sets dl_tls_dtv_gaps), the next dlopen
  //     reuses the first gap at slot N (dso1a), while slot N+1 (dso3) remains
  //     an unmapped gap below dl_tls_max_dtv_idx (N+2).
  //   - On older glibc (where dlclose does not set dl_tls_dtv_gaps unless the
  //     highest modid shrinks), the next dlopen increments dl_tls_max_dtv_idx
  //     from N+2 to N+3, reusing slot N+3 (dso1b), while slots N and N+1 remain
  //     unmapped gaps.
  dlclose(h1a);
  dlclose(h3);
  dlclose(h1b);

  // Open dso2. It reuses an already-recorded DTV slot (slot N on newer glibc,
  // slot N+3 on older glibc) and increments dl_tls_generation. Touching dso2's
  // DTLS triggers _dl_update_slotinfo, which:
  //   1. Unmaps the old DTLS at the reused slot and allocates dso2's DTLS in
  //      its place, verifying DTLS_on_tls_get_addr updates the existing DTV
  //      slot.
  //   2. Unmaps dso3's DTLS at slot N+1 without reusing slot N+1, verifying
  //      LSan skips the inaccessible DTLS range at exit instead of crashing.
  void *h2 = dlopen(dso2_path, RTLD_LAZY | RTLD_LOCAL);
  assert(h2);
  auto set2 = reinterpret_cast<set_fn>(dlsym(h2, "set_dso2_tls"));
  auto get2 = reinterpret_cast<get_fn>(dlsym(h2, "get_dso2_tls"));
  auto addr2 = reinterpret_cast<get_fn>(dlsym(h2, "get_dso2_tls_addr"));
  assert(set2 && get2 && addr2);
  set2(malloc(256));
  assert(get2() != nullptr);
  fprintf(stderr, "DSO2_DTLS=%p\n", addr2());
  fflush(stderr);

  return 0;
}
#endif

// CHECK: DSO2_DTLS=[[DSO2:0x[0-9a-f]+]]
// CHECK-DAG: DTLS {{[0-9]+}} at [[DSO2]]-0x{{[0-9a-f]+}}.
// CHECK-DAG: DTLS {{[0-9]+}} at {{.*}} is not accessible.
// CHECK-NOT: ERROR: LeakSanitizer
