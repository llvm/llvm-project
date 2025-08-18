// This test case checks for a bug where anonymous code results in
// out-of-order stack frame numbers.

// UNSUPPORTED: android
// UNSUPPORTED: aarch64
// UNSUPPORTED: darwin
// UNSUPPORTED: ios

// RUN: %clangxx_asan -O0 -g %s -o %t
// RUN: %env_asan_opts=symbolize=0 not %run %t DUMMY_ARG > %t.asan_report 2>&1
// RUN: %asan_symbolize -d --log-level debug --log-dest %t_debug_log_output.txt -l %t.asan_report > %t.asan_report_sym
// RUN: FileCheck --input-file=%t.asan_report_sym %s

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>

static void call_via_anon_page(void (*fn)()) {
  const size_t pagesz = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  uint8_t *mem =
      static_cast<uint8_t *>(mmap(nullptr, pagesz, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (mem == MAP_FAILED)
    perror("mmap");

#if defined(__x86_64__)
  // x86_64: mov rax, imm64; call rax; ret
  // 48 B8 <imm64> FF D0 C3
  uint8_t stub[2 + 8 + 2 + 1] = {0x48, 0xB8};
  std::memcpy(stub + 2, &fn, sizeof(fn)); // imm64
  stub[2 + 8] = 0xFF;
  stub[2 + 9] = 0xD0;  // call rax
  stub[2 + 10] = 0xC3; // ret
#else
#  error "unsupported platform"
#endif

  std::memcpy(mem, stub, sizeof(stub));
  mprotect(mem, pagesz, PROT_READ | PROT_EXEC);

  using Thunk = void (*)();
  reinterpret_cast<Thunk>(mem)();

  munmap(mem, pagesz);
}

static void crash() {
  char p[8], *s = p;

  // out-of-bounds write to trigger ASan
  s[16] = 42;
}

int main() {
  call_via_anon_page(crash);
  return 0;
}

// Check that the numbering of the stackframes is correct.

// CHECK: AddressSanitizer: stack-buffer-overflow
// CHECK-NEXT: WRITE of size
// CHECK-NEXT: #0 0x{{[0-9a-fA-F]+}} in crash
// CHECK-NEXT: #1 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: #2 0x{{[0-9a-fA-F]+}} in main
