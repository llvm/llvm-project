// Test that SYCL device compilation inherits type properties from the host
// architecture via AuxTriple/HostTriple.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fsycl-is-device \
// RUN:   -fsyntax-only -verify=linux %s

// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -aux-triple x86_64-pc-windows-msvc -fsycl-is-device \
// RUN:   -fsyntax-only -verify=windows %s

// linux-no-diagnostics
// windows-no-diagnostics

void test_long_size() {
#if defined(_WIN64)
  static_assert(sizeof(long) == 4, "long should be 4 bytes on Windows");
#else
  static_assert(sizeof(long) == 8, "long should be 8 bytes on Linux");
#endif
}

