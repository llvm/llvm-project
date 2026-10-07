// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -Wno-stringop-overread -Werror=fortify-source -verify=fortify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -Wno-stringop-overread -Werror=fortify-source -verify=fortify
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c %s -Wno-fortify-source -Wstringop-overread -verify=disabled
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ %s -Wno-fortify-source -Wstringop-overread -verify=disabled
// disabled-no-diagnostics

typedef __SIZE_TYPE__ size_t;

#ifdef __cplusplus
extern "C" {
#endif
long write(int, const void *, size_t);
long pwrite(int, const void *, size_t, long);
long pwrite64(int, const void *, size_t, long long);
#ifdef __cplusplus
}
#endif

// These overread diagnostics belong to -Wfortify-source, independently of
// whether -Wstringop-overread is enabled.
void test_write(size_t n) {
  char buf[4];
  write(0, buf, 4);
  pwrite(0, buf, 4, 0);
  pwrite64(0, buf, 4, 0);
  write(0, buf, n);
  pwrite(0, buf, n, 0);
  pwrite64(0, buf, n, 0);
  write(0, buf, 8); // fortify-error {{'write' will always read past the end of the source buffer; source buffer has size 4, but the size is 8}}
  pwrite(0, buf, 8, 0); // fortify-error {{'pwrite' will always read past the end of the source buffer; source buffer has size 4, but the size is 8}}
  pwrite64(0, buf, 8, 0); // fortify-error {{'pwrite64' will always read past the end of the source buffer; source buffer has size 4, but the size is 8}}
}
