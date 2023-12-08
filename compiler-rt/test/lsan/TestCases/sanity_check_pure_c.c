// Check that we can build C code.
// RUN: %clang_lsan %s -o %t
#ifdef __cplusplus
#error "This test must be built in C mode"
#endif

int main() {
  // FIXME: ideally this should somehow check that we don't have libstdc++
  return 0;
}
