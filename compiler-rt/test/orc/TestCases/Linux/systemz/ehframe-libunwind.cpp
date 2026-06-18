// REQUIRES: libunwind-available
// RUN: %clangxx -fexceptions -fPIC -c -o %t %s
// RUN: env LD_PRELOAD=%shared_libunwind %llvm_jitlink %t

extern "C" void llvm_jitlink_setTestResultOverride(long Value);

int main(int argc, char *argv[]) {
  llvm_jitlink_setTestResultOverride(1);
  try {
    throw 0;
  } catch (int X) {
    llvm_jitlink_setTestResultOverride(X);
  }
  return 0;
}
