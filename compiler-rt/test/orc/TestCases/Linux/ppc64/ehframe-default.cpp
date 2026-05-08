// RUN: %clangxx -fexceptions -fPIC -c -o %t %s
// RUN: %llvm_jitlink %t

// REQUIRES: disabled
//// https://github.com/llvm/llvm-project/issues/175094

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
