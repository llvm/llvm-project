// RUN: %clang_cl -EHsc -MD -c -o %t %s
// RUN: %llvm_jitlink %t

extern "C" __declspec(dllimport) void llvm_jitlink_setTestResultOverride(
    long Value);

int main(int argc, char *argv[]) {
  llvm_jitlink_setTestResultOverride(1);
  try {
    throw 0;
  } catch (int X) {
    llvm_jitlink_setTestResultOverride(X);
  }
  return 0;
}
