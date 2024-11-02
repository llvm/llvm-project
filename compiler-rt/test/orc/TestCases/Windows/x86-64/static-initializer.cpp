// RUN: %clang_cl -MD -c -o %t %s
// RUN: %llvm_jitlink %t

extern "C" __declspec(dllimport) void llvm_jitlink_setTestResultOverride(
    long Value);

extern "C" int init() { return -1; }
extern "C" int init2() { return -2; }

extern "C" int a = init();
extern "C" int b = 2;
extern "C" int c = init2();

int main(int argc, char *argv[]) {
  llvm_jitlink_setTestResultOverride(a + b + c + 1);
  return 0;
}
