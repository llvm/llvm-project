void __llvm_profile_initialize_file(void);
int __llvm_orderfile_dump(void);

__attribute__((noinline)) int f(int a);

__attribute__((noinline)) int g(int a);

int main(int argc, const char *argv[]) {
  int a = f(argc);
  int t = 0;
  for (int i = 0; i < argc; i++)
    t += g(a);
  f(t);
  __llvm_profile_initialize_file();
  __llvm_orderfile_dump();
  return 0;
}
