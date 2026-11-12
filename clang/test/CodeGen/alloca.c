// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

typedef __SIZE_TYPE__ size_t;
void *alloca(size_t size);
char *strcpy(char *restrict s1, const char *restrict s2);
int puts(const char *s);
int main(int argc, char **argv) {
  char *C = (char*)alloca(argc);
  strcpy(C, argv[0]);
  puts(C);
}
