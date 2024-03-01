void test(char *ptr) { asm volatile("clflushopt %0" : "+m"(*ptr)::"memory"); }

int main(int argc, char **argv) {
  test(argv[0]);
  return 0;
}
