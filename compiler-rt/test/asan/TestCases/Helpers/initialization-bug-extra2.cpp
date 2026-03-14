// 'z' is dynamically initialized global from different TU.
extern int z;
int __attribute__((noinline)) initY() {
  return z + 1;
}
int y = initY();
