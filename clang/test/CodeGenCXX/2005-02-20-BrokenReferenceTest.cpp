// RUN: %clang_cc1 -emit-llvm %s -o /dev/null

void test(unsigned char *b, int rb) {
  typedef unsigned char imgfoo[10][rb];
  imgfoo &br = *(imgfoo *)b;

  br[0][0] = 1;

  rb = br[0][0];
}
