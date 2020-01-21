#include <cstdio>

const char* build = BUILD;

int main(int argc, char **argv) {
  printf("argc: %d\n", argc);
  return argv[0][0];
}
