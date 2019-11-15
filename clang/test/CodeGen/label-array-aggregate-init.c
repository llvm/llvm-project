// RUN: %clang -cc1 -emit-llvm %s -o /dev/null

int main() {
L:
  (void)(void *[]){ &&L, 0, 0 };
}
