// RUN: %clangxx -c -o %t %s
// RUN: %llvm_jitlink %t
//
// REQUIRES: system-darwin && host-arch-compatible

static int x = 1;

class Init {
public:
  Init() { x = 0; }
};

static Init I;

int main(int argc, char *argv[]) {
  return x;
}
