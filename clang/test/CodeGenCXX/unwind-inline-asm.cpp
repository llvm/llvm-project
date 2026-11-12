// RUN: %clang_cc1 -triple x86_64-unknown-linux -emit-llvm -DUNWIND -fcxx-exceptions -fexceptions -o - %s | FileCheck -check-prefixes CHECK,CHECK-UNWIND %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -emit-llvm -fcxx-exceptions -fexceptions -o - %s | FileCheck -check-prefixes CHECK,CHECK-NO-UNWIND %s

extern "C" void printf(const char *fmt, ...);

struct DropBomb {
  bool defused = false;

  ~DropBomb() {
    if (defused) {
      return;
    }
    printf("Boom!\n");
  }
};

extern "C" void trap() {
  throw "Trap";
}

// CHECK: define dso_local void @test()
extern "C" void test() {
  DropBomb bomb;
// CHECK-UNWIND: invoke void asm sideeffect unwind "call trap"
// CHECK-NO-UNWIND: call void asm sideeffect "call trap"
#ifdef UNWIND
  asm volatile("call trap" ::
                   : "unwind");
#else
  asm volatile("call trap" ::
                   :);
#endif
  bomb.defused = true;
}
