// Ensure that the frontend adds the proper metadata when CFI is
// enabled.
// RUN: %clang --target=x86_64-scei-ps4 -funified-lto -flto -fsanitize=cfi -fvisibility=hidden -fno-sanitize-ignorelist -c %s -o %t.o
// RUN: llvm-dis %t.o -o %t1
// RUN: FileCheck <%t1.0 %s

typedef int (*FuncPtr)();

int a() { return 1; }
int b() { return 2; }
int c() { return 3; }

FuncPtr func[3] = {a,b,c};

int
main(int argc, char *argv[]) {
  // CHECK: call i1 @llvm.type.test
  return func[argc]();
  // CHECK-LABEL: trap
}

// CHECK: typeTests:
