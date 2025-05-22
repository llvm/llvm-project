// REQUIRES: host-supports-jit
// UNSUPPORTED: system-aix
//
// We disable RTTI to avoid problems on Windows for non-RTTI builds of LLVM
// where the JIT cannot find ??_7type_info@@6B@.
// RUN: cat %s | clang-repl -Xcc -fno-rtti | FileCheck %s
// RUN: cat %s | clang-repl -Xcc -fno-rtti -Xcc -O2 | FileCheck %s

extern "C" int printf(const char *, ...);

struct A { int a; A(int a) : a(a) {} virtual ~A(); };

// Then define the virtual destructor as inline out-of-line, in a separate
// PartialTranslationUnit.
inline A::~A() { printf("~A(%d)\n", a); }

// Create one instance with new and delete it.
A *a1 = new A(1);
delete a1;
// CHECK: ~A(1)

// Also create one global that will be auto-destructed.
A a2(2);
// CHECK: ~A(2)
