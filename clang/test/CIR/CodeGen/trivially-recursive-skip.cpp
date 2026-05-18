// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -O2 -disable-llvm-passes %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -O2 -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

extern "C++" {
extern char *strrchr(char *__s, int __c) __attribute__((__nothrow__))
__attribute__((__pure__)) __attribute__((__nonnull__(1)))
__asm__("strrchr");
extern char *strrchr(char *__s, int __c) __attribute__((__nothrow__))
    __asm__("strrchr") __attribute__((__always_inline__))
    __attribute__((__gnu_inline__));
extern __inline __attribute__((__nothrow__))
    __attribute__((__always_inline__)) __attribute__((__gnu_inline__))
char *strrchr(char *__s, int __c) {
  return __builtin_strrchr(__s, __c);
}
}

extern "C" int puts(const char *);

extern "C" void caller(char *s) {
  if (s) {
    const char *base = strrchr(s, '/');
    puts(base ? base + 1 : s);
  } else {
    puts("(null)");
  }
}

// CIR-NOT: cir.func {{.*}}available_externally @strrchr
// LLVM-NOT: define {{.*}}available_externally{{.*}}@strrchr
// OGCG-NOT: define {{.*}}available_externally{{.*}}@strrchr
