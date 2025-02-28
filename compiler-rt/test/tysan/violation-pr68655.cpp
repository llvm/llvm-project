// RUN: %clang_tysan -O0 %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

// https://github.com/llvm/llvm-project/issues/68655
struct S1 {
  long long a;
  long long b;
};

// CHECK: TypeSanitizer: type-aliasing-violation on address
// CHECK-NEXT:  READ of size 4 at {{.+}} with type int accesses an existing object of type long long (in {{.*}}S1 at offset 0)
// CHECK-NEXT: in copyMem(S1*, S1*) {{.*/?}}violation-pr68655.cpp:19

void inline copyMem(S1 *dst, S1 *src) {
  unsigned *d = reinterpret_cast<unsigned *>(dst);
  unsigned *s = reinterpret_cast<unsigned *>(src);

  for (int i = 0; i < sizeof(S1) / sizeof(unsigned); i++) {
    *d = *s;
    d++;
    s++;
  }
}

void math(S1 *dst, int *srcA, int idx_t) {
  S1 zero[4];
  for (int i = 0; i < 2; i++) {
    zero[i].a = i + idx_t;
    zero[i].b = i * idx_t;
  }

  copyMem(&dst[idx_t], &zero[srcA[idx_t]]);
}

int main() {
  S1 dst = {0};
  int Src[2] = {0, 0};
  math(&dst, &Src[0], 0);
  return 0;
}
