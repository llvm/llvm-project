// RUN: %clang_cc1 -triple x86_64-gnu-linux -emit-llvm -o - %s | FileCheck %s -check-prefix=PLAIN
// RUN: %clang_cc1 -triple x86_64-gnu-linux -emit-llvm -o - -fsanitize=address %s | FileCheck %s -check-prefix=ASAN

typedef __typeof__(sizeof(0)) size_t;
namespace std {
  struct nothrow_t {};
  std::nothrow_t nothrow;
}
void *operator new[](size_t, const std::nothrow_t &) throw();
void *operator new[](size_t, char *);

struct C {
  int x;
  ~C();
};

C *CallNew() {
  return new C[10];
}
// PLAIN-LABEL: CallNew
// PLAIN-NOT: nosanitize
// PLAIN-NOT: __asan_poison_cxx_array_cookie
// ASAN-LABEL: CallNew
// ASAN: store{{.*}}nosanitize
// ASAN-NOT: nosanitize
// ASAN: call void @__asan_poison_cxx_array_cookie

C *CallNewNoThrow() {
  return new (std::nothrow) C[10];
}
// PLAIN-LABEL: CallNewNoThrow
// PLAIN-NOT: nosanitize
// PLAIN-NOT: __asan_poison_cxx_array_cookie
// ASAN-LABEL: CallNewNoThrow
// ASAN: store{{.*}}nosanitize
// ASAN-NOT: nosanitize
// ASAN: call void @__asan_poison_cxx_array_cookie

void CallDelete(C *c) {
  delete [] c;
}

// PLAIN-LABEL: CallDelete
// PLAIN-NOT: nosanitize
// ASAN-LABEL: CallDelete
// ASAN-NOT: nosanitize
// ASAN: call i64 @__asan_load_cxx_array_cookie
// ASAN-NOT: nosanitize

char Buffer[20];
C *CallPlacementNew() {
  return new (Buffer) C[20];
}
// ASAN-LABEL: CallPlacementNew
// ASAN-NOT: __asan_poison_cxx_array_cookie
