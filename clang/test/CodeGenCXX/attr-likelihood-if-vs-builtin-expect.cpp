// RUN: %clang_cc1 -O1 -emit-llvm %s -o - -triple=x86_64-linux-gnu | FileCheck %s

// Verifies the output of __builtin_expect versus the output of the likelihood
// attributes. They should generate the same probabilities for the branches.

extern bool a();
extern bool b();
extern bool c();

void ab1(int &i) {
  // CHECK-LABEL: define{{.*}}ab1
  // CHECK: br {{.*}} !prof !2
  // CHECK: br {{.*}} !prof !2
  // CHECK: br {{.*}} !prof !2
  if (__builtin_expect(a() && b() && a(), 1)) {
    ++i;
  } else {
    --i;
  }
}

void al(int &i) {
  // CHECK-LABEL: define{{.*}}al
  // CHECK: br {{.*}} !prof !2
  // CHECK: br {{.*}} !prof !2
  // CHECK: br {{.*}} !prof !2
  if (a() && b() && c()) [[likely]] {
    ++i;
  } else {
    --i;
  }
}

void ab0(int &i) {
  // CHECK-LABEL: define{{.*}}ab0
  // CHECK: br {{.*}}else{{$}}
  // CHECK: br {{.*}}else{{$}}
  // CHECK: br {{.*}} !prof !8
  if (__builtin_expect(a() && b() && c(), 0)) {
    ++i;
  } else {
    --i;
  }
}

void au(int &i) {
  // CHECK-LABEL: define{{.*}}au
  // CHECK: br {{.*}}else{{$}}
  // CHECK: br {{.*}}else{{$}}
  // CHECK: br {{.*}} !prof !8
  if (a() && b() && c()) [[unlikely]] {
    ++i;
  } else {
    --i;
  }
}

void ob1(int &i) {
  // CHECK-LABEL: define{{.*}}ob1
  // CHECK: br {{.*}}false{{$}}
  // CHECK: br {{.*}}rhs{{$}}
  // CHECK: br {{.*}} !prof !2
  if (__builtin_expect(a() || b() || a(), 1)) {
    i = 0;
  } else {
    --i;
  }
}

void ol(int &i) {
  // CHECK-LABEL: define{{.*}}ol
  // CHECK: br {{.*}}false{{$}}
  // CHECK: br {{.*}}false2{{$}}
  // CHECK: br {{.*}} !prof !2
  if (a() || b() || c()) [[likely]] {
    i = 0;
  } else {
    --i;
  }
}

void ob0(int &i) {
  // CHECK-LABEL: define{{.*}}ob0
  // CHECK: br {{.*}} !prof !8
  // CHECK: br {{.*}} !prof !8
  // CHECK: br {{.*}} !prof !8
  if (__builtin_expect(a() || b() || c(), 0)) {
    i = 0;
  } else {
    --i;
  }
}

void ou(int &i) {
  // CHECK-LABEL: define{{.*}}ou
  // CHECK: br {{.*}} !prof !8
  // CHECK: br {{.*}} !prof !8
  // CHECK: br {{.*}} !prof !8
  if (a() || b() || c()) [[unlikely]] {
    i = 0;
  } else {
    --i;
  }
}

void nb1(int &i) {
  // CHECK-LABEL: define{{.*}}nb1
  // CHECK: storemerge{{.*}} !prof !8
  if (__builtin_expect(!a(), 1)) {
    ++i;
  } else {
    --i;
  }
}

void nl(int &i) {
  // CHECK-LABEL: define{{.*}}nl
  // CHECK: storemerge{{.*}} !prof !8
  if (!a()) [[likely]] {
    ++i;
  } else {
    --i;
  }
}

void nb0(int &i) {
  // CHECK-LABEL: define{{.*}}nb0
  // CHECK: storemerge{{.*}} !prof !2
  if (__builtin_expect(!a(), 0)) {
    ++i;
  } else {
    --i;
  }
}

void nu(int &i) {
  // CHECK-LABEL: define{{.*}}nu
  // CHECK: storemerge{{.*}} !prof !2
  if (!a()) [[unlikely]] {
    ++i;
  } else {
    --i;
  }
}

void tb1(int &i) {
  // CHECK-LABEL: define{{.*}}tb1
  // CHECK: br {{.*}}false{{$}}
  // CHECK: br {{.*}}end{{$}}
  // CHECK: br {{.*}}end{{$}}
  // CHECK: storemerge{{.*}} !prof !2
  if (__builtin_expect(a() ? b() : c(), 1)) {
    ++i;
  } else {
    --i;
  }
}

void tl(int &i) {
  // CHECK-LABEL: define{{.*}}tl
  // CHECK: br {{.*}}false{{$}}
  // CHECK: br {{.*}}end{{$}}
  // CHECK: br {{.*}}end{{$}}
  // CHECK: storemerge{{.*}} !prof !2
  if (bool d = a() ? b() : c()) [[likely]] {
    ++i;
  } else {
    --i;
  }
}

void tl2(int &i) {
  // CHECK-LABEL: define{{.*}}tl
  // CHECK: br {{.*}}false{{$}}
  // CHECK: br {{.*}} !prof !2
  // CHECK: br {{.*}} !prof !2
  if (a() ? b() : c()) [[likely]] {
    ++i;
  } else {
    --i;
  }
}

void tb0(int &i) {
  // CHECK-LABEL: define{{.*}}tb0
  // CHECK: br {{.*}}false{{$}}
  // CHECK: br {{.*}}end{{$}}
  // CHECK: br {{.*}}end{{$}}
  // CHECK: storemerge{{.*}} !prof !8
  if (__builtin_expect(a() ? b() : c(), 0)) {
    ++i;
  } else {
    --i;
  }
}

void tu(int &i) {
  // CHECK-LABEL: define{{.*}}tu
  // CHECK: br {{.*}}false{{$}}
  // CHECK: br {{.*}}end{{$}}
  // CHECK: br {{.*}}end{{$}}
  // CHECK: storemerge{{.*}} !prof !8
  if (bool d = a() ? b() : c()) [[unlikely]] {
    ++i;
  } else {
    --i;
  }
}

void tu2(int &i) {
  // CHECK-LABEL: define{{.*}}tu
  // CHECK: br {{.*}}false{{$}}
  // CHECK: br {{.*}} !prof !8
  // CHECK: br {{.*}} !prof !8
  if (a() ? b() : c()) [[unlikely]] {
    ++i;
  } else {
    --i;
  }
}

// CHECK: !2 = !{!"branch_weights", i32 2000, i32 1}
// CHECK: !8 = !{!"branch_weights", i32 1, i32 2000}
