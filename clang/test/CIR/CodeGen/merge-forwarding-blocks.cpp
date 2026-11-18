// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -O1 -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-OPT --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -O0 -disable-O0-optnone -disable-llvm-passes %s -o %t-o0.ll
// RUN: FileCheck --check-prefix=LLVM-O0 --input-file=%t-o0.ll %s

// Blocks with a single predecessor produced when CIR's structured control
// flow is flattened are merged into their predecessors by the
// llvm-simplify-cfg pass at -O1+, in the LLVM dialect before translation.
// At -O0 the pass is off and the raw shape is preserved.

// LLVM-OPT-LABEL: define {{.*}} @_Z11fwd_if_elseiii(
// LLVM-OPT:       store i32 %{{.*}}, ptr {{%.*}}, align 4
// LLVM-OPT-NEXT:  store i32 %{{.*}}, ptr {{%.*}}, align 4
// LLVM-OPT-NEXT:  store i32 %{{.*}}, ptr {{%.*}}, align 4
// LLVM-OPT-NEXT:  [[LOAD:%.*]] = load i32, ptr {{%.*}}, align 4
// LLVM-OPT-NEXT:  [[CMP:%.*]] = icmp ne i32 [[LOAD]], 0
// LLVM-OPT-NEXT:  br i1 [[CMP]], label {{%.*}}, label {{%.*}}
int fwd_if_else(int c, int a, int b) {
  if (c)
    return a;
  else
    return b;
}

// The loop header block directly follows the entry branch: the unmerged
// lowering threads the entry through forwarding blocks before reaching
// the header, and the loop body through a forwarding block before the
// increment.

// LLVM-OPT-LABEL: define {{.*}} @_Z8fwd_loopi(
// LLVM-OPT:       store i32 0, ptr {{%.*}}, align 4
// LLVM-OPT:       store i32 0, ptr {{%.*}}, align 4
// LLVM-OPT:       br label {{%.*}}
// LLVM-OPT-EMPTY:
// LLVM-OPT-NEXT:  {{^[0-9]+:.*$}}
// LLVM-OPT-NEXT:  {{%.*}} = load i32, ptr {{%.*}}, align 4
// LLVM-OPT:       {{%.*}} = add nsw i32 {{%.*}}, {{%.*}}
// LLVM-OPT-NOT:   br label
// LLVM-OPT:       {{%.*}} = add nsw i32 {{%.*}}, 1
int fwd_loop(int n) {
  int total = 0;
  for (int i = 0; i < n; ++i)
    total += i;
  return total;
}

// LLVM-OPT-LABEL: define {{.*}} @_Z11fwd_optnoneiii(
// LLVM-OPT:       store i32 %2, ptr {{%.*}}, align 4
// LLVM-OPT-NEXT:  br label {{%.*}}
__attribute__((optnone)) int fwd_optnone(int c, int a, int b) {
  if (c)
    return a;
  else
    return b;
}

// LLVM-O0-LABEL: define {{.*}} @_Z11fwd_if_elseiii(
// LLVM-O0:       store i32 %{{.*}}, ptr {{%.*}}, align 4
// LLVM-O0-NEXT:  store i32 %{{.*}}, ptr {{%.*}}, align 4
// LLVM-O0-NEXT:  store i32 %{{.*}}, ptr {{%.*}}, align 4
// LLVM-O0-NEXT:  br label {{%.*}}
