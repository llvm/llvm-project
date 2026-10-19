// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -O2 -fno-unroll-loops -falign-loops=32 -emit-cir %s -o %t-opts.cir
// RUN: FileCheck --input-file=%t-opts.cir %s --check-prefix=OPTS

void use(int);
int arr[3];

// C++11 and later loops are 'mustprogress', unless they are trivial infinite
// loops.
void must_progress(int n) {
// CIR-LABEL: cir.func {{.*}} @_Z13must_progressi
  // CIR: cir.for
  for (int i = 0; i < n; ++i)
    use(i);
  // CIR: } loop_hints < must_progress>

  // CIR: cir.while
  while (n--)
    use(n);
  // CIR: } loop_hints < must_progress>

  // CIR: cir.do
  do {
    use(n);
  } while (n > 0);
  // CIR: } loop_hints < must_progress>

  // Like classic codegen, range-for loops are not 'mustprogress'.
  // CIR: cir.for
  for (int x : arr)
    use(x);
  // CIR-NOT: loop_hints

  // CIR: cir.for
  for (;;) {
  }
  // CIR-NOT: loop_hints
  // CIR: cir.return
}


void pragmas(int n) {
// CIR-LABEL: cir.func {{.*}} @_Z7pragmasi
// OPTS-LABEL: cir.func {{.*}} @_Z7pragmasi
#pragma clang loop unroll_count(4) vectorize(enable) interleave_count(2)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress vectorize_enable interleave_count = 2 unroll_count = 4>
// OPTS: } loop_hints < must_progress vectorize_enable interleave_count = 2 unroll_count = 4 code_align = 32>
#pragma nounroll
  while (n)
    --n;
// CIR: } loop_hints < must_progress unroll = disable>
// OPTS: } loop_hints < must_progress unroll = disable code_align = 32>
#pragma unroll
  do {
    --n;
  } while (n > 0);
// CIR: } loop_hints < must_progress unroll = enable>
// OPTS: } loop_hints < must_progress unroll = enable code_align = 32>
#pragma unroll 8
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress unroll_count = 8>
// OPTS: } loop_hints < must_progress unroll_count = 8 code_align = 32>
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; ++i)
    use(i);
// CIR: } loop_hints < must_progress unroll = full>
// OPTS: } loop_hints < must_progress unroll = full code_align = 32>
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress vectorize_width = 1 vectorize_scalable = false interleave_count = 1>
// OPTS: } loop_hints < must_progress vectorize_width = 1 vectorize_scalable = false interleave_count = 1 unroll = disable code_align = 32>
#pragma clang loop vectorize_width(8)
  for (int x : arr)
    use(x);
// CIR: } loop_hints < vectorize_width = 8 vectorize_scalable = false>
#pragma clang loop vectorize_width(4, scalable)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress vectorize_width = 4 vectorize_scalable = true>
#pragma clang loop vectorize_width(4, fixed)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress vectorize_width = 4 vectorize_scalable = false>
#pragma clang loop vectorize(assume_safety)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress vectorize_enable parallel>
#pragma clang loop vectorize_predicate(enable)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress vectorize_predicate = true>
#pragma clang loop vectorize_predicate(disable)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress vectorize_predicate = false>
#pragma clang loop distribute(enable)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress distribute = true>
#pragma clang loop distribute(disable)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress distribute = false>
#pragma unroll_and_jam
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress unroll_and_jam = enable>
#pragma nounroll_and_jam
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress unroll_and_jam = disable>
#pragma unroll_and_jam(4)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress unroll_and_jam_count = 4>
#pragma clang loop pipeline(disable)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress pipeline_disabled>
#pragma clang loop pipeline_initiation_interval(3)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress pipeline_initiation_interval = 3>
#pragma clang loop licm(disable)
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress licm_disabled>
  [[clang::code_align(16)]]
  for (int i = 0; i < n; ++i)
    use(i);
// CIR: } loop_hints < must_progress code_align = 16>
// OPTS: } loop_hints < must_progress unroll = disable code_align = 16>
}

