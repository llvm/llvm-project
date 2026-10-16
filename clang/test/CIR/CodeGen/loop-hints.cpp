// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -O2 -fno-unroll-loops -falign-loops=32 -emit-cir %s -o %t-opts.cir
// RUN: FileCheck --input-file=%t-opts.cir %s --check-prefix=OPTS

void use(int);
int arr[3];

// C++11 and later loops are 'mustprogress', unless they are trivial infinite
// loops.
void must_progress(int n) {
  for (int i = 0; i < n; ++i)
    use(i);
  while (n--)
    use(n);
  do {
    use(n);
  } while (n > 0);
  // Like classic codegen, range-for loops are not 'mustprogress'.
  for (int x : arr)
    use(x);
  for (;;) {
  }
}

// CIR-LABEL: cir.func {{.*}} @_Z13must_progressi
// CIR: cir.for
// CIR: } loop_hints < must_progress>
// CIR: cir.while
// CIR: } loop_hints < must_progress>
// CIR: cir.do
// CIR: } loop_hints < must_progress>
// CIR: cir.for
// CIR-NOT: loop_hints
// CIR: cir.for
// CIR-NOT: loop_hints
// CIR: cir.return

void pragmas(int n) {
#pragma clang loop unroll_count(4) vectorize(enable) interleave_count(2)
  for (int i = 0; i < n; ++i)
    use(i);
#pragma nounroll
  while (n)
    --n;
#pragma unroll
  do {
    --n;
  } while (n > 0);
#pragma unroll 8
  for (int i = 0; i < n; ++i)
    use(i);
#pragma clang loop unroll(full)
  for (int i = 0; i < 4; ++i)
    use(i);
#pragma clang loop vectorize(disable) interleave(disable)
  for (int i = 0; i < n; ++i)
    use(i);
#pragma clang loop vectorize_width(8)
  for (int x : arr)
    use(x);
#pragma clang loop vectorize_width(4, scalable)
  for (int i = 0; i < n; ++i)
    use(i);
#pragma clang loop vectorize_width(4, fixed)
  for (int i = 0; i < n; ++i)
    use(i);
#pragma clang loop vectorize(assume_safety)
  for (int i = 0; i < n; ++i)
    use(i);
#pragma clang loop vectorize_predicate(enable)
  for (int i = 0; i < n; ++i)
    use(i);
#pragma clang loop vectorize_predicate(disable)
  for (int i = 0; i < n; ++i)
    use(i);
#pragma clang loop distribute(enable)
  for (int i = 0; i < n; ++i)
    use(i);
#pragma clang loop distribute(disable)
  for (int i = 0; i < n; ++i)
    use(i);
#pragma unroll_and_jam
  for (int i = 0; i < n; ++i)
    use(i);
#pragma nounroll_and_jam
  for (int i = 0; i < n; ++i)
    use(i);
#pragma unroll_and_jam(4)
  for (int i = 0; i < n; ++i)
    use(i);
#pragma clang loop pipeline(disable)
  for (int i = 0; i < n; ++i)
    use(i);
#pragma clang loop pipeline_initiation_interval(3)
  for (int i = 0; i < n; ++i)
    use(i);
#pragma clang loop licm(disable)
  for (int i = 0; i < n; ++i)
    use(i);
  [[clang::code_align(16)]]
  for (int i = 0; i < n; ++i)
    use(i);
}

// CIR-LABEL: cir.func {{.*}} @_Z7pragmasi
// CIR: } loop_hints < must_progress vectorize_enable interleave_count = 2 unroll_count = 4>
// CIR: } loop_hints < must_progress unroll = disable>
// CIR: } loop_hints < must_progress unroll = enable>
// CIR: } loop_hints < must_progress unroll_count = 8>
// CIR: } loop_hints < must_progress unroll = full>
// CIR: } loop_hints < must_progress vectorize_width = 1 vectorize_scalable = false interleave_count = 1>
// CIR: } loop_hints < vectorize_width = 8 vectorize_scalable = false>
// CIR: } loop_hints < must_progress vectorize_width = 4 vectorize_scalable = true>
// CIR: } loop_hints < must_progress vectorize_width = 4 vectorize_scalable = false>
// CIR: } loop_hints < must_progress vectorize_enable parallel>
// CIR: } loop_hints < must_progress vectorize_predicate = true>
// CIR: } loop_hints < must_progress vectorize_predicate = false>
// CIR: } loop_hints < must_progress distribute = true>
// CIR: } loop_hints < must_progress distribute = false>
// CIR: } loop_hints < must_progress unroll_and_jam = enable>
// CIR: } loop_hints < must_progress unroll_and_jam = disable>
// CIR: } loop_hints < must_progress unroll_and_jam_count = 4>
// CIR: } loop_hints < must_progress pipeline_disabled>
// CIR: } loop_hints < must_progress pipeline_initiation_interval = 3>
// CIR: } loop_hints < must_progress licm_disabled>
// CIR: } loop_hints < must_progress code_align = 16>

// -falign-loops applies to every loop, unless [[clang::code_align]] overrides
// it. -fno-unroll-loops disables unrolling of every loop that no pragma
// overrides.
// OPTS-LABEL: cir.func {{.*}} @_Z7pragmasi
// OPTS: } loop_hints < must_progress vectorize_enable interleave_count = 2 unroll_count = 4 code_align = 32>
// OPTS: } loop_hints < must_progress unroll = disable code_align = 32>
// OPTS: } loop_hints < must_progress unroll = enable code_align = 32>
// OPTS: } loop_hints < must_progress unroll_count = 8 code_align = 32>
// OPTS: } loop_hints < must_progress unroll = full code_align = 32>
// OPTS: } loop_hints < must_progress vectorize_width = 1 vectorize_scalable = false interleave_count = 1 unroll = disable code_align = 32>
// OPTS: } loop_hints < must_progress unroll = disable code_align = 16>
