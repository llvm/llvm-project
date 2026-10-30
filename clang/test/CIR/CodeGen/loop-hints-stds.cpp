// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR,CXX17
// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-98.cir
// RUN: FileCheck --input-file=%t-98.cir %s --check-prefixes=CIR,CXX98

// C++17 has different mustprogress guarantees that dont' exist before this, so
// mustprogress changes behavior.

void use(int);

// CIR-LABEL: cir.func {{.*}} @_Z5hintsi
void hints(int n) {
#pragma GCC unroll 4
  for (int i = 0; i < n; ++i)
    use(i);
// CXX17: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll_count = 4>
// CXX98: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < unroll_count = 4>
#pragma clang loop unroll(disable)
  for (int i = 0; i < n; ++i)
    use(i);
// CXX17: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable>
// CXX98: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < unroll = disable>
#pragma clang loop unroll(enable)
  for (int i = 0; i < n; ++i)
    use(i);
// CXX17: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = enable>
// CXX98: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < unroll = enable>
#pragma clang loop interleave(enable)
  for (int i = 0; i < n; ++i)
    use(i);
// CXX17: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_enable>
// CXX98: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < vectorize_enable>
#pragma clang loop vectorize_width(4) unroll_count(2)
  for (int i = 0; i < n; ++i)
    use(i);
// CXX17: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_width = 4 vectorize_scalable = false unroll_count = 2>
// CXX98: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < vectorize_width = 4 vectorize_scalable = false unroll_count = 2>
  for (int i = 0; i < n; ++i)
    use(i);
// CXX17: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress>
// CXX98-NOT: loop_hints

// CIR: cir.return
}


// CIR-LABEL: cir.func {{.*}} @_Z6nestedi
void nested(int n) {
#pragma clang loop unroll(full)
  for (int i = 0; i < n; ++i) {
#pragma clang loop vectorize(enable)
    for (int j = 0; j < n; ++j)
      use(j);
// CXX17: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress vectorize_enable>
// CXX98: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < vectorize_enable>
  }
// CXX17: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = full>
// CXX98: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < unroll = full>
}


template <typename T> void tmpl(T n) {
#pragma clang loop unroll_count(3)
  while (n--)
    use(n);
}

void instantiate() {
  tmpl(4);
// CIR-LABEL: cir.func {{.*}} @_Z4tmplIiEvT_
// CXX17: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll_count = 3>
// CXX98: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < unroll_count = 3>
  auto l = [](int n) {
#pragma nounroll
    do {
      use(n);
    } while (--n);
  };
  l(2);
// CIR-LABEL: cir.func {{.*}} @_ZZ11instantiatevENK3$_0clEi
// CXX17: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < must_progress unroll = disable>
// CXX98: } loop_id distinct{{\[[0-9]+\]}}<> loop_hints < unroll = disable>
}

