struct map {
  void find(int);
  void find();
};

int main() {
  map *m;
  m->find(10);
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:8:11 %s -o - | FileCheck %s
  // CHECK: OVERLOAD: [#void#]find(<#int#>)

  // Also check when the lhs is an explicit pr-value.
  (m+0)->find(10);
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:13:15 %s -o - | FileCheck %s
}
