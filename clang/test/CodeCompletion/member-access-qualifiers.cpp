struct deque_base {
  int &size();
  const int &size() const;
};

struct deque : private deque_base {
    int size() const;
};

auto x = deque().
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):18 %s -o - | FileCheck %s
// CHECK: COMPLETION: size : [#int#]size()[# const#]
// CHECK: COMPLETION: size (Hidden,InBase,Inaccessible) : [#int &#]deque_base::size()
