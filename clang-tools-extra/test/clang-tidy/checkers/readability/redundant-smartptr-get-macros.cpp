// RUN: %check_clang_tidy %s readability-redundant-smartptr-get %t -- \
// RUN:   -config="{CheckOptions: {readability-redundant-smartptr-get.IgnoreMacros: false}}"

#include <memory>

#define MACRO(p) p.get()

void Positive() {
  std::shared_ptr<int> x;
  if (MACRO(x) == nullptr)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:13: warning: redundant get() call on smart pointer
};
