// RUN: not %clang_cc1 -std=c++20 -fsyntax-only -code-completion-at=%s:7:39 %s -DTEST_TEMPLATE
// RUN: not %clang_cc1 -std=c++20 -fsyntax-only -code-completion-at=%s:12:19 %s -DTEST_CONSTRAINT

#ifdef TEST_TEMPLATE
void template_cutoff() {
  [=]() mutable -> decltype(y + x)
  requires(is_same<decltype((y)), int /*invoke completion here*/ &>
#endif

#ifdef TEST_CONSTRAINT
void constraint_cutoff() {
  []() requires x /*invoke completion here*/
#endif
