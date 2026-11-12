// RUN: c-index-test -code-completion-at=%s:1:1 -Wunknown-foo-bar-warning -Werror %s

void f();
