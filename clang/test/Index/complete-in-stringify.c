const char *func(const char *);

#define MORE __FILE__

#define M(x) "1"#x
#define N(x) func("2"#x MORE)

void foo(const char *);

int test() {
    foo(M(x()));
    foo(N(x()));
}

// RUN: c-index-test -code-completion-at=%s:11:11 %s | FileCheck %s
// RUN: c-index-test -code-completion-at=%s:12:11 %s | FileCheck %s
// CHECK: Natural language
