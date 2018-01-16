namespace ns {
int bar;
}

int main() { return ns:: }

// RUN: echo "namespace ns { int foo; }" > %t.h
// RUN: c-index-test -write-pch %t.h.pch -x c++-header %t.h
//
// RUN: c-index-test -code-completion-at=%s:5:26 -include %t.h %s | FileCheck -check-prefix=WITH-PCH %s
// WITH-PCH: {TypedText bar}
// WITH-PCH: {TypedText foo}

// RUN: env CINDEXTEST_COMPLETION_SKIP_PREAMBLE=1 c-index-test -code-completion-at=%s:5:26 -include %t.h %s | FileCheck -check-prefix=SKIP-PCH %s
// SKIP-PCH-NOT: foo
// SKIP-PCH: {TypedText bar}
// SKIP-PCH-NOT: foo

// Verify that with *no* preamble (no -include flag) we still get local results.
// SkipPreamble used to break this, by making lookup *too* lazy.
// RUN: env CINDEXTEST_COMPLETION_SKIP_PREAMBLE=1 c-index-test -code-completion-at=%s:5:26 %s | FileCheck -check-prefix=NO-PCH %s
// NO-PCH-NOT: foo
// NO-PCH: {TypedText bar}
// NO-PCH-NOT: foo

