// DEFINE: %{filecheck} = FileCheck %s --match-full-lines --check-prefix

// RUN: not %clang -fsyntax-only %s --ssaf-tu-summary-file=foobar 2>&1 | %{filecheck}=DEFAULT-ERROR
// RUN:     %clang -fsyntax-only %s --ssaf-tu-summary-file=foobar -Wno-error=scalable-static-analysis-framework 2>&1 | %{filecheck}=DEMOTED-TO-WARNING
// RUN:     %clang -fsyntax-only %s --ssaf-tu-summary-file=foobar -Wno-scalable-static-analysis-framework 2>&1 | count 0

// This test demonstrates that the "scalable-static-analysis-framework" diagnostics can be downgraded or completely silenced with the right flags.

void empty() {}

// DEFAULT-ERROR: error: failed to parse the value of '--ssaf-tu-summary-file=foobar' the value must follow the '<path>.<format>' pattern [-Wscalable-static-analysis-framework]
// DEFAULT-ERROR: 1 error generated.

// DEMOTED-TO-WARNING: warning: failed to parse the value of '--ssaf-tu-summary-file=foobar' the value must follow the '<path>.<format>' pattern [-Wscalable-static-analysis-framework]
// DEMOTED-TO-WARNING: 1 warning generated.
