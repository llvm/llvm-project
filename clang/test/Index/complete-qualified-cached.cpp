namespace a_namespace {};
class Class { static void foo(); };
Class::
// Completion for a_namespace should be available at the start of the line.
// START-OF-LINE: a_namespace
// START-OF-LINE: Class
// -- Using cached completions.
// RUN: env CINDEXTEST_EDITING=1 c-index-test -code-completion-at=%s:3:1 %s \
// RUN: | FileCheck --check-prefix=START-OF-LINE %s
// -- Without cached completions.
// RUN: c-index-test -code-completion-at=%s:3:1 %s \
// RUN: | FileCheck --check-prefix=START-OF-LINE %s
//
//
// ... and should not be available after 'Class::^'
// AFTER-QUALIFIER: Class
// -- Using cached completions.
// RUN: env CINDEXTEST_EDITING=1 c-index-test -code-completion-at=%s:3:8 %s \
// RUN: | FileCheck --implicit-check-not=a_namespace --check-prefix=AFTER-QUALIFIER %s
// -- Without cached completions.
// RUN: c-index-test -code-completion-at=%s:3:8 %s \
// RUN: | FileCheck --implicit-check-not=a_namespace --check-prefix=AFTER-QUALIFIER %s
