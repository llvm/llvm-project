// RUN: clang-tidy -checks='-*,google*' -config='{}' -dump-config - -- | FileCheck %s
// CHECK: CheckOptions:
// CHECK: {{- key: *google-readability-braces-around-statements.ShortStatementLines}}
// CHECK-NEXT: {{value: *'1'}}
// CHECK: {{- key: *google-readability-function-size.StatementThreshold}}
// CHECK-NEXT: {{value: *'800'}}
// CHECK: {{- key: *google-readability-namespace-comments.ShortNamespaceLines}}
// CHECK-NEXT: {{value: *'10'}}
// CHECK: {{- key: *google-readability-namespace-comments.SpacesBeforeComments}}
// CHECK-NEXT: {{value: *'2'}}
