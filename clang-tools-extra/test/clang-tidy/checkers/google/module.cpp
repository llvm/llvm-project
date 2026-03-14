// RUN: clang-tidy -checks='-*,google*' -config='{}' -dump-config - -- | FileCheck %s
// CHECK: CheckOptions:
// CHECK-DAG: {{google-readability-braces-around-statements.ShortStatementLines: '1'}}
// CHECK-DAG: {{google-readability-function-size.StatementThreshold: '800'}}
// CHECK-DAG: {{google-readability-namespace-comments.ShortNamespaceLines: '10'}}
// CHECK-DAG: {{google-readability-namespace-comments.SpacesBeforeComments: '2'}}
