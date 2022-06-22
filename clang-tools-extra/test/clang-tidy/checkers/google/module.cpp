// RUN: clang-tidy -checks='-*,google*' -config='{}' -dump-config - -- | FileCheck %s
// CHECK: CheckOptions:
// CHECK-DAG: {{- key: *google-readability-braces-around-statements.ShortStatementLines *[[:space:]] *value: *'1'}}
// CHECK-DAG: {{- key: *google-readability-function-size.StatementThreshold *[[:space:]] *value: *'800'}}
// CHECK-DAG: {{- key: *google-readability-namespace-comments.ShortNamespaceLines *[[:space:]] *value: *'10'}}
// CHECK-DAG: {{- key: *google-readability-namespace-comments.SpacesBeforeComments *[[:space:]] *value: *'2'}}
