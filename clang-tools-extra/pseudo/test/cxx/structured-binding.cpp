// RUN: clang-pseudo -grammar=cxx -source=%s --start-symbol=statement-seq --print-forest | FileCheck %s

// Verify there is no false parse of the structured binding declaration.
ABC[post] = abc;
// CHECK: statement-seq~expression-statement := expression ;
// CHECK: postfix-expression [ expr-or-braced-init-list ]
