// RUN: %check_clang_tidy -std=c++11-or-later %s modernize-use-trailing-return-type %t \
// RUN:   -config="{CheckOptions: {modernize-use-trailing-return-type.TransformLambdas: none, \
// RUN:                            modernize-use-trailing-return-type.TransformFunctions: false}}"

// CHECK-MESSAGES: warning: The check 'modernize-use-trailing-return-type' will not perform any analysis because 'TransformFunctions' and 'TransformLambdas' are disabled. [clang-tidy-config]
