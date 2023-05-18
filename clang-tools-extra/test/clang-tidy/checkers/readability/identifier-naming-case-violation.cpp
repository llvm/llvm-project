// RUN: clang-tidy %s -checks=readability-identifier-naming \
// RUN:   -config="{CheckOptions: {\
// RUN:   readability-identifier-naming.FunctionCase: camelback, \
// RUN:   readability-identifier-naming.VariableCase: camelBack, \
// RUN:   readability-identifier-naming.ClassCase: UUPER_CASE, \
// RUN:   readability-identifier-naming.StructCase: CAMEL, \
// RUN:   readability-identifier-naming.EnumCase: AnY_cASe, \
// RUN:   readability-identifier-naming.VirtualMethodCase: lEaDiNg_upper_SNAKE_CaSe, \
// RUN:   }}" -- 2>&1 | FileCheck %s --implicit-check-not="{{warning|error}}:"

// CHECK-DAG: warning: invalid configuration value 'camelback' for option 'readability-identifier-naming.FunctionCase'; did you mean 'camelBack'? [clang-tidy-config]
// CHECK-DAG: warning: invalid configuration value 'UUPER_CASE' for option 'readability-identifier-naming.ClassCase'; did you mean 'UPPER_CASE'? [clang-tidy-config]
// Don't try to suggest an alternative for 'CAMEL'
// CHECK-DAG: warning: invalid configuration value 'CAMEL' for option 'readability-identifier-naming.StructCase' [clang-tidy-config]
// This fails on the EditDistance, but as it matches ignoring case suggest the correct value
// CHECK-DAG: warning: invalid configuration value 'AnY_cASe' for option 'readability-identifier-naming.EnumCase'; did you mean 'aNy_CasE'? [clang-tidy-config]
// CHECK-DAG: warning: invalid configuration value 'lEaDiNg_upper_SNAKE_CaSe' for option 'readability-identifier-naming.VirtualMethodCase'; did you mean 'Leading_upper_snake_case'? [clang-tidy-config]
