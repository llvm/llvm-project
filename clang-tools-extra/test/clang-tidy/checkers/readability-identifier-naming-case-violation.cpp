// RUN: clang-tidy %s -checks=readability-identifier-naming \
// RUN:   -config="{CheckOptions: [\
// RUN:   {key: readability-identifier-naming.FunctionCase, value: camelback}, \
// RUN:   {key: readability-identifier-naming.VariableCase, value: camelBack}, \
// RUN:   {key: readability-identifier-naming.ClassCase, value: UUPER_CASE}, \
// RUN:   {key: readability-identifier-naming.StructCase, value: CAMEL}, \
// RUN:   {key: readability-identifier-naming.EnumCase, value: AnY_cASe}, \
// RUN:   ]}" 2>&1 | FileCheck %s --implicit-check-not warning

// CHECK-DAG: warning: invalid configuration value 'camelback' for option 'readability-identifier-naming.FunctionCase'; did you mean 'camelBack'?{{$}}
// CHECK-DAG: warning: invalid configuration value 'UUPER_CASE' for option 'readability-identifier-naming.ClassCase'; did you mean 'UPPER_CASE'?{{$}}
// Don't try to suggest an alternative for 'CAMEL'
// CHECK-DAG: warning: invalid configuration value 'CAMEL' for option 'readability-identifier-naming.StructCase'{{$}}
// This fails on the EditDistance, but as it matches ignoring case suggest the correct value
// CHECK-DAG: warning: invalid configuration value 'AnY_cASe' for option 'readability-identifier-naming.EnumCase'; did you mean 'aNy_CasE'?{{$}}
