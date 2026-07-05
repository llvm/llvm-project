// RUN: clang-tidy -checks=-*,modernize-use-nullptr -explain-config | FileCheck --check-prefix=CHECK-MESSAGE1 %s
// RUN: clang-tidy -config="{Checks: '-*,modernize-use-nullptr'}" -explain-config | FileCheck --check-prefix=CHECK-MESSAGE2 %s
// RUN: clang-tidy -checks=modernize-use-nullptr -config="{Checks: '-*,modernize-use-nullptr'}" -explain-config | FileCheck --check-prefix=CHECK-MESSAGE3 %s
// RUN: clang-tidy -checks=modernize-use-nullptr -config="{Checks: '-*,-modernize-use-nullptr'}" %S/Inputs/explain-config/a.cc -explain-config -- | FileCheck --check-prefix=CHECK-MESSAGE4 %s
// RUN: clang-tidy -checks=modernize-use-nullptr -config="{Checks: '-*,modernize-*'}" -explain-config | FileCheck --check-prefix=CHECK-MESSAGE5 %s
// RUN: clang-tidy -explain-config %S/Inputs/explain-config/a.cc -- | grep "'modernize-use-nullptr' is enabled in the .*[/\\]Inputs[/\\]explain-config[/\\].clang-tidy."

// CHECK-MESSAGE1: 'modernize-use-nullptr' is enabled in the command-line option '-checks'.
// CHECK-MESSAGE2: 'modernize-use-nullptr' is enabled in the command-line option '-config'.
// CHECK-MESSAGE3: 'modernize-use-nullptr' is enabled in the command-line option '-checks'.
// CHECK-MESSAGE4: 'modernize-use-nullptr' is enabled in the command-line option '-checks'.
// CHECK-MESSAGE5: 'modernize-use-nullptr' is enabled in the command-line option '-checks'.
