// RUN: %check_clang_tidy -std=c++98-or-later %s readability-trailing-comma %t -- \
// RUN:   -config='{CheckOptions: {readability-trailing-comma.SingleLineCommaPolicy: Ignore, readability-trailing-comma.MultiLineCommaPolicy: Ignore}}'


// CHECK-MESSAGES: warning: The check 'readability-trailing-comma' will not perform any analysis because 'SingleLineCommaPolicy' and 'MultiLineCommaPolicy' are both set to 'Ignore'. [clang-tidy-config]

enum E1 {
  A
};

enum E2 { B, };
