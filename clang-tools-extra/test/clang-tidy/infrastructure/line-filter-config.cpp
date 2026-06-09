// RUN: clang-tidy -checks='-*,modernize-use-using' -config="{LineFilter: [{name: 'line-filter-config.cpp', lines: [[8, 8]]}]}" %s -- 2>&1 | FileCheck --check-prefix=CONFIG %s
// RUN: clang-tidy -checks='-*,modernize-use-using' -config="{LineFilter: [{name: 'line-filter-config.cpp', lines: [[8, 8]]}]}" -line-filter="[{name: 'line-filter-config.cpp', lines: [[12, 12]]}]" %s -- 2>&1 | FileCheck --check-prefix=CLI %s

typedef int BeforeLineFilter;
// CONFIG-NOT: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CLI-NOT: :[[@LINE-2]]:1: warning: use 'using' instead of 'typedef'

typedef int ConfigWarn;
// CONFIG: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]
// CLI-NOT: :[[@LINE-2]]:1: warning: use 'using' instead of 'typedef'

typedef int CliWarn;
// CONFIG-NOT: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CLI: :[[@LINE-2]]:1: warning: use 'using' instead of 'typedef' [modernize-use-using]

typedef int AfterLineFilter;
// CONFIG-NOT: :[[@LINE-1]]:1: warning: use 'using' instead of 'typedef'
// CLI-NOT: :[[@LINE-2]]:1: warning: use 'using' instead of 'typedef'
