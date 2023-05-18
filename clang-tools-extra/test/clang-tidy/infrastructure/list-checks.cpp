// RUN: mkdir -p %T/clang-tidy/list-checks/
// RUN: echo '{Checks: "-*,google-*"}' > %T/clang-tidy/.clang-tidy
// RUN: cd %T/clang-tidy/list-checks
// RUN: clang-tidy -list-checks | grep "^ *google-"
