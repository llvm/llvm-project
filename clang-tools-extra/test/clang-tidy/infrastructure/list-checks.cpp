// RUN: mkdir -p %t.dir/clang-tidy/list-checks/
// RUN: echo '{Checks: "-*,google-*"}' > %t.dir/clang-tidy/.clang-tidy
// RUN: cd %t.dir/clang-tidy/list-checks
// RUN: clang-tidy -list-checks | grep "^ *google-"
