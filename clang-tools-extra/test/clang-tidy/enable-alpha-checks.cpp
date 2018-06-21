// Check if '-allow-enabling-analyzer-alpha-checkers' is visible for users.
// RUN: clang-tidy -help | not grep 'allow-enabling-analyzer-alpha-checkers'

// Check if '-allow-enabling-analyzer-alpha-checkers' enables alpha checks.
// RUN: clang-tidy -checks=* -list-checks | not grep 'clang-analyzer-alpha'
// RUN: clang-tidy -checks=* -list-checks -allow-enabling-analyzer-alpha-checkers | grep 'clang-analyzer-alpha'
