// RUN: not clang-tidy %s -checks='-*'
// RUN: clang-tidy %s -checks='-*' --allow-no-checks -- | count 0
