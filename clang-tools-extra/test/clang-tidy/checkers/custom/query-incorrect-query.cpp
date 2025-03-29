// RUN: %check_clang_tidy %s custom-* %t --config-file=%S/Inputs/incorrect-clang-tidy.yml

// CHECK-MESSAGES: warning: unsupported querry kind [clang-tidy-config]
