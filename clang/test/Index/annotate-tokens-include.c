#include "annotate-tokens-include.h"

// RUN: c-index-test -test-annotate-tokens=%s:1:1:2:1 %s | FileCheck %s
// CHECK: Punctuation: "#" [1:1 - 1:2] inclusion directive=annotate-tokens-include.h
// CHECK: Identifier: "include" [1:2 - 1:9] inclusion directive=annotate-tokens-include.h
// CHECK: Literal: ""annotate-tokens-include.h"" [1:10 - 1:37] inclusion directive=annotate-tokens-include.h

