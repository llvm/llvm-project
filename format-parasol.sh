#!/usr/bin/env bash

# Modified by Sunscreen under the AGPLv3 license; see the README at the
# repository root for more information

GIT_ROOT=$(git rev-parse --show-toplevel)

find ./llvm/lib/Target/Parasol -name '*.cpp' -o -name '*.h' -o -name "*.def" | xargs "$GIT_ROOT/build/bin/clang-format" -i
