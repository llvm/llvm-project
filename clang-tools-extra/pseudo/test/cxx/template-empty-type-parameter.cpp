// RUN: clang-pseudo -grammar=%cxx-bnf-file -source=%s --print-forest | FileCheck %s
template <typename> struct MatchParents;
// CHECK: template-parameter-list~TYPENAME := tok[2]
