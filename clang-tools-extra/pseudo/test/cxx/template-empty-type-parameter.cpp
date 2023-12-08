// RUN: clang-pseudo -grammar=cxx -source=%s --print-forest | FileCheck %s
template <typename> struct MatchParents;
// CHECK: template-parameter-list~TYPENAME := tok[2]
