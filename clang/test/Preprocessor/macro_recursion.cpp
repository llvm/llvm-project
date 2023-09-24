// RUN: %clang_cc1 -E %s -std=c++20 | FileCheck %s

#define merge_all_expand2(a, b) a ## b
#define merge_all_expand(a, b) merge_all_expand2(a, b)
#define concat_all(head, ...)merge_all_expand(head,__VA_OPT__(__THIS_MACRO__(__VA_ARGS__)))
0: concat_all(aa, bb, cc)
1: [concat_all()]
// CHECK: 0: aabbcc
// CHECK: 1: []

#undef merge_all_expand
#undef merge_all_expand2
#undef concat_all

#define reverse(head, ...)__VA_OPT__(__THIS_MACRO__(__VA_ARGS__),)head

2: reverse(1,2,3)

// CHECK: 2: 3,2,1

#undef reverse

#define fold_left(op, head, ...)(__VA_OPT__(__THIS_MACRO__(op,__VA_ARGS__)op)head)

3: fold_left(+, 1, 2, 3, 4)
// CHECK: 3: ((((4)+3)+2)+1)

#undef fold_left
