// RUN: %clang_cc1 -E %s | FileCheck %s
#define hash_hash # ## # 
#define mkstr(a) # a 
#define in_between(a) mkstr(a) 
#define join(c, d) in_between(c hash_hash d) 
// CHECK: "x ## y";
join(x, y);

#define FOO(x) A x B
// CHECK: A ## B;
FOO(##);
