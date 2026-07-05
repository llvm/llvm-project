// RUN: %clang_cc1 -E %s | FileCheck -strict-whitespace %s

#define hash_hash # ## # 
#define mkstr(a) # a 
#define in_between(a) mkstr(a) 
#define join(c, d) in_between(c hash_hash d) 
char p[] = join(x, y);

// CHECK: char p[] = "x ## y";

