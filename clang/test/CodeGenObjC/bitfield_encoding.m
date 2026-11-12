// RUN: %clang_cc1 -triple i386-unknown-unknown -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o %t %s
// RUN: grep "ib1b14" %t | count 1
// RUN: %clang_cc1 -triple i386-unknown-unknown -fobjc-runtime=macosx-fragile-10.5 -fobjc-runtime=gcc -emit-llvm -o %t %s
// RUN: grep "ib32i1b33i14" %t | count 1

struct foo{
	int a;
	int b:1;
	int c:14;
};

const char *encoding = @encode(struct foo);
