// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.2 %s -emit-llvm -o - | FileCheck %s
// <rdar://problem/10551376>

struct FOO {
	unsigned int x;
};

#pragma pack(push, 2)

// CHECK: %struct.BAR = type <{ %struct.FOO, i8, i8 }>
struct BAR : FOO {
	char y;
};

#pragma pack(pop)

BAR* x = 0;