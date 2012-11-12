// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %t.mm -o %t-rw.cpp
// RUN: FileCheck -check-prefix LP --input-file=%t-rw.cpp %s
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o %t-modern-rw.cpp
// RUN: FileCheck -check-prefix LP --input-file=%t-modern-rw.cpp %s
// radar 7607781

typedef struct {
	int a;
	int b;
} mystruct;
	
void g(int (^block)(mystruct s)) {
	mystruct x;
	int v = block(x);
}

void f(const void **arg) {
	__block const void **q = arg;
	g(^(mystruct s){
		*q++ = (void*)s.a;
		return 314;
		});
}

// CHECK-LP: (__Block_byref_q_0 *)&q
