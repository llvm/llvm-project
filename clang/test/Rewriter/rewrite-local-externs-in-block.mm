// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

extern "C" int printf(const char*, ...);

void bar(void (^block)()) {
	block();
}

int main() {
	static int myArr[3] = {1, 2, 3};
	printf ("%d %d %d\n", myArr[0], myArr[1], myArr[2]);
	
	bar(^{
		printf ("%d %d %d\n", myArr[0], myArr[1], myArr[2]);
		myArr[0] = 42;
		myArr[2] = 100;
		printf ("%d %d %d\n", myArr[0], myArr[1], myArr[2]);
	});

	printf ("%d %d %d\n", myArr[0], myArr[1], myArr[2]);
}
