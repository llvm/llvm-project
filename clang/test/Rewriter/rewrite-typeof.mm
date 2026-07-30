// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: FileCheck -check-prefix CHECK-LP --input-file=%t-rw.cpp %s

extern "C" {
extern "C" void *_Block_copy(const void *aBlock);
extern "C" void _Block_release(const void *aBlock);
}

int main() {
    __attribute__((__blocks__(byref))) int a = 42;
    int save_a = a;

    void (^b)(void) = ^{
        ((__typeof(^{ a = 2; }))_Block_copy((const void *)(^{ a = 2; })));
    };

    ((__typeof(b))_Block_copy((const void *)(b)));

    return 0;
}

// CHECK-LP: ((void (^)(void))_Block_copy((const void *)(b)))

void f() {
	int a;	
	__typeof__(a) aVal = a;
	char *a1t = (char *)@encode(__typeof__(a));
        __typeof__(aVal) bVal;
	char *a2t = (char *)@encode(__typeof__(bVal));
        __typeof__(bVal) cVal = bVal;
	char *a3t = (char *)@encode(__typeof__(cVal));

}


// CHECK-LP: int aVal =  a;

// CHECK-LP: int bVal;
