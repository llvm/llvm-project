// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

@class NSArray;
int main(void) {
	NSArray *foo;
	for (Class c in foo) { }
}
