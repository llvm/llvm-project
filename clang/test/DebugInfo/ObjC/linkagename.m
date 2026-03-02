// RUN: %clang_cc1  -debug-info-kind=limited -S -o %t %s
// RUN: not grep "001-[F bar" %t
// Linkage name should not use 001 prefix in debug info.

@interface F 
-(int) bar;
@end

@implementation F
-(int) bar {
	return 42;
}
@end

extern int f(F *fn) {
	return [fn bar];
}
	
