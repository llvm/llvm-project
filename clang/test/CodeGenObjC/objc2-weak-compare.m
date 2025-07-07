// RUN: %clang_cc1 -Wno-error=return-type -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: %clang_cc1 -Wno-error=return-type -x objective-c++ -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s

@interface PBXTarget 
{

PBXTarget * __weak _lastKnownTarget;
PBXTarget * __weak _KnownTarget;
PBXTarget * result;
}
- Meth;
@end

extern void foo(void);
@implementation PBXTarget
- Meth {
	if (_lastKnownTarget != result)
	 foo();
	if (result != _lastKnownTarget)
	 foo();

 	if (_lastKnownTarget != _KnownTarget)
	  foo();
}

@end
