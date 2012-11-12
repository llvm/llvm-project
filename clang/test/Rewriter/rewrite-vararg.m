// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-address-of-temporary -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp

// rdar://9056351
void *sel_registerName(const char *);

@interface NSObject @end
@class NSString;

@protocol P
  -(void)ParliamentFunkadelic;
@end
	
@interface Foo {
  NSObject <P> *_dataSource;
}
@end
	
@interface Bar { }
+(void)WhateverBar:(NSString*)format, ...;
@end
	
@implementation Foo
-(void)WhateverFoo {
	[Bar WhateverBar:@"ISyncSessionDriverDataSource %@ responded poorly", _dataSource];
}
@end
