// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -Wno-address-of-temporary -Did="void *" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// radar 9254348
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -Wno-address-of-temporary -Did="void *" -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp
// rdar://11259664

// rdar://11375908
typedef unsigned long size_t;

void *sel_registerName(const char *);
typedef void (^BLOCK_TYPE)(void);

@interface CoreDAVTaskGroup 
{
  int IVAR;
}
@property int IVAR;
- (void) setCompletionBlock : (BLOCK_TYPE) arg;
@end

@implementation CoreDAVTaskGroup
- (void)_finishInitialSync {
                    CoreDAVTaskGroup *folderPost;
  folderPost.completionBlock = ^{
    self.IVAR = 0;
    [self _finishInitialSync];
  };

  [folderPost setCompletionBlock : (^{
    self.IVAR = 0;
  })];
}
@dynamic IVAR;
- (void) setCompletionBlock : (BLOCK_TYPE) arg {}
@end


