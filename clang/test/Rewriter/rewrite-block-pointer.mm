// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc %s -o %t-modern-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -std=gnu++98 -Wno-address-of-temporary -D"SEL=void*" -D"__declspec(X)=" %t-modern-rw.cpp

typedef unsigned long size_t;

typedef void * id;
void *sel_registerName(const char *);

@interface X
@end

void foo(void (^block)(int));

@implementation X
static void enumerateIt(void (^block)(id, id, char *)) {
      foo(^(int idx) { });
}
@end

void apply(void (^block)(int));

static void x(int (^cmp)(int, int)) {
	x(cmp);
}

static void y(int (^cmp)(int, int)) {
	apply(^(int sect) {
		x(cmp);
    });
}

void *_Block_copy(const void *aBlock);
void x(void (^block)(void)) {
        block = ((__typeof(block))_Block_copy((const void *)(block)));
}

@interface Y {
@private
    id _private;
}
- (void (^)(void))f;
@end

typedef void (^void_block_t)(void);

@interface YY {
    void_block_t __completion;
}
@property (copy) void_block_t f;
@end

@implementation Y
- (void (^)(void))f {
    return [_private f];
}

@end

@protocol CoreDAVAccountInfoProvider;
@protocol CodeProvider;
typedef void (^BDVDiscoveryCompletionHandler)(int success, id<CoreDAVAccountInfoProvider> discoveredInfo);
typedef void (^BDVDiscoveryCompletion)(id<CodeProvider> codeInfo, int success, id<CoreDAVAccountInfoProvider> discoveredInfo);
typedef void (^BDVDiscovery)(int success);
typedef void (^BDVDisc)(id<CoreDAVAccountInfoProvider> discoveredInfo, id<CodeProvider> codeInfo, 
                        int success, id<CoreDAVAccountInfoProvider, CodeProvider> Info);
typedef void (^BLOCK)(id, id<CoreDAVAccountInfoProvider>, id<CodeProvider> codeInfo);
typedef void (^EMPTY_BLOCK)();
typedef void (^  BDVDiscoveryCompletion1  )(id<CodeProvider> codeInfo, int success, id<CoreDAVAccountInfoProvider> discoveredInfo);

void (^BL)(void(^arg1)(), int i1, void(^arg)(int));

typedef void (^iscoveryCompletionHandler)(void(^arg1)(), id<CoreDAVAccountInfoProvider> discoveredInfo);

typedef void (^DVDisc)(id<CoreDAVAccountInfoProvider> discoveredInfo, id<CodeProvider> codeInfo,
			void(^arg1)(), int i1, void(^arg)(id<CoreDAVAccountInfoProvider>),
                        int success, id<CoreDAVAccountInfoProvider, CodeProvider> Info);


@interface I @end
@interface INTF @end
void (^BLINT)(I<CoreDAVAccountInfoProvider>* ARG, INTF<CodeProvider, CoreDAVAccountInfoProvider>* ARG1);

void  test8608902() {
  BDVDiscoveryCompletionHandler ppp;
  ppp(1, 0);
}

void test9204669() {
   __attribute__((__blocks__(byref))) char (^addChangeToData)();

   addChangeToData = ^() {
      return 'b';
   };
   addChangeToData();
}

void test9204669_1() {
   __attribute__((__blocks__(byref))) void (^addChangeToData)();

   addChangeToData = ^() {
    addChangeToData();
   };
}

