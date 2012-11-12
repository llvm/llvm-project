// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fms-extensions -fobjc-default-synthesize-properties -rewrite-objc %t.mm -o %t-rw.cpp 
// RUN: FileCheck --input-file=%t-rw.cpp %s
// RUN: %clang_cc1 -fsyntax-only  -Werror -DSEL="void *" -Did="struct objc_object *" -Wno-attributes -Wno-address-of-temporary -U__declspec -D"__declspec(X)=" %t-rw.cpp
// rdar://11374235

extern "C" void *sel_registerName(const char *);

@interface NSObject 
- (void) release;
- (id) retain;
@end
@class NSString;

@interface SynthItAll : NSObject
@property int howMany;
@property (retain) NSString* what; 
@end

@implementation SynthItAll
@end


@interface SynthSetter : NSObject
@property (nonatomic) int howMany; 
@property (nonatomic, retain) NSString* what; 
@end

@implementation SynthSetter

- (int) howMany {
    return _howMany;
}
// - (void) setHowMany: (int) value

- (NSString*) what {
    return _what;
}
// - (void) setWhat: (NSString*) value    
@end


@interface SynthGetter : NSObject
@property (nonatomic) int howMany;
@property (nonatomic, retain) NSString* what;
@end

@implementation SynthGetter
// - (int) howMany
- (void) setHowMany: (int) value {
    _howMany = value;
}

// - (NSString*) what
- (void) setWhat: (NSString*) value {
    if (_what != value) {
        [_what release];
        _what = [value retain];
    }
}
@end

typedef struct {
        int x:1;
        int y:1;
} TBAR;

@interface NONAME
{
  TBAR _bar;
}
@property TBAR bad;
@end

@implementation NONAME
@end

// CHECK: (*(int *)((char *)self + OBJC_IVAR_$_SynthItAll$_howMany)) = howMany;
// CHECK: return (*(int *)((char *)self + OBJC_IVAR_$_SynthGetter$_howMany));
// CHECK: (*(TBAR *)((char *)self + OBJC_IVAR_$_NONAME$_bad)) = bad;
