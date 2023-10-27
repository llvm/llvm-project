// RUN: %clang_cc1 -fblocks -rewrite-objc -fms-extensions %s -o %t-rw.cpp
// RUN: %clang_cc1 -Werror -fsyntax-only -Wno-address-of-temporary -Wno-c++11-narrowing -std=c++11 -D"Class=void*" -D"id=void*" -D"SEL=void*" -U__declspec -D"__declspec(X)=" %t-rw.cpp

typedef unsigned long NSUInteger;

typedef struct _NSRange {
    NSUInteger location;
    NSUInteger length;
} NSRange;

typedef struct {
    NSUInteger _capacity;
    NSRange _ranges[0];
} _NSRangeInfo;

@interface Foo{
    @protected 
    struct _bar {
        int x:1;
        int y:1;
    } bar;
    union {
        struct {
            NSRange _range;
        } _singleRange;
        struct {
            void *  _data;
            void *_reserved;
        } _multipleRanges;
    } _internal;    
}
@end
@implementation Foo
- (void)x:(Foo *)other {
  bar.x = 0;
  bar.y = -1;
  self->_internal._singleRange._range = (( other ->bar.x) ? &( other ->_internal._singleRange._range) : ((NSRange *)(&(((_NSRangeInfo *)( other ->_internal._multipleRanges._data))->_ranges))))[0];
}
@end
@interface FooS : Foo
@end
@implementation FooS
- (void)y {

  NSUInteger asdf =  (( self ->bar.x) ? 1 : ((_NSRangeInfo *)( self ->_internal._multipleRanges._data))->_capacity ); 
}
@end
