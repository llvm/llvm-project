// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

@protocol MyProto1 
@end

@protocol MyProto2
@end

@interface INTF @end

INTF <MyProto1> *g1;

INTF <MyProto1, MyProto2> *g2, *g3;

INTF <MyProto1> * Func(INTF <MyProto1> *p2, INTF<MyProto1> *p3, INTF *p4, INTF<MyProto1> *p5)
{
	return p2;
}

INTF <MyProto1, MyProto2> * Func1(INTF *p2, INTF<MyProto1, MyProto2> *p3, INTF *p4, INTF<MyProto1> *p5)
{
	return p3;
}

@interface Foo
@property int (*hashFunction)(const void *item, int (*size)(const void *item));
@end
