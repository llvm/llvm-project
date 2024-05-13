// RUN: %clang_cc1 -ast-print %s

@protocol P1 @end
@protocol P2 @end
@protocol P3 @end

@interface INTF 
- (INTF<P1>*) METH;
@end

void foo(void)
{
        INTF *pintf;
	INTF<P1>* p1;
	INTF<P1, P1>* p2;
	INTF<P1, P3>* p3;
	INTF<P1, P3, P2>* p4;
	INTF<P2,P2, P3, P1>* p5;
}
