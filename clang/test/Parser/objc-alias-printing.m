// RUN: %clang_cc1 -ast-print %s

@protocol P1 @end
@protocol P2 @end

@interface INTF @end

@compatibility_alias alias INTF;


int foo (void)
{
	INTF *pi;
	INTF<P2,P1> *pi2;
	alias *p;
	alias<P1,P2> *p2;
	return pi2 == p2;
}
