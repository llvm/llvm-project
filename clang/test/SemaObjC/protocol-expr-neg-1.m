// RUN: %clang_cc1 -fsyntax-only -verify %s

@class Protocol;

@protocol fproto;

@protocol p1 
@end

@class cl;

int main()
{
	Protocol *proto = @protocol(p1);
        Protocol *fproto = @protocol(fproto);
	Protocol *pp = @protocol(i); // expected-error {{cannot find protocol declaration for 'i'}}
	Protocol *p1p = @protocol(cl); // expected-error {{cannot find protocol declaration for 'cl'}}
}

