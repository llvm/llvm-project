// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

@protocol fproto @end

@protocol p1 
@end

@class cl;

int main(void)
{
	Protocol *proto = @protocol(p1);
        Protocol *fproto = @protocol(fproto);
}

