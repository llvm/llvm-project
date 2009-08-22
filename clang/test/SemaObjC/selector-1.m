// RUN: clang-cc -verify %s 

@interface Lancelot @end
@implementation Lancelot

- (void):(int)x {}
- (void)xx:(int)x :(int)y { }

@end

@interface I
- (id) compare: (char) arg1;
@end

@interface J
- (id) compare: (id) arg1;
@end

SEL foo()
{
	return @selector(compare:);	// Non warning on multiple selector found.
}

int main() {
 SEL s = @selector(retain);
 SEL s1 = @selector(meth1:);
 SEL s2 = @selector(retainArgument::);
 SEL s3 = @selector(retainArgument:::::);
 SEL s4 = @selector(retainArgument:with:);
 SEL s5 = @selector(meth1:with:with:);
 SEL s6 = @selector(getEnum:enum:bool:);
 SEL s7 = @selector(char:float:double:unsigned:short:long:);

 SEL s9 = @selector(:enum:bool:);
}
