// RUN: %clang_cc1  -Wselector-type-mismatch -verify %s 

@interface I
- (id) compare: (char) arg1; // expected-note {{method 'compare:' declared here}}
- length;
@end

@interface J
- (id) compare: (id) arg1; // expected-note {{method 'compare:' declared here}}
@end

SEL func(void)
{
	return @selector(compare:);	// expected-warning {{several methods with selector 'compare:' of mismatched types are found for the @selector expression}}
}

void Test16458579(void) {
 SEL s = @selector((retain));
 SEL s1 = @selector((meth1:));
 SEL s2 = @selector((retainArgument::));
 SEL s3 = @selector((retainArgument:::::));
 SEL s4 = @selector((retainArgument:with:));
 SEL s5 = @selector((meth1:with:with:));
 SEL s6 = @selector((getEnum:enum:bool:));
 SEL s7 = @selector((char:float:double:unsigned:short:long:));
 SEL s9 = @selector((:enum:bool:));
}
int main(void) {
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

@interface NSObject @end

@class NSNumber;

@interface XBRecipe : NSObject
@property (nonatomic, assign) float finalVolume; // expected-note {{method 'setFinalVolume:' declared here}}
@end

@interface XBDocument : NSObject
@end

@interface XBDocument ()
- (void)setFinalVolume:(NSNumber *)finalVolumeNumber; // expected-note {{method 'setFinalVolume:' declared here}}
@end

@implementation XBDocument
- (void)setFinalVolume:(NSNumber *)finalVolumeNumber
{
    (void)@selector(setFinalVolume:); // expected-warning {{several methods with selector 'setFinalVolume:' of mismatched types are found for the @selector expression}}
}
@end
