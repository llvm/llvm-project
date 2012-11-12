// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

@interface MyDerived 
{
@public
	int IVAR;
}
@end

MyDerived *pd;
int main() {
	return pd->IVAR;
}


