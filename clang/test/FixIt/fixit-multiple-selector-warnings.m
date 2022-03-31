/* RUN: cp %s %t
   RUN: %clang_cc1 -x objective-c -Wselector-type-mismatch -fixit %t
   RUN: %clang_cc1 -x objective-c -Wselector-type-mismatch -Werror %t
*/
// rdar://16458579

@interface I
- (id) compare: (char) arg1;
- length;
@end

@interface J
- (id) compare: (id) arg1;
@end

SEL func(void)
{
        (void)@selector( compare: );
        (void)@selector (compare:);
        (void)@selector( compare:);
        (void)@selector(compare: );
        (void)@selector ( compare: );
	return @selector(compare:);
}


