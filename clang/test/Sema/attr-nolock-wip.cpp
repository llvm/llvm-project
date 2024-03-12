// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s
// R UN: %clang_cc1 -fsyntax-only -fblocks -verify -x c -std=c23 %s

#if !__has_attribute(clang_nolock)
#error "the 'nolock' attribute is not available"
#endif

void unannotated(void);
void nolock(void) [[clang::nolock]];
void noalloc(void) [[clang::noalloc]];
void type_conversions(void)
{
	// It's fine to remove a performance constraint.
	void (*fp_plain)(void);

	fp_plain = unannotated;
	fp_plain = nolock;
	fp_plain = noalloc;
}
