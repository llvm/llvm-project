// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s
// R UN: %clang_cc1 -fsyntax-only -fblocks -verify -x c -std=c23 %s

#if !__has_attribute(clang_nolock)
#error "the 'nolock' attribute is not available"
#endif

void unannotated();
void nolock() [[clang::nolock]];
void noalloc() [[clang::noalloc]];


void callthis(void (*fp)());


void type_conversions()
{
// 	callthis(nolock);

	// It's fine to remove a performance constraint.
	void (*fp_plain)();

// 	fp_plain = unannotated;
	fp_plain = nolock;
// 	fp_plain = noalloc;
}
