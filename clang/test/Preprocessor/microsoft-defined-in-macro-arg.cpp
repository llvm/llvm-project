// RUN: %clang_cc1 -E -fms-compatibility %s | FileCheck %s --check-prefix=MS
// RUN: %clang_cc1 -E %s | FileCheck %s --check-prefix=STD

// A 'defined' generated while pre-expanding a macro argument in an #if
// condition. MSVC does not pre-expand macro arguments, so it never expands the
// operand of such a 'defined'; with -fms-compatibility, clang does the same.

#define FOO(x) (defined __foo_##x && __foo_##x)
#define BAR(x) (defined(__foo_##x) && __foo_##x)
#define __foo_hello true
#define NOT(X) (!(X))
#define ID(X) X

// Without a surrounding macro, both modes see 'defined __foo_hello'.
#if FOO(hello)
direct_defined
#else
direct_undefined
#endif
// MS: direct_defined
// STD: direct_defined

#if NOT(FOO(hello))
not_true
#else
not_false
#endif
// MS: not_false
// STD: not_true

#if NOT(BAR(hello))
paren_not_true
#else
paren_not_false
#endif
// MS: paren_not_false
// STD: paren_not_true

#if ID(NOT(ID(FOO(hello))))
nested_true
#else
nested_false
#endif
// MS: nested_false
// STD: nested_true

#if 0
#elif ID(FOO(hello))
elif_defined
#else
elif_undefined
#endif
// MS: elif_defined
// STD: elif_undefined

// An operand that is not a macro is unaffected.
#if ID(FOO(world))
world_defined
#else
world_undefined
#endif
// MS: world_undefined
// STD: world_undefined

// Outside #if, the argument is still pre-expanded as usual.
ID(defined __foo_hello)
// MS: defined true
// STD: defined true
