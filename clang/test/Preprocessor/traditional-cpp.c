/* Clang supports a very limited subset of -traditional-cpp, basically we only
 * intend to add support for things that people actually rely on when doing
 * things like using /usr/bin/cpp to preprocess non-source files. */

/*
 RUN: %clang_cc1 -traditional-cpp %s -E | FileCheck -strict-whitespace %s
 RUN: %clang_cc1 -traditional-cpp %s -E -C | FileCheck -check-prefix=CHECK-COMMENTS %s
 RUN: %clang_cc1 -traditional-cpp -x c++ %s -E | FileCheck -check-prefix=CHECK-CXX %s
*/

/* -traditional-cpp should eliminate all C89 comments. */
/* CHECK-NOT: /*
 * CHECK-COMMENTS: {{^}}/* -traditional-cpp should eliminate all C89 comments. *{{/$}}
 */

/* -traditional-cpp should only eliminate "//" comments in C++ mode. */
/* CHECK: {{^}}foo // bar{{$}}
 * CHECK-CXX: {{^}}foo {{$}}
 */
foo // bar


/* The lines in this file contain hard tab characters and trailing whitespace; 
 * do not change them! */

/* CHECK: {{^}}	indented!{{$}}
 * CHECK: {{^}}tab	separated	values{{$}}
 */
	indented!
tab	separated	values

#define bracket(x) >>>x<<<
bracket(|  spaces  |)
/* CHECK: {{^}}>>>|  spaces  |<<<{{$}}
 */

/* This is still a preprocessing directive. */
# define foo bar
foo!
-
	foo!	foo!	
/* CHECK: {{^}}bar!{{$}}
 * CHECK: {{^}}	bar!	bar!	{{$}}
 */

/* Deliberately check a leading newline with spaces on that line. */
   
# define foo bar
foo!
-
	foo!	foo!	
/* CHECK: {{^}}bar!{{$}}
 * CHECK: {{^}}	bar!	bar!	{{$}}
 */

/* FIXME: -traditional-cpp should not consider this a preprocessing directive
 * because the # isn't in the first column.
 */
 #define foo2 bar
foo2!
/* If this were working, both of these checks would be on.
 * CHECK-NOT: {{^}} #define foo2 bar{{$}}
 * CHECK-NOT: {{^}}foo2!{{$}}
 */

/* FIXME: -traditional-cpp should not homogenize whitespace in macros.
 */
#define bracket2(x) >>>  x  <<<
bracket2(spaces)
/* If this were working, this check would be on.
 * CHECK-NOT: {{^}}>>>  spaces  <<<{{$}}
 */


/* Check that #if 0 blocks work as expected */
#if 0
#error "this is not an error"

#if 1
a b c in skipped block
#endif

/* Comments are whitespace too */

#endif
/* CHECK-NOT: {{^}}a b c in skipped block{{$}}
 * CHECK-NOT: {{^}}/* Comments are whitespace too
 */

Preserve URLs: http://clang.llvm.org
/* CHECK: {{^}}Preserve URLs: http://clang.llvm.org{{$}}
 */

/* The following tests ensure we ignore # and ## in macro bodies */

#define FOO_NO_STRINGIFY(a) test(# a)
FOO_NO_STRINGIFY(foobar)
/* CHECK: {{^}}test(# foobar){{$}}
 */

#define FOO_NO_PASTE(a, b) test(b##a)
FOO_NO_PASTE(xxx,yyy)
/* CHECK: {{^}}test(yyy##xxx){{$}}
 */

#define BAR_NO_STRINGIFY(a) test(#a)
BAR_NO_STRINGIFY(foobar)
/* CHECK: {{^}}test(#foobar){{$}}
 */
