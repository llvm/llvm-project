#line 42
int foo; // This should appear as at line-directive-nofilename.h:42

#line 100 "foobar.h"
int bar; // This should appear as at foobar.h:100
