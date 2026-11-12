// expected-warning 0-1 {{umbrella header}}

// FIXME: The "umbrella header" warning should be moved to a separate test.
// This "0-1" is only here because the warning is only emitted when the
// module is (otherwise) successfully included.

#ifndef MODULE_H
#define MODULE_H
const char *getModuleVersion(void);

#ifdef FOO
#  error Module should have been built without -DFOO
#endif

@interface Module
+(const char *)version; // retrieve module version
+alloc;
@end

#define MODULE_H_MACRO 1
#__private_macro MODULE_H_MACRO

#include <Module/Sub.h>
#include <Module/Buried/Treasure.h>

__asm("foo");

typedef	struct __sFILE {
  int _offset;
} FILE;

extern FILE *myFile;

#define SOME_MACRO_ATTR_GETTING_UNDEFINED __attribute__((objc_method_family(none)))
#undef SOME_MACRO_ATTR_GETTING_UNDEFINED

#endif // MODULE_H
