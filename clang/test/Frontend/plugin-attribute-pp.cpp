// RUN: %clang -fplugin=%llvmshlibdir/Attribute%pluginext -E %s | FileCheck %s
// RUN: %clang -fplugin=%llvmshlibdir/Attribute%pluginext -E %s -x c | FileCheck %s
// REQUIRES: plugins, examples

#ifdef __cplusplus
# define HAS_ATTR(a) __has_cpp_attribute (a)
#else
# define HAS_ATTR(a) __has_c_attribute (a)
#endif

#if __has_attribute(example)
// CHECK: has_attribute(example) was true
has_attribute(example) was true
#endif
#if HAS_ATTR(example)
// CHECK: has_$LANG_attribute(example) was true
has_$LANG_attribute(example) was true
#endif

#if __has_attribute(doesnt_exist)
// CHECK-NOT: has_attribute(doesnt_exist) unexpectedly was true
has_attribute(doesnt_exist) unexpectedly was true
#endif

#if HAS_ATTR(doesnt_exist)
// CHECK-NOT: has_$LANG_attribute(doesnt_exist) unexpectedly was true
has_$LANG_attribute(doesnt_exist) unexpectedly was true
#endif
