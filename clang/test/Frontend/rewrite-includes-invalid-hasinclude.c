// RUN: not %clang_cc1 -E -frewrite-includes -DFIRST -I %S/Inputs %s -o - | FileCheck -strict-whitespace %s

#if __has_include bar.h
#endif

#if __has_include(bar.h)
#endif

#if __has_include(<bar.h)
#endif

// CHECK: #if __has_include bar.h
// CHECK: #endif
// CHECK: #if __has_include(bar.h)
// CHECK: #endif
// CHECK: #if __has_include(<bar.h)
// CHECK: #endif
