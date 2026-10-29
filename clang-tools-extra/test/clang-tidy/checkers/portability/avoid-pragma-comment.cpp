// Only the "lib" pragma comment type is implemented on Linux, the rest are
// Windows-only.  We test for both platform targets.
// RUN: %check_clang_tidy -check-suffixes=LINUX %s portability-avoid-pragma-comment %t -- -- -target x86_64-unknown-linux-gnu
// RUN: %check_clang_tidy -check-suffixes=WINDOWS %s portability-avoid-pragma-comment %t -- -- -target x86_64-pc-windows-msvc

#pragma comment(lib, "some_lib")
// CHECK-MESSAGES-LINUX:   :[[@LINE-1]]:9: warning: avoid 'pragma comment' directive; use the build system to link libraries [portability-avoid-pragma-comment]
// CHECK-MESSAGES-WINDOWS: :[[@LINE-2]]:9: warning: avoid 'pragma comment' directive; use the build system to link libraries [portability-avoid-pragma-comment]

_Pragma("comment(lib, \"some_lib\")")
// CHECK-MESSAGES-LINUX:   :[[@LINE-1]]:1: warning: avoid 'pragma comment' directive; use the build system to link libraries [portability-avoid-pragma-comment]
// CHECK-MESSAGES-WINDOWS: :[[@LINE-2]]:1: warning: avoid 'pragma comment' directive; use the build system to link libraries [portability-avoid-pragma-comment]

// The rest are Windows-only and should be caught by "-Wunknown-pragmas" or
// "-Wignored-pragmas" on Linux.  On Linux they won't show up in the AST, so
// portability-avoid-pragma-comment won't detect them.

#pragma comment(linker, "some_linker_flag")
// CHECK-MESSAGES-WINDOWS: :[[@LINE-1]]:9: warning: avoid 'pragma comment' directive; use the build system to set linker options [portability-avoid-pragma-comment]

#pragma comment(compiler)
// CHECK-MESSAGES-WINDOWS: :[[@LINE-1]]:9: warning: avoid 'pragma comment' directive [portability-avoid-pragma-comment]

#pragma comment(user, "Some string")
// CHECK-MESSAGES-WINDOWS: :[[@LINE-1]]:9: warning: avoid 'pragma comment' directive [portability-avoid-pragma-comment]

_Pragma("comment(linker, \"some_linker_flag\")")
// CHECK-MESSAGES-WINDOWS: :[[@LINE-1]]:1: warning: avoid 'pragma comment' directive; use the build system to set linker options [portability-avoid-pragma-comment]

_Pragma("comment(compiler)")
// CHECK-MESSAGES-WINDOWS: :[[@LINE-1]]:1: warning: avoid 'pragma comment' directive [portability-avoid-pragma-comment]

_Pragma("comment(user, \"Some string\")")
// CHECK-MESSAGES-WINDOWS: :[[@LINE-1]]:1: warning: avoid 'pragma comment' directive [portability-avoid-pragma-comment]

// __pragma() is a Microsoft-specific extension
#ifdef _MSC_VER 
__pragma(comment(lib, "some_lib"))
// CHECK-MESSAGES-WINDOWS: :[[@LINE-1]]:10: warning: avoid 'pragma comment' directive; use the build system to link libraries [portability-avoid-pragma-comment]

__pragma(comment(linker, "some_linker_flag"))
// CHECK-MESSAGES-WINDOWS: :[[@LINE-1]]:10: warning: avoid 'pragma comment' directive; use the build system to set linker options [portability-avoid-pragma-comment]

__pragma(comment(compiler))
// CHECK-MESSAGES-WINDOWS: :[[@LINE-1]]:10: warning: avoid 'pragma comment' directive [portability-avoid-pragma-comment]

__pragma(comment(user, "Some string"))
// CHECK-MESSAGES-WINDOWS: :[[@LINE-1]]:10: warning: avoid 'pragma comment' directive [portability-avoid-pragma-comment]
#endif
