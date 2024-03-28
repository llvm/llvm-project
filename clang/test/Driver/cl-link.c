// Note: %s must be preceded by -- or bound to another option, otherwise it may
// be interpreted as a command-line option, e.g. on Mac where %s is commonly
// under /Users.

// RUN: %clang_cl /Tc%s -fuse-ld=link -### /link foo bar baz 2>&1 | FileCheck --check-prefix=LINK %s
// RUN: %clang_cl /Tc%s -fuse-ld=link -### /linkfoo bar baz 2>&1 | FileCheck --check-prefix=LINK %s
// LINK: link.exe
// LINK: "foo"
// LINK: "bar"
// LINK: "baz"

// RUN: %clang_cl -m32 -arch:IA32 --target=i386-pc-win32 /Tc%s -fuse-ld=link -### -fsanitize=address 2>&1 | FileCheck --check-prefix=ASAN %s
// ASAN: link.exe
// ASAN: "-debug"
// ASAN: "-incremental:no"
// ASAN: "{{[^"]*}}clang_rt.asan_dynamic-i386.lib"
// ASAN: "-wholearchive:{{.*}}clang_rt.asan_static_runtime_thunk-i386.lib"
// ASAN: "{{.*}}cl-link{{.*}}.obj"

// RUN: %clang_cl -m32 -arch:IA32 --target=i386-pc-win32 /MD /Tc%s -fuse-ld=link -### -fsanitize=address 2>&1 | FileCheck --check-prefix=ASAN-MD %s
// ASAN-MD: link.exe
// ASAN-MD: "-debug"
// ASAN-MD: "-incremental:no"
// ASAN-MD: "{{.*}}clang_rt.asan_dynamic-i386.lib"
// ASAN-MD: "-include:___asan_seh_interceptor"
// ASAN-MD: "-wholearchive:{{.*}}clang_rt.asan_dynamic_runtime_thunk-i386.lib"
// ASAN-MD: "{{.*}}cl-link{{.*}}.obj"

// RUN: %clang_cl /LD -fuse-ld=link -### /Tc%s 2>&1 | FileCheck --check-prefix=DLL %s
// RUN: %clang_cl /LDd -fuse-ld=link -### /Tc%s 2>&1 | FileCheck --check-prefix=DLL %s
// DLL: link.exe
// "-dll"

// RUN: %clang_cl -m32 -arch:IA32 --target=i386-pc-win32 /LD /Tc%s -fuse-ld=link -### -fsanitize=address 2>&1 | FileCheck --check-prefix=ASAN-DLL %s
// RUN: not %clang_cl -m32 -arch:IA32 --target=i386-pc-win32 /LDd /Tc%s -fuse-ld=link -### -fsanitize=address 2>&1 | FileCheck --check-prefix=ASAN-DLL %s
// ASAN-DLL: link.exe
// ASAN-DLL: "-dll"
// ASAN-DLL: "-debug"
// ASAN-DLL: "-incremental:no"
// ASAN-DLL: "{{.*}}clang_rt.asan_dynamic-i386.lib"
// ASAN-DLL: "-wholearchive:{{.*}}clang_rt.asan_static_runtime_thunk-i386.lib"
// ASAN-DLL: "{{.*}}cl-link{{.*}}.obj"

// RUN: %clang_cl /Zi /Tc%s -fuse-ld=link -### 2>&1 | FileCheck --check-prefix=DEBUG %s
// DEBUG: link.exe
// DEBUG: "-debug"

// Don't pass through /libpath: if it's not after a /link flag:
// RUN: not %clang_cl /Tc%s /libpath:foo -fuse-ld=link -### /link /libpath:bar 2>&1 | FileCheck --check-prefix=LIBPATH %s
// LIBPATH: error: no such file or directory: '/libpath:foo'
// LIBPATH: libpath:bar

// PR27234
// RUN: %clang_cl /Tc%s nonexistent.obj -fuse-ld=link -### /link /libpath:somepath 2>&1 | FileCheck --check-prefix=NONEXISTENT %s
// RUN: %clang_cl /Tc%s nonexistent.lib -fuse-ld=link -### /link /libpath:somepath 2>&1 | FileCheck --check-prefix=NONEXISTENT %s
// RUN: %clang_cl /Tc%s nonexistent.obj -fuse-ld=link -### /winsysroot somepath 2>&1 | FileCheck --check-prefix=NONEXISTENT %s
// RUN: %clang_cl /Tc%s nonexistent.lib -fuse-ld=link -### /winsysroot somepath 2>&1 | FileCheck --check-prefix=NONEXISTENT %s
// RUN: %clang_cl /Tc%s nonexistent.obj -fuse-ld=link -### 2>&1 | FileCheck --check-prefix=NONEXISTENT %s
// RUN: %clang_cl /Tc%s nonexistent.lib -fuse-ld=link -### 2>&1 | FileCheck --check-prefix=NONEXISTENT %s
// NONEXISTENT-NOT: no such file
// NONEXISTENT: link.exe
// NONEXISTENT: nonexistent

// RUN: %clang_cl /Tc%s -fuse-ld=lld -### 2>&1 | FileCheck --check-prefix=USE_LLD %s
// USE_LLD: lld-link

// RUN: %clang_cl -m32 -arch:IA32 --target=i386-pc-win32 /Tc%s -fuse-ld=link -### -fsanitize=address 2>&1 | FileCheck --check-prefix=INFER-LINK %s
// INFER-LINK: link.exe
// INFER-LINK: /INFERASANLIBS:NO

// RUN: %clang_cl -m32 -arch:IA32 --target=i386-pc-win32 /Tc%s -fuse-ld=lld -### -fsanitize=address 2>&1 | FileCheck --check-prefix=INFER-LLD %s
// INFER-LLD: lld-link
// INFER-LLD-NOT: INFERASANLIBS
