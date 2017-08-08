// Note: %s must be preceded by -- or bound to another option, otherwise it may
// be interpreted as a command-line option, e.g. on Mac where %s is commonly
// under /Users.

// RUN: %clang_cl /Tc%s -### /link foo bar baz 2>&1 | FileCheck --check-prefix=LINK %s
// RUN: %clang_cl /Tc%s -### /linkfoo bar baz 2>&1 | FileCheck --check-prefix=LINK %s
// LINK: link.exe
// LINK: "foo"
// LINK: "bar"
// LINK: "baz"

// RUN: %clang_cl -m32 -arch:IA32 --target=i386-pc-win32 /Tc%s -### -fsanitize=address 2>&1 | FileCheck --check-prefix=ASAN %s
// ASAN: link.exe
// ASAN: "-debug"
// ASAN: "-incremental:no"
// ASAN: "{{[^"]*}}clang_rt.asan-i386.lib"
// ASAN: "-wholearchive:{{.*}}clang_rt.asan-i386.lib"
// ASAN: "{{[^"]*}}clang_rt.asan_cxx-i386.lib"
// ASAN: "-wholearchive:{{.*}}clang_rt.asan_cxx-i386.lib"
// ASAN: "{{.*}}cl-link{{.*}}.obj"

// RUN: %clang_cl -m32 -arch:IA32 --target=i386-pc-win32 /MD /Tc%s -### -fsanitize=address 2>&1 | FileCheck --check-prefix=ASAN-MD %s
// ASAN-MD: link.exe
// ASAN-MD: "-debug"
// ASAN-MD: "-incremental:no"
// ASAN-MD: "{{.*}}clang_rt.asan_dynamic-i386.lib"
// ASAN-MD: "{{[^"]*}}clang_rt.asan_dynamic_runtime_thunk-i386.lib"
// ASAN-MD: "-include:___asan_seh_interceptor"
// ASAN-MD: "-wholearchive:{{.*}}clang_rt.asan_dynamic_runtime_thunk-i386.lib"
// ASAN-MD: "{{.*}}cl-link{{.*}}.obj"

// RUN: %clang_cl /LD -### /Tc%s 2>&1 | FileCheck --check-prefix=DLL %s
// RUN: %clang_cl /LDd -### /Tc%s 2>&1 | FileCheck --check-prefix=DLL %s
// DLL: link.exe
// "-dll"

// RUN: %clang_cl -m32 -arch:IA32 --target=i386-pc-win32 /LD /Tc%s -### -fsanitize=address 2>&1 | FileCheck --check-prefix=ASAN-DLL %s
// RUN: %clang_cl -m32 -arch:IA32 --target=i386-pc-win32 /LDd /Tc%s -### -fsanitize=address 2>&1 | FileCheck --check-prefix=ASAN-DLL %s
// ASAN-DLL: link.exe
// ASAN-DLL: "-dll"
// ASAN-DLL: "-debug"
// ASAN-DLL: "-incremental:no"
// ASAN-DLL: "{{.*}}clang_rt.asan_dll_thunk-i386.lib"
// ASAN-DLL: "{{.*}}cl-link{{.*}}.obj"

// RUN: %clang_cl /Zi /Tc%s -### 2>&1 | FileCheck --check-prefix=DEBUG %s
// DEBUG: link.exe
// DEBUG: "-debug"

// PR27234
// RUN: %clang_cl /Tc%s nonexistent.obj -### /link /libpath:somepath 2>&1 | FileCheck --check-prefix=NONEXISTENT %s
// RUN: %clang_cl /Tc%s nonexistent.lib -### /link /libpath:somepath 2>&1 | FileCheck --check-prefix=NONEXISTENT %s
// NONEXISTENT-NOT: no such file
// NONEXISTENT: link.exe
// NONEXISTENT: "/libpath:somepath"
// NONEXISTENT: nonexistent

// RUN: %clang_cl /Tc%s -fuse-ld=lld -### 2>&1 | FileCheck --check-prefix=USE_LLD %s
// USE_LLD: lld-link
