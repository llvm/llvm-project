// General tests that the driver handles combinations of --rtlib=XXX and
// --unwindlib=XXX properly.
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=libgcc \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=RTLIB-GCC %s
// RTLIB-GCC: "{{.*}}lgcc"
// RTLIB-GCC: "{{.*}}lgcc_s"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=libgcc --unwindlib=libunwind \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=RTLIB-GCC-UNWINDLIB-COMPILER-RT %s
// RTLIB-GCC-UNWINDLIB-COMPILER-RT: "{{.*}}lgcc"
// RTLIB-GCC-UNWINDLIB-COMPILER-RT: "{{.*}}l:libunwind.so"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=libgcc --unwindlib=libunwind \
// RUN:     -static-libgcc \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=RTLIB-GCC-STATIC-UNWINDLIB-COMPILER-RT %s
// RTLIB-GCC-STATIC-UNWINDLIB-COMPILER-RT: "{{.*}}lgcc"
// RTLIB-GCC-STATIC-UNWINDLIB-COMPILER-RT: "{{.*}}l:libunwind.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1   \
// RUN:     --target=x86_64-unknown-linux -rtlib=compiler-rt \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=RTLIB-COMPILER-RT %s
// RTLIB-COMPILER-RT: "{{.*}}libclang_rt.builtins-x86_64.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1   \
// RUN:     --target=x86_64-unknown-linux -rtlib=compiler-rt --unwindlib=libgcc \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=RTLIB-COMPILER-RT-UNWINDLIB-GCC %s
// RTLIB-COMPILER-RT-UNWINDLIB-GCC: "{{.*}}libclang_rt.builtins-x86_64.a"
// RTLIB-COMPILER-RT-UNWINDLIB-GCC: "{{.*}}lgcc_s"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1              \
// RUN:     --target=x86_64-unknown-linux -rtlib=compiler-rt --unwindlib=libgcc \
// RUN:     -static --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=RTLIB-COMPILER-RT-UNWINDLIB-GCC-STATIC %s
// RTLIB-COMPILER-RT-UNWINDLIB-GCC-STATIC: "{{.*}}libclang_rt.builtins-x86_64.a"
// RTLIB-COMPILER-RT-UNWINDLIB-GCC-STATIC: "{{.*}}lgcc_eh"
//
// RUN: not %clang -no-canonical-prefixes %s -o %t.o 2> %t.err              \
// RUN:     --target=x86_64-unknown-linux -rtlib=libgcc --unwindlib=libunwind \
// RUN:     --gcc-toolchain="" \
// RUN: FileCheck --input-file=%t.err --check-prefix=RTLIB-GCC-UNWINDLIB-COMPILER_RT %s
// RTLIB-GCC-UNWINDLIB-COMPILER_RT: "{{[.|\\\n]*}}--rtlib=libgcc requires --unwindlib=libgcc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-w64-mingw32 -rtlib=compiler-rt --unwindlib=libunwind \
// RUN:     -shared-libgcc \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=MINGW-RTLIB-COMPILER-RT-SHARED-UNWINDLIB-COMPILER-RT %s
// MINGW-RTLIB-COMPILER-RT-SHARED-UNWINDLIB-COMPILER-RT: "{{.*}}libclang_rt.builtins-x86_64.a"
// MINGW-RTLIB-COMPILER-RT-SHARED-UNWINDLIB-COMPILER-RT: "{{.*}}l:libunwind.dll.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-w64-mingw32 -rtlib=compiler-rt --unwindlib=libunwind \
// RUN:     -static-libgcc \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=MINGW-RTLIB-COMPILER-RT-STATIC-UNWINDLIB-COMPILER-RT %s
// MINGW-RTLIB-COMPILER-RT-STATIC-UNWINDLIB-COMPILER-RT: "{{.*}}libclang_rt.builtins-x86_64.a"
// MINGW-RTLIB-COMPILER-RT-STATIC-UNWINDLIB-COMPILER-RT: "{{.*}}l:libunwind.a"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-w64-mingw32 -rtlib=compiler-rt --unwindlib=libunwind \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=MINGW-RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT %s
// RUN: %clangxx -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     --target=x86_64-w64-mingw32 -rtlib=compiler-rt --unwindlib=libunwind \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=MINGW-RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT %s
// MINGW-RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT: "{{.*}}libclang_rt.builtins-x86_64.a"
// MINGW-RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT: "{{.*}}lunwind"
