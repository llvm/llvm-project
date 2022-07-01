// General tests that the driver handles combinations of --rtlib=XXX and
// --unwindlib=XXX properly.
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=libgcc --unwindlib=platform \
// RUN:     --gcc-toolchain="" -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=RTLIB-GCC %s
// RTLIB-GCC: "-lgcc"
// RTLIB-GCC-SAME: "--as-needed"
// RTLIB-GCC-SAME: "-lgcc_s"
// RTLIB-GCC-SAME: "--no-as-needed"
//
// RUN: %clangxx -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=libgcc --unwindlib=platform \
// RUN:     --gcc-toolchain="" -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=RTLIB-GXX %s
// RTLIB-GXX: "-lgcc"
// RTLIB-GXX-NOT: "--as-needed"
// RTLIB-GXX-SAME: "-lgcc_s"
// RTLIB-GXX-NOT: "--no-as-needed"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=libgcc --unwindlib=libunwind \
// RUN:     --gcc-toolchain="" -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=RTLIB-GCC-UNWINDLIB-COMPILER-RT %s
// RUN: %clangxx -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=libgcc --unwindlib=libunwind \
// RUN:     --gcc-toolchain="" -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=RTLIB-GCC-UNWINDLIB-COMPILER-RT %s
// RTLIB-GCC-UNWINDLIB-COMPILER-RT: "-lgcc"
// RTLIB-GCC-UNWINDLIB-COMPILER-RT-SAME: "--as-needed"
// RTLIB-GCC-UNWINDLIB-COMPILER-RT-SAME: "-lunwind"
// RTLIB-GCC-UNWINDLIB-COMPILER-RT-SAME: "--no-as-needed"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=libgcc --unwindlib=libunwind \
// RUN:     -shared-libgcc \
// RUN:     --gcc-toolchain="" -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=RTLIB-GCC-SHARED-UNWINDLIB-COMPILER-RT %s
// RTLIB-GCC-SHARED-UNWINDLIB-COMPILER-RT: "-l:libunwind.so"
// RTLIB-GCC-SHARED-UNWINDLIB-COMPILER-RT-SAME: "-lgcc"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=libgcc --unwindlib=libunwind \
// RUN:     -static-libgcc \
// RUN:     --gcc-toolchain="" -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=RTLIB-GCC-STATIC-UNWINDLIB-COMPILER-RT %s
// RTLIB-GCC-STATIC-UNWINDLIB-COMPILER-RT: "-lgcc"
// RTLIB-GCC-STATIC-UNWINDLIB-COMPILER-RT-SAME: "-l:libunwind.a"
//
// RUN: %clang -### %s 2>&1   \
// RUN:     --target=x86_64-unknown-linux -rtlib=compiler-rt \
// RUN:     --gcc-toolchain="" -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=RTLIB-COMPILER-RT %s
// RTLIB-COMPILER-RT: "{{.*}}libclang_rt.builtins-x86_64.a"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=compiler-rt --unwindlib=libunwind \
// RUN:     --gcc-toolchain="" -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT %s
// RUN: %clangxx -### %s 2>&1 \
// RUN:     --target=x86_64-unknown-linux -rtlib=compiler-rt --unwindlib=libunwind \
// RUN:     --gcc-toolchain="" -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT %s
// RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT: "{{.*}}libclang_rt.builtins-x86_64.a"
// RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT-SAME: "--as-needed"
// RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT-SAME: "-lunwind"
// RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT-SAME: "--no-as-needed"
//
// RUN: %clang -### %s 2>&1   \
// RUN:     --target=x86_64-unknown-linux -rtlib=compiler-rt --unwindlib=libgcc \
// RUN:     --gcc-toolchain="" -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=RTLIB-COMPILER-RT-UNWINDLIB-GCC %s
// RTLIB-COMPILER-RT-UNWINDLIB-GCC: "{{.*}}libclang_rt.builtins-x86_64.a"
// RTLIB-COMPILER-RT-UNWINDLIB-GCC-SAME: "-lgcc_s"
//
// RUN: %clang -### %s 2>&1              \
// RUN:     --target=x86_64-unknown-linux -rtlib=compiler-rt --unwindlib=libgcc \
// RUN:     -static --gcc-toolchain="" -resource-dir=%S/Inputs/resource_dir \
// RUN:   | FileCheck --check-prefix=RTLIB-COMPILER-RT-UNWINDLIB-GCC-STATIC %s
// RTLIB-COMPILER-RT-UNWINDLIB-GCC-STATIC: "{{.*}}libclang_rt.builtins-x86_64.a"
// RTLIB-COMPILER-RT-UNWINDLIB-GCC-STATIC-SAME: "-lgcc_eh"
//
// RUN: not %clang %s 2> %t.err              \
// RUN:     --target=x86_64-unknown-linux -rtlib=libgcc --unwindlib=libunwind \
// RUN:     --gcc-toolchain="" -resource-dir=%S/Inputs/resource_dir \
// RUN: FileCheck --input-file=%t.err --check-prefix=RTLIB-GCC-UNWINDLIB-COMPILER_RT %s
// RTLIB-GCC-UNWINDLIB-COMPILER_RT: "{{[.|\\\n]*}}--rtlib=libgcc requires --unwindlib=libgcc"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-w64-mingw32 -rtlib=compiler-rt --unwindlib=libunwind \
// RUN:     -shared-libgcc \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=MINGW-RTLIB-COMPILER-RT-SHARED-UNWINDLIB-COMPILER-RT %s
// MINGW-RTLIB-COMPILER-RT-SHARED-UNWINDLIB-COMPILER-RT: "{{.*}}libclang_rt.builtins-x86_64.a"
// MINGW-RTLIB-COMPILER-RT-SHARED-UNWINDLIB-COMPILER-RT-SAME: "-l:libunwind.dll.a"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-w64-mingw32 -rtlib=compiler-rt --unwindlib=libunwind \
// RUN:     -static-libgcc \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=MINGW-RTLIB-COMPILER-RT-STATIC-UNWINDLIB-COMPILER-RT %s
// MINGW-RTLIB-COMPILER-RT-STATIC-UNWINDLIB-COMPILER-RT: "{{.*}}libclang_rt.builtins-x86_64.a"
// MINGW-RTLIB-COMPILER-RT-STATIC-UNWINDLIB-COMPILER-RT-SAME: "-l:libunwind.a"
//
// RUN: %clang -### %s 2>&1 \
// RUN:     --target=x86_64-w64-mingw32 -rtlib=compiler-rt --unwindlib=libunwind \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=MINGW-RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT %s
// RUN: %clangxx -### %s 2>&1 \
// RUN:     --target=x86_64-w64-mingw32 -rtlib=compiler-rt --unwindlib=libunwind \
// RUN:     --gcc-toolchain="" \
// RUN:   | FileCheck --check-prefix=MINGW-RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT %s
// MINGW-RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT: "{{.*}}libclang_rt.builtins-x86_64.a"
// MINGW-RTLIB-COMPILER-RT-UNWINDLIB-COMPILER-RT-SAME: "-lunwind"
