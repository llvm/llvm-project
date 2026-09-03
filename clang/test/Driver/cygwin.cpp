// RUN: %clang -### %s --target=i686-pc-windows-cygnus --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir \
// RUN:   --stdlib=platform 2>&1 | FileCheck --check-prefix=CHECK %s
// CHECK:      "-cc1"
// CHECK-SAME: "-fno-use-init-array"
// CHECK-SAME: "-resource-dir" "[[RESOURCE:[^"]+]]"
// CHECK-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-pc-cygwin/10/../../../../include/c++/10"
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-pc-cygwin/10/../../../../include/i686-pc-cygwin/c++/10"
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-pc-cygwin/10/../../../../include/c++/10/backward"
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]{{(/|\\\\)}}include"
// CHECK-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/i686-pc-cygwin/10/../../../../i686-pc-cygwin/include"
// CHECK-SAME: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// CHECK-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/usr/include/w32api"
// CHECK-SAME: "-femulated-tls"
// CHECK-SAME: "-exception-model=dwarf"
// CHECK:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-SAME: "-m" "i386pe"
// CHECK-SAME: "{{.*}}{{/|\\\\}}crt0.o"
// CHECK-SAME: "{{.*}}i686-pc-cygwin{{/|\\\\}}{{[0-9.]*}}{{/|\\\\}}crtbegin.o"
// CHECK-SAME: "{{.*}}i686-pc-cygwin{{/|\\\\}}{{[0-9.]*}}{{/|\\\\}}crtend.o"

// RUN: %clang -### %s --target=i686-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   --stdlib=platform -static 2>&1 | FileCheck --check-prefix=CHECK-STATIC %s
// CHECK-STATIC:      "-cc1" "-triple" "i686-pc-windows-cygnus"
// CHECK-STATIC-SAME: "-static-define"
// CHECK-STATIC:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-STATIC-SAME: "-Bstatic"

// RUN: %clang -### %s --target=i686-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -shared 2>&1 | FileCheck --check-prefix=CHECK-SHARED %s
// CHECK-SHARED:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-SHARED-SAME: "--shared"
// CHECK-SHARED-SAME: "-e" "__cygwin_dll_entry@12"

// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:     --gcc-toolchain=%S/Inputs/basic_cross_cygwin_tree/usr \
// RUN:     --target=i686-pc-cygwin \
// RUN:   | FileCheck --check-prefix=CHECK-CROSS %s
// CHECK-CROSS: "-cc1" "-triple" "i686-pc-windows-cygnus"
// CHECK-CROSS: "{{.*}}/Inputs/basic_cross_cygwin_tree/usr/lib/gcc/i686-pc-msys/10/../../../../i686-pc-msys/bin{{(/|\\\\)}}as" "--32"
// CHECK-CROSS:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-CROSS-SAME: "{{.*}}{{/|\\\\}}crt0.o"
// CHECK-CROSS-SAME: "{{.*}}i686-pc-msys{{/|\\\\}}{{[0-9.]*}}{{/|\\\\}}crtbegin.o"
// CHECK-CROSS-SAME: "{{.*}}i686-pc-msys{{/|\\\\}}{{[0-9.]*}}{{/|\\\\}}crtend.o"

// RUN: %clang -### %s --target=x86_64-pc-windows-cygnus --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir \
// RUN:   --stdlib=platform 2>&1 | FileCheck --check-prefix=CHECK-64 %s
// CHECK-64:      "-cc1"
// CHECK-64-SAME: "-resource-dir" "[[RESOURCE:[^"]+]]"
// CHECK-64-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-msys/10/../../../../include/c++/10"
// CHECK-64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-msys/10/../../../../include/x86_64-pc-msys/c++/10"
// CHECK-64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-msys/10/../../../../include/c++/10/backward"
// CHECK-64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/local/include"
// CHECK-64-SAME: {{^}} "-internal-isystem" "[[RESOURCE]]{{(/|\\\\)}}include"
// CHECK-64-SAME: {{^}} "-internal-isystem" "[[SYSROOT]]/usr/lib/gcc/x86_64-pc-msys/10/../../../../x86_64-pc-msys/include"
// CHECK-64-SAME: "-internal-externc-isystem" "[[SYSROOT]]/include"
// CHECK-64-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/usr/include"
// CHECK-64-SAME: {{^}} "-internal-externc-isystem" "[[SYSROOT]]/usr/include/w32api"
// CHECK-64-SAME: "-femulated-tls"
// CHECK-64-SAME: "-exception-model=seh"
// CHECK-64:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-SAME: "-m" "i386pep"
// CHECK-64-SAME: "{{.*}}{{/|\\\\}}crt0.o"
// CHECK-64-SAME: "{{.*}}x86_64-pc-msys{{/|\\\\}}{{[0-9.]*}}{{/|\\\\}}crtbegin.o"
// CHECK-64-SAME: "{{.*}}x86_64-pc-msys{{/|\\\\}}{{[0-9.]*}}{{/|\\\\}}crtend.o"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   --stdlib=platform -static 2>&1 | FileCheck --check-prefix=CHECK-64-STATIC %s
// CHECK-64-STATIC:      "-cc1" "-triple" "x86_64-pc-windows-cygnus"
// CHECK-64-STATIC-SAME: "-static-define"
// CHECK-64-STATIC:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-STATIC-SAME: "-Bstatic"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -shared 2>&1 | FileCheck --check-prefix=CHECK-64-SHARED %s
// CHECK-64-SHARED:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-SHARED-SAME: "--shared"
// CHECK-64-SHARED-SAME: "-e" "_cygwin_dll_entry"

// RUN: %clang -### -o %t %s 2>&1 -no-integrated-as -fuse-ld=ld \
// RUN:     --gcc-toolchain=%S/Inputs/basic_cross_cygwin_tree/usr \
// RUN:     --target=x86_64-pc-cygwin \
// RUN:   | FileCheck --check-prefix=CHECK-64-CROSS %s
// CHECK-64-CROSS: "-cc1" "-triple" "x86_64-pc-windows-cygnus"
// CHECK-64-CROSS: "{{.*}}/Inputs/basic_cross_cygwin_tree/usr/lib/gcc/x86_64-pc-cygwin/10/../../../../x86_64-pc-cygwin/bin{{(/|\\\\)}}as" "--64"
// CHECK-64-CROSS:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-CROSS-SAME: "{{.*}}{{/|\\\\}}crt0.o"
// CHECK-64-CROSS-SAME: "{{.*}}x86_64-pc-cygwin{{/|\\\\}}{{[0-9.]*}}{{/|\\\\}}crtbegin.o"
// CHECK-64-CROSS-SAME: "{{.*}}x86_64-pc-cygwin{{/|\\\\}}{{[0-9.]*}}{{/|\\\\}}crtend.o"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -mdll 2>&1 | FileCheck --check-prefix=CHECK-64-DLL %s
// CHECK-64-DLL:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-DLL-SAME: "--dll"
// CHECK-64-DLL-SAME: "-e" "_cygwin_dll_entry"

// RUN: %clang -### %s --target=i686-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -mdll 2>&1 | FileCheck --check-prefix=CHECK-DLL %s
// CHECK-DLL:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-DLL-SAME: "--dll"
// CHECK-DLL-SAME: "-e" "__cygwin_dll_entry@12"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -mwindows 2>&1 | FileCheck --check-prefix=CHECK-64-WINDOWS %s
// CHECK-64-WINDOWS:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-WINDOWS-SAME: "--subsystem" "windows"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -mconsole 2>&1 | FileCheck --check-prefix=CHECK-64-CONSOLE %s
// CHECK-64-CONSOLE:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-CONSOLE-SAME: "--subsystem" "console"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -o a 2>&1 | FileCheck --check-prefix=CHECK-64-EXENAME %s
// CHECK-64-EXENAME:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-EXENAME-SAME: "-o" "a.exe"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -o a.out 2>&1 | FileCheck --check-prefix=CHECK-64-EXENAME-WITH-EXT %s
// CHECK-64-EXENAME-WITH-EXT:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-EXENAME-WITH-EXT-SAME: "-o" "a.out"

// RUN: %clang -### %s --target=i686-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   2>&1 | FileCheck --check-prefix=CHECK-ASLR-DEFAULT %s
// CHECK-ASLR-DEFAULT:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-ASLR-DEFAULT-NOT:  "--disable-high-entropy-va"

// RUN: %clang -### %s --target=i686-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   2>&1 | FileCheck --check-prefix=CHECK-NXCOMPAT-DEFAULT %s
// CHECK-NXCOMPAT-DEFAULT:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-NXCOMPAT-DEFAULT-SAME: "--disable-nxcompat"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   2>&1 | FileCheck --check-prefix=CHECK-64-ASLR-DEFAULT %s
// CHECK-64-ASLR-DEFAULT:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-ASLR-DEFAULT-SAME: "--disable-high-entropy-va"
// CHECK-64-ASLR-DEFAULT-SAME: "--disable-nxcompat"

// RUN: %clang -### %s --target=i686-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   2>&1 | FileCheck --check-prefix=CHECK-WRAP %s
// CHECK-WRAP:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-WRAP-SAME: "--wrap=_Znwj"
// CHECK-WRAP-SAME: "--wrap=_Znaj"
// CHECK-WRAP-SAME: "--wrap=_ZdlPv"
// CHECK-WRAP-SAME: "--wrap=_ZdaPv"
// CHECK-WRAP-SAME: "--wrap=_ZnwjRKSt9nothrow_t"
// CHECK-WRAP-SAME: "--wrap=_ZnajRKSt9nothrow_t"
// CHECK-WRAP-SAME: "--wrap=_ZdlPvRKSt9nothrow_t"
// CHECK-WRAP-SAME: "--wrap=_ZdaPvRKSt9nothrow_t"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   2>&1 | FileCheck --check-prefix=CHECK-64-WRAP %s
// CHECK-64-WRAP:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-WRAP-SAME: "--wrap=_Znwm"
// CHECK-64-WRAP-SAME: "--wrap=_Znam"
// CHECK-64-WRAP-SAME: "--wrap=_ZdlPv"
// CHECK-64-WRAP-SAME: "--wrap=_ZdaPv"
// CHECK-64-WRAP-SAME: "--wrap=_ZnwmRKSt9nothrow_t"
// CHECK-64-WRAP-SAME: "--wrap=_ZnamRKSt9nothrow_t"
// CHECK-64-WRAP-SAME: "--wrap=_ZdlPvRKSt9nothrow_t"
// CHECK-64-WRAP-SAME: "--wrap=_ZdaPvRKSt9nothrow_t"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:                -resource-dir=%S/Inputs/resource_dir -rtlib=compiler-rt \
// RUN:   2>&1 | FileCheck --check-prefix=CHECK-64-RTLIB %s
// CHECK-64-RTLIB:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-RTLIB-SAME: "{{.*}}{{/|\\\\}}lib{{/|\\\\}}cygwin{{/|\\\\}}libclang_rt.builtins-x86_64.a"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:                -resource-dir=%S/Inputs/resource_dir_with_per_target_subdir -rtlib=compiler-rt \
// RUN:   2>&1 | FileCheck --check-prefix=CHECK-64-RTLIB-PER-TARGET %s
// CHECK-64-RTLIB-PER-TARGET:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-RTLIB-PER-TARGET-SAME: "{{.*}}{{/|\\\\}}lib{{/|\\\\}}x86_64-pc-windows-cygnus{{/|\\\\}}libclang_rt.builtins.a"
