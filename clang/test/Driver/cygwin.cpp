// RUN: %clang -### %s --target=i686-pc-windows-cygnus --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -resource-dir=%S/Inputs/resource_dir \
// RUN:   --stdlib=platform 2>&1 | FileCheck --check-prefix=CHECK %s
// CHECK:      "-cc1"
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
// CHECK-64-SAME: "--disable-high-entropy-va"

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
// RUN:   -Wl,--high-entropy-va 2>&1 | FileCheck --check-prefix=CHECK-64-ASLR-ON %s
// CHECK-64-ASLR-ON:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-ASLR-ON-SAME: "--high-entropy-va"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -Wl,--disable-high-entropy-va 2>&1 | FileCheck --check-prefix=CHECK-64-ASLR-OFF %s
// CHECK-64-ASLR-OFF:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-ASLR-OFF-SAME: "--disable-high-entropy-va"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -o a 2>&1 | FileCheck --check-prefix=CHECK-64-EXENAME %s
// CHECK-64-EXENAME:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-EXENAME-SAME: "-o" "a.exe"

// RUN: %clang -### %s --target=x86_64-pc-cygwin --sysroot=%S/Inputs/basic_cygwin_tree \
// RUN:   -o a.out 2>&1 | FileCheck --check-prefix=CHECK-64-EXENAME-WITH-EXT %s
// CHECK-64-EXENAME-WITH-EXT:      "{{.*}}ld{{(\.exe)?}}"
// CHECK-64-EXENAME-WITH-EXT-SAME: "-o" "a.out"
