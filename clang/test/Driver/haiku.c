// UNSUPPORTED: system-windows

// Check the C header paths
// RUN: %clang --target=x86_64-unknown-haiku -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-C-HEADER-PATH %s
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/non-packaged/develop/headers"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/app"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/device"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/drivers"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/game"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/interface"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/kernel"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/locale"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/mail"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/media"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/midi"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/midi2"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/net"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/opengl"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/storage"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/support"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/translation"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/add-ons/graphics"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/add-ons/input_server"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/add-ons/mail_daemon"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/add-ons/registrar"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/add-ons/screen_saver"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/add-ons/tracker"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/be_apps/Deskbar"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/be_apps/NetPositive"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/os/be_apps/Tracker"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/3rdparty"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/bsd"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/glibc"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/gnu"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers/posix"
// CHECK-C-HEADER-PATH: "-internal-isystem" "/boot/system/develop/headers"

// Check x86_64-unknown-haiku, X86_64
// RUN: %clang -### %s 2>&1 --target=x86_64-unknown-haiku \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/haiku_x86_64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-LD-X86_64 %s
// CHECK-LD-X86_64: "-cc1" "-triple" "x86_64-unknown-haiku"
// CHECK-LD-X86_64-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-LD-X86_64: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-X86_64-SAME: "--no-undefined"
// CHECK-LD-X86_64-SAME: "[[SYSROOT]]/boot/system/develop/lib/crti.o"
// CHECK-LD-X86_64-SAME: {{^}} "[[SYSROOT]]/boot/system/develop/tools/lib/gcc/x86_64-unknown-haiku/13.2.0/crtbeginS.o"
// CHECK-LD-X86_64-SAME: {{^}} "[[SYSROOT]]/boot/system/develop/lib/start_dyn.o"
// CHECK-LD-X86_64-SAME: {{^}} "[[SYSROOT]]/boot/system/develop/lib/init_term_dyn.o"
// CHECK-LD-X86_64-SAME: "-lgcc" "--push-state" "--as-needed" "-lgcc_s" "--no-as-needed" "--pop-state"
// CHECK-LD-X86_64-SAME: {{^}} "-lroot"
// CHECK-LD-X86_64-SAME: {{^}} "-lgcc" "--push-state" "--as-needed" "-lgcc_s" "--no-as-needed" "--pop-state"
// CHECK-LD-X86_64-SAME: {{^}} "[[SYSROOT]]/boot/system/develop/tools/lib/gcc/x86_64-unknown-haiku/13.2.0/crtendS.o"
// CHECK-LD-X86_64-SAME: {{^}} "[[SYSROOT]]/boot/system/develop/lib/crtn.o"

// Check the right flags are present with -shared
// RUN: %clang -### %s -shared 2>&1 --target=x86_64-unknown-haiku \
// RUN:     --gcc-toolchain="" \
// RUN:     --sysroot=%S/Inputs/haiku_x86_64_tree \
// RUN:   | FileCheck --check-prefix=CHECK-X86_64-SHARED %s
// CHECK-X86_64-SHARED: "-cc1" "-triple" "x86_64-unknown-haiku"
// CHECK-X86_64-SHARED-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// CHECK-X86_64-SHARED: "{{.*}}ld{{(.exe)?}}"
// CHECK-X86_64-SHARED-NOT: "[[SYSROOT]]/boot/system/develop/lib/start_dyn.o"

// Check default ARM CPU, ARMv6
// RUN: %clang -### %s 2>&1 --target=arm-unknown-haiku \
// RUN:   | FileCheck --check-prefix=CHECK-ARM-CPU %s
// CHECK-ARM-CPU: "-target-cpu" "arm1176jzf-s"
