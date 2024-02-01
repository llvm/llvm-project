// RUN: %clang -target i686-windows-gnu -rtlib=platform -c -### --sysroot=%S/Inputs/mingw_clang_tree/mingw32 %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_CLANG_TREE %s
// CHECK_MINGW_CLANG_TREE: "[[BASE:[^"]+]]/Inputs/mingw_clang_tree/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include"
// CHECK_MINGW_CLANG_TREE: "[[BASE:[^"]+]]/Inputs/mingw_clang_tree/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}usr{{/|\\\\}}include"
// CHECK_MINGW_CLANG_TREE: "[[BASE]]/Inputs/mingw_clang_tree/mingw32{{/|\\\\}}include"


// RUN: %clang -target i686-windows-gnu -rtlib=platform -stdlib=libc++ -c -### --sysroot=%S/Inputs/mingw_clang_tree/mingw32 %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_CLANG_TREE_LIBCXX %s
// CHECK_MINGW_CLANG_TREE_LIBCXX: "[[BASE:[^"]+]]/Inputs/mingw_clang_tree/mingw32{{/|\\\\}}include{{/|\\\\}}i686-unknown-windows-gnu{{/|\\\\}}c++{{/|\\\\}}v1"
// CHECK_MINGW_CLANG_TREE_LIBCXX: "[[BASE:[^"]+]]/Inputs/mingw_clang_tree/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}v1"


// RUN: %clang -target i686-pc-windows-gnu -rtlib=platform -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_mingw_org_tree/mingw %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_ORG_TREE %s
// CHECK_MINGW_ORG_TREE: "[[BASE:[^"]+]]/Inputs/mingw_mingw_org_tree/mingw{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}mingw32{{/|\\\\}}4.8.1{{/|\\\\}}include{{/|\\\\}}c++"
// CHECK_MINGW_ORG_TREE: "[[BASE]]/Inputs/mingw_mingw_org_tree/mingw{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}mingw32{{/|\\\\}}4.8.1{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}mingw32"
// CHECK_MINGW_ORG_TREE: "[[BASE]]/Inputs/mingw_mingw_org_tree/mingw{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}mingw32{{/|\\\\}}4.8.1{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}backward"
// CHECK_MINGW_ORG_TREE: "[[BASE]]/Inputs/mingw_mingw_org_tree/mingw{{/|\\\\}}mingw32{{/|\\\\}}include"
// CHECK_MINGW_ORG_TREE: "[[BASE]]/Inputs/mingw_mingw_org_tree/mingw{{/|\\\\}}include"


// RUN: %clang -target i686-pc-windows-gnu -rtlib=platform -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_mingw_builds_tree/mingw32 %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_BUILDS_TREE %s
// CHECK_MINGW_BUILDS_TREE: "[[BASE:[^"]+]]/Inputs/mingw_mingw_builds_tree/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++"
// CHECK_MINGW_BUILDS_TREE: "[[BASE]]/Inputs/mingw_mingw_builds_tree/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}i686-w64-mingw32"
// CHECK_MINGW_BUILDS_TREE: "[[BASE]]/Inputs/mingw_mingw_builds_tree/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}backward"
// CHECK_MINGW_BUILDS_TREE: "[[BASE]]/Inputs/mingw_mingw_builds_tree/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include"


// RUN: %clang -target i686-pc-windows-gnu -rtlib=platform -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_msys2_tree/msys64/mingw32 %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_MSYS_TREE %s
// CHECK_MINGW_MSYS_TREE: "[[BASE:[^"]+]]/Inputs/mingw_msys2_tree/msys64{{/|\\\\}}mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}4.9.2"
// CHECK_MINGW_MSYS_TREE: "[[BASE]]/Inputs/mingw_msys2_tree/msys64/mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}4.9.2{{/|\\\\}}i686-w64-mingw32"
// CHECK_MINGW_MSYS_TREE: "[[BASE]]/Inputs/mingw_msys2_tree/msys64/mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}4.9.2{{/|\\\\}}backward"
// CHECK_MINGW_MSYS_TREE: "[[BASE]]/Inputs/mingw_msys2_tree/msys64/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include"
// CHECK_MINGW_MSYS_TREE: "[[BASE]]/Inputs/mingw_msys2_tree/msys64/mingw32{{/|\\\\}}include"


// RUN: %clang -target x86_64-pc-windows-gnu -rtlib=platform -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_opensuse_tree/usr %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_OPENSUSE_TREE %s
// CHECK_MINGW_OPENSUSE_TREE: "[[BASE:[^"]+]]/Inputs/mingw_opensuse_tree/usr{{/|\\\\}}lib64{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}5.1.0{{/|\\\\}}include{{/|\\\\}}c++"
// CHECK_MINGW_OPENSUSE_TREE: "[[BASE]]/Inputs/mingw_opensuse_tree/usr{{/|\\\\}}lib64{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}5.1.0{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}x86_64-w64-mingw32"
// CHECK_MINGW_OPENSUSE_TREE: "[[BASE]]/Inputs/mingw_opensuse_tree/usr{{/|\\\\}}lib64{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}5.1.0{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}backward"
// CHECK_MINGW_OPENSUSE_TREE: "[[BASE]]/Inputs/mingw_opensuse_tree/usr{{/|\\\\}}x86_64-w64-mingw32/sys-root/mingw{{/|\\\\}}include"


// RUN: %clang -target x86_64-pc-windows-gnu -rtlib=platform -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_fedora_tree/usr %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_FEDORA_TREE %s
// CHECK_MINGW_FEDORA_TREE: "[[BASE:[^"]+]]/Inputs/mingw_fedora_tree/usr{{/|\\\\}}x86_64-w64-mingw32ucrt/sys-root/mingw{{/|\\\\}}include{{/|\\\\}}c++"
// CHECK_MINGW_FEDORA_TREE: "[[BASE]]/Inputs/mingw_fedora_tree/usr{{/|\\\\}}x86_64-w64-mingw32ucrt/sys-root/mingw{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}x86_64-w64-mingw32ucrt"
// CHECK_MINGW_FEDORA_TREE: "[[BASE]]/Inputs/mingw_fedora_tree/usr{{/|\\\\}}x86_64-w64-mingw32ucrt/sys-root/mingw{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}backward"
// CHECK_MINGW_FEDORA_TREE: "[[BASE]]/Inputs/mingw_fedora_tree/usr{{/|\\\\}}x86_64-w64-mingw32ucrt/sys-root/mingw{{/|\\\\}}include"


// RUN: %clang -target i686-pc-windows-gnu -rtlib=platform -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_arch_tree/usr %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_ARCH_TREE %s
// CHECK_MINGW_ARCH_TREE: "[[BASE:[^"]+]]/Inputs/mingw_arch_tree/usr{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}5.1.0"
// CHECK_MINGW_ARCH_TREE: "[[BASE]]/Inputs/mingw_arch_tree/usr{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}5.1.0{{/|\\\\}}i686-w64-mingw32"
// CHECK_MINGW_ARCH_TREE: "[[BASE]]/Inputs/mingw_arch_tree/usr{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}5.1.0{{/|\\\\}}backward"
// CHECK_MINGW_ARCH_TREE: "[[BASE]]/Inputs/mingw_arch_tree/usr{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include"


// RUN: %clang -target x86_64-pc-windows-gnu -rtlib=platform -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_ubuntu_tree/usr %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_UBUNTU_TREE %s
// CHECK_MINGW_UBUNTU_TREE: "[[BASE:[^"]+]]/Inputs/mingw_ubuntu_tree/usr{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}4.8"
// CHECK_MINGW_UBUNTU_TREE: "[[BASE]]/Inputs/mingw_ubuntu_tree/usr{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}4.8{{/|\\\\}}x86_64-w64-mingw32"
// CHECK_MINGW_UBUNTU_TREE: "[[BASE]]/Inputs/mingw_ubuntu_tree/usr{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}4.8{{/|\\\\}}backward"
// CHECK_MINGW_UBUNTU_TREE: "[[BASE]]/Inputs/mingw_ubuntu_tree/usr{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"


// RUN: %clang -target x86_64-pc-windows-gnu -rtlib=platform -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_ubuntu_posix_tree/usr %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_UBUNTU_POSIX_TREE %s
// CHECK_MINGW_UBUNTU_POSIX_TREE: "[[BASE:[^"]+]]/Inputs/mingw_ubuntu_posix_tree/usr{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}c++"
// CHECK_MINGW_UBUNTU_POSIX_TREE: "[[BASE]]/Inputs/mingw_ubuntu_posix_tree/usr{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}x86_64-w64-mingw32"
// CHECK_MINGW_UBUNTU_POSIX_TREE: "[[BASE]]/Inputs/mingw_ubuntu_posix_tree/usr{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}10.2-posix{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}backward"
// CHECK_MINGW_UBUNTU_POSIX_TREE: "[[BASE]]/Inputs/mingw_ubuntu_posix_tree/usr{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"

// RUN: %clang -target i686-windows-gnu -E -### %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_NO_UNICODE %s
// RUN: %clang -target i686-windows-gnu -E -### %s -municode 2>&1 | FileCheck -check-prefix=CHECK_MINGW_UNICODE %s
// CHECK_MINGW_NO_UNICODE-NOT: "-DUNICODE"
// CHECK_MINGW_UNICODE: "-DUNICODE"

// RUN: %clang -target i686-windows-gnu -### %s 2>&1 | FileCheck -check-prefix=CHECK_NO_SUBSYS %s
// RUN: %clang -target i686-windows-gnu -### %s -mwindows -mconsole 2>&1 | FileCheck -check-prefix=CHECK_SUBSYS_CONSOLE %s
// RUN: %clang -target i686-windows-gnu -### %s -mconsole -mwindows 2>&1 | FileCheck -check-prefix=CHECK_SUBSYS_WINDOWS %s
// CHECK_NO_SUBSYS-NOT: "--subsystem"
// CHECK_SUBSYS_CONSOLE: "--subsystem" "console"
// CHECK_SUBSYS_WINDOWS: "--subsystem" "windows"

// RUN: %clang -target i686-windows-gnu -### %s 2>&1 | FileCheck -check-prefix=CHECK_NO_INIT_ARRAY %s
// CHECK_NO_INIT_ARRAY: "-fno-use-init-array"

// RUN: %clang -target arm64ec-windows-gnu -### -o /dev/null %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix CHECK_MINGW_EC_LINK
// CHECK_MINGW_EC_LINK: "-m" "arm64ecpe"
