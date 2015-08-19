// RUN: %clang -target i686-windows-gnu -c -### --sysroot=%S/Inputs/mingw_clang_tree/mingw32 %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_CLANG_TREE %s
// CHECK_MINGW_CLANG_TREE: "{{.*}}/Inputs/mingw_clang_tree/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include"
// CHECK_MINGW_CLANG_TREE: "{{.*}}/Inputs/mingw_clang_tree/mingw32{{/|\\\\}}include"


// RUN: %clang -target i686-pc-windows-gnu -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_mingw_org_tree/mingw %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_ORG_TREE %s
// CHECK_MINGW_ORG_TREE: "{{.*}}/Inputs/mingw_mingw_org_tree/mingw{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}mingw32{{/|\\\\}}4.8.1{{/|\\\\}}include{{/|\\\\}}c++"
// CHECK_MINGW_ORG_TREE: "{{.*}}/Inputs/mingw_mingw_org_tree/mingw{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}mingw32{{/|\\\\}}4.8.1{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}mingw32"
// CHECK_MINGW_ORG_TREE: "{{.*}}{{/|\\\\}}Inputs/mingw_mingw_org_tree/mingw{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}mingw32{{/|\\\\}}4.8.1{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}backward"
// CHECK_MINGW_ORG_TREE: "{{.*}}/Inputs/mingw_mingw_org_tree/mingw{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}mingw32{{/|\\\\}}4.8.1{{/|\\\\}}include"
// CHECK_MINGW_ORG_TREE: "{{.*}}/Inputs/mingw_mingw_org_tree/mingw{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}mingw32{{/|\\\\}}4.8.1{{/|\\\\}}include-fixed"
// CHECK_MINGW_ORG_TREE: "{{.*}}/Inputs/mingw_mingw_org_tree/mingw{{/|\\\\}}mingw32{{/|\\\\}}include"
// CHECK_MINGW_ORG_TREE: {{.*}}/Inputs/mingw_mingw_org_tree/mingw{{/|\\\\}}include


// RUN: %clang -target i686-pc-windows-gnu -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_mingw_builds_tree/mingw32 %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_BUILDS_TREE %s
// CHECK_MINGW_BUILDS_TREE: "{{.*}}/Inputs/mingw_mingw_builds_tree/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++"
// CHECK_MINGW_BUILDS_TREE: "{{.*}}/Inputs/mingw_mingw_builds_tree/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}i686-w64-mingw32"
// CHECK_MINGW_BUILDS_TREE: "{{.*}}/Inputs/mingw_mingw_builds_tree/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}backward"
// CHECK_MINGW_BUILDS_TREE: "{{.*}}/Inputs/mingw_mingw_builds_tree/mingw32{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}4.9.1{{/|\\\\}}include"
// CHECK_MINGW_BUILDS_TREE: "{{.*}}/Inputs/mingw_mingw_builds_tree/mingw32{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}4.9.1{{/|\\\\}}include-fixed"
// CHECK_MINGW_BUILDS_TREE: "{{.*}}/Inputs/mingw_mingw_builds_tree/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include"


// RUN: %clang -target i686-pc-windows-gnu -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_msys2_tree/msys64/mingw32 %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_MSYS_TREE %s
// CHECK_MINGW_MSYS_TREE: "{{.*}}/Inputs/mingw_msys2_tree/msys64{{/|\\\\}}mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}4.9.2"
// CHECK_MINGW_MSYS_TREE: "{{.*}}/Inputs/mingw_msys2_tree/msys64/mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}4.9.2{{/|\\\\}}i686-w64-mingw32"
// CHECK_MINGW_MSYS_TREE: "{{.*}}/Inputs/mingw_msys2_tree/msys64/mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}4.9.2{{/|\\\\}}backward"
// CHECK_MINGW_MSYS_TREE:  "{{.*}}/Inputs/mingw_msys2_tree/msys64/mingw32{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}4.9.2{{/|\\\\}}include"
// CHECK_MINGW_MSYS_TREE:  "{{.*}}/Inputs/mingw_msys2_tree/msys64/mingw32{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}4.9.2{{/|\\\\}}include-fixed"
// CHECK_MINGW_MSYS_TREE: "{{.*}}/Inputs/mingw_msys2_tree/msys64/mingw32{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include"
// CHECK_MINGW_MSYS_TREE: "{{.*}}/Inputs/mingw_msys2_tree/msys64/mingw32{{/|\\\\}}include"


// RUN: %clang -target x86_64-pc-windows-gnu -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_opensuse_tree/usr %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_OPENSUSE_TREE %s
// CHECK_MINGW_OPENSUSE_TREE: "{{.*}}/Inputs/mingw_opensuse_tree/usr{{/|\\\\}}lib64{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}5.1.0{{/|\\\\}}include{{/|\\\\}}c++"
// CHECK_MINGW_OPENSUSE_TREE: "{{.*}}/Inputs/mingw_opensuse_tree/usr{{/|\\\\}}lib64{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}5.1.0{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}x86_64-w64-mingw32"
// CHECK_MINGW_OPENSUSE_TREE: "{{.*}}/Inputs/mingw_opensuse_tree/usr{{/|\\\\}}lib64{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}5.1.0{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}backward"
// CHECK_MINGW_OPENSUSE_TREE: "{{.*}}/Inputs/mingw_opensuse_tree/usr{{/|\\\\}}lib64{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}5.1.0{{/|\\\\}}include"
// CHECK_MINGW_OPENSUSE_TREE: "{{.*}}/Inputs/mingw_opensuse_tree/usr{{/|\\\\}}x86_64-w64-mingw32/sys-root/mingw/include"
// CHECK_MINGW_OPENSUSE_TREE: "{{.*}}/Inputs/mingw_opensuse_tree/usr{{/|\\\\}}lib64{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}5.1.0{{/|\\\\}}include-fixed"


// RUN: %clang -target i686-pc-windows-gnu -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_arch_tree/usr %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_ARCH_TREE %s
// CHECK_MINGW_ARCH_TREE: "{{.*}}/Inputs/mingw_arch_tree/usr{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}5.1.0"
// CHECK_MINGW_ARCH_TREE: "{{.*}}/Inputs/mingw_arch_tree/usr{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}5.1.0{{/|\\\\}}i686-w64-mingw32"
// CHECK_MINGW_ARCH_TREE: "{{.*}}/Inputs/mingw_arch_tree/usr{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}5.1.0{{/|\\\\}}backward"
// CHECK_MINGW_ARCH_TREE: "{{.*}}/Inputs/mingw_arch_tree/usr{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}5.1.0{{/|\\\\}}include"
// CHECK_MINGW_ARCH_TREE: "{{.*}}/Inputs/mingw_arch_tree/usr{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}5.1.0{{/|\\\\}}include-fixed"
// CHECK_MINGW_ARCH_TREE: "{{.*}}/Inputs/mingw_arch_tree/usr{{/|\\\\}}i686-w64-mingw32{{/|\\\\}}include"


// RUN: %clang -target x86_64-pc-windows-gnu -stdlib=libstdc++ -c -### --sysroot=%S/Inputs/mingw_ubuntu_tree/usr %s 2>&1 | FileCheck -check-prefix=CHECK_MINGW_UBUNTU_TREE %s
// CHECK_MINGW_UBUNTU_TREE: "{{.*}}/Inputs/mingw_ubuntu_tree/usr{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}4.8"
// CHECK_MINGW_UBUNTU_TREE: "{{.*}}/Inputs/mingw_ubuntu_tree/usr{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}4.8{{/|\\\\}}x86_64-w64-mingw32"
// CHECK_MINGW_UBUNTU_TREE: "{{.*}}/Inputs/mingw_ubuntu_tree/usr{{/|\\\\}}include{{/|\\\\}}c++{{/|\\\\}}4.8{{/|\\\\}}backward"
// CHECK_MINGW_UBUNTU_TREE: "{{.*}}/Inputs/mingw_ubuntu_tree/usr{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}4.8{{/|\\\\}}include"
// CHECK_MINGW_UBUNTU_TREE: "{{.*}}/Inputs/mingw_ubuntu_tree/usr{{/|\\\\}}lib{{/|\\\\}}gcc{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}4.8{{/|\\\\}}include-fixed"
// CHECK_MINGW_UBUNTU_TREE: "{{.*}}/Inputs/mingw_ubuntu_tree/usr{{/|\\\\}}x86_64-w64-mingw32{{/|\\\\}}include"
