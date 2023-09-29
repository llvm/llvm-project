/// Check the behavior of toolchain for NEC Aurora VE
/// REQUIRES: ve-registered-target
/// UNSUPPORTED: system-windows

///-----------------------------------------------------------------------------
/// Checking dwarf-version

// RUN: %clangxx -### -g --target=ve-unknown-linux-gnu \
// RUN:     %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s
// DWARF_VER: "-dwarf-version=5"

///-----------------------------------------------------------------------------
/// Checking include-path

// RUN: %clangxx -### --target=ve-unknown-linux-gnu \
// RUN:     --sysroot %S/Inputs/basic_ve_tree %s \
// RUN:     -ccc-install-dir %S/Inputs/basic_ve_tree/bin \
// RUN:     -resource-dir=%S/Inputs/basic_ve_tree/resource_dir \
// RUN:     2>&1 | FileCheck -check-prefix=DEFINC %s
// DEFINC: "-cc1"
// DEFINC-SAME: "-nostdsysteminc"
// DEFINC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// DEFINC-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// DEFINC-SAME: "-internal-isystem" "{{.*}}/bin/../include/ve-unknown-linux-gnu/c++/v1"
// DEFINC-SAME: "-internal-isystem" "{{.*}}/bin/../include/c++/v1"
// DEFINC-SAME: "-internal-isystem" "[[RESOURCE_DIR]]/include"
// DEFINC-SAME: "-internal-isystem" "[[SYSROOT]]/opt/nec/ve/include"
// DEFINC: nld"
// DEFINC-SAME: "-rpath" "[[SYSROOT]]/bin/../lib/ve-unknown-linux-gnu"

// RUN: %clangxx -### --target=ve-unknown-linux-gnu \
// RUN:     --sysroot %S/Inputs/basic_ve_tree %s \
// RUN:     -ccc-install-dir %S/Inputs/basic_ve_tree/bin \
// RUN:     -resource-dir=%S/Inputs/basic_ve_tree/resource_dir \
// RUN:     -nostdlibinc 2>&1 | FileCheck -check-prefix=NOSTDLIBINC %s
// NOSTDLIBINC: "-cc1"
// NOSTDLIBINC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// NOSTDLIBINC-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// NOSTDLIBINC-NOT: "-internal-isystem" "{{.*}}/bin/../include/ve-unknown-linux-gnu/c++/v1"
// NOSTDLIBINC-NOT: "-internal-isystem" "{{.*}}/bin/../include/c++/v1"
// NOSTDLIBINC-SAME: "-internal-isystem" "[[RESOURCE_DIR]]/include"
// NOSTDLIBINC-NOT: "-internal-isystem" "[[SYSROOT]]/opt/nec/ve/include"

// RUN: %clangxx -### --target=ve-unknown-linux-gnu \
// RUN:     --sysroot %S/Inputs/basic_ve_tree %s \
// RUN:     -ccc-install-dir %S/Inputs/basic_ve_tree/bin \
// RUN:     -resource-dir=%S/Inputs/basic_ve_tree/resource_dir \
// RUN:     -nobuiltininc 2>&1 | FileCheck -check-prefix=NOBUILTININC %s
// NOBUILTININC: "-cc1"
// NOBUILTININC-SAME: "-nobuiltininc"
// NOBUILTININC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// NOBUILTININC-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// NOBUILTININC-SAME: "-internal-isystem" "{{.*}}/bin/../include/ve-unknown-linux-gnu/c++/v1"
// NOBUILTININC-SAME: "-internal-isystem" "{{.*}}/bin/../include/c++/v1"
// NOBUILTININC-NOT: "-internal-isystem" "[[RESOURCE_DIR]]/include"
// NOBUILTININC-SAME: "-internal-isystem" "[[SYSROOT]]/opt/nec/ve/include"

// RUN: %clangxx -### --target=ve-unknown-linux-gnu \
// RUN:     --sysroot %S/Inputs/basic_ve_tree %s \
// RUN:     -ccc-install-dir %S/Inputs/basic_ve_tree/bin \
// RUN:     -resource-dir=%S/Inputs/basic_ve_tree/resource_dir \
// RUN:     -nostdinc 2>&1 | FileCheck -check-prefix=NOSTDINC %s
// NOSTDINC: "-cc1"
// NOSTDINC-SAME: "-nobuiltininc"
// NOSTDINC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// NOSTDINC-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// NOSTDINC-NOT: "-internal-isystem" "{{.*}}/bin/../include/ve-unknown-linux-gnu/c++/v1"
// NOSTDINC-NOT: "-internal-isystem" "{{.*}}/bin/../include/c++/v1"
// NOSTDINC-NOT: "-internal-isystem" "[[RESOURCE_DIR]]/include"
// NOSTDINC-NOT: "-internal-isystem" "[[SYSROOT]]/opt/nec/ve/include"

// RUN: %clangxx -### --target=ve-unknown-linux-gnu \
// RUN:     --sysroot %S/Inputs/basic_ve_tree %s \
// RUN:     -ccc-install-dir %S/Inputs/basic_ve_tree/bin \
// RUN:     -resource-dir=%S/Inputs/basic_ve_tree/resource_dir \
// RUN:     -nostdinc++ 2>&1 | FileCheck -check-prefix=NOSTDINCXX %s
// NOSTDINCXX: "-cc1"
// NOSTDINCXX-SAME: "-nostdinc++"
// NOSTDINCXX-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// NOSTDINCXX-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// NOSTDINCXX-NOT: "-internal-isystem" "{{.*}}/bin/../include/ve-unknown-linux-gnu/c++/v1"
// NOSTDINCXX-NOT: "-internal-isystem" "{{.*}}/bin/../include/c++/v1"
// NOSTDINCXX-SAME: "-internal-isystem" "[[RESOURCE_DIR]]/include"
// NOSTDINCXX-SAME: "-internal-isystem" "[[SYSROOT]]/opt/nec/ve/include"

///-----------------------------------------------------------------------------
/// Checking environment variable NCC_CPLUS_INCLUDE_PATH

// RUN: env NCC_CPLUS_INCLUDE_PATH=/test/test %clangxx -### \
// RUN:     --target=ve-unknown-linux-gnu %s \
// RUN:     --sysroot %S/Inputs/basic_ve_tree \
// RUN:     -resource-dir=%S/Inputs/basic_ve_tree/resource_dir \
// RUN:     2>&1 | FileCheck -check-prefix=DEFINCENV %s

// DEFINCENV: "-cc1"
// DEFINCENV-SAME: "-nostdsysteminc"
// DEFINCENV-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// DEFINCENV-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// DEFINCENV-SAME: "-internal-isystem" "/test/test"
// DEFINCENV-SAME: "-internal-isystem" "[[RESOURCE_DIR]]/include"
// DEFINCENV-SAME: "-internal-isystem" "[[SYSROOT]]/opt/nec/ve/include"

///-----------------------------------------------------------------------------
/// Checking -faddrsig

// RUN: %clangxx -### --target=ve-unknown-linux-gnu \
// RUN:     %s 2>&1 | FileCheck -check-prefix=DEFADDRSIG %s
// DEFADDRSIG: "-cc1"
// DEFADDRSIG-NOT: "-faddrsig"

///-----------------------------------------------------------------------------
/// Checking -fintegrated-as

// RUN: %clangxx -### --target=ve-unknown-linux-gnu \
// RUN:     -x assembler -fuse-ld=ld %s 2>&1 | \
// RUN:    FileCheck -check-prefix=AS %s
// RUN: %clangxx -### --target=ve-unknown-linux-gnu \
// RUN:     -fno-integrated-as -x assembler -fuse-ld=ld %s 2>&1 | \
// RUN:    FileCheck -check-prefix=NAS %s

// AS: "-cc1as"
// AS: nld{{.*}}

// NAS: nas{{.*}}
// NAS: nld{{.*}}

///-----------------------------------------------------------------------------
/// Checking default behavior:
///  - dynamic linker
///  - library paths
///  - nld VE specific options
///  - sjlj exception

// RUN: %clangxx -### --target=ve-unknown-linux-gnu \
// RUN:     --sysroot %S/Inputs/basic_ve_tree \
// RUN:     -fuse-ld=ld \
// RUN:     -resource-dir=%S/Inputs/basic_ve_tree/resource_dir \
// RUN:     --unwindlib=none \
// RUN:     --stdlib=libc++ %s 2>&1 | FileCheck -check-prefix=DEF %s

// DEF:      "-cc1"
// DEF-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// DEF-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// DEF-SAME: "-exception-model=sjlj"
// DEF:      nld"
// DEF-SAME: "--sysroot=[[SYSROOT]]"
// DEF-SAME: "-dynamic-linker" "/opt/nec/ve/lib/ld-linux-ve.so.1"
// DEF-SAME: "[[SYSROOT]]/opt/nec/ve/lib/crt1.o"
// DEF-SAME: "[[SYSROOT]]/opt/nec/ve/lib/crti.o"
// DEF-SAME: "-z" "max-page-size=0x4000000"
// DEF-SAME: "[[RESOURCE_DIR]]/lib/ve-unknown-linux-gnu/clang_rt.crtbegin.o"
// DEF-SAME: "-lc++" "-lc++abi" "-lunwind" "-lpthread" "-ldl"
// DEF-SAME: "[[RESOURCE_DIR]]/lib/ve-unknown-linux-gnu/libclang_rt.builtins.a" "-lc"
// DEF-SAME: "[[RESOURCE_DIR]]/lib/ve-unknown-linux-gnu/libclang_rt.builtins.a"
// DEF-SAME: "[[RESOURCE_DIR]]/lib/ve-unknown-linux-gnu/clang_rt.crtend.o"
// DEF-SAME: "[[SYSROOT]]/opt/nec/ve/lib/crtn.o"
