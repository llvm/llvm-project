/// Check the behavior of toolchain for NEC Aurora VE
/// REQUIRES: ve-registered-target
/// UNSUPPORTED: system-windows

///-----------------------------------------------------------------------------
/// Checking dwarf-version

// RUN: %clang -### -g -target ve %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s
// DWARF_VER: "-dwarf-version=4"

///-----------------------------------------------------------------------------
/// Checking include-path

// RUN: %clang -### -target ve --sysroot %S/Inputs/basic_ve_tree %s \
// RUN:     -resource-dir=%S/Inputs/basic_ve_tree/resource_dir \
// RUN:     2>&1 | FileCheck -check-prefix=DEFINC %s
// DEFINC: clang{{.*}} "-cc1"
// DEFINC-SAME: "-nostdsysteminc"
// DEFINC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// DEFINC-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// DEFINC-SAME: "-internal-isystem" "[[RESOURCE_DIR]]/include"
// DEFINC-SAME: "-internal-isystem" "[[SYSROOT]]/opt/nec/ve/include"

// RUN: %clang -### -target ve --sysroot %S/Inputs/basic_ve_tree %s \
// RUN:     -resource-dir=%S/Inputs/basic_ve_tree/resource_dir \
// RUN:     -nostdlibinc 2>&1 | FileCheck -check-prefix=NOSTDLIBINC %s
// NOSTDLIBINC: clang{{.*}} "-cc1"
// NOSTDLIBINC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// NOSTDLIBINC-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// NOSTDLIBINC-SAME: "-internal-isystem" "[[RESOURCE_DIR]]/include"
// NOSTDLIBINC-NOT: "-internal-isystem" "[[SYSROOT]]/opt/nec/ve/include"

// RUN: %clang -### -target ve --sysroot %S/Inputs/basic_ve_tree %s \
// RUN:     -resource-dir=%S/Inputs/basic_ve_tree/resource_dir \
// RUN:     -nobuiltininc 2>&1 | FileCheck -check-prefix=NOBUILTININC %s
// NOBUILTININC: clang{{.*}} "-cc1"
// NOBUILTININC-SAME: "-nobuiltininc"
// NOBUILTININC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// NOBUILTININC-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// NOBUILTININC-NOT: "-internal-isystem" "[[RESOURCE_DIR]]/include"
// NOBUILTININC-SAME: "-internal-isystem" "[[SYSROOT]]/opt/nec/ve/include"

// RUN: %clang -### -target ve --sysroot %S/Inputs/basic_ve_tree %s \
// RUN:     -resource-dir=%S/Inputs/basic_ve_tree/resource_dir \
// RUN:     -nostdinc 2>&1 | FileCheck -check-prefix=NOSTDINC %s
// NOSTDINC: clang{{.*}} "-cc1"
// NOSTDINC-SAME: "-nobuiltininc"
// NOSTDINC-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// NOSTDINC-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// NOSTDINC-NOT: "-internal-isystem" "[[RESOURCE_DIR]]/include"
// NOSTDINC-NOT: "-internal-isystem" "[[SYSROOT]]/opt/nec/ve/include"

///-----------------------------------------------------------------------------
/// Checking -faddrsig

// RUN: %clang -### -target ve %s 2>&1 | FileCheck -check-prefix=DEFADDRSIG %s
// DEFADDRSIG: clang{{.*}} "-cc1"
// DEFADDRSIG-NOT: "-faddrsig"

///-----------------------------------------------------------------------------
/// Checking -fintegrated-as

// RUN: %clang -### -target ve \
// RUN:    -x assembler -fuse-ld=ld %s 2>&1 | \
// RUN:    FileCheck -check-prefix=AS %s
// RUN: %clang -### -target ve \
// RUN:    -fno-integrated-as -fuse-ld=ld -x assembler %s 2>&1 | \
// RUN:    FileCheck -check-prefix=NAS %s

// AS: clang{{.*}} "-cc1as"
// AS: nld{{.*}}

// NAS: nas{{.*}}
// NAS: nld{{.*}}

///-----------------------------------------------------------------------------
/// Checking default behavior:
///  - dynamic linker
///  - library paths
///  - nld VE specific options
///  - sjlj exception

// RUN: %clang -### -target ve-unknown-linux-gnu \
// RUN:     --sysroot %S/Inputs/basic_ve_tree \
// RUN:     -resource-dir=%S/Inputs/basic_ve_tree/resource_dir \
// RUN:     -fuse-ld=ld \
// RUN:     %s 2>&1 | FileCheck -check-prefix=DEF %s

// DEF:      clang{{.*}}" "-cc1"
// DEF-SAME: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// DEF-SAME: "-isysroot" "[[SYSROOT:[^"]+]]"
// DEF-SAME: "-exception-model=sjlj"
// DEF:      nld"
// DEF-SAME: "--sysroot=[[SYSROOT]]"
// DEF-SAME: "-dynamic-linker" "/opt/nec/ve/lib/ld-linux-ve.so.1"
// DEF-SAME: "[[SYSROOT]]/opt/nec/ve/lib/crt1.o"
// DEF-SAME: "[[SYSROOT]]/opt/nec/ve/lib/crti.o"
// DEF-SAME: "-z" "max-page-size=0x4000000"
// DEF-SAME: "[[RESOURCE_DIR]]/lib/linux/clang_rt.crtbegin-ve.o"
// DEF-SAME: "[[RESOURCE_DIR]]/lib/linux/libclang_rt.builtins-ve.a" "-lc"
// DEF-SAME: "[[RESOURCE_DIR]]/lib/linux/libclang_rt.builtins-ve.a"
// DEF-SAME: "[[RESOURCE_DIR]]/lib/linux/clang_rt.crtend-ve.o"
// DEF-SAME: "[[SYSROOT]]/opt/nec/ve/lib/crtn.o"
