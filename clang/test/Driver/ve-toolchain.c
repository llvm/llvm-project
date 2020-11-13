/// Check the behavior of toolchain for NEC Aurora VE
/// REQUIRES: ve-registered-target

///-----------------------------------------------------------------------------
/// Checking dwarf-version

// RUN: %clang -### -g -target ve %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s
// DWARF_VER: "-dwarf-version=4"

///-----------------------------------------------------------------------------
/// Checking dynamic-linker

// RUN: %clang -### -target ve %s 2>&1 | FileCheck -check-prefix=DYNLINKER %s
// DYNLINKER: nld{{.*}} "-dynamic-linker" "/opt/nec/ve/lib/ld-linux-ve.so.1"

///-----------------------------------------------------------------------------
/// Checking VE specific option

// RUN: %clang -### -target ve %s 2>&1 | FileCheck -check-prefix=VENLDOPT %s
// VENLDOPT: nld{{.*}} "-z" "max-page-size=0x4000000"

///-----------------------------------------------------------------------------
/// Checking include-path

// RUN: %clang -### -target ve %s 2>&1 | FileCheck -check-prefix=DEFINC %s
// DEFINC: clang{{.*}} "-cc1"
// DEFINC: "-nostdsysteminc"
// DEFINC: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include"
// DEFINC: "-internal-isystem" "/opt/nec/ve/include"

// RUN: %clang -### -target ve %s -nostdlibinc 2>&1 | \
// RUN:    FileCheck -check-prefix=NOSTDLIBINC %s
// NOSTDLIBINC: clang{{.*}} "-cc1"
// NOSTDLIBINC: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include"
// NOSTDLIBINC-NOT: "-internal-isystem" "/opt/nec/ve/include"

// RUN: %clang -### -target ve %s -nobuiltininc 2>&1 | \
// RUN:    FileCheck -check-prefix=NOBUILTININC %s
// NOBUILTININC: clang{{.*}} "-cc1"
// NOBUILTININC: "-nobuiltininc"
// NOBUILTININC-NOT: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include"
// NOBUILTININC: "-internal-isystem" "/opt/nec/ve/include"

// RUN: %clang -### -target ve %s -nostdinc 2>&1 | \
// RUN:    FileCheck -check-prefix=NOSTDINC %s
// NOSTDINC: clang{{.*}} "-cc1"
// NOSTDINC: "-nobuiltininc"
// NOSTDINC-NOT: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include"
// NOSTDINC-NOT: "-internal-isystem" "/opt/nec/ve/include"

///-----------------------------------------------------------------------------
/// Checking -fuse-init-array

// RUN: %clang -### -target ve %s 2>&1 | FileCheck -check-prefix=DEFINITARRAY %s
// DEFINITARRAY: clang{{.*}} "-cc1"
// DEFINITARRAY-NOT: "-fuse-init-array"

// RUN: %clang -### -target ve %s -fno-use-init-array 2>&1 | \
// RUN:     FileCheck -check-prefix=NOTINITARRAY %s
// NOTINITARRAY: clang{{.*}} "-cc1"
// NOTINITARRAY: "-fno-use-init-array"

///-----------------------------------------------------------------------------
/// Checking -faddrsig

// RUN: %clang -### -target ve %s 2>&1 | FileCheck -check-prefix=DEFADDESIG %s
// DEFADDESIG: clang{{.*}} "-cc1"
// DEFADDESIG-NOT: "-faddrsig"

// RUN: %clang -### -target ve %s -faddrsig 2>&1 | \
// RUN:     FileCheck -check-prefix=ADDRSIG %s
// ADDRSIG: clang{{.*}} "-cc1"
// ADDRSIG: "-faddrsig"

// RUN: %clang -### -target ve %s -fno-addrsig 2>&1 | \
// RUN:     FileCheck -check-prefix=NOADDRSIG %s
// NOADDRSIG: clang{{.*}} "-cc1"
// NOADDRSIG-NOT: "-faddrsig"

///-----------------------------------------------------------------------------
/// Checking exceptions

// RUN: %clang -### -target ve %s 2>&1 | FileCheck -check-prefix=DEFEXCEPTION %s
// DEFEXCEPTION: clang{{.*}} "-cc1"
// DEFEXCEPTION: "-fsjlj-exceptions"

///-----------------------------------------------------------------------------
/// Passing -fintegrated-as

// RUN: %clang -### -target ve -x assembler %s 2>&1 | \
// RUN:    FileCheck -check-prefix=AS_LINK %s
// RUN: %clang -### -target ve -fno-integrated-as -x assembler %s 2>&1 | \
// RUN:    FileCheck -check-prefix=NAS_LINK %s

// AS_LINK: clang{{.*}} "-cc1as"
// AS_LINK: nld{{.*}}

// NAS_LINK: nas{{.*}}
// NAS_LINK: nld{{.*}}
