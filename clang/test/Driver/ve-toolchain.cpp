/// Check the behavior of toolchain for NEC Aurora VE
/// REQUIRES: ve-registered-target

///-----------------------------------------------------------------------------
/// Checking dwarf-version

// RUN: %clangxx -### -g -target ve %s 2>&1 | FileCheck -check-prefix=DWARF_VER %s
// DWARF_VER: "-dwarf-version=4"

///-----------------------------------------------------------------------------
/// Checking dynamic-linker

// RUN: %clangxx -### -target ve %s 2>&1 | FileCheck -check-prefix=DYNLINKER %s
// DYNLINKER: nld{{.*}} "-dynamic-linker" "/opt/nec/ve/lib/ld-linux-ve.so.1"

///-----------------------------------------------------------------------------
/// Checking VE specific option

// RUN: %clangxx -### -target ve %s 2>&1 | FileCheck -check-prefix=VENLDOPT %s
// VENLDOPT: nld{{.*}} "-z" "max-page-size=0x4000000"

///-----------------------------------------------------------------------------
/// Checking include-path

// RUN: %clangxx -### -target ve %s 2>&1 | FileCheck -check-prefix=DEFINC %s
// DEFINC: clang{{.*}} "-cc1"
// DEFINC: "-nostdsysteminc"
// DEFINC: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include/c++/v1"
// DEFINC: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include"
// DEFINC: "-internal-isystem" "/opt/nec/ve/include"

// RUN: %clangxx -### -target ve %s -nostdlibinc 2>&1 | \
// RUN:    FileCheck -check-prefix=NOSTDLIBINC %s
// NOSTDLIBINC: clang{{.*}} "-cc1"
// NOSTDLIBINC-NOT: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include/c++/v1"
// NOSTDLIBINC: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include"
// NOSTDLIBINC-NOT: "-internal-isystem" "/opt/nec/ve/include"

// RUN: %clangxx -### -target ve %s -nobuiltininc 2>&1 | \
// RUN:    FileCheck -check-prefix=NOBUILTININC %s
// NOBUILTININC: clang{{.*}} "-cc1"
// NOBUILTININC: "-nobuiltininc"
// NOBUILTININC: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include/c++/v1"
// NOBUILTININC-NOT: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include"
// NOBUILTININC: "-internal-isystem" "/opt/nec/ve/include"

// RUN: %clangxx -### -target ve %s -nostdinc 2>&1 | \
// RUN:    FileCheck -check-prefix=NOSTDINC %s
// NOSTDINC: clang{{.*}} "-cc1"
// NOSTDINC: "-nobuiltininc"
// NOSTDINC-NOT: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include/c++/v1"
// NOSTDINC-NOT: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include"
// NOSTDINC-NOT: "-internal-isystem" "/opt/nec/ve/include"

// RUN: %clangxx -### -target ve %s -nostdinc++ 2>&1 | \
// RUN:    FileCheck -check-prefix=NOSTDINCXX %s
// NOSTDINCXX: clang{{.*}} "-cc1"
// NOSTDINCXX: "-nostdinc++"
// NOSTDINCXX-NOT: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include/c++/v1"
// NOSTDINCXX: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include"
// NOSTDINCXX: "-internal-isystem" "/opt/nec/ve/include"

///-----------------------------------------------------------------------------
/// Checking environment variable NCC_CPLUS_INCLUDE_PATH

// RUN: env NCC_CPLUS_INCLUDE_PATH=/test/test %clangxx -### -target ve %s \
// RUN:    2>&1 | FileCheck -check-prefix=DEFINCENV %s

// DEFINCENV: clang{{.*}} "-cc1"
// DEFINCENV: "-nostdsysteminc"
// DEFINCENV: "-internal-isystem" "/test/test"
// DEFINCENV: "-internal-isystem" "{{.*}}/lib/clang/{{[0-9.]*}}/include"
// DEFINCENV: "-internal-isystem" "/opt/nec/ve/include"

///-----------------------------------------------------------------------------
/// Checking -fuse-init-array

// RUN: %clangxx -### -target ve %s 2>&1 | FileCheck -check-prefix=DEFINITARRAY %s
// DEFINITARRAY: clang{{.*}} "-cc1"
// DEFINITARRAY-NOT: "-fuse-init-array"

// RUN: %clangxx -### -target ve %s -fno-use-init-array 2>&1 | \
// RUN:     FileCheck -check-prefix=NOTINITARRAY %s
// NOTINITARRAY: clang{{.*}} "-cc1"
// NOTINITARRAY: "-fno-use-init-array"

///-----------------------------------------------------------------------------
/// Checking -faddrsig

// RUN: %clangxx -### -target ve %s 2>&1 | FileCheck -check-prefix=DEFADDESIG %s
// DEFADDESIG: clang{{.*}} "-cc1"
// DEFADDESIG-NOT: "-faddrsig"

// RUN: %clangxx -### -target ve %s -faddrsig 2>&1 | \
// RUN:     FileCheck -check-prefix=ADDRSIG %s
// ADDRSIG: clang{{.*}} "-cc1"
// ADDRSIG: "-faddrsig"

// RUN: %clangxx -### -target ve %s -fno-addrsig 2>&1 | \
// RUN:     FileCheck -check-prefix=NOADDRSIG %s
// NOADDRSIG: clang{{.*}} "-cc1"
// NOADDRSIG-NOT: "-faddrsig"

///-----------------------------------------------------------------------------
/// Checking exceptions

// RUN: %clangxx -### -target ve %s 2>&1 | FileCheck -check-prefix=DEFEXCEPTION %s
// DEFEXCEPTION: clang{{.*}} "-cc1"
// DEFEXCEPTION: "-fsjlj-exceptions"

///-----------------------------------------------------------------------------
/// Passing -fintegrated-as

// RUN: %clangxx -### -target ve -x assembler %s 2>&1 | \
// RUN:    FileCheck -check-prefix=AS_LINK %s
// RUN: %clangxx -### -target ve -fno-integrated-as -x assembler %s 2>&1 | \
// RUN:    FileCheck -check-prefix=NAS_LINK %s

// AS_LINK: clang{{.*}} "-cc1as"
// AS_LINK: nld{{.*}}

// NAS_LINK: nas{{.*}}
// NAS_LINK: nld{{.*}}

///-----------------------------------------------------------------------------
/// Checking default libraries

// RUN: %clangxx -### -target ve --stdlib=c++ %s 2>&1 | \
// RUN:    FileCheck -check-prefix=LINK %s

// LINK: clang{{.*}} "-cc1"
// LINK: nld{{.*}} "{{.*}}/crt1.o" "{{.*}}/crti.o"{{.*}}"crtbegin.o"{{.*}}"-lc++" "-lc++abi" "-lunwind" "-lpthread" "-ldl"
