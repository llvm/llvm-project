// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.


// Alias options:

// RUN: %clang_cl /c -### -- %s 2>&1 | FileCheck -check-prefix=c %s
// c: -c

// RUN: not %clang_cl /C -### -- %s 2>&1 | FileCheck -check-prefix=C %s
// C: error: invalid argument '/C' only allowed with '/E, /P or /EP'

// RUN: %clang_cl /C /P -### -- %s 2>&1 | FileCheck -check-prefix=C_P %s
// C_P: "-E"
// C_P: "-C"

// RUN: %clang_cl /d1reportAllClassLayout -### /c /WX -- %s 2>&1 | \
// RUN:     FileCheck -check-prefix=d1reportAllClassLayout %s
// d1reportAllClassLayout-NOT: warning:
// d1reportAllClassLayout-NOT: error:
// d1reportAllClassLayout: -fdump-record-layouts

// RUN: %clang_cl /Dfoo=bar /D bar=baz /DMYDEF#value /DMYDEF2=foo#bar /DMYDEF3#a=b /DMYDEF4# \
// RUN:    -### -- %s 2>&1 | FileCheck -check-prefix=D %s
// D: "-D" "foo=bar"
// D: "-D" "bar=baz"
// D: "-D" "MYDEF=value"
// D: "-D" "MYDEF2=foo#bar"
// D: "-D" "MYDEF3=a=b"
// D: "-D" "MYDEF4="

// RUN: %clang_cl /E -### -- %s 2>&1 | FileCheck -check-prefix=E %s
// E: "-E"
// E: "-o" "-"

// RUN: %clang_cl /EP -### -- %s 2>&1 | FileCheck -check-prefix=EP %s
// EP: "-E"
// EP: "-P"
// EP: "-o" "-"

// RUN: %clang_cl /external:Ipath  -### -- %s 2>&1 | FileCheck -check-prefix=EXTERNAL_I %s
// RUN: %clang_cl /external:I path -### -- %s 2>&1 | FileCheck -check-prefix=EXTERNAL_I %s
// EXTERNAL_I: "-isystem" "path"

// RUN: %clang_cl /fp:fast /fp:except -### -- %s 2>&1 | FileCheck -check-prefix=fpexcept %s
// fpexcept-NOT: -funsafe-math-optimizations
// fpexcept: -ffp-exception-behavior=strict

// RUN: %clang_cl /fp:fast /fp:except /fp:except- -### -- %s 2>&1 | FileCheck -check-prefix=fpexcept_ %s
// fpexcept_: -funsafe-math-optimizations
// fpexcept_: -ffp-exception-behavior=ignore

// RUN: %clang_cl /fp:precise /fp:fast -### -- %s 2>&1 | FileCheck -check-prefix=fpfast %s
// fpfast: -funsafe-math-optimizations
// fpfast: -ffast-math

// RUN: %clang_cl /fp:fast /fp:precise -Wno-overriding-complex-range -### -- %s 2>&1 | \
// RUN:   FileCheck -check-prefix=fpprecise %s
// fpprecise-NOT: -funsafe-math-optimizations
// fpprecise-NOT: -ffast-math

// RUN: %clang_cl /fp:fast /fp:strict -Wno-overriding-complex-range -### -- %s 2>&1 | \
// RUN:   FileCheck -check-prefix=fpstrict %s
// fpstrict-NOT: -funsafe-math-optimizations
// fpstrict-NOT: -ffast-math
// fpstrict: -ffp-contract=off

// RUN: %clang_cl /fp:strict /fp:contract -### -- %s 2>&1 | FileCheck -check-prefix=fpcontract %s
// fpcontract: -ffp-contract=on

// RUN: %clang_cl /fsanitize=address -### -- %s 2>&1 | FileCheck -check-prefix=fsanitize_address %s
// fsanitize_address: -fsanitize=address

// RUN: %clang_cl -### /FA -fprofile-instr-generate -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-INSTR-GENERATE %s
// RUN: %clang_cl -### /FA -fprofile-instr-generate -fno-rtlib-defaultlib -frtlib-defaultlib -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-INSTR-GENERATE %s
// RUN: %clang_cl -### /FA -fprofile-instr-generate=/tmp/somefile.profraw -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-INSTR-GENERATE-FILE %s
// RUN: %clang_cl -### /FAcsu -fprofile-instr-generate -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-INSTR-GENERATE %s
// RUN: %clang_cl -### /FAcsu -fprofile-instr-generate=/tmp/somefile.profraw -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-INSTR-GENERATE-FILE %s
// CHECK-PROFILE-INSTR-GENERATE: "-fprofile-instrument=clang" "--dependent-lib=clang_rt.profile{{[^"]*}}.lib"
// CHECK-PROFILE-INSTR-GENERATE-FILE: "-fprofile-instrument-path=/tmp/somefile.profraw"

// RUN: %clang_cl -### /FA -fprofile-instr-generate -fno-rtlib-defaultlib -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-INSTR-GENERATE-NODEF %s
// CHECK-PROFILE-INSTR-GENERATE-NODEF-NOT: "--dependent-lib=clang_rt.profile{{[^"]*}}.lib"

// RUN: %clang_cl -### /FA -fprofile-generate -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-GENERATE %s
// RUN: %clang_cl -### /FAcsu -fprofile-generate -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-GENERATE %s
// CHECK-PROFILE-GENERATE: "-fprofile-instrument=llvm" "--dependent-lib=clang_rt.profile{{[^"]*}}.lib"

// RUN: not %clang_cl -### /FA -fprofile-instr-generate -fprofile-instr-use -- %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: not %clang_cl -### /FA -fprofile-instr-generate -fprofile-instr-use=file -- %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: not %clang_cl -### /FAcsu -fprofile-instr-generate -fprofile-instr-use -- %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: not %clang_cl -### /FAcsu -fprofile-instr-generate -fprofile-instr-use=file -- %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// CHECK-NO-MIX-GEN-USE: '{{[a-z=-]*}}' not allowed with '{{[a-z=-]*}}'

// RUN: rm -rf %t && mkdir %t
// RUN: llvm-profdata merge -o %t/somefile.prof %S/Inputs/a.proftext
// RUN: llvm-profdata merge -o %t/default.profdata %S/Inputs/a.proftext
// RUN: cd %t

// RUN: %clang_cl -### /FA -fprofile-instr-use -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE %s
// RUN: %clang_cl -### /FA -fprofile-use -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE %s
// RUN: %clang_cl -### /FA -fprofile-instr-use=%t/somefile.prof -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE-FILE %s
// RUN: %clang_cl -### /FA -fprofile-use=%t/somefile.prof -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE-FILE %s
// RUN: %clang_cl -### /FAcsu -fprofile-instr-use -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE %s
// RUN: %clang_cl -### /FAcsu -fprofile-use -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE %s
// RUN: %clang_cl -### /FAcsu -fprofile-instr-use=%t/somefile.prof -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE-FILE %s
// RUN: %clang_cl -### /FAcsu -fprofile-use=%t/somefile.prof -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE-FILE %s
// CHECK-PROFILE-USE: "-fprofile-instrument-use-path={{.*}}default.profdata"
// CHECK-PROFILE-USE-FILE: "-fprofile-instrument-use-path={{.*}}somefile.prof"

// RUN: %clang_cl /GA -### -- %s 2>&1 | FileCheck -check-prefix=GA %s
// GA: -ftls-model=local-exec

// RTTI is on by default; just check that we don't error.
// RUN: %clang_cl /Zs /GR -- %s 2>&1

// RUN: %clang_cl /GR- -### -- %s 2>&1 | FileCheck -check-prefix=GR_ %s
// GR_: -fno-rtti

// Security Buffer Check is on by default.
// RUN: %clang_cl -### -- %s 2>&1 | FileCheck -check-prefix=GS-default %s
// GS-default: "-stack-protector" "2"

// RUN: %clang_cl /GS -### -- %s 2>&1 | FileCheck -check-prefix=GS %s
// GS: "-stack-protector" "2"

// RUN: %clang_cl /GS- -### -- %s 2>&1 | FileCheck -check-prefix=GS_ %s
// GS_-NOT: -stack-protector

// RUN: %clang_cl /Gy -### -- %s 2>&1 | FileCheck -check-prefix=Gy %s
// Gy: -ffunction-sections

// RUN: %clang_cl /Gy /Gy- -### -- %s 2>&1 | FileCheck -check-prefix=Gy_ %s
// Gy_-NOT: -ffunction-sections

// RUN: %clang_cl /Gs -### -- %s 2>&1 | FileCheck -check-prefix=Gs %s
// Gs: "-mstack-probe-size=4096"
// RUN: %clang_cl /Gs0 -### -- %s 2>&1 | FileCheck -check-prefix=Gs0 %s
// Gs0: "-mstack-probe-size=0"
// RUN: %clang_cl /Gs4096 -### -- %s 2>&1 | FileCheck -check-prefix=Gs4096 %s
// Gs4096: "-mstack-probe-size=4096"

// RUN: %clang_cl /Gw -### -- %s 2>&1 | FileCheck -check-prefix=Gw %s
// Gw: -fdata-sections

// RUN: %clang_cl /Gw /Gw- -### -- %s 2>&1 | FileCheck -check-prefix=Gw_ %s
// Gw_-NOT: -fdata-sections

// RUN: %clang_cl /hotpatch -### -- %s 2>&1 | FileCheck -check-prefix=hotpatch %s
// hotpatch: -fms-hotpatch

// RUN: %clang_cl /Imyincludedir -### -- %s 2>&1 | FileCheck -check-prefix=SLASH_I %s
// RUN: %clang_cl /I myincludedir -### -- %s 2>&1 | FileCheck -check-prefix=SLASH_I %s
// SLASH_I: "-I" "myincludedir"

// RUN: %clang_cl /imsvcmyincludedir -### -- %s 2>&1 | FileCheck -check-prefix=SLASH_imsvc %s
// RUN: %clang_cl /imsvc myincludedir -### -- %s 2>&1 | FileCheck -check-prefix=SLASH_imsvc %s
// Clang's resource header directory should be first:
// SLASH_imsvc: "-resource-dir" "[[RESOURCE_DIR:[^"]+]]"
// SLASH_imsvc: "-internal-isystem" "[[RESOURCE_DIR]]{{[/\\]+}}include"
// SLASH_imsvc: "-internal-isystem" "myincludedir"

// RUN: %clang_cl /J -### -- %s 2>&1 | FileCheck -check-prefix=J %s
// J: -fno-signed-char

// RUN: %clang_cl /Ofoo -### -- %s 2>&1 | FileCheck -check-prefix=O %s
// O: /Ofoo

// RUN: %clang_cl /Ob0 -### -- %s 2>&1 | FileCheck -check-prefix=Ob0 %s
// Ob0: -fno-inline

// RUN: %clang_cl /Ob2 -### -- %s 2>&1 | FileCheck -check-prefix=Ob2 %s
// RUN: %clang_cl /Ob3 -### -- %s 2>&1 | FileCheck -check-prefix=Ob2 %s
// RUN: %clang_cl /Odb2 -### -- %s 2>&1 | FileCheck -check-prefix=Ob2 %s
// RUN: %clang_cl /Odb3 -### -- %s 2>&1 | FileCheck -check-prefix=Ob2 %s
// RUN: %clang_cl /O2 /Ob2 -### -- %s 2>&1 | FileCheck -check-prefix=Ob2 %s
// Ob2-NOT: warning: argument unused during compilation: '/O2'
// Ob2: -finline-functions

// RUN: %clang_cl /Ob1 -### -- %s 2>&1 | FileCheck -check-prefix=Ob1 %s
// RUN: %clang_cl /Odb1 -### -- %s 2>&1 | FileCheck -check-prefix=Ob1 %s
// Ob1: -finline-hint-functions

// RUN: %clang_cl /Od -### -- %s 2>&1 | FileCheck -check-prefix=Od %s
// Od: -O0

// RUN: %clang_cl /Oi- /Oi -### -- %s 2>&1 | FileCheck -check-prefix=Oi %s
// Oi-NOT: -fno-builtin

// RUN: %clang_cl /Oi- -### -- %s 2>&1 | FileCheck -check-prefix=Oi_ %s
// Oi_: -fno-builtin

// RUN: %clang_cl /Os --target=i686-pc-windows-msvc -### -- %s 2>&1 | FileCheck -check-prefix=Os %s
// RUN: %clang_cl /Os --target=x86_64-pc-windows-msvc -### -- %s 2>&1 | FileCheck -check-prefix=Os %s
// Os: -Os
// Os: -mframe-pointer=none

// RUN: %clang_cl /Ot --target=i686-pc-windows-msvc -### -- %s 2>&1 | FileCheck -check-prefix=Ot %s
// RUN: %clang_cl /Ot --target=x86_64-pc-windows-msvc -### -- %s 2>&1 | FileCheck -check-prefix=Ot %s
// Ot: -O3
// Ot: -mframe-pointer=none

// RUN: %clang_cl /Ox --target=i686-pc-windows-msvc -### -- %s 2>&1 | FileCheck -check-prefix=Ox %s
// RUN: %clang_cl /Ox --target=x86_64-pc-windows-msvc -### -- %s 2>&1 | FileCheck -check-prefix=Ox %s
// Ox: -O3
// Ox: -mframe-pointer=none

// RUN: %clang_cl --target=i686-pc-win32 /O2sy- -### -- %s 2>&1 | FileCheck -check-prefix=PR24003 %s
// PR24003: -Os
// PR24003: -mframe-pointer=all

// RUN: %clang_cl --target=i686-pc-win32 -Werror -Wno-msvc-not-found /Oy- /O2 -### -- %s 2>&1 | FileCheck -check-prefix=Oy_2 %s
// Oy_2: -O3
// Oy_2: -mframe-pointer=all

// RUN: %clang_cl --target=aarch64-pc-windows-msvc -Werror -Wno-msvc-not-found /Oy- /O2 -### -- %s 2>&1 | FileCheck -check-prefix=Oy_aarch64 %s
// Oy_aarch64: -O3
// Oy_aarch64: -mframe-pointer=non-leaf

// RUN: %clang_cl --target=i686-pc-win32 -Werror -Wno-msvc-not-found /O2 /O2 -### -- %s 2>&1 | FileCheck -check-prefix=O2O2 %s
// O2O2: "-O3"

// RUN: %clang_cl /Zs -Werror /Oy -- %s 2>&1

// RUN: %clang_cl --target=i686-pc-win32 -Werror -Wno-msvc-not-found /Oy- -### -- %s 2>&1 | FileCheck -check-prefix=Oy_ %s
// Oy_: -mframe-pointer=all

// RUN: %clang_cl /Qvec -### -- %s 2>&1 | FileCheck -check-prefix=Qvec %s
// Qvec: -vectorize-loops

// RUN: %clang_cl /Qvec /Qvec- -### -- %s 2>&1 | FileCheck -check-prefix=Qvec_ %s
// Qvec_-NOT: -vectorize-loops

// RUN: %clang_cl /showIncludes -### -- %s 2>&1 | FileCheck -check-prefix=showIncludes_ %s
// showIncludes_: --show-includes
// showIncludes_: -sys-header-deps

// RUN: %clang_cl /showIncludes:user -### -- %s 2>&1 | FileCheck -check-prefix=showIncludesUser %s
// showIncludesUser: --show-includes
// showIncludesUser-NOT: -sys-header-deps

// RUN: %clang_cl /E /showIncludes -### -- %s 2>&1 | FileCheck -check-prefix=showIncludes_E %s
// RUN: %clang_cl /E /showIncludes:user -### -- %s 2>&1 | FileCheck -check-prefix=showIncludes_E %s
// RUN: %clang_cl /EP /showIncludes -### -- %s 2>&1 | FileCheck -check-prefix=showIncludes_E %s
// RUN: %clang_cl /E /EP /showIncludes -### -- %s 2>&1 | FileCheck -check-prefix=showIncludes_E %s
// RUN: %clang_cl /EP /P /showIncludes -### -- %s 2>&1 | FileCheck -check-prefix=showIncludes_E %s
// showIncludes_E-NOT: warning: argument unused during compilation: '--show-includes'

// /source-charset: should warn on everything except UTF-8.
// RUN: not %clang_cl /source-charset:utf-16 -### -- %s 2>&1 | FileCheck -check-prefix=source-charset-utf-16 %s
// source-charset-utf-16: invalid value 'utf-16' in '/source-charset:utf-16'

// /execution-charset: should warn on everything except UTF-8.
// RUN: not %clang_cl /execution-charset:utf-16 -### -- %s 2>&1 | FileCheck -check-prefix=execution-charset-utf-16 %s
// execution-charset-utf-16: invalid value 'utf-16' in '/execution-charset:utf-16'
//
// RUN: %clang_cl /Umymacro -### -- %s 2>&1 | FileCheck -check-prefix=U %s
// RUN: %clang_cl /U mymacro -### -- %s 2>&1 | FileCheck -check-prefix=U %s
// U: "-U" "mymacro"

// RUN: %clang_cl /validate-charset -### -- %s 2>&1 | FileCheck -check-prefix=validate-charset %s
// validate-charset: -Winvalid-source-encoding

// RUN: %clang_cl /validate-charset- -### -- %s 2>&1 | FileCheck -check-prefix=validate-charset_ %s
// validate-charset_: -Wno-invalid-source-encoding

// RUN: %clang_cl /vd2 -### -- %s 2>&1 | FileCheck -check-prefix=VD2 %s
// VD2: -vtordisp-mode=2

// RUN: %clang_cl /vmg -### -- %s 2>&1 | FileCheck -check-prefix=VMG %s
// VMG: "-fms-memptr-rep=virtual"

// RUN: %clang_cl /vmg /vms -### -- %s 2>&1 | FileCheck -check-prefix=VMS %s
// VMS: "-fms-memptr-rep=single"

// RUN: %clang_cl /vmg /vmm -### -- %s 2>&1 | FileCheck -check-prefix=VMM %s
// VMM: "-fms-memptr-rep=multiple"

// RUN: %clang_cl /vmg /vmv -### -- %s 2>&1 | FileCheck -check-prefix=VMV %s
// VMV: "-fms-memptr-rep=virtual"

// RUN: not %clang_cl /vmg /vmb -### -- %s 2>&1 | FileCheck -check-prefix=VMB %s
// VMB: '/vmg' not allowed with '/vmb'

// RUN: not %clang_cl /vmg /vmm /vms -### -- %s 2>&1 | FileCheck -check-prefix=VMX %s
// VMX: '/vms' not allowed with '/vmm'

// RUN: %clang_cl --target=i686-pc-win32 /volatile:iso -### -- %s 2>&1 | FileCheck -check-prefix=VOLATILE-ISO %s
// RUN: %clang_cl --target=aarch64-pc-win32 -### -- %s 2>&1 | FileCheck -check-prefix=VOLATILE-ISO %s
// VOLATILE-ISO-NOT: "-fms-volatile"

// RUN: %clang_cl --target=aarch64-pc-win32 /volatile:ms -### -- %s 2>&1 | FileCheck -check-prefix=VOLATILE-MS %s
// RUN: %clang_cl --target=i686-pc-win32 -### -- %s 2>&1 | FileCheck -check-prefix=VOLATILE-MS %s
// VOLATILE-MS: "-fms-volatile"

// RUN: %clang_cl /W0 -### -- %s 2>&1 | FileCheck -check-prefix=W0 %s
// W0: -w

// RUN: %clang_cl /W1 -### -- %s 2>&1 | FileCheck -check-prefix=W1 %s
// RUN: %clang_cl /W2 -### -- %s 2>&1 | FileCheck -check-prefix=W1 %s
// RUN: %clang_cl /W3 -### -- %s 2>&1 | FileCheck -check-prefix=W1 %s
// RUN: %clang_cl /W4 -### -- %s 2>&1 | FileCheck -check-prefix=W4 %s
// RUN: %clang_cl /Wall -### -- %s 2>&1 | FileCheck -check-prefix=Weverything %s
// W1: -Wall
// W4: -WCL4
// Weverything: -Weverything

// RUN: %clang_cl /WX -Wno-msvc-not-found -### -- %s 2>&1 | FileCheck -check-prefix=WX %s
// WX: -Werror

// RUN: %clang_cl /WX- -### -- %s 2>&1 | FileCheck -check-prefix=WX_ %s
// WX_: -Wno-error

// RUN: %clang_cl /w -### -- %s 2>&1 | FileCheck -check-prefix=w %s
// w: -w

// RUN: %clang_cl /Zp -### -- %s 2>&1 | FileCheck -check-prefix=ZP %s
// ZP: -fpack-struct=1

// RUN: %clang_cl /Zp2 -### -- %s 2>&1 | FileCheck -check-prefix=ZP2 %s
// ZP2: -fpack-struct=2

// RUN: %clang_cl /Zs -### -- %s 2>&1 | FileCheck -check-prefix=Zs %s
// Zs: -fsyntax-only

// RUN: %clang_cl /FIasdf.h -### -- %s 2>&1 | FileCheck -check-prefix=FI %s
// FI: "-include" "asdf.h"

// RUN: %clang_cl /FI asdf.h -### -- %s 2>&1 | FileCheck -check-prefix=FI_ %s
// FI_: "-include" "asdf.h"

// RUN: %clang_cl /TP /c -### -- %s 2>&1 | FileCheck -check-prefix=NO-GX %s
// NO-GX-NOT: "-fcxx-exceptions" "-fexceptions"

// RUN: %clang_cl /TP /c /GX -### -- %s 2>&1 | FileCheck -check-prefix=GX %s
// GX: "-fcxx-exceptions" "-fexceptions"

// RUN: %clang_cl /TP /c /GX /GX- -### -- %s 2>&1 | FileCheck -check-prefix=GX_ %s
// GX_-NOT: "-fcxx-exceptions" "-fexceptions"

// RUN: %clang_cl /d1PP -### -- %s 2>&1 | FileCheck -check-prefix=d1PP %s
// d1PP: -dD

// RUN: %clang_cl --target=i686-pc-windows-msvc /c /QIntel-jcc-erratum -### -- %s 2>&1 | FileCheck -check-prefix=jcceratum %s
// jcceratum: "-mllvm" "-x86-branches-within-32B-boundaries"


// We forward any unrecognized -W diagnostic options to cc1.
// RUN: %clang_cl -Wunused-pragmas -### -- %s 2>&1 | FileCheck -check-prefix=WJoined %s
// WJoined: "-cc1"
// WJoined: "-Wunused-pragmas"

// We recognize -f[no-]strict-aliasing.
// RUN: %clang_cl -c -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULTSTRICT %s
// DEFAULTSTRICT: "-relaxed-aliasing"
// RUN: %clang_cl -c -fstrict-aliasing -### -- %s 2>&1 | FileCheck -check-prefix=STRICT %s
// STRICT-NOT: "-relaxed-aliasing"
// RUN: %clang_cl -c -fno-strict-aliasing -### -- %s 2>&1 | FileCheck -check-prefix=NOSTRICT %s
// NOSTRICT: "-relaxed-aliasing"

// We recognize -f[no-]delete-null-pointer-checks.
// RUN: %clang_cl -c -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULTNULL %s
// DEFAULTNULL-NOT: "-fno-delete-null-pointer-checks"
// RUN: %clang_cl -c -fdelete-null-pointer-checks -### -- %s 2>&1 | FileCheck -check-prefix=NULL %s
// NULL-NOT: "-fno-delete-null-pointer-checks"
// RUN: %clang_cl -c -fno-delete-null-pointer-checks -### -- %s 2>&1 | FileCheck -check-prefix=NONULL %s
// NONULL: "-fno-delete-null-pointer-checks"

// RUN: %clang_cl -c -### /std:c11 -- %s 2>&1 | FileCheck -check-prefix CHECK-C11 %s
// CHECK-C11: -std=c11

// RUN: %clang_cl -c -### /std:c17 -- %s 2>&1 | FileCheck -check-prefix CHECK-C17 %s
// CHECK-C17: -std=c17

// RUN: %clang_cl -c -### /std:clatest -- %s 2>&1 | FileCheck -check-prefix CHECK-CLATEST %s
// CHECK-CLATEST: -std=c23

// For some warning ids, we can map from MSVC warning to Clang warning.
// RUN: %clang_cl -wd4005 -wd4100 -wd4910 -wd4996 -wd12345678 -### -- %s 2>&1 | FileCheck -check-prefix=Wno %s
// Wno: "-cc1"
// Wno: "-Wno-macro-redefined"
// Wno: "-Wno-unused-parameter"
// Wno: "-Wno-dllexport-explicit-instantiation-decl"
// Wno: "-Wno-deprecated-declarations"
// Wno-NOT: "-wd

// Ignored options. Check that we don't get "unused during compilation" errors.
// RUN: %clang_cl /c \
// RUN:    /analyze- \
// RUN:    /bigobj \
// RUN:    /cgthreads4 \
// RUN:    /cgthreads8 \
// RUN:    /d2FastFail \
// RUN:    /d2Zi+ \
// RUN:    /errorReport:foo \
// RUN:    /execution-charset:utf-8 \
// RUN:    /FC \
// RUN:    /Fdfoo \
// RUN:    /FS \
// RUN:    /Gd \
// RUN:    /GF \
// RUN:    /GS- \
// RUN:    /kernel- \
// RUN:    /nologo \
// RUN:    /Og \
// RUN:    /openmp- \
// RUN:    /permissive- \
// RUN:    /RTC1 \
// RUN:    /sdl \
// RUN:    /sdl- \
// RUN:    /source-charset:utf-8 \
// RUN:    /utf-8 \
// RUN:    /vmg \
// RUN:    /volatile:iso \
// RUN:    /w12345 \
// RUN:    /wd1234 \
// RUN:    /Wv \
// RUN:    /Wv:17 \
// RUN:    /Zm \
// RUN:    /Zo \
// RUN:    /Zo- \
// RUN:    -### -- %s 2>&1 | FileCheck -DMSG=%errc_ENOENT -check-prefix=IGNORED %s
// IGNORED-NOT: argument unused during compilation
// IGNORED-NOT: [[MSG]]
// Don't confuse /openmp- with the /o flag:
// IGNORED-NOT: "-o" "penmp-.obj"

// Ignored options and compile-only options are ignored for link jobs.
// RUN: touch %t.obj
// RUN: %clang_cl /nologo -### -- %t.obj 2>&1 | FileCheck -check-prefix=LINKUNUSED %s
// RUN: %clang_cl /Dfoo -### -- %t.obj 2>&1 | FileCheck -check-prefix=LINKUNUSED %s
// RUN: %clang_cl /MD -### -- %t.obj 2>&1 | FileCheck -check-prefix=LINKUNUSED %s
// LINKUNUSED-NOT: argument unused during compilation

// Support ignoring warnings about unused arguments.
// RUN: not %clang_cl /Abracadabra -Qunused-arguments -### -- %s 2>&1 | FileCheck -check-prefix=UNUSED %s
// UNUSED-NOT: argument unused during compilation

// Unsupported but parsed options. Check that we don't error on them.
// (/Zs is for syntax-only)
// RUN: %clang_cl /Zs \
// RUN:     /await \
// RUN:     /await:strict \
// RUN:     /constexpr:depth1000 /constexpr:backtrace1000 /constexpr:steps1000 \
// RUN:     /AIfoo \
// RUN:     /AI foo_does_not_exist \
// RUN:     /Bt \
// RUN:     /Bt+ \
// RUN:     /clr:pure \
// RUN:     /d1import_no_registry \
// RUN:     /d1nodatetime \
// RUN:     /d2FH4 \
// RUN:     /d2TrimInlines \
// RUN:     /docname \
// RUN:     /dynamicdeopt \
// RUN:     /experimental:external \
// RUN:     /experimental:module \
// RUN:     /experimental:preprocessor \
// RUN:     /exportHeader /headerName:foo \
// RUN:     /external:anglebrackets \
// RUN:     /external:env:var \
// RUN:     /external:W0 \
// RUN:     /external:W1 \
// RUN:     /external:W2 \
// RUN:     /external:W3 \
// RUN:     /external:W4 \
// RUN:     /external:templates- \
// RUN:     /headerUnit foo.h=foo.ifc /headerUnit:quote foo.h=foo.ifc /headerUnit:angle foo.h=foo.ifc \
// RUN:     /EHsc \
// RUN:     /F 42 \
// RUN:     /FA \
// RUN:     /FAc \
// RUN:     /Fafilename \
// RUN:     /FAs \
// RUN:     /FAu \
// RUN:     /favor:blend \
// RUN:     /fno-sanitize-address-vcasan-lib \
// RUN:     /Fifoo \
// RUN:     /Fmfoo \
// RUN:     /FpDebug\main.pch \
// RUN:     /Frfoo \
// RUN:     /FRfoo \
// RUN:     /FU foo \
// RUN:     /Fx \
// RUN:     /G1 \
// RUN:     /G2 \
// RUN:     /GA \
// RUN:     /Gd \
// RUN:     /Ge \
// RUN:     /Gh \
// RUN:     /GH \
// RUN:     /GL \
// RUN:     /GL- \
// RUN:     /Gm \
// RUN:     /Gm- \
// RUN:     /Gr \
// RUN:     /GS \
// RUN:     /GT \
// RUN:     /GX \
// RUN:     /Gv \
// RUN:     /Gz \
// RUN:     /GZ \
// RUN:     /H \
// RUN:     /homeparams \
// RUN:     /kernel \
// RUN:     /LN \
// RUN:     /MP \
// RUN:     /o foo.obj \
// RUN:     /ofoo.obj \
// RUN:     /openmp \
// RUN:     /openmp:experimental \
// RUN:     /Qfast_transcendentals \
// RUN:     /QIfist \
// RUN:     /Qimprecise_fwaits \
// RUN:     /Qpar \
// RUN:     /Qpar-report:1 \
// RUN:     /Qsafe_fp_loads \
// RUN:     /Qspectre \
// RUN:     /Qspectre-load \
// RUN:     /Qspectre-load-cf \
// RUN:     /Qvec-report:2 \
// RUN:     /reference foo=foo.ifc /reference foo.ifc \
// RUN:     /sourceDependencies foo.json \
// RUN:     /sourceDependencies:directives foo.json \
// RUN:     /translateInclude \
// RUN:     /u \
// RUN:     /V \
// RUN:     /volatile:ms \
// RUN:     /wfoo \
// RUN:     /WL \
// RUN:     /Wp64 \
// RUN:     /X \
// RUN:     /Y- \
// RUN:     /Yc \
// RUN:     /Ycstdafx.h \
// RUN:     /Yd \
// RUN:     /Yl- \
// RUN:     /Ylfoo \
// RUN:     /Yustdafx.h \
// RUN:     /Z7 \
// RUN:     /Za \
// RUN:     /Ze \
// RUN:     /Zg \
// RUN:     /Zi \
// RUN:     /ZI \
// RUN:     /Zl \
// RUN:     /ZW:nostdlib \
// RUN:     -- %s 2>&1

// We support -Xclang for forwarding options to cc1.
// RUN: %clang_cl -Xclang hellocc1 -### -- %s 2>&1 | FileCheck -check-prefix=Xclang %s
// Xclang: "-cc1"
// Xclang: "hellocc1"

// Files under /Users are often confused with the /U flag. (This could happen
// for other flags too, but this is the one people run into.)
// RUN: %clang_cl /c /Users/me/myfile.c -### 2>&1 | FileCheck -check-prefix=SlashU %s
// SlashU: warning: '/Users/me/myfile.c' treated as the '/U' option
// SlashU: note: use '--' to treat subsequent arguments as filenames

// RTTI is on by default. /GR- controls -fno-rtti-data.
// RUN: %clang_cl /c /GR- -### -- %s 2>&1 | FileCheck -check-prefix=NoRTTI %s
// NoRTTI: "-fno-rtti-data"
// NoRTTI-NOT: "-fno-rtti"
// RUN: %clang_cl /c /GR -### -- %s 2>&1 | FileCheck -check-prefix=RTTI %s
// RTTI-NOT: "-fno-rtti-data"
// RTTI-NOT: "-fno-rtti"

// RUN: %clang_cl -target x86_64-windows /Zi /c -### -- %s 2>&1 | FileCheck -check-prefix=Zi %s
// Zi: "-gcodeview"
// Zi: "-debug-info-kind=constructor"

// RUN: %clang_cl -target x86_64-windows /Z7 /c -### -- %s 2>&1 | FileCheck -check-prefix=Z7 %s
// Z7: "-gcodeview"
// Z7: "-debug-info-kind=constructor"

// RUN: %clang_cl -target x86_64-windows -gline-tables-only /c -### -- %s 2>&1 | FileCheck -check-prefix=ZGMLT %s
// ZGMLT: "-gcodeview"
// ZGMLT: "-debug-info-kind=line-tables-only"

// RUN: %clang_cl /c -### -- %s 2>&1 | FileCheck -check-prefix=BreproDefault %s
// BreproDefault: "-mincremental-linker-compatible"

// RUN: %clang_cl /Brepro- /Brepro /c '-###' -- %s 2>&1 | FileCheck -check-prefix=Brepro %s
// Brepro-NOT: "-mincremental-linker-compatible"

// RUN: %clang_cl /Brepro /Brepro- /c '-###' -- %s 2>&1 | FileCheck -check-prefix=Brepro_ %s
// Brepro_: "-mincremental-linker-compatible"

// If We specify both /Z7 and -gdwarf we should get dwarf, not codeview.
// RUN: %clang_cl /Z7 -gdwarf /c -### -- %s 2>&1 | FileCheck -check-prefix=Z7_gdwarf %s
// RUN: %clang_cl -gdwarf /Z7 /c -### -- %s 2>&1 | FileCheck -check-prefix=Z7_gdwarf %s
// Z7_gdwarf-NOT: "-gcodeview"
// Z7_gdwarf: "-debug-info-kind=constructor"
// Z7_gdwarf: "-dwarf-version=

// RUN: %clang_cl /ZH:MD5 /c -### -- %s 2>&1 | FileCheck -check-prefix=ZH_MD5 %s
// ZH_MD5: "-gsrc-hash=md5"

// RUN: %clang_cl /ZH:SHA1 /c -### -- %s 2>&1 \
// RUN:     | FileCheck -check-prefix=ZH_SHA1 %s
// ZH_SHA1: "-gsrc-hash=sha1"

// RUN: %clang_cl /ZH:SHA_256 /c -### -- %s 2>&1 \
// RUN:     | FileCheck -check-prefix=ZH_SHA256 %s
// ZH_SHA256: "-gsrc-hash=sha256"

// RUN: %clang_cl -fmsc-version=1800 -TP -### -- %s 2>&1 | FileCheck -check-prefix=CXX11 %s
// CXX11: -std=c++11

// RUN: %clang_cl -fmsc-version=1900 -TP -### -- %s 2>&1 | FileCheck -check-prefix=CXX14 %s
// CXX14: -std=c++14

// RUN: %clang_cl -fmsc-version=1900 -TP -std:c++14 -### -- %s 2>&1 | FileCheck -check-prefix=STDCXX14 %s
// STDCXX14: -std=c++14

// RUN: %clang_cl -fmsc-version=1900 -TP -std:c++17 -### -- %s 2>&1 | FileCheck -check-prefix=STDCXX17 %s
// STDCXX17: -std=c++17

// RUN: %clang_cl -fmsc-version=1900 -TP -std:c++20 -### -- %s 2>&1 | FileCheck -check-prefix=STDCXX20 %s
// STDCXX20: -std=c++20

// RUN: %clang_cl -fmsc-version=1900 -TP -std:c++23preview -### -- %s 2>&1 | FileCheck -check-prefix=STDCXX23PREVIEW %s
// STDCXX23PREVIEW: -std=c++23

// RUN: %clang_cl -fmsc-version=1900 -TP -std:c++latest -### -- %s 2>&1 | FileCheck -check-prefix=STDCXXLATEST %s
// STDCXXLATEST: -std=c++26

// RUN: env CL="/Gy" %clang_cl -### -- %s 2>&1 | FileCheck -check-prefix=ENV-CL %s
// ENV-CL: "-ffunction-sections"

// RUN: env CL="/Gy" _CL_="/Gy- -- %s" %clang_cl -### 2>&1 | FileCheck -check-prefix=ENV-_CL_ %s
// ENV-_CL_-NOT: "-ffunction-sections"

// RUN: env CL="%s" _CL_="%s" not %clang --rsp-quoting=windows -c

// RUN: %clang_cl -### /c -flto -- %s 2>&1 | FileCheck -check-prefix=LTO %s
// LTO: -flto

// RUN: %clang_cl -### /c -flto -fno-lto -- %s 2>&1 | FileCheck -check-prefix=LTO-NO %s
// LTO-NO-NOT: "-flto"

// RUN: %clang_cl -### /c -flto=thin -- %s 2>&1 | FileCheck -check-prefix=LTO-THIN %s
// LTO-THIN: -flto=thin

// RUN: not %clang_cl -### -Fe%t.exe -entry:main -flto -- %s 2>&1 | FileCheck -check-prefix=LTO-WITHOUT-LLD %s
// RUN: not %clang_cl -### -fuse-ld=link -Fe%t.exe -entry:main -flto -- %s 2>&1 | FileCheck -check-prefix=LTO-WITHOUT-LLD %s
// RUN: not %clang -### --target=x86_64-windows-msvc -fuse-ld=link -Fe%t.exe -entry:main -flto -- %s 2>&1 | FileCheck -check-prefix=LTO-WITHOUT-LLD %s
// LTO-WITHOUT-LLD: LTO requires -fuse-ld=lld

// RUN: %clang_cl  -### -- %s 2>&1 | FileCheck -check-prefix=NOCFGUARD %s
// RUN: %clang_cl /guard:cf- -### -- %s 2>&1 | FileCheck -check-prefix=NOCFGUARD %s
// NOCFGUARD-NOT: -cfguard

// RUN: %clang_cl /guard:cf -### -- %s 2>&1 | FileCheck -check-prefix=CFGUARD %s
// CFGUARD: -cfguard
// CFGUARD-NOT: -cfguard-no-checks

// RUN: %clang_cl /guard:cf,nochecks -### -- %s 2>&1 | FileCheck -check-prefix=CFGUARDNOCHECKS %s
// CFGUARDNOCHECKS: -cfguard-no-checks

// RUN: not %clang_cl /guard:nochecks -### -- %s 2>&1 | FileCheck -check-prefix=CFGUARDNOCHECKSINVALID %s
// CFGUARDNOCHECKSINVALID: invalid value 'nochecks' in '/guard:'

// RUN: %clang_cl  -### -- %s 2>&1 | FileCheck -check-prefix=NOEHCONTGUARD %s
// RUN: %clang_cl /guard:ehcont- -### -- %s 2>&1 | FileCheck -check-prefix=NOEHCONTGUARD %s
// NOEHCONTGUARD-NOT: -ehcontguard

// RUN: %clang_cl /guard:ehcont -### -- %s 2>&1 | FileCheck -check-prefix=EHCONTGUARD %s
// EHCONTGUARD: -ehcontguard

// RUN: %clang_cl /guard:cf /guard:ehcont -Wall -Wno-msvc-not-found -### -- %s 2>&1 | \
// RUN:   FileCheck -check-prefix=BOTHGUARD %s --implicit-check-not=warning:
// BOTHGUARD: -cfguard
// BOTHGUARD-SAME: -ehcontguard
// BOTHGUARD: -guard:cf
// BOTHGUARD-SAME: -guard:ehcont

// RUN: not %clang_cl /guard:foo -### -- %s 2>&1 | FileCheck -check-prefix=CFGUARDINVALID %s
// CFGUARDINVALID: invalid value 'foo' in '/guard:'

// Accept "core" clang options.
// (/Zs is for syntax-only, -Werror makes it fail hard on unknown options)
// RUN: %clang_cl \
// RUN:     --driver-mode=cl \
// RUN:     -fblocks \
// RUN:     -fcrash-diagnostics-dir=/foo \
// RUN:     -fno-crash-diagnostics \
// RUN:     -fno-blocks \
// RUN:     -fbuiltin \
// RUN:     -fno-builtin \
// RUN:     -fno-builtin-strcpy \
// RUN:     -fcolor-diagnostics \
// RUN:     -fno-color-diagnostics \
// RUN:     -fcoverage-mapping \
// RUN:     -fno-coverage-mapping \
// RUN:     -fdiagnostics-color \
// RUN:     -fdiagnostics-color=auto \
// RUN:     -fno-diagnostics-color \
// RUN:     -fdebug-compilation-dir . \
// RUN:     -fdebug-compilation-dir=. \
// RUN:     -ffile-compilation-dir=. \
// RUN:     -fdiagnostics-parseable-fixits \
// RUN:     -fdiagnostics-absolute-paths \
// RUN:     -ferror-limit=10 \
// RUN:     -fident \
// RUN:     -fno-ident \
// RUN:     -fmsc-version=1800 \
// RUN:     -fno-strict-aliasing \
// RUN:     -fstrict-aliasing \
// RUN:     -fsyntax-only \
// RUN:     -fms-compatibility \
// RUN:     -fno-ms-compatibility \
// RUN:     -fms-extensions \
// RUN:     -fno-ms-extensions \
// RUN:     -Xclang -disable-llvm-passes \
// RUN:     -resource-dir asdf \
// RUN:     -resource-dir=asdf \
// RUN:     -Wunused-variable \
// RUN:     -fmacro-backtrace-limit=0 \
// RUN:     -fstandalone-debug \
// RUN:     -feliminate-unused-debug-types \
// RUN:     -fno-eliminate-unused-debug-types \
// RUN:     -flimit-debug-info \
// RUN:     -flto \
// RUN:     -fmerge-all-constants \
// RUN:     -no-canonical-prefixes \
// RUN:     -march=skylake \
// RUN:     -fbracket-depth=123 \
// RUN:     -fprofile-generate \
// RUN:     -fprofile-generate=dir \
// RUN:     -fprofile-sample-use=%S/Inputs/file.prof \
// RUN:     -fno-profile-generate \
// RUN:     -fno-profile-instr-generate \
// RUN:     -fno-profile-instr-use \
// RUN:     -fcs-profile-generate \
// RUN:     -fcs-profile-generate=dir \
// RUN:     -fpseudo-probe-for-profiling \
// RUN:     -ftime-trace \
// RUN:     -fmodules \
// RUN:     -fno-modules \
// RUN:     -fimplicit-module-maps \
// RUN:     -fmodule-maps \
// RUN:     -fmodule-name=foo \
// RUN:     -fmodule-implementation-of \
// RUN:     -fsystem-module \
// RUN:     -fmodule-map-file=foo \
// RUN:     -fmodule-file=foo \
// RUN:     -fmodules-ignore-macro=foo \
// RUN:     -fmodules-strict-decluse \
// RUN:     -fmodules-decluse \
// RUN:     -fno-modules-decluse \
// RUN:     -fmodules-search-all \
// RUN:     -fno-modules-search-all \
// RUN:     -fimplicit-modules \
// RUN:     -fno-implicit-modules \
// RUN:     -fstrict-overflow \
// RUN:     -fno-strict-overflow \
// RUN:     -ftrivial-auto-var-init=zero \
// RUN:     -fwrapv \
// RUN:     -fno-wrapv \
// RUN:     -fwrapv-pointer \
// RUN:     -fno-wrapv-pointer \
// RUN:     --version \
// RUN:     --warning-suppression-mappings=foo \
// RUN:     -Werror /Zs -- %s 2>&1

// Accept clang options under the /clang: flag.
// The first test case ensures that the SLP vectorizer is on by default and that
// it's being turned off by the /clang:-fno-slp-vectorize flag.

// RUN: %clang_cl -O2 -### -- %s 2>&1 | FileCheck -check-prefix=NOCLANG %s
// NOCLANG: "--dependent-lib=libcmt"
// NOCLANG-SAME: "-vectorize-slp"
// NOCLANG-NOT: "--dependent-lib=msvcrt"

// RUN: %clang_cl -O2 -MD /clang:-fno-slp-vectorize /clang:-MD /clang:-MF /clang:my_dependency_file.dep /c /Fo%/t/cl-options.obj -### -- %s 2>&1 | FileCheck -DPREFIX=%/t -check-prefix=CLANG %s
// CLANG: "--dependent-lib=msvcrt"
// CLANG-SAME: "-dependency-file" "my_dependency_file.dep"
// CLANG-SAME: "-MT" "[[PREFIX]]/cl-options.obj"
// CLANG-NOT: "--dependent-lib=libcmt"
// CLANG-NOT: "-vectorize-slp"

// Cover PR42501: clang-cl /clang: pass-through causes read-after-free with aliased options.
// RUN: %clang_cl /clang:-save-temps /clang:-Wl,test1,test2 -### -- %s 2>&1 | FileCheck -check-prefix=SAVETEMPS %s
// SAVETEMPS: "-save-temps=cwd"
// SAVETEMPS: "test1" "test2"

// Validate that the default triple is used when run an empty tools dir is specified
// RUN: %clang_cl -vctoolsdir "" -### -- %s 2>&1 | FileCheck %s --check-prefix VCTOOLSDIR
// VCTOOLSDIR: "-triple" "{{[a-zA-Z0-9_-]*}}-pc-windows-msvc19.33.0"

// Validate that built-in include paths are based on the supplied path
// RUN: %clang_cl --target=aarch64-pc-windows-msvc -vctoolsdir "/fake" -winsdkdir "/foo" -winsdkversion 10.0.12345.0 -### -- %s 2>&1 | FileCheck %s --check-prefix FAKEDIR
// FAKEDIR: "-internal-isystem" "/fake{{/|\\\\}}include"
// FAKEDIR: "-internal-isystem" "/fake{{/|\\\\}}atlmfc{{/|\\\\}}include"
// FAKEDIR: "-internal-isystem" "/foo{{/|\\\\}}Include{{/|\\\\}}10.0.12345.0{{/|\\\\}}ucrt"
// FAKEDIR: "-internal-isystem" "/foo{{/|\\\\}}Include{{/|\\\\}}10.0.12345.0{{/|\\\\}}shared"
// FAKEDIR: "-internal-isystem" "/foo{{/|\\\\}}Include{{/|\\\\}}10.0.12345.0{{/|\\\\}}um"
// FAKEDIR: "-internal-isystem" "/foo{{/|\\\\}}Include{{/|\\\\}}10.0.12345.0{{/|\\\\}}winrt"
// FAKEDIR: "-libpath:/fake{{/|\\\\}}lib{{/|\\\\}}
// FAKEDIR: "-libpath:/fake{{/|\\\\}}atlmfc{{/|\\\\}}lib{{/|\\\\}}
// FAKEDIR: "-libpath:/foo{{/|\\\\}}Lib{{/|\\\\}}10.0.12345.0{{/|\\\\}}ucrt
// FAKEDIR: "-libpath:/foo{{/|\\\\}}Lib{{/|\\\\}}10.0.12345.0{{/|\\\\}}um

// Accept both the -target and --target= spellings.
// RUN: %clang_cl --target=i686-pc-windows-msvc19.14.0 -### -- %s 2>&1 | FileCheck -check-prefix=TARGET %s
// RUN: %clang_cl -target i686-pc-windows-msvc19.14.0  -### -- %s 2>&1 | FileCheck -check-prefix=TARGET %s
// TARGET: "-triple" "i686-pc-windows-msvc19.14.0"

// RUN: %clang_cl /JMC /c -### -- %s 2>&1 | FileCheck %s --check-prefix JMCWARN
// JMCWARN: /JMC requires debug info. Use '/Zi', '/Z7' or debug options that enable debugger's stepping function; option ignored

// RUN: %clang_cl /JMC /c -### -- %s 2>&1 | FileCheck %s --check-prefix NOJMC
// RUN: %clang_cl /JMC /Z7 /JMC- /c -### -- %s 2>&1 | FileCheck %s --check-prefix NOJMC
// NOJMC-NOT: -fjmc

// RUN: %clang_cl /JMC /Z7 /c -### -- %s 2>&1 | FileCheck %s --check-prefix JMC
// JMC: -fjmc

// RUN: %clang_cl /external:W0 /c -### -- %s 2>&1 | FileCheck -check-prefix=EXTERNAL_W0 %s
// RUN: %clang_cl /external:W1 /c -### -- %s 2>&1 | FileCheck -check-prefix=EXTERNAL_Wn %s
// RUN: %clang_cl /external:W2 /c -### -- %s 2>&1 | FileCheck -check-prefix=EXTERNAL_Wn %s
// RUN: %clang_cl /external:W3 /c -### -- %s 2>&1 | FileCheck -check-prefix=EXTERNAL_Wn %s
// RUN: %clang_cl /external:W4 /c -### -- %s 2>&1 | FileCheck -check-prefix=EXTERNAL_Wn %s
// EXTERNAL_W0: "-Wno-system-headers"
// EXTERNAL_Wn: "-Wsystem-headers"

// RUN: %clang_cl -vctoolsdir "" /arm64EC /c -### -- %s 2>&1 | FileCheck --check-prefix=ARM64EC %s 
// ARM64EC-NOT: /arm64EC has been overridden by specified target
// ARM64EC: "-triple" "arm64ec-pc-windows-msvc19.33.0"
// ARM64EC-SAME: "--dependent-lib=softintrin"

// RUN: %clang_cl -vctoolsdir "" /arm64EC /c -target x86_64-pc-windows-msvc  -### -- %s 2>&1 | FileCheck --check-prefix=ARM64EC_OVERRIDE %s
// ARM64EC_OVERRIDE: warning: /arm64EC has been overridden by specified target: x86_64-pc-windows-msvc; option ignored

// RUN: %clang_cl /d2epilogunwind /c -### -- %s 2>&1 | FileCheck %s --check-prefix=EPILOGUNWIND
// EPILOGUNWIND: -fwinx64-eh-unwindv2=best-effort

// RUN: %clang_cl /d2epilogunwindrequirev2 /c -### -- %s 2>&1 | FileCheck %s --check-prefix=EPILOGUNWINDREQUIREV2
// RUN: %clang_cl /d2epilogunwindrequirev2 /d2epilogunwind /c -### -- %s 2>&1 | FileCheck %s --check-prefix=EPILOGUNWINDREQUIREV2
// EPILOGUNWINDREQUIREV2: -fwinx64-eh-unwindv2=require

// RUN: %clang_cl /funcoverride:override_me1 /funcoverride:override_me2 /c -### -- %s 2>&1 | FileCheck %s --check-prefix=FUNCOVERRIDE
// FUNCOVERRIDE: -loader-replaceable-function=override_me1
// FUNCOVERRIDE-SAME: -loader-replaceable-function=override_me2

void f(void) { }
