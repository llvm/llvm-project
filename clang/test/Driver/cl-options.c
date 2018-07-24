// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.


// Alias options:

// RUN: %clang_cl /c -### -- %s 2>&1 | FileCheck -check-prefix=c %s
// c: -c

// RUN: %clang_cl /C -### -- %s 2>&1 | FileCheck -check-prefix=C %s
// C: error: invalid argument '-C' only allowed with '/E, /P or /EP'

// RUN: %clang_cl /C /P -### -- %s 2>&1 | FileCheck -check-prefix=C_P %s
// C_P: "-E"
// C_P: "-C"

// RUN: %clang_cl /d1reportAllClassLayout -### -- %s 2>&1 | FileCheck -check-prefix=d1reportAllClassLayout %s
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

// RUN: %clang_cl /fp:fast /fp:except -### -- %s 2>&1 | FileCheck -check-prefix=fpexcept %s
// fpexcept-NOT: -menable-unsafe-fp-math

// RUN: %clang_cl /fp:fast /fp:except /fp:except- -### -- %s 2>&1 | FileCheck -check-prefix=fpexcept_ %s
// fpexcept_: -menable-unsafe-fp-math

// RUN: %clang_cl /fp:precise /fp:fast -### -- %s 2>&1 | FileCheck -check-prefix=fpfast %s
// fpfast: -menable-unsafe-fp-math
// fpfast: -ffast-math

// RUN: %clang_cl /fp:fast /fp:precise -### -- %s 2>&1 | FileCheck -check-prefix=fpprecise %s
// fpprecise-NOT: -menable-unsafe-fp-math
// fpprecise-NOT: -ffast-math

// RUN: %clang_cl /fp:fast /fp:strict -### -- %s 2>&1 | FileCheck -check-prefix=fpstrict %s
// fpstrict-NOT: -menable-unsafe-fp-math
// fpstrict-NOT: -ffast-math

// RUN: %clang_cl /Z7 -gcolumn-info -### -- %s 2>&1 | FileCheck -check-prefix=gcolumn %s
// gcolumn: -dwarf-column-info

// RUN: %clang_cl /Z7 -gno-column-info -### -- %s 2>&1 | FileCheck -check-prefix=gnocolumn %s
// gnocolumn-NOT: -dwarf-column-info

// RUN: %clang_cl /Z7 -### -- %s 2>&1 | FileCheck -check-prefix=gdefcolumn %s
// gdefcolumn-NOT: -dwarf-column-info

// RUN: %clang_cl -### /FA -fprofile-instr-generate -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-GENERATE %s
// RUN: %clang_cl -### /FA -fprofile-instr-generate=/tmp/somefile.profraw -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-GENERATE-FILE %s
// RUN: %clang_cl -### /FA -fprofile-instr-generate -fprofile-instr-use -- %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// RUN: %clang_cl -### /FA -fprofile-instr-generate -fprofile-instr-use=file -- %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MIX-GEN-USE %s
// CHECK-PROFILE-GENERATE: "-fprofile-instrument=clang"
// CHECK-PROFILE-GENERATE-FILE: "-fprofile-instrument-path=/tmp/somefile.profraw"
// CHECK-NO-MIX-GEN-USE: '{{[a-z=-]*}}' not allowed with '{{[a-z=-]*}}'

// RUN: %clang_cl -### /FA -fprofile-instr-use -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE %s
// RUN: %clang_cl -### /FA -fprofile-instr-use=/tmp/somefile.prof -- %s 2>&1 | FileCheck -check-prefix=CHECK-PROFILE-USE-FILE %s
// CHECK-PROFILE-USE: "-fprofile-instrument-use-path=default.profdata"
// CHECK-PROFILE-USE-FILE: "-fprofile-instrument-use-path=/tmp/somefile.prof"

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
// Gs: "-mstack-probe-size=0"
// RUN: %clang_cl /Gs0 -### -- %s 2>&1 | FileCheck -check-prefix=Gs0 %s
// Gs0: "-mstack-probe-size=0"
// RUN: %clang_cl /Gs4096 -### -- %s 2>&1 | FileCheck -check-prefix=Gs4096 %s
// Gs4096: "-mstack-probe-size=4096"

// RUN: %clang_cl /Gw -### -- %s 2>&1 | FileCheck -check-prefix=Gw %s
// Gw: -fdata-sections

// RUN: %clang_cl /Gw /Gw- -### -- %s 2>&1 | FileCheck -check-prefix=Gw_ %s
// Gw_-NOT: -fdata-sections

// RUN: %clang_cl /Imyincludedir -### -- %s 2>&1 | FileCheck -check-prefix=SLASH_I %s
// RUN: %clang_cl /I myincludedir -### -- %s 2>&1 | FileCheck -check-prefix=SLASH_I %s
// SLASH_I: "-I" "myincludedir"

// RUN: %clang_cl /imsvcmyincludedir -### -- %s 2>&1 | FileCheck -check-prefix=SLASH_imsvc %s
// RUN: %clang_cl /imsvc myincludedir -### -- %s 2>&1 | FileCheck -check-prefix=SLASH_imsvc %s
// Clang's resource header directory should be first:
// SLASH_imsvc: "-internal-isystem" "{{[^"]*}}lib{{(64)?/|\\\\}}clang{{[^"]*}}include"
// SLASH_imsvc: "-internal-isystem" "myincludedir"

// RUN: %clang_cl /J -### -- %s 2>&1 | FileCheck -check-prefix=J %s
// J: -fno-signed-char

// RUN: %clang_cl /Ofoo -### -- %s 2>&1 | FileCheck -check-prefix=O %s
// O: /Ofoo

// RUN: %clang_cl /Ob0 -### -- %s 2>&1 | FileCheck -check-prefix=Ob0 %s
// Ob0: -fno-inline

// RUN: %clang_cl /Ob2 -### -- %s 2>&1 | FileCheck -check-prefix=Ob2 %s
// RUN: %clang_cl /Odb2 -### -- %s 2>&1 | FileCheck -check-prefix=Ob2 %s
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
// Os-NOT: -mdisable-fp-elim
// Os: -momit-leaf-frame-pointer
// Os: -Os

// RUN: %clang_cl /Ot --target=i686-pc-windows-msvc -### -- %s 2>&1 | FileCheck -check-prefix=Ot %s
// RUN: %clang_cl /Ot --target=x86_64-pc-windows-msvc -### -- %s 2>&1 | FileCheck -check-prefix=Ot %s
// Ot-NOT: -mdisable-fp-elim
// Ot: -momit-leaf-frame-pointer
// Ot: -O2

// RUN: %clang_cl /Ox --target=i686-pc-windows-msvc -### -- %s 2>&1 | FileCheck -check-prefix=Ox %s
// RUN: %clang_cl /Ox --target=x86_64-pc-windows-msvc -### -- %s 2>&1 | FileCheck -check-prefix=Ox %s
// Ox-NOT: -mdisable-fp-elim
// Ox: -momit-leaf-frame-pointer
// Ox: -O2

// RUN: %clang_cl --target=i686-pc-win32 /O2sy- -### -- %s 2>&1 | FileCheck -check-prefix=PR24003 %s
// PR24003: -mdisable-fp-elim
// PR24003: -momit-leaf-frame-pointer
// PR24003: -Os

// RUN: %clang_cl --target=i686-pc-win32 -Werror /Oy- /O2 -### -- %s 2>&1 | FileCheck -check-prefix=Oy_2 %s
// Oy_2: -momit-leaf-frame-pointer
// Oy_2: -O2

// RUN: %clang_cl --target=i686-pc-win32 -Werror /O2 /O2 -### -- %s 2>&1 | FileCheck -check-prefix=O2O2 %s
// O2O2: "-O2"

// RUN: %clang_cl /Zs -Werror /Oy -- %s 2>&1

// RUN: %clang_cl --target=i686-pc-win32 -Werror /Oy- -### -- %s 2>&1 | FileCheck -check-prefix=Oy_ %s
// Oy_: -mdisable-fp-elim

// RUN: %clang_cl /Qvec -### -- %s 2>&1 | FileCheck -check-prefix=Qvec %s
// Qvec: -vectorize-loops

// RUN: %clang_cl /Qvec /Qvec- -### -- %s 2>&1 | FileCheck -check-prefix=Qvec_ %s
// Qvec_-NOT: -vectorize-loops

// RUN: %clang_cl /showIncludes -### -- %s 2>&1 | FileCheck -check-prefix=showIncludes %s
// showIncludes: --show-includes

// RUN: %clang_cl /E /showIncludes -### -- %s 2>&1 | FileCheck -check-prefix=showIncludes_E %s
// RUN: %clang_cl /EP /showIncludes -### -- %s 2>&1 | FileCheck -check-prefix=showIncludes_E %s
// RUN: %clang_cl /E /EP /showIncludes -### -- %s 2>&1 | FileCheck -check-prefix=showIncludes_E %s
// RUN: %clang_cl /EP /P /showIncludes -### -- %s 2>&1 | FileCheck -check-prefix=showIncludes_E %s
// showIncludes_E-NOT: warning: argument unused during compilation: '--show-includes'

// /source-charset: should warn on everything except UTF-8.
// RUN: %clang_cl /source-charset:utf-16 -### -- %s 2>&1 | FileCheck -check-prefix=source-charset-utf-16 %s
// source-charset-utf-16: invalid value 'utf-16'

// /execution-charset: should warn on everything except UTF-8.
// RUN: %clang_cl /execution-charset:utf-16 -### -- %s 2>&1 | FileCheck -check-prefix=execution-charset-utf-16 %s
// execution-charset-utf-16: invalid value 'utf-16'
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

// RUN: %clang_cl /vmg /vmb -### -- %s 2>&1 | FileCheck -check-prefix=VMB %s
// VMB: '/vmg' not allowed with '/vmb'

// RUN: %clang_cl /vmg /vmm /vms -### -- %s 2>&1 | FileCheck -check-prefix=VMX %s
// VMX: '/vms' not allowed with '/vmm'

// RUN: %clang_cl /volatile:iso -### -- %s 2>&1 | FileCheck -check-prefix=VOLATILE-ISO %s
// VOLATILE-ISO-NOT: "-fms-volatile"

// RUN: %clang_cl /volatile:ms -### -- %s 2>&1 | FileCheck -check-prefix=VOLATILE-MS %s
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

// RUN: %clang_cl /WX -### -- %s 2>&1 | FileCheck -check-prefix=WX %s
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

// We recognize -f[no-]delayed-template-parsing.
// /Zc:twoPhase[-] has the opposite meaning.
// RUN: %clang_cl -c -### -- %s 2>&1 | FileCheck -check-prefix=DELAYEDDEFAULT %s
// DELAYEDDEFAULT: "-fdelayed-template-parsing"
// RUN: %clang_cl -c -fdelayed-template-parsing -### -- %s 2>&1 | FileCheck -check-prefix=DELAYEDON %s
// RUN: %clang_cl -c /Zc:twoPhase- -### -- %s 2>&1 | FileCheck -check-prefix=DELAYEDON %s
// DELAYEDON: "-fdelayed-template-parsing"
// RUN: %clang_cl -c -fno-delayed-template-parsing -### -- %s 2>&1 | FileCheck -check-prefix=DELAYEDOFF %s
// RUN: %clang_cl -c /Zc:twoPhase -### -- %s 2>&1 | FileCheck -check-prefix=DELAYEDOFF %s
// DELAYEDOFF-NOT: "-fdelayed-template-parsing"

// For some warning ids, we can map from MSVC warning to Clang warning.
// RUN: %clang_cl -wd4005 -wd4100 -wd4910 -wd4996 -### -- %s 2>&1 | FileCheck -check-prefix=Wno %s
// Wno: "-cc1"
// Wno: "-Wno-macro-redefined"
// Wno: "-Wno-unused-parameter"
// Wno: "-Wno-dllexport-explicit-instantiation-decl"
// Wno: "-Wno-deprecated-declarations"

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
// RUN:    /Zc:__cplusplus \
// RUN:    /Zc:auto \
// RUN:    /Zc:forScope \
// RUN:    /Zc:inline \
// RUN:    /Zc:rvalueCast \
// RUN:    /Zc:ternary \
// RUN:    /Zc:wchar_t \
// RUN:    /Zm \
// RUN:    /Zo \
// RUN:    /Zo- \
// RUN:    -### -- %s 2>&1 | FileCheck -check-prefix=IGNORED %s
// IGNORED-NOT: argument unused during compilation
// IGNORED-NOT: no such file or directory
// Don't confuse /openmp- with the /o flag:
// IGNORED-NOT: "-o" "penmp-.obj"

// Ignored options and compile-only options are ignored for link jobs.
// RUN: touch %t.obj
// RUN: %clang_cl /nologo -### -- %t.obj 2>&1 | FileCheck -check-prefix=LINKUNUSED %s
// RUN: %clang_cl /Dfoo -### -- %t.obj 2>&1 | FileCheck -check-prefix=LINKUNUSED %s
// RUN: %clang_cl /MD -### -- %t.obj 2>&1 | FileCheck -check-prefix=LINKUNUSED %s
// LINKUNUSED-NOT: argument unused during compilation

// Support ignoring warnings about unused arguments.
// RUN: %clang_cl /Abracadabra -Qunused-arguments -### -- %s 2>&1 | FileCheck -check-prefix=UNUSED %s
// UNUSED-NOT: argument unused during compilation

// Unsupported but parsed options. Check that we don't error on them.
// (/Zs is for syntax-only)
// RUN: %clang_cl /Zs \
// RUN:     /AIfoo \
// RUN:     /Bt \
// RUN:     /Bt+ \
// RUN:     /clr:pure \
// RUN:     /docname \
// RUN:     /EHsc \
// RUN:     /F \
// RUN:     /FA \
// RUN:     /FAc \
// RUN:     /Fafilename \
// RUN:     /FAs \
// RUN:     /FAu \
// RUN:     /favor:blend \
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
// RUN:     /guard:cf \
// RUN:     /guard:cf- \
// RUN:     /GX \
// RUN:     /Gv \
// RUN:     /Gz \
// RUN:     /GZ \
// RUN:     /H \
// RUN:     /homeparams \
// RUN:     /hotpatch \
// RUN:     /kernel \
// RUN:     /LN \
// RUN:     /MP \
// RUN:     /o foo.obj \
// RUN:     /ofoo.obj \
// RUN:     /openmp \
// RUN:     /Qfast_transcendentals \
// RUN:     /QIfist \
// RUN:     /Qimprecise_fwaits \
// RUN:     /Qpar \
// RUN:     /Qvec-report:2 \
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
// SlashU: note: Use '--' to treat subsequent arguments as filenames

// RTTI is on by default. /GR- controls -fno-rtti-data.
// RUN: %clang_cl /c /GR- -### -- %s 2>&1 | FileCheck -check-prefix=NoRTTI %s
// NoRTTI: "-fno-rtti-data"
// NoRTTI-NOT: "-fno-rtti"
// RUN: %clang_cl /c /GR -### -- %s 2>&1 | FileCheck -check-prefix=RTTI %s
// RTTI-NOT: "-fno-rtti-data"
// RTTI-NOT: "-fno-rtti"

// thread safe statics are off for versions < 19.
// RUN: %clang_cl /c -### -fms-compatibility-version=18 -- %s 2>&1 | FileCheck -check-prefix=NoThreadSafeStatics %s
// RUN: %clang_cl /Zc:threadSafeInit /Zc:threadSafeInit- /c -### -- %s 2>&1 | FileCheck -check-prefix=NoThreadSafeStatics %s
// NoThreadSafeStatics: "-fno-threadsafe-statics"

// RUN: %clang_cl /Zc:threadSafeInit /c -### -- %s 2>&1 | FileCheck -check-prefix=ThreadSafeStatics %s
// ThreadSafeStatics-NOT: "-fno-threadsafe-statics"

// RUN: %clang_cl /Zi /c -### -- %s 2>&1 | FileCheck -check-prefix=Zi %s
// Zi: "-gcodeview"
// Zi: "-debug-info-kind=limited"

// RUN: %clang_cl /Z7 /c -### -- %s 2>&1 | FileCheck -check-prefix=Z7 %s
// Z7: "-gcodeview"
// Z7: "-debug-info-kind=limited"

// RUN: %clang_cl /Zd /c -### -- %s 2>&1 | FileCheck -check-prefix=Z7GMLT %s
// Z7GMLT: "-gcodeview"
// Z7GMLT: "-debug-info-kind=line-tables-only"

// RUN: %clang_cl -gline-tables-only /c -### -- %s 2>&1 | FileCheck -check-prefix=ZGMLT %s
// ZGMLT: "-gcodeview"
// ZGMLT: "-debug-info-kind=line-tables-only"

// RUN: %clang_cl /c -### -- %s 2>&1 | FileCheck -check-prefix=BreproDefault %s
// BreproDefault: "-mincremental-linker-compatible"

// RUN: %clang_cl /Brepro- /Brepro /c '-###' -- %s 2>&1 | FileCheck -check-prefix=Brepro %s
// Brepro-NOT: "-mincremental-linker-compatible"

// RUN: %clang_cl /Brepro /Brepro- /c '-###' -- %s 2>&1 | FileCheck -check-prefix=Brepro_ %s
// Brepro_: "-mincremental-linker-compatible"

// This test was super sneaky: "/Z7" means "line-tables", but "-gdwarf" occurs
// later on the command line, so it should win. Interestingly the cc1 arguments
// came out right, but had wrong semantics, because an invariant assumed by
// CompilerInvocation was violated: it expects that at most one of {gdwarfN,
// line-tables-only} appear. If you assume that, then you can safely use
// Args.hasArg to test whether a boolean flag is present without caring
// where it appeared. And for this test, it appeared to the left of -gdwarf
// which made it "win". This test could not detect that bug.
// RUN: %clang_cl /Z7 -gdwarf /c -### -- %s 2>&1 | FileCheck -check-prefix=Z7_gdwarf %s
// Z7_gdwarf: "-gcodeview"
// Z7_gdwarf: "-debug-info-kind=limited"
// Z7_gdwarf: "-dwarf-version=4"

// RUN: %clang_cl -fmsc-version=1800 -TP -### -- %s 2>&1 | FileCheck -check-prefix=CXX11 %s
// CXX11: -std=c++11

// RUN: %clang_cl -fmsc-version=1900 -TP -### -- %s 2>&1 | FileCheck -check-prefix=CXX14 %s
// CXX14: -std=c++14

// RUN: %clang_cl -fmsc-version=1900 -TP -std:c++14 -### -- %s 2>&1 | FileCheck -check-prefix=STDCXX14 %s
// STDCXX14: -std=c++14

// RUN: %clang_cl -fmsc-version=1900 -TP -std:c++17 -### -- %s 2>&1 | FileCheck -check-prefix=STDCXX17 %s
// STDCXX17: -std=c++17

// RUN: %clang_cl -fmsc-version=1900 -TP -std:c++latest -### -- %s 2>&1 | FileCheck -check-prefix=STDCXXLATEST %s
// STDCXXLATEST: -std=c++2a

// RUN: env CL="/Gy" %clang_cl -### -- %s 2>&1 | FileCheck -check-prefix=ENV-CL %s
// ENV-CL: "-ffunction-sections"

// RUN: env CL="/Gy" _CL_="/Gy- -- %s" %clang_cl -### 2>&1 | FileCheck -check-prefix=ENV-_CL_ %s
// ENV-_CL_-NOT: "-ffunction-sections"

// RUN: env CL="%s" _CL_="%s" not %clang --rsp-quoting=windows -c

// RUN: %clang_cl -### /c -flto -- %s 2>&1 | FileCheck -check-prefix=LTO %s
// LTO: -flto

// RUN: %clang_cl -### /c -flto=thin -- %s 2>&1 | FileCheck -check-prefix=LTO-THIN %s
// LTO-THIN: -flto=thin

// RUN: %clang_cl -### -Fe%t.exe -entry:main -flto -- %s 2>&1 | FileCheck -check-prefix=LTO-WITHOUT-LLD %s
// LTO-WITHOUT-LLD: LTO requires -fuse-ld=lld

// Accept "core" clang options.
// (/Zs is for syntax-only, -Werror makes it fail hard on unknown options)
// RUN: %clang_cl \
// RUN:     --driver-mode=cl \
// RUN:     -fblocks \
// RUN:     -fno-blocks \
// RUN:     -fbuiltin \
// RUN:     -fno-builtin \
// RUN:     -fno-builtin-strcpy \
// RUN:     -fcolor-diagnostics \
// RUN:     -fno-color-diagnostics \
// RUN:     -fcoverage-mapping \
// RUN:     -fno-coverage-mapping \
// RUN:     -fdiagnostics-color \
// RUN:     -fno-diagnostics-color \
// RUN:     -fdiagnostics-parseable-fixits \
// RUN:     -fdiagnostics-absolute-paths \
// RUN:     -ferror-limit=10 \
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
// RUN:     -flimit-debug-info \
// RUN:     -flto \
// RUN:     -fmerge-all-constants \
// RUN:     -no-canonical-prefixes \
// RUN:     --version \
// RUN:     -Werror /Zs -- %s 2>&1


void f() { }
