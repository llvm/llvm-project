// NOTE: This lit test was automatically generated to validate unintentionally exposed arguments to various driver flavours.
// NOTE: To make changes, see /Users/georgeasante/llvm-project/clang/utils/generate_unsupported_in_drivermode.py from which it was generated.

// RUN: not %clang -cc1as -A - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -A -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -A -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -A -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -A- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -A- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -A- -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -A- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -B - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -B -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -B -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -B -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -C - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -C -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -C -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -CC - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -CC -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -CC -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -D - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -E - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -E -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -cc1as -EB - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -EB -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -EB -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -cc1as -EL - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -EL -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -EL -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -cc1as -Eonly - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -Eonly -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -Eonly -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -F - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -F -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -faapcs-bitfield-load - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -faapcs-bitfield-load -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -faapcs-bitfield-load -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -G - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -G -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -G -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -G= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -G= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -G= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -H - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -H -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -H -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -J - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -J -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -J -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -J -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -J -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -K - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -K -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -K -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -K -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -L - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -L -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -L -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -L -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -M - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -M -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -M -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -M -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -MD - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -MD -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -MD -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -MD -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -MF - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -MF -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -MF -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -MF -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -MG - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -MG -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -MG -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -MJ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -MJ -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -MJ -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -MJ -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -MM - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -MM -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -MM -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -MM -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -MMD - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -MMD -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -MMD -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -MMD -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -MP - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -MP -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -MQ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -MQ -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -MQ -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -MT - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -MT -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -MT -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -MV - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -MV -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -MV -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Mach - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Mach -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Mach -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Mach -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -O - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -O0 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -O4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -O - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -ObjC - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -ObjC++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Ofast - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -P - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -P -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -P -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Q - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Q -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Q -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Q -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Qn - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -Qn -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Qn -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Qunused-arguments - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Qunused-arguments -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -Qy - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -Qy -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Qy -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -R - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -R -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -R -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Rpass= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -Rpass= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Rpass= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Rpass-analysis= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -Rpass-analysis= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Rpass-analysis= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Rpass-missed= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -Rpass-missed= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Rpass-missed= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -S - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -S -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -S -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -T - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -T -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -T -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -cc1as -U - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -U -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -V - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -V -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -V -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -V -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -WCL4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -W - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wa, - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wall - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wdeprecated - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wframe-larger-than - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wframe-larger-than= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Winvalid-constexpr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Winvalid-gnu-asm-cast - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wl, - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wlarge-by-value-copy= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wlarge-by-value-copy - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wlarger-than- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wno-deprecated - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wno-invalid-constexpr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wno-nonportable-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wno-rewrite-macros - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wno-system-headers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wno-write-strings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wnonportable-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wp, - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wsystem-headers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wsystem-headers-in-module= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wundef-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wwrite-strings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -X - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -X -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -X -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -X -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xanalyzer - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xanalyzer -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xanalyzer -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xanalyzer -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xarch_ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xarch_ -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xarch_ -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xarch_ -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xarch_device - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xarch_device -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xarch_device -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xarch_device -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xarch_host - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xarch_host -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xarch_host -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xarch_host -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xassembler - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xassembler -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xassembler -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xassembler -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xclang - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xclang -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -Xcuda-fatbinary - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xcuda-fatbinary -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xcuda-fatbinary -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xcuda-fatbinary -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xcuda-ptxas - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xcuda-ptxas -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -Xcuda-ptxas -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xflang - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xflang -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -Xflang -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -Xflang -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -Xlinker - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xlinker -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -Xoffload-linker - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xoffload-linker -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -Xopenmp-target - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xopenmp-target -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xopenmp-target -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xopenmp-target -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xopenmp-target= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xopenmp-target= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xopenmp-target= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xopenmp-target= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xpreprocessor - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xpreprocessor -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xpreprocessor -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xpreprocessor -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Z - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Z -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Z -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Z -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Z-Xlinker-no-demangle - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Z-Xlinker-no-demangle -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Z-Xlinker-no-demangle -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Z-Xlinker-no-demangle -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Z-reserved-lib-cckext - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Z-reserved-lib-cckext -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Z-reserved-lib-cckext -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Z-reserved-lib-cckext -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Z-reserved-lib-stdc++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Z-reserved-lib-stdc++ -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Z-reserved-lib-stdc++ -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Z-reserved-lib-stdc++ -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Zlinker-input - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Zlinker-input -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Zlinker-input -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Zlinker-input -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --CLASSPATH - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --CLASSPATH -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --CLASSPATH -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --CLASSPATH -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --CLASSPATH= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --CLASSPATH= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --CLASSPATH= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --CLASSPATH= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -### - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -### -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as /AI - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /AI -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /AI -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Brepro - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Brepro -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Brepro -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Brepro- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Brepro- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Brepro- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Bt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Bt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Bt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Bt+ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Bt+ -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Bt+ -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /C - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /C -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /C -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /C -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /D - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /E - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /E -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: %clang /E -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /EH - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /EH -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: %clang /EH -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /EP - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /EP -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: %clang /EP -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /F - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /F -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /FA - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /FA -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /FC - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /FC -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /FI - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /FI -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /FR - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /FR -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /FS - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /FS -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /FU - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /FU -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Fa - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /Fa -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Fd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /Fd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Fe - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /Fe -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Fe: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /Fe: -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Fi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /Fi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Fi: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /Fi: -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Fm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /Fm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Fo - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Fo: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Fp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /Fp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Fp: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /Fp: -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Fr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /Fr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Fx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /Fx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /G1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /G1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /G1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /G2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /G2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /G2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GA - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GA -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GA -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GF - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GF -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GF -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GF- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GF- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GF- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GH - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GH -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GH -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GL - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GL -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GL -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GL- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GL- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GL- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GR - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GR -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GR -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GR- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GR- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GR- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GS - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GS -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GS -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GS- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GS- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GS- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GT - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GT -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GT -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GX - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GX -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GX -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GX- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GX- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GX- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /GZ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /GZ -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /GZ -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Ge - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Ge -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Ge -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gh - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gh -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gh -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gm- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gm- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gm- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gregcall - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gregcall -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gregcall -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gregcall4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gregcall4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gregcall4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gv - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gv -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gv -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gw - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gw -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gw -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gw- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gw- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gw- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gy - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gy -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gy -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gy- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gy- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gy- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gz - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gz -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gz -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /H - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /H -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /H -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /H -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /HELP - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /HELP -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /HELP -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /HELP -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /J - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /J -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /J -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /J -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /JMC - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /JMC -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /JMC -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /JMC -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /JMC- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /JMC- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /JMC- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /JMC- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /LD - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /LD -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /LD -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /LDd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /LDd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /LDd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /LN - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /LN -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /LN -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /MD - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /MD -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /MD -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /MD -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /MDd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /MDd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /MDd -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /MDd -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /MP - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /MP -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /MP -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /MP -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /MT - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /MT -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /MTd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /MTd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /O - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /P - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /P -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /P -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /P -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /QIfist - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /QIfist -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /QIfist -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /QIfist -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /QIntel-jcc-erratum - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /QIntel-jcc-erratum -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /QIntel-jcc-erratum -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /QIntel-jcc-erratum -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /? - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /? -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /? -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /? -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Qfast_transcendentals - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qfast_transcendentals -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qfast_transcendentals -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qfast_transcendentals -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Qimprecise_fwaits - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qimprecise_fwaits -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qimprecise_fwaits -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qimprecise_fwaits -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Qpar - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qpar -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qpar -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qpar -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Qpar-report - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qpar-report -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qpar-report -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qpar-report -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Qsafe_fp_loads - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qsafe_fp_loads -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qsafe_fp_loads -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qsafe_fp_loads -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Qspectre - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qspectre -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qspectre -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qspectre -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Qspectre-load - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qspectre-load -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qspectre-load -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qspectre-load -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Qspectre-load-cf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qspectre-load-cf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qspectre-load-cf -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qspectre-load-cf -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Qvec - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qvec -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qvec -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qvec -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Qvec- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qvec- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qvec- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qvec- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Qvec-report - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qvec-report -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qvec-report -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qvec-report -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /RTC - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /RTC -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /TC - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /TC -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as /TP - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /TP -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as /Tc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Tc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as /Tp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Tp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as /U - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc /U -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /V - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /V -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /V -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /W0 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /W1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /W2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /W3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /W4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /WL - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /WX - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /WX- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Wall - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Wp64 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Wv - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /X - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /X -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /X -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /X -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Y- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Y- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Y- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Y- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Yc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Yc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Yc -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Yc -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Yd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Yd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Yd -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Yd -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Yl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Yl -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Yl -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Yl -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Yu - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Yu -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Yu -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Yu -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Z7 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Z7 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Z7 -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Z7 -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /ZH:MD5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /ZH:MD5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /ZH:MD5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /ZH:MD5 -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /ZH:SHA1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /ZH:SHA1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /ZH:SHA1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /ZH:SHA1 -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /ZH:SHA_256 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /ZH:SHA_256 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /ZH:SHA_256 -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /ZH:SHA_256 -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /ZI - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /ZI -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /ZI -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /ZI -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /ZW - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /ZW -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /ZW -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /ZW -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Za - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Za -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Za -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Za -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:__STDC__ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:__STDC__ -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:__STDC__ -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:__STDC__ -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:__cplusplus - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:__cplusplus -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:__cplusplus -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:__cplusplus -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:alignedNew - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:alignedNew -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:alignedNew -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:alignedNew -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:alignedNew- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:alignedNew- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:alignedNew- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:alignedNew- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:auto - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:auto -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:auto -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:auto -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:char8_t - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:char8_t -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:char8_t -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:char8_t -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:char8_t- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:char8_t- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:char8_t- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:char8_t- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:dllexportInlines - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:dllexportInlines -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:dllexportInlines -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:dllexportInlines -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:dllexportInlines- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:dllexportInlines- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:dllexportInlines- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:dllexportInlines- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:forScope - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:forScope -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:forScope -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:forScope -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:inline - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:inline -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:inline -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:inline -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:rvalueCast - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:rvalueCast -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:rvalueCast -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:rvalueCast -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:sizedDealloc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:sizedDealloc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:sizedDealloc -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:sizedDealloc -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:sizedDealloc- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:sizedDealloc- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:sizedDealloc- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:sizedDealloc- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:strictStrings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:strictStrings -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:strictStrings -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:strictStrings -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:ternary - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:ternary -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:ternary -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:ternary -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:threadSafeInit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:threadSafeInit -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:threadSafeInit -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:threadSafeInit -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:threadSafeInit- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:threadSafeInit- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:threadSafeInit- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:threadSafeInit- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:tlsGuards - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:tlsGuards -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:tlsGuards -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:tlsGuards -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:tlsGuards- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:tlsGuards- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:tlsGuards- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:tlsGuards- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:trigraphs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:trigraphs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:trigraphs -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:trigraphs -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:trigraphs- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:trigraphs- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:trigraphs- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:trigraphs- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:twoPhase - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:twoPhase -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:twoPhase -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:twoPhase -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:twoPhase- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:twoPhase- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:twoPhase- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:twoPhase- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:wchar_t - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:wchar_t -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:wchar_t -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:wchar_t -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zc:wchar_t- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zc:wchar_t- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zc:wchar_t- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zc:wchar_t- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Ze - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Ze -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Ze -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Ze -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zg - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zg -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zg -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zg -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zi -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: %clang /Zi -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zl -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zl -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zl -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zm -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zm -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zo - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zo -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zo -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zo -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zo- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zo- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zo- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zo- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zp -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zp -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zp -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zp -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Zs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Zs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Zs -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Zs -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /analyze- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /analyze- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /analyze- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /analyze- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /arch: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /arch: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /arch: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /arch: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /arm64EC - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /arm64EC -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /arm64EC -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /arm64EC -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /await - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /await -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /await -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /await -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /await: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /await: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /await: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /await: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /bigobj - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /bigobj -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /bigobj -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /c - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /c -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /c -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /c -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /cgthreads - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /cgthreads -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /cgthreads -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /cgthreads -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /clang: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /clang: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /clang: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /clang: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /clr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /clr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /clr -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /clr -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /constexpr: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /constexpr: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /constexpr: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /constexpr: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /d1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /d1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /d1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /d1PP - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /d1PP -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /d1PP -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /d1reportAllClassLayout - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /d1reportAllClassLayout -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /d1reportAllClassLayout -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /d2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /d2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /d2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /d2FastFail - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /d2FastFail -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /d2FastFail -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /d2Zi+ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /d2Zi+ -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /d2Zi+ -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /diagnostics:caret - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /diagnostics:caret -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /diagnostics:caret -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /diagnostics:classic - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /diagnostics:classic -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /diagnostics:classic -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /diagnostics:column - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /diagnostics:column -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /diagnostics:column -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /diasdkdir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /diasdkdir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /diasdkdir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /doc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /doc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /doc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /errorReport - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /errorReport -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /errorReport -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /errorReport -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /execution-charset: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /execution-charset: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /execution-charset: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /execution-charset: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /experimental: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /experimental: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /experimental: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /experimental: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /exportHeader - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /exportHeader -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /exportHeader -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /exportHeader -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /external: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /external: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /external: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /external: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /external:I - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /external:I -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /external:I -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /external:I -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /external:W0 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /external:W0 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /external:W0 -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /external:W0 -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /external:W1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /external:W1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /external:W1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /external:W1 -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /external:W2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /external:W2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /external:W2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /external:W2 -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /external:W3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /external:W3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /external:W3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /external:W3 -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /external:W4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /external:W4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /external:W4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /external:W4 -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /external:env: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /external:env: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /external:env: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /external:env: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /favor - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /favor -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /favor -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /favor -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /fno-sanitize-address-vcasan-lib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /fno-sanitize-address-vcasan-lib -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /fno-sanitize-address-vcasan-lib -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /fno-sanitize-address-vcasan-lib -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /fp:contract - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /fp:contract -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /fp:contract -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /fp:contract -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /fp:except - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /fp:except -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /fp:except -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /fp:except -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /fp:except- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /fp:except- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /fp:except- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /fp:except- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /fp:fast - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /fp:fast -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /fp:fast -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /fp:fast -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /fp:precise - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /fp:precise -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /fp:precise -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /fp:precise -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /fp:strict - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /fp:strict -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /fp:strict -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /fp:strict -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /fsanitize=address - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /fsanitize=address -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /fsanitize=address -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /fsanitize=address -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /fsanitize-address-use-after-return - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /fsanitize-address-use-after-return -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /fsanitize-address-use-after-return -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /fsanitize-address-use-after-return -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /guard: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /guard: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /guard: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /guard: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /headerUnit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /headerUnit -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /headerUnit -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /headerUnit -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /headerUnit:angle - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /headerUnit:angle -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /headerUnit:angle -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /headerUnit:angle -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /headerUnit:quote - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /headerUnit:quote -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /headerUnit:quote -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /headerUnit:quote -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /headerName: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /headerName: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /headerName: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /headerName: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /help - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /help -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: %clang /help -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /homeparams - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /homeparams -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /homeparams -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /homeparams -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /hotpatch - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /hotpatch -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /hotpatch -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /hotpatch -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /imsvc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /imsvc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /imsvc -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /imsvc -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /kernel - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /kernel -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /kernel -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /kernel -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /kernel- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /kernel- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /kernel- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /kernel- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /link - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /link -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /link -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /nologo - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /nologo -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /nologo -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /nologo -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang_dxc /o -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc /openmp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc /openmp- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc /openmp:experimental -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /permissive - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /permissive -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /permissive -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /permissive -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /permissive- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /permissive- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /permissive- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /permissive- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /reference - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /reference -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /reference -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /reference -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /sdl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /sdl -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /sdl -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /sdl -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /sdl- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /sdl- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /sdl- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /sdl- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /showFilenames - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /showFilenames -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /showFilenames -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /showFilenames -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /showFilenames- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /showFilenames- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /showFilenames- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /showFilenames- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /showIncludes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /showIncludes -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /showIncludes -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /showIncludes -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /showIncludes:user - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /showIncludes:user -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /showIncludes:user -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /showIncludes:user -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /sourceDependencies - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /sourceDependencies -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /sourceDependencies -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /sourceDependencies -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /sourceDependencies:directives - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /sourceDependencies:directives -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /sourceDependencies:directives -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /sourceDependencies:directives -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /source-charset: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /source-charset: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /source-charset: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /source-charset: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /std: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /std: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /std: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /std: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /translateInclude - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /translateInclude -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /translateInclude -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /translateInclude -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /tune: - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /tune: -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /tune: -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /tune: -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /u - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /u -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /u -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /utf-8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /utf-8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /utf-8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /validate-charset - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /validate-charset -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /validate-charset -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /validate-charset -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /validate-charset- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /validate-charset- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /validate-charset- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /validate-charset- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /vctoolsdir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /vctoolsdir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /vctoolsdir -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /vctoolsdir -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /vctoolsversion - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /vctoolsversion -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /vctoolsversion -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /vctoolsversion -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /vd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /vd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /vd -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /vd -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /vmb - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /vmb -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /vmb -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /vmb -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /vmg - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /vmg -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /vmg -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /vmg -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /vmm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /vmm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /vmm -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /vmm -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /vms - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /vms -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /vms -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /vms -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /vmv - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /vmv -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /vmv -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /vmv -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /volatile:iso - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /volatile:iso -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /volatile:iso -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /volatile:iso -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /volatile:ms - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /volatile:ms -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /volatile:ms -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /volatile:ms -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /w - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /w -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /w -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /w -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /w - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /w -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /w -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /w -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /wd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /wd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /wd -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /wd -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /winsdkdir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /winsdkdir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /winsdkdir -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /winsdkdir -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /winsdkversion - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /winsdkversion -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /winsdkversion -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /winsdkversion -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /winsysroot - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /winsysroot -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /winsysroot -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /winsysroot -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --all-warnings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --all-warnings -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --all-warnings -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --all-warnings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --analyze - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --analyze -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --analyze -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --analyzer-no-default-checks - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --analyzer-no-default-checks -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --analyzer-no-default-checks -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --analyzer-no-default-checks -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --analyzer-output - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --analyzer-output -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --analyzer-output -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --analyzer-output -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --assemble - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --assemble -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --assemble -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --assemble -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --assert - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --assert -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --assert -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --assert -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --assert= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --assert= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --assert= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --assert= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --bootclasspath - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --bootclasspath -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --bootclasspath -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --bootclasspath -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --bootclasspath= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --bootclasspath= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --bootclasspath= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --bootclasspath= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --classpath - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --classpath -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --classpath -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --classpath -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --classpath= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --classpath= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --classpath= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --classpath= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --comments - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --comments -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --comments -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --comments -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --comments-in-macros - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --comments-in-macros -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --comments-in-macros -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --comments-in-macros -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --compile - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --compile -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --compile -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --compile -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --constant-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --constant-cfstrings -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --constant-cfstrings -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --constant-cfstrings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --debug - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --debug -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --debug -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --debug -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --debug= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --debug= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --debug= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --debug= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --define-macro - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --define-macro -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --define-macro -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --define-macro -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --define-macro= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --define-macro= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --define-macro= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --define-macro= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --dependencies -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --dependencies -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --dependencies -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --dyld-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --dyld-prefix -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --dyld-prefix -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --dyld-prefix -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --dyld-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --dyld-prefix= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --dyld-prefix= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --dyld-prefix= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --encoding - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --encoding -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --encoding -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --encoding -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --encoding= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --encoding= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --encoding= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --encoding= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --entry - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --entry -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --entry -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --entry -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --extdirs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --extdirs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --extdirs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --extdirs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --extdirs= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --extdirs= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --extdirs= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --extdirs= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --extra-warnings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --extra-warnings -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --extra-warnings -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --extra-warnings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --for-linker - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --for-linker -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --for-linker -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --for-linker -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --for-linker= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --for-linker= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --for-linker= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --for-linker= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --force-link - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --force-link -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --force-link -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --force-link -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --force-link= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --force-link= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --force-link= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --force-link= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --help-hidden - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --help-hidden -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --help-hidden -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --help-hidden -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --imacros= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --imacros= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --imacros= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-barrier - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-barrier -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-barrier -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-directory - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-directory -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-directory -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-directory= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-directory= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-directory= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-directory-after - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-directory-after -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-directory-after -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-directory-after= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-directory-after= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-directory-after= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-prefix -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-prefix -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-prefix= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-prefix= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-with-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-with-prefix -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-with-prefix -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-with-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-with-prefix= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-with-prefix= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-with-prefix-after - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-with-prefix-after -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-with-prefix-after -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-with-prefix-after= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-with-prefix-after= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-with-prefix-after= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-with-prefix-before - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-with-prefix-before -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-with-prefix-before -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --include-with-prefix-before= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --include-with-prefix-before= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --include-with-prefix-before= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --language - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --language -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --language -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --language -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --language= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --language= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --language= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --language= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --library-directory - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --library-directory -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --library-directory -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --library-directory -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --library-directory= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --library-directory= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --library-directory= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --library-directory= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --mhwdiv - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --mhwdiv -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --mhwdiv -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --mhwdiv -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --mhwdiv= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --mhwdiv= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --mhwdiv= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --migrate - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --migrate -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --migrate -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --migrate -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-line-commands - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-line-commands -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --no-line-commands -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --no-line-commands -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-standard-includes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-standard-includes -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --no-standard-includes -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --no-standard-includes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-standard-libraries - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-standard-libraries -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --no-standard-libraries -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --no-standard-libraries -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-undefined - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-undefined -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --no-undefined -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --no-undefined -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-warnings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --no-warnings -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --no-warnings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --optimize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --optimize= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --output -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --output= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --output-class-directory -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --output-class-directory= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --param - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --param -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --param -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --param -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --param= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --param= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --param= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --param= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --precompile - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --precompile -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --precompile -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --prefix - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --prefix -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --prefix -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --prefix -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --prefix= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --prefix= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --prefix= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --preprocess - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --preprocess -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --preprocess -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --preprocess -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --print-diagnostic-categories - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --print-diagnostic-categories -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --print-diagnostic-categories -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --print-diagnostic-categories -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --print-file-name - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --print-file-name -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --print-file-name -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --print-file-name -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --print-missing-file-dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --print-missing-file-dependencies -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --print-missing-file-dependencies -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --print-missing-file-dependencies -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --print-prog-name - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --print-prog-name -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --print-prog-name -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --print-prog-name -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --profile - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --profile -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --profile -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --profile -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --resource - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --resource -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --resource -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --resource -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --resource= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --resource= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --resource= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --resource= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --rtlib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --rtlib -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --rtlib -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --rtlib -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -serialize-diagnostics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -serialize-diagnostics -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -serialize-diagnostics -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -serialize-diagnostics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --signed-char - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --signed-char -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --signed-char -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --signed-char -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --std - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --std -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --std -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --std -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --stdlib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --stdlib -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --stdlib -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --stdlib -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --sysroot - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --sysroot -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --sysroot -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --sysroot -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --sysroot= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --sysroot= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --sysroot= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --sysroot= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --target-help - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --target-help -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --target-help -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --target-help -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --trace-includes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --trace-includes -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --trace-includes -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --trace-includes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --undefine-macro - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --undefine-macro -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --undefine-macro -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --undefine-macro -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --undefine-macro= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --undefine-macro= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --undefine-macro= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --undefine-macro= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --unsigned-char - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --unsigned-char -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --unsigned-char -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --unsigned-char -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --user-dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --user-dependencies -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --user-dependencies -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --user-dependencies -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --verbose - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --verbose -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --verbose -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --verbose -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --version - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --version -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as --warn- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --warn- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --warn- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --warn-= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --warn-= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --warn-= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --write-dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --write-dependencies -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --write-dependencies -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --write-user-dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --write-user-dependencies -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --write-user-dependencies -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -add-plugin - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -add-plugin -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -add-plugin -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -add-plugin -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -alias_list - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -alias_list -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -alias_list -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -alias_list -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -faligned-alloc-unavailable - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -faligned-alloc-unavailable -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -faligned-alloc-unavailable -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -faligned-alloc-unavailable -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -all_load - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -all_load -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -all_load -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -all_load -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -allowable_client - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -allowable_client -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -allowable_client -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -allowable_client -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --amdgpu-arch-tool= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --amdgpu-arch-tool= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --amdgpu-arch-tool= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cfg-add-implicit-dtors - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cfg-add-implicit-dtors -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cfg-add-implicit-dtors -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cfg-add-implicit-dtors -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -unoptimized-cfg - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -unoptimized-cfg -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -unoptimized-cfg -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -analyze - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyze -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyze -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyze -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyze-function - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyze-function -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyze-function -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyze-function -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyze-function= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyze-function= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyze-function= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyze-function= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-checker - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-checker -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-checker -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-checker -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-checker= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-checker= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-checker= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-checker= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-checker-help - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-checker-help -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-checker-help -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-checker-help -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-checker-help-alpha - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-checker-help-alpha -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-checker-help-alpha -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-checker-help-alpha -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-checker-help-developer - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-checker-help-developer -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-checker-help-developer -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-checker-help-developer -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-checker-option-help - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-checker-option-help -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-checker-option-help -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-checker-option-help -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-checker-option-help-alpha - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-checker-option-help-alpha -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-checker-option-help-alpha -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-checker-option-help-alpha -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-checker-option-help-developer - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-checker-option-help-developer -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-checker-option-help-developer -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-checker-option-help-developer -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-config - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-config -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-config -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-config -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-config-compatibility-mode - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-config-compatibility-mode -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-config-compatibility-mode -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-config-compatibility-mode -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-config-compatibility-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-config-compatibility-mode= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-config-compatibility-mode= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-config-compatibility-mode= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-config-help - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-config-help -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-config-help -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-config-help -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-constraints - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-constraints -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-constraints -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-constraints -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-constraints= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-constraints= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-constraints= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-constraints= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-disable-all-checks - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-disable-all-checks -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-disable-all-checks -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-disable-all-checks -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-disable-checker - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-disable-checker -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-disable-checker -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-disable-checker -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-disable-checker= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-disable-checker= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-disable-checker= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-disable-checker= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-disable-retry-exhausted - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-disable-retry-exhausted -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-disable-retry-exhausted -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-disable-retry-exhausted -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-display-progress - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-display-progress -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-display-progress -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-display-progress -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-dump-egraph - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-dump-egraph -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-dump-egraph -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-dump-egraph -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-dump-egraph= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-dump-egraph= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-dump-egraph= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-dump-egraph= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-inline-max-stack-depth - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-inline-max-stack-depth -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-inline-max-stack-depth -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-inline-max-stack-depth -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-inline-max-stack-depth= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-inline-max-stack-depth= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-inline-max-stack-depth= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-inline-max-stack-depth= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-inlining-mode - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-inlining-mode -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-inlining-mode -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-inlining-mode -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-inlining-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-inlining-mode= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-inlining-mode= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-inlining-mode= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-list-enabled-checkers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-list-enabled-checkers -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-list-enabled-checkers -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-list-enabled-checkers -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-max-loop - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-max-loop -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-max-loop -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-max-loop -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-note-analysis-entry-points - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-note-analysis-entry-points -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-note-analysis-entry-points -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-note-analysis-entry-points -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-opt-analyze-headers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-opt-analyze-headers -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-opt-analyze-headers -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-opt-analyze-headers -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-output - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-output -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-output -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -analyzer-output= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-output= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-output= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -analyzer-purge - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-purge -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-purge -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-purge -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-purge= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-purge= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-purge= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-purge= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-stats - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-stats -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-stats -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-stats -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-viz-egraph-graphviz - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-viz-egraph-graphviz -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-viz-egraph-graphviz -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-viz-egraph-graphviz -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -analyzer-werror - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -analyzer-werror -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -analyzer-werror -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -analyzer-werror -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fnew-alignment - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fnew-alignment -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fnew-alignment -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fnew-alignment -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -faligned-new - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -faligned-new -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -faligned-new -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -faligned-new -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-aligned-new - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-aligned-new -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-aligned-new -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-aligned-new -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ftree-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ftree-vectorize -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ftree-vectorize -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ftree-vectorize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-tree-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-tree-vectorize -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-tree-vectorize -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-tree-vectorize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ftree-slp-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ftree-slp-vectorize -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ftree-slp-vectorize -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ftree-slp-vectorize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-tree-slp-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-tree-slp-vectorize -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-tree-slp-vectorize -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-tree-slp-vectorize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fterminated-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fterminated-vtables -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fterminated-vtables -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fterminated-vtables -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcuda-rdc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcuda-rdc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fcuda-rdc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-cuda-rdc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-cuda-rdc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-cuda-rdc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --hip-device-lib-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --hip-device-lib-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --hip-device-lib-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -grecord-gcc-switches - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -grecord-gcc-switches -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -grecord-gcc-switches -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -grecord-gcc-switches -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gno-record-gcc-switches - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gno-record-gcc-switches -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gno-record-gcc-switches -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gno-record-gcc-switches -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -miphoneos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -miphoneos-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -miphoneos-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -miphonesimulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -miphonesimulator-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -miphonesimulator-version-min= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -miphonesimulator-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mllvm= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mllvm= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -mmacosx-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmacosx-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mmacosx-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nocudainc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nocudainc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -nocudainc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nocudainc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nocudalib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nocudalib -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -nocudalib -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nocudalib -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-multiarch - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-multiarch -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -print-multiarch -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -print-multiarch -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --system-header-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --system-header-prefix -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --system-header-prefix -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --system-header-prefix -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-system-header-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-system-header-prefix -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --no-system-header-prefix -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --no-system-header-prefix -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcpu=help - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mcpu=help -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtune=help - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mtune=help -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -integrated-as - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -integrated-as -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -integrated-as -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -integrated-as -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -no-integrated-as - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-integrated-as -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-integrated-as -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -coverage-data-file= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -coverage-data-file= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -coverage-data-file= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -coverage-data-file= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -coverage-notes-file= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -coverage-notes-file= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -coverage-notes-file= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -coverage-notes-file= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fopenmp-is-device - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fopenmp-is-device -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fopenmp-is-device -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fopenmp-is-device -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-cuda-approx-transcendentals - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-cuda-approx-transcendentals -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-cuda-approx-transcendentals -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-cuda-approx-transcendentals -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Gs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Gs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Gs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /O1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /O2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Ob0 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Ob1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Ob2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Ob3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Od - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Og - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Oi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Oi- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Os - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Ot - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Ox - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Oy - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Oy- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Qgather- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qgather- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qgather- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qgather- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /Qscatter- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qscatter- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc /Qscatter- -### | FileCheck -check-prefix=DXCOption %s
// RUN: %clang /Qscatter- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -Xmicrosoft-visualc-tools-root - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xmicrosoft-visualc-tools-root -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xmicrosoft-visualc-tools-root -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xmicrosoft-visualc-tools-root -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xmicrosoft-visualc-tools-version - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xmicrosoft-visualc-tools-version -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xmicrosoft-visualc-tools-version -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xmicrosoft-visualc-tools-version -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xmicrosoft-windows-sdk-root - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xmicrosoft-windows-sdk-root -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xmicrosoft-windows-sdk-root -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xmicrosoft-windows-sdk-root -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xmicrosoft-windows-sdk-version - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xmicrosoft-windows-sdk-version -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xmicrosoft-windows-sdk-version -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xmicrosoft-windows-sdk-version -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xmicrosoft-windows-sys-root - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xmicrosoft-windows-sys-root -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xmicrosoft-windows-sys-root -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xmicrosoft-windows-sys-root -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Qembed_debug - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Qembed_debug -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl /Qembed_debug -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: %clang /Qembed_debug -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -shared-libasan - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -shared-libasan -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -shared-libasan -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -shared-libasan -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -static-libasan - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -static-libasan -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -static-libasan -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -static-libasan -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-whitelist-dir-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-white-list-dir-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fslp-vectorize-aggressive - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fslp-vectorize-aggressive -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fslp-vectorize-aggressive -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fslp-vectorize-aggressive -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-slp-vectorize-aggressive - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-slp-vectorize-aggressive -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-slp-vectorize-aggressive -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-slp-vectorize-aggressive -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -frecord-gcc-switches - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -frecord-gcc-switches -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -frecord-gcc-switches -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -frecord-gcc-switches -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-record-gcc-switches - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-record-gcc-switches -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-record-gcc-switches -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-record-gcc-switches -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xclang= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xclang= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -fexpensive-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fexpensive-optimizations -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fexpensive-optimizations -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fexpensive-optimizations -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-expensive-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-expensive-optimizations -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-expensive-optimizations -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-expensive-optimizations -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdefer-pop - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdefer-pop -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdefer-pop -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdefer-pop -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-defer-pop - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-defer-pop -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-defer-pop -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-defer-pop -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xparser - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xparser -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xparser -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xparser -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Xcompiler - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -Xcompiler -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -Xcompiler -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -Xcompiler -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-blacklist= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-blacklist= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-blacklist - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-blacklist -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fhonor-infinites - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fhonor-infinites -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fhonor-infinites -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fhonor-infinites -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-honor-infinites - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-honor-infinites -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-honor-infinites -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-honor-infinites -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -findirect-virtual-calls - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -findirect-virtual-calls -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -findirect-virtual-calls -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -findirect-virtual-calls -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --config - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --config -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -ansi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ansi -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ansi -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ansi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -arch - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -arch -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -arch -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -arch -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -arch_errors_fatal - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -arch_errors_fatal -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -arch_errors_fatal -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -arch_errors_fatal -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -arch_only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -arch_only -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -arch_only -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -arch_only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -arcmt-action= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -arcmt-action= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -arcmt-action= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -arcmt-action= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -arcmt-migrate-emit-errors - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -arcmt-migrate-emit-errors -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -arcmt-migrate-emit-errors -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -arcmt-migrate-report-output - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -arcmt-migrate-report-output -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -arcmt-migrate-report-output -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -as-secure-log-file -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -as-secure-log-file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -as-secure-log-file -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ast-dump - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ast-dump -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ast-dump -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ast-dump -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ast-dump= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ast-dump= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ast-dump= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ast-dump= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ast-dump-all - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ast-dump-all -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ast-dump-all -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ast-dump-all -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ast-dump-all= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ast-dump-all= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ast-dump-all= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ast-dump-all= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ast-dump-decl-types - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ast-dump-decl-types -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ast-dump-decl-types -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ast-dump-decl-types -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ast-dump-filter - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ast-dump-filter -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ast-dump-filter -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ast-dump-filter -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ast-dump-filter= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ast-dump-filter= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ast-dump-filter= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ast-dump-filter= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ast-dump-lookups - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ast-dump-lookups -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ast-dump-lookups -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ast-dump-lookups -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ast-list - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ast-list -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ast-list -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ast-list -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ast-merge - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ast-merge -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ast-merge -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ast-merge -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ast-print - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ast-print -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ast-print -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ast-print -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ast-view - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ast-view -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ast-view -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ast-view -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --autocomplete= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --autocomplete= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --autocomplete= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --autocomplete= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -aux-target-cpu - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -aux-target-cpu -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -aux-target-cpu -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -aux-target-cpu -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -aux-target-feature - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -aux-target-feature -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -aux-target-feature -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -aux-target-feature -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -aux-triple - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -aux-triple -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -aux-triple -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -aux-triple -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -b - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -b -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -b -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -b -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -bind_at_load - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -bind_at_load -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -bind_at_load -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -bind_at_load -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -building-pch-with-obj - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -building-pch-with-obj -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -building-pch-with-obj -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -bundle - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -bundle -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -bundle -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -bundle -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -bundle_loader - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -bundle_loader -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -bundle_loader -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -bundle_loader -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -c - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -c -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -c -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -c -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -c-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -c-isystem -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -c-isystem -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -c-isystem -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -canonical-prefixes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -canonical-prefixes -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -ccc- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ccc- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ccc- -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ccc- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ccc-arcmt-check - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ccc-arcmt-check -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ccc-arcmt-check -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ccc-arcmt-check -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ccc-arcmt-migrate - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ccc-arcmt-migrate -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ccc-arcmt-migrate -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ccc-arcmt-migrate -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ccc-arcmt-modify - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ccc-arcmt-modify -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ccc-arcmt-modify -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ccc-arcmt-modify -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ccc-gcc-name - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ccc-gcc-name -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ccc-gcc-name -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ccc-gcc-name -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ccc-install-dir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ccc-install-dir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -ccc-objcmt-migrate - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ccc-objcmt-migrate -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ccc-objcmt-migrate -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ccc-objcmt-migrate -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ccc-print-bindings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ccc-print-bindings -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -ccc-print-phases - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ccc-print-phases -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -cfguard - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cfguard -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cfguard -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cfguard -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -cfguard-no-checks - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cfguard-no-checks -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cfguard-no-checks -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cfguard-no-checks -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -chain-include - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -chain-include -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -chain-include -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -chain-include -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -cl-denorms-are-zero - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -cl-denorms-are-zero -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -cl-denorms-are-zero -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-denorms-are-zero -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-ext= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-ext= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-ext= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-fast-relaxed-math - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-fast-relaxed-math -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-fast-relaxed-math -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-finite-math-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-finite-math-only -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-finite-math-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-fp32-correctly-rounded-divide-sqrt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-fp32-correctly-rounded-divide-sqrt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-fp32-correctly-rounded-divide-sqrt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-kernel-arg-info - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-kernel-arg-info -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-kernel-arg-info -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-mad-enable - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-mad-enable -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-mad-enable -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-no-signed-zeros - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-no-signed-zeros -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-no-signed-zeros -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-no-stdinc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -cl-no-stdinc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -cl-no-stdinc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-no-stdinc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-opt-disable - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-opt-disable -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-opt-disable -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-single-precision-constant - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-single-precision-constant -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-single-precision-constant -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-std= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-std= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-std= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-strict-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-strict-aliasing -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-strict-aliasing -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-uniform-work-group-size - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-uniform-work-group-size -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-uniform-work-group-size -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cl-unsafe-math-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cl-unsafe-math-optimizations -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cl-unsafe-math-optimizations -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -clear-ast-before-backend - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -clear-ast-before-backend -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -clear-ast-before-backend -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -clear-ast-before-backend -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -client_name - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -client_name -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -client_name -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -client_name -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -code-completion-at - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -code-completion-at -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -code-completion-at -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -code-completion-at -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -code-completion-at= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -code-completion-at= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -code-completion-at= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -code-completion-at= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -code-completion-brief-comments - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -code-completion-brief-comments -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -code-completion-brief-comments -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -code-completion-brief-comments -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -code-completion-macros - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -code-completion-macros -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -code-completion-macros -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -code-completion-macros -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -code-completion-patterns - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -code-completion-patterns -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -code-completion-patterns -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -code-completion-patterns -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -code-completion-with-fixits - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -code-completion-with-fixits -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -code-completion-with-fixits -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -code-completion-with-fixits -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -combine - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -combine -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -combine -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -combine -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -compatibility_version - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -compatibility_version -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -compatibility_version -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -compatibility_version -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -compiler-options-dump - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -compiler-options-dump -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -compiler-options-dump -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -compiler-options-dump -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang_cl -compress-debug-sections -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -compress-debug-sections -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -compress-debug-sections -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang_cl -compress-debug-sections= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -compress-debug-sections= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -compress-debug-sections= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --config= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --config= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as --config-system-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --config-system-dir= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as --config-user-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --config-user-dir= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -coverage - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -coverage -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -coverage -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -coverage-version= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -coverage-version= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -coverage-version= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -coverage-version= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang_cl --crel -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --crel -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang --crel -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --cuda-compile-host-device - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --cuda-compile-host-device -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --cuda-compile-host-device -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --cuda-device-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --cuda-device-only -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --cuda-device-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --cuda-feature= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --cuda-feature= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --cuda-feature= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --cuda-gpu-arch= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --cuda-gpu-arch= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --cuda-gpu-arch= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --cuda-host-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --cuda-host-only -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --cuda-host-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --cuda-include-ptx= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --cuda-include-ptx= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --cuda-include-ptx= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --cuda-noopt-device-debug - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --cuda-noopt-device-debug -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --cuda-noopt-device-debug -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --cuda-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --cuda-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --cuda-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --cuda-path-ignore-env - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --cuda-path-ignore-env -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --cuda-path-ignore-env -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cuid= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -cuid= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -current_version - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -current_version -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -current_version -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -current_version -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -cxx-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -cxx-isystem -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -cxx-isystem -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dA - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -dA -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -dA -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dA -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dD - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dD -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dD -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dE - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dE -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dE -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dI - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dI -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dI -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dM - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dM -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dM -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -d - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -d -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -d -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -d -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -d - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -d -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -d -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -d -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -darwin-target-variant - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -darwin-target-variant -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -darwin-target-variant -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -darwin-target-variant-sdk-version= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -darwin-target-variant-sdk-version= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -darwin-target-variant-triple -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -darwin-target-variant-triple -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dead_strip - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -dead_strip -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -dead_strip -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dead_strip -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -debug-forward-template-params - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -debug-forward-template-params -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -debug-forward-template-params -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -debug-info-kind= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -debug-info-kind= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -debug-info-macro -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -debug-info-macro -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -debugger-tuning= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -debugger-tuning= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -default-function-attr -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -default-function-attr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1 --defsym -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --defsym -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --defsym -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dependency-dot - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dependency-dot -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dependency-dot -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dependency-file - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dependency-file -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dependency-file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --dependent-lib= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --dependent-lib= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --dependent-lib= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -detailed-preprocessing-record - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -detailed-preprocessing-record -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -detailed-preprocessing-record -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -diagnostic-log-file - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -diagnostic-log-file -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -diagnostic-log-file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -serialize-diagnostic-file - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -serialize-diagnostic-file -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -serialize-diagnostic-file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -serialize-diagnostic-file -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -disable-O0-optnone - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -disable-O0-optnone -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -disable-O0-optnone -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -disable-free - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -disable-free -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -disable-free -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -disable-lifetime-markers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -disable-lifetime-markers -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -disable-lifetime-markers -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -disable-llvm-optzns - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -disable-llvm-optzns -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -disable-llvm-optzns -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -disable-llvm-passes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -disable-llvm-passes -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -disable-llvm-passes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -disable-llvm-verifier - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -disable-llvm-verifier -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -disable-llvm-verifier -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -disable-objc-default-synthesize-properties - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -disable-objc-default-synthesize-properties -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -disable-objc-default-synthesize-properties -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -disable-pragma-debug-crash - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -disable-pragma-debug-crash -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -disable-pragma-debug-crash -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -disable-red-zone - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -disable-red-zone -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -disable-red-zone -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -discard-value-names - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -discard-value-names -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -discard-value-names -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --driver-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --driver-mode= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -dsym-dir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -dsym-dir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -dsym-dir -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dsym-dir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dump-coverage-mapping - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dump-coverage-mapping -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dump-coverage-mapping -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dump-deserialized-decls - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dump-deserialized-decls -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dump-deserialized-decls -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dump-raw-tokens - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dump-raw-tokens -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dump-raw-tokens -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dump-tokens - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dump-tokens -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dump-tokens -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dumpdir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dumpdir -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dumpdir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dumpmachine - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -dumpmachine -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -dumpmachine -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dumpmachine -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dumpspecs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -dumpspecs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -dumpspecs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dumpspecs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dumpversion - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -dumpversion -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -dumpversion -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dumpversion -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -dwarf-debug-flags -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dwarf-debug-flags -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1 -dwarf-debug-producer -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -dwarf-debug-producer -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dwarf-debug-producer -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dwarf-explicit-import - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dwarf-explicit-import -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dwarf-explicit-import -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dwarf-ext-refs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -dwarf-ext-refs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dwarf-ext-refs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -dwarf-version= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dwarf-version= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /Fc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Fo - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as /Vd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /Vd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl /Vd -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -cc1as --E - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --E -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --E -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang --E -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /HV - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /HV -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl /HV -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: %clang /HV -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /hlsl-no-stdinc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /hlsl-no-stdinc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl /hlsl-no-stdinc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: %clang /hlsl-no-stdinc -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --dxv-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --dxv-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --dxv-path= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -cc1as /validator-version - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl /validator-version -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -cc1as -dylib_file - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -dylib_file -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -dylib_file -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dylib_file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dylinker - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -dylinker -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -dylinker -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dylinker -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dylinker_install_name - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -dylinker_install_name -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -dylinker_install_name -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dylinker_install_name -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dynamic - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -dynamic -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -dynamic -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dynamic -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -dynamiclib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -dynamiclib -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -dynamiclib -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -dynamiclib -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -e - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -e -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -e -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -e -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ehcontguard - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ehcontguard -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ehcontguard -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ehcontguard -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --embed-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --embed-dir= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --embed-dir= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -emit-ast - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -emit-ast -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -emit-cir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-cir -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-cir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -emit-codegen-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-codegen-only -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-codegen-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-codegen-only -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --emit-extension-symbol-graphs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --emit-extension-symbol-graphs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --emit-extension-symbol-graphs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -emit-fir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -emit-fir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -emit-fir -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-fir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-fir -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -emit-header-unit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-header-unit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-header-unit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-header-unit -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -emit-hlfir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -emit-hlfir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -emit-hlfir -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-hlfir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-hlfir -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -emit-html - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-html -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-html -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-html -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -emit-interface-stubs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-interface-stubs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-interface-stubs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -emit-llvm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-llvm -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-llvm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -emit-llvm-bc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-llvm-bc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-llvm-bc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-llvm-bc -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -emit-llvm-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-llvm-only -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-llvm-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-llvm-only -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -emit-llvm-uselists - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-llvm-uselists -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-llvm-uselists -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-llvm-uselists -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -emit-merged-ifs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-merged-ifs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-merged-ifs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -emit-mlir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -emit-mlir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -emit-mlir -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-mlir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-mlir -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -emit-module - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-module -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-module -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-module -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -emit-module-interface - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-module-interface -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-module-interface -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-module-interface -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -emit-obj - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-obj -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-obj -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-obj -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -emit-pch - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-pch -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-pch -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-pch -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --pretty-sgf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --pretty-sgf -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --pretty-sgf -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /emit-pristine-llvm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /emit-pristine-llvm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl /emit-pristine-llvm -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: %clang /emit-pristine-llvm -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -emit-reduced-module-interface - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-reduced-module-interface -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-reduced-module-interface -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -emit-reduced-module-interface -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --emit-sgf-symbol-labels-for-testing - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --emit-sgf-symbol-labels-for-testing -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --emit-sgf-symbol-labels-for-testing -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --emit-static-lib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --emit-static-lib -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --emit-static-lib -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --emit-static-lib -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -emit-symbol-graph - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -emit-symbol-graph -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -emit-symbol-graph -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /enable-16bit-types - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /enable-16bit-types -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl /enable-16bit-types -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: %clang /enable-16bit-types -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -enable-noundef-analysis - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -enable-noundef-analysis -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -enable-noundef-analysis -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -enable-noundef-analysis -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -enable-tlsdesc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -enable-tlsdesc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -enable-tlsdesc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -enable-tlsdesc -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --end-no-unused-arguments - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --end-no-unused-arguments -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -error-on-deserialized-decl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -error-on-deserialized-decl -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -error-on-deserialized-decl -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -error-on-deserialized-decl -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -error-on-deserialized-decl= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -error-on-deserialized-decl= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -error-on-deserialized-decl= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -error-on-deserialized-decl= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -exception-model - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -exception-model -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -exception-model -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -exception-model -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -exception-model= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -exception-model= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -exception-model= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -exception-model= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -exported_symbols_list - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -exported_symbols_list -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -exported_symbols_list -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -exported_symbols_list -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -extract-api - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -extract-api -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -extract-api -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --extract-api-ignores= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --extract-api-ignores= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --extract-api-ignores= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -faapcs-bitfield-width - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -faapcs-bitfield-width -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -faapcs-bitfield-width -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -faddress-space-map-mangling= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -faddress-space-map-mangling= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -faddress-space-map-mangling= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -faddress-space-map-mangling= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -faggressive-function-elimination - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -faggressive-function-elimination -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -faggressive-function-elimination -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -faggressive-function-elimination -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -falign-commons - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -falign-commons -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -falign-commons -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -falign-commons -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -falign-jumps - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -falign-jumps -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -falign-jumps -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -falign-jumps -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -falign-jumps= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -falign-jumps= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -falign-jumps= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -falign-jumps= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -falign-labels - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -falign-labels -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -falign-labels -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -falign-labels -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -falign-labels= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -falign-labels= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -falign-labels= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -falign-labels= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -falign-loops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -falign-loops -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -falign-loops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -falign-loops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -faligned-new= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -faligned-new= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -faligned-new= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -faligned-new= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fall-intrinsics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fall-intrinsics -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fall-intrinsics -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fall-intrinsics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fallow-pch-with-different-modules-cache-path - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fallow-pch-with-different-modules-cache-path -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fallow-pch-with-different-modules-cache-path -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fallow-pch-with-different-modules-cache-path -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fallow-pch-with-compiler-errors - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fallow-pch-with-compiler-errors -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fallow-pch-with-compiler-errors -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fallow-pch-with-compiler-errors -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fallow-pcm-with-compiler-errors - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fallow-pcm-with-compiler-errors -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fallow-pcm-with-compiler-errors -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fallow-pcm-with-compiler-errors -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fkeep-inline-functions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fkeep-inline-functions -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fkeep-inline-functions -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fkeep-inline-functions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -funit-at-a-time - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -funit-at-a-time -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -funit-at-a-time -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -funit-at-a-time -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fapinotes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fapinotes -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fapinotes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fapinotes-modules - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fapinotes-modules -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fapinotes-modules -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fapinotes-swift-version= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fapinotes-swift-version= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fapinotes-swift-version= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fapply-global-visibility-to-externs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fapply-global-visibility-to-externs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fapply-global-visibility-to-externs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fapply-global-visibility-to-externs -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fauto-profile= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fauto-profile= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fauto-profile= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fauto-profile= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fautomatic - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fautomatic -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fautomatic -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fautomatic -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fbacktrace - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fbacktrace -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fbacktrace -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fbacktrace -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fblas-matmul-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fblas-matmul-limit= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fblas-matmul-limit= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fblas-matmul-limit= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fblocks-runtime-optional - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fblocks-runtime-optional -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fblocks-runtime-optional -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fblocks-runtime-optional -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fbounds-check - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fbounds-check -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fbounds-check -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fbounds-check -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fbracket-depth - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fbracket-depth -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fbracket-depth -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fbracket-depth -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fbranch-count-reg - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fbranch-count-reg -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fbranch-count-reg -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fbranch-count-reg -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fbuild-session-file= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fbuild-session-file= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fbuild-session-file= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fbuild-session-file= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fbuild-session-timestamp= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fbuild-session-timestamp= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fbuild-session-timestamp= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcall-saved-x10 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcall-saved-x10 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcall-saved-x10 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcall-saved-x10 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcall-saved-x11 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcall-saved-x11 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcall-saved-x11 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcall-saved-x11 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcall-saved-x12 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcall-saved-x12 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcall-saved-x12 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcall-saved-x12 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcall-saved-x13 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcall-saved-x13 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcall-saved-x13 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcall-saved-x13 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcall-saved-x14 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcall-saved-x14 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcall-saved-x14 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcall-saved-x14 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcall-saved-x15 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcall-saved-x15 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcall-saved-x15 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcall-saved-x15 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcall-saved-x18 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcall-saved-x18 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcall-saved-x18 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcall-saved-x18 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcall-saved-x8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcall-saved-x8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcall-saved-x8 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcall-saved-x8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcall-saved-x9 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcall-saved-x9 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcall-saved-x9 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcall-saved-x9 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcaller-saves - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcaller-saves -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcaller-saves -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcaller-saves -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /fcgl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /fcgl -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl /fcgl -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: %clang /fcgl -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fcheck= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcheck= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcheck= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcheck= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcheck-array-temporaries - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcheck-array-temporaries -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcheck-array-temporaries -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcheck-array-temporaries -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcheck-new - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fcheck-new -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcheck-new -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fclang-abi-compat= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fclang-abi-compat= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fclang-abi-compat= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcoarray= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcoarray= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcoarray= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcoarray= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcomment-block-commands= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fcomment-block-commands= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcomment-block-commands= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcompatibility-qualified-id-block-type-checking - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fcompatibility-qualified-id-block-type-checking -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcompatibility-qualified-id-block-type-checking -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fcompatibility-qualified-id-block-type-checking -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fcomplete-member-pointers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fcomplete-member-pointers -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fconst-strings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fconst-strings -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fconst-strings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fconst-strings -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fconstant-string-class - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fconstant-string-class -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fconstant-string-class -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fconstant-string-class -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fconvergent-functions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fconvergent-functions -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fconvergent-functions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcrash-diagnostics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcrash-diagnostics -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -fcrash-diagnostics= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcrash-diagnostics= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -fcrash-diagnostics-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcrash-diagnostics-dir= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -fcray-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcray-pointer -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fcray-pointer -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcray-pointer -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fctor-dtor-return-this - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fctor-dtor-return-this -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fctor-dtor-return-this -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fctor-dtor-return-this -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fcuda-allow-variadic-functions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fcuda-allow-variadic-functions -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcuda-allow-variadic-functions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fcuda-allow-variadic-functions -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fcuda-flush-denormals-to-zero - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fcuda-flush-denormals-to-zero -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fcuda-flush-denormals-to-zero -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcuda-include-gpubinary - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fcuda-include-gpubinary -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcuda-include-gpubinary -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fcuda-include-gpubinary -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fcuda-is-device - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fcuda-is-device -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcuda-is-device -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fcuda-is-device -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fcuda-short-ptr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fcuda-short-ptr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcx-fortran-rules - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fcx-fortran-rules -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcx-fortran-rules -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fcx-limited-range - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fcx-limited-range -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fcx-limited-range -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fc++-abi= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fc++-abi= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fc++-abi= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fd-lines-as-code - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fd-lines-as-code -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fd-lines-as-code -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fd-lines-as-code -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fd-lines-as-comments - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fd-lines-as-comments -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fd-lines-as-comments -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fd-lines-as-comments -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdebug-dump-all - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-dump-all -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-dump-all -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-dump-all -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-dump-all -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-dump-parse-tree - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-dump-parse-tree -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-dump-parse-tree -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-dump-parse-tree -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-dump-parse-tree -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-dump-parse-tree-no-sema - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-dump-parse-tree-no-sema -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-dump-parse-tree-no-sema -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-dump-parse-tree-no-sema -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-dump-parse-tree-no-sema -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-dump-parsing-log - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-dump-parsing-log -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-dump-parsing-log -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-dump-parsing-log -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-dump-parsing-log -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-dump-pft - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-dump-pft -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-dump-pft -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-dump-pft -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-dump-pft -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-dump-provenance - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-dump-provenance -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-dump-provenance -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-dump-provenance -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-dump-provenance -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-dump-symbols - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-dump-symbols -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-dump-symbols -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-dump-symbols -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-dump-symbols -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-measure-parse-tree - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-measure-parse-tree -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-measure-parse-tree -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-measure-parse-tree -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-measure-parse-tree -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-module-writer - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-module-writer -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-module-writer -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-module-writer -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-module-writer -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-pass-manager - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdebug-pass-manager -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-pass-manager -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-pass-manager -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-pre-fir-tree - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-pre-fir-tree -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-pre-fir-tree -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-pre-fir-tree -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-pre-fir-tree -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-unparse - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-unparse -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-unparse -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-unparse -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-unparse -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-unparse-no-sema - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-unparse-no-sema -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-unparse-no-sema -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-unparse-no-sema -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-unparse-no-sema -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-unparse-with-modules - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-unparse-with-modules -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-unparse-with-modules -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-unparse-with-modules -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-unparse-with-modules -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebug-unparse-with-symbols - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdebug-unparse-with-symbols -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdebug-unparse-with-symbols -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebug-unparse-with-symbols -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebug-unparse-with-symbols -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebugger-cast-result-to-id - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdebugger-cast-result-to-id -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebugger-cast-result-to-id -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebugger-cast-result-to-id -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebugger-objc-literal - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdebugger-objc-literal -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebugger-objc-literal -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebugger-objc-literal -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdebugger-support - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdebugger-support -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdebugger-support -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdebugger-support -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdeclare-opencl-builtins - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdeclare-opencl-builtins -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdeclare-opencl-builtins -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdeclare-opencl-builtins -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdeclspec - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdeclspec -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdeclspec -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdefault-calling-conv= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdefault-calling-conv= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdefault-calling-conv= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdefault-calling-conv= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdefault-inline - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdefault-inline -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdefault-inline -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdefault-inline -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdepfile-entry= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdepfile-entry= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdepfile-entry= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdeprecated-macro - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdeprecated-macro -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdeprecated-macro -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdeprecated-macro -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdevirtualize - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdevirtualize -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdevirtualize -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdevirtualize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdevirtualize-speculatively - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdevirtualize-speculatively -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdevirtualize-speculatively -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdevirtualize-speculatively -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdiagnostics-fixit-info - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdiagnostics-fixit-info -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdiagnostics-fixit-info -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdiagnostics-fixit-info -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdiagnostics-format - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdiagnostics-format -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdiagnostics-format -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdiagnostics-format -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdiagnostics-format= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdiagnostics-format= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdiagnostics-format= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdiagnostics-format= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdiagnostics-parseable-fixits - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -fdiagnostics-print-source-range-info - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdiagnostics-print-source-range-info -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdiagnostics-print-source-range-info -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdiagnostics-show-category - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdiagnostics-show-category -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdiagnostics-show-category -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdiagnostics-show-category -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdiagnostics-show-category= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdiagnostics-show-category= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdiagnostics-show-category= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdiagnostics-show-category= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdisable-module-hash - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdisable-module-hash -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdisable-module-hash -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdisable-module-hash -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdiscard-value-names - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdiscard-value-names -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdiscard-value-names -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -cc1as -fdollar-ok - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdollar-ok -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdollar-ok -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdollar-ok -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdriver-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdriver-only -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -fdump-fortran-optimized - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdump-fortran-optimized -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdump-fortran-optimized -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdump-fortran-optimized -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdump-fortran-original - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdump-fortran-original -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdump-fortran-original -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdump-fortran-original -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdump-parse-tree - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fdump-parse-tree -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fdump-parse-tree -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdump-parse-tree -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fdump-record-layouts - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdump-record-layouts -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdump-record-layouts -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdump-record-layouts -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdump-record-layouts-canonical - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdump-record-layouts-canonical -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdump-record-layouts-canonical -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdump-record-layouts-canonical -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdump-record-layouts-complete - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdump-record-layouts-complete -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdump-record-layouts-complete -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdump-record-layouts-complete -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdump-record-layouts-simple - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdump-record-layouts-simple -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdump-record-layouts-simple -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdump-record-layouts-simple -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fdump-vtable-layouts - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fdump-vtable-layouts -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fdump-vtable-layouts -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fdump-vtable-layouts -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fembed-bitcode-marker - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fembed-bitcode-marker -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fembed-bitcode-marker -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fembed-bitcode-marker -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fencode-extended-block-signature - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fencode-extended-block-signature -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fencode-extended-block-signature -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fencode-extended-block-signature -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ferror-limit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ferror-limit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ferror-limit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ferror-limit -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fexperimental-isel - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fexperimental-isel -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fexperimental-isel -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fexperimental-isel -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fexperimental-relative-c++-abi-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fexperimental-relative-c++-abi-vtables -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fexperimental-relative-c++-abi-vtables -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fexperimental-sanitize-metadata=atomics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fexperimental-sanitize-metadata=atomics -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fexperimental-sanitize-metadata=atomics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fexperimental-sanitize-metadata=covered - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fexperimental-sanitize-metadata=covered -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fexperimental-sanitize-metadata=covered -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fexperimental-sanitize-metadata=uar - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fexperimental-sanitize-metadata=uar -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fexperimental-sanitize-metadata=uar -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fexperimental-strict-floating-point - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fexperimental-strict-floating-point -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fexperimental-strict-floating-point -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fexternal-blas - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fexternal-blas -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fexternal-blas -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fexternal-blas -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fexternc-nounwind - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fexternc-nounwind -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fexternc-nounwind -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fexternc-nounwind -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ff2c - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ff2c -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ff2c -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ff2c -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffake-address-space-map - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ffake-address-space-map -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffake-address-space-map -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ffake-address-space-map -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fimplicit-modules-use-lock - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fimplicit-modules-use-lock -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fimplicit-modules-use-lock -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fimplicit-modules-use-lock -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ffine-grained-bitfield-accesses - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ffine-grained-bitfield-accesses -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffine-grained-bitfield-accesses -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffinite-math-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ffinite-math-only -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffinite-math-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -finline-limit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -finline-limit -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -finline-limit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finline-limit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-a0 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-a0 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-a0 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-a0 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-a1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-a1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-a1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-a1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-a2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-a2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-a2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-a2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-a3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-a3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-a3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-a3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-a4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-a4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-a4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-a4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-a5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-a5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-a5 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-a5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-a6 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-a6 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-a6 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-a6 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-d0 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-d0 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-d0 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-d0 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-d1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-d1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-d1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-d1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-d2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-d2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-d2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-d2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-d3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-d3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-d3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-d3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-d4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-d4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-d4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-d4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-d5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-d5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-d5 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-d5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-d6 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-d6 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-d6 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-d6 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-d7 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-d7 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-d7 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-d7 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-g1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-g1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-g1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-g1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-g2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-g2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-g2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-g2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-g3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-g3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-g3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-g3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-g4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-g4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-g4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-g4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-g5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-g5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-g5 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-g5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-g6 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-g6 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-g6 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-g6 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-g7 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-g7 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-g7 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-g7 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-i0 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-i0 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-i0 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-i0 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-i1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-i1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-i1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-i1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-i2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-i2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-i2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-i2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-i3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-i3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-i3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-i3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-i4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-i4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-i4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-i4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-i5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-i5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-i5 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-i5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-l0 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-l0 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-l0 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-l0 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-l1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-l1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-l1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-l1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-l2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-l2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-l2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-l2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-l3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-l3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-l3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-l3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-l4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-l4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-l4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-l4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-l5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-l5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-l5 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-l5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-l6 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-l6 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-l6 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-l6 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-l7 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-l7 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-l7 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-l7 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-o0 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-o0 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-o0 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-o0 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-o1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-o1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-o1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-o1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-o2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-o2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-o2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-o2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-o3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-o3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-o3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-o3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-o4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-o4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-o4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-o4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-o5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-o5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-o5 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-o5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-r9 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-r9 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffixed-r9 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffixed-r9 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x10 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x10 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x10 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x11 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x11 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x11 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x12 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x12 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x12 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x13 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x13 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x13 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x14 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x14 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x14 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x15 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x15 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x15 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x17 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x17 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x17 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x18 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x18 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x18 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x19 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x19 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x19 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x20 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x20 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x20 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x21 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x21 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x21 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x22 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x22 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x22 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x23 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x23 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x23 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x24 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x24 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x24 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x25 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x25 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x25 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x26 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x26 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x26 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x27 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x27 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x27 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x28 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x28 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x28 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x29 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x29 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x29 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x30 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x30 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x30 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x31 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x31 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x31 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x6 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x6 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x6 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x7 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x7 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x7 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffixed-x9 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffixed-x9 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -ffixed-x9 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffloat-store - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffloat-store -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffloat-store -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffloat-store -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fforbid-guard-variables - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fforbid-guard-variables -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fforbid-guard-variables -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fforbid-guard-variables -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ffpe-trap= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffpe-trap= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffpe-trap= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffpe-trap= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffree-line-length- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffree-line-length- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffree-line-length- -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffree-line-length- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffrontend-optimize - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ffrontend-optimize -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ffrontend-optimize -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ffrontend-optimize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ffuchsia-api-level= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -ffuchsia-api-level= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fgcse - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fgcse -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fgcse -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fgcse -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fgcse-after-reload - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fgcse-after-reload -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fgcse-after-reload -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fgcse-after-reload -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fgcse-las - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fgcse-las -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fgcse-las -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fgcse-las -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fgcse-sm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fgcse-sm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fgcse-sm -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fgcse-sm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fget-definition - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fget-definition -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fget-definition -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fget-definition -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fget-definition -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fget-symbols-sources - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fget-symbols-sources -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fget-symbols-sources -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fget-symbols-sources -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fget-symbols-sources -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fglobal-isel - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fglobal-isel -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fglobal-isel -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fglobal-isel -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fgpu-allow-device-init - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fgpu-allow-device-init -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fgpu-default-stream= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fgpu-default-stream= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fgpu-defer-diag - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fgpu-defer-diag -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fgpu-exclude-wrong-side-overloads - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fgpu-exclude-wrong-side-overloads -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fgpu-flush-denormals-to-zero - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fgpu-flush-denormals-to-zero -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fgpu-flush-denormals-to-zero -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fgpu-inline-threshold= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fgpu-inline-threshold= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fgpu-inline-threshold= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fgpu-rdc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fgpu-rdc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fgpu-sanitize - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fgpu-sanitize -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fgpu-sanitize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fhalf-no-semantic-interposition - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fhalf-no-semantic-interposition -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fhalf-no-semantic-interposition -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fhalf-no-semantic-interposition -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fhip-dump-offload-linker-script - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fhip-dump-offload-linker-script -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fhip-dump-offload-linker-script -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fhip-emit-relocatable - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fhip-emit-relocatable -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fhip-emit-relocatable -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fhip-fp32-correctly-rounded-divide-sqrt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fhip-fp32-correctly-rounded-divide-sqrt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fhip-fp32-correctly-rounded-divide-sqrt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fhip-kernel-arg-name - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fhip-kernel-arg-name -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fhip-new-launch-api - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fhip-new-launch-api -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fhlsl-strict-availability - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fhlsl-strict-availability -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fhlsl-strict-availability -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -filelist - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -filelist -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -filelist -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -filelist -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1 -filetype -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -filetype -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -filetype -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -filetype -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -finclude-default-header - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -finclude-default-header -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finclude-default-header -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -finclude-default-header -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -finit-character= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -finit-character= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -finit-character= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finit-character= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -finit-integer= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -finit-integer= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -finit-integer= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finit-integer= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -finit-local-zero - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -finit-local-zero -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -finit-local-zero -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finit-local-zero -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -finit-logical= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -finit-logical= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -finit-logical= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finit-logical= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -finit-real= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -finit-real= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -finit-real= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finit-real= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -finline-functions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -finline-functions -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finline-functions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -finline-functions-called-once - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -finline-functions-called-once -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -finline-functions-called-once -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finline-functions-called-once -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -finline-hint-functions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -finline-hint-functions -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finline-hint-functions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -finline-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -finline-limit= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -finline-limit= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finline-limit= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -finline-small-functions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -finline-small-functions -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -finline-small-functions -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finline-small-functions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -finteger-4-integer-8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -finteger-4-integer-8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -finteger-4-integer-8 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -finteger-4-integer-8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fipa-cp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fipa-cp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fipa-cp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fipa-cp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fivopts - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fivopts -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fivopts -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fivopts -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fix-only-warnings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fix-only-warnings -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fix-only-warnings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fix-only-warnings -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fix-what-you-can - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fix-what-you-can -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fix-what-you-can -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fix-what-you-can -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fixit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fixit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fixit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fixit -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fixit= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fixit= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fixit= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fixit= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fixit-recompile - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fixit-recompile -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fixit-recompile -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fixit-recompile -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fixit-to-temporary - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fixit-to-temporary -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fixit-to-temporary -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fixit-to-temporary -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -flang-deprecated-no-hlfir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -flang-deprecated-no-hlfir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -flang-deprecated-no-hlfir -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -flang-deprecated-no-hlfir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -flang-deprecated-no-hlfir -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -flang-experimental-hlfir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -flang-experimental-hlfir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -flang-experimental-hlfir -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -flang-experimental-hlfir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -flang-experimental-hlfir -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -flang-experimental-integer-overflow - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -flang-experimental-integer-overflow -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -flang-experimental-integer-overflow -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -flang-experimental-integer-overflow -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -flang-experimental-integer-overflow -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -flat_namespace - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -flat_namespace -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -flat_namespace -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -flat_namespace -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -flimit-debug-info - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -flimit-debug-info -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -fversion-loops-for-stride - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fversion-loops-for-stride -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fversion-loops-for-stride -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fversion-loops-for-stride -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fversion-loops-for-stride -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -flto-unit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -flto-unit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -flto-unit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -flto-unit -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -flto-visibility-public-std - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -flto-visibility-public-std -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -flto-visibility-public-std -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -flto-visibility-public-std -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fmax-array-constructor= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fmax-array-constructor= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fmax-array-constructor= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmax-array-constructor= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmax-errors= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fmax-errors= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fmax-errors= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmax-errors= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmax-identifier-length - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fmax-identifier-length -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fmax-identifier-length -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmax-identifier-length -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmax-stack-var-size= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fmax-stack-var-size= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fmax-stack-var-size= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmax-stack-var-size= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmax-subrecord-length= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fmax-subrecord-length= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fmax-subrecord-length= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmax-subrecord-length= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmerge-constants - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fmerge-constants -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fmerge-constants -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmerge-constants -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmerge-functions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmerge-functions -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmerge-functions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fmerge-functions -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fmodule-feature - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodule-feature -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodule-feature -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fmodule-feature -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fmodule-file= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fmodule-file= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodule-file-home-is-cwd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodule-file-home-is-cwd -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodule-file-home-is-cwd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fmodule-file-home-is-cwd -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fmodule-format= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodule-format= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodule-format= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fmodule-format= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fmodule-implementation-of - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fmodule-implementation-of -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodule-map-file-home-is-cwd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodule-map-file-home-is-cwd -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodule-map-file-home-is-cwd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fmodule-map-file-home-is-cwd -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fmodule-maps - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fmodule-maps -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fmodule-maps -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodule-output - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fmodule-output -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodule-output= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fmodule-output= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodule-private - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fmodule-private -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fmodule-private -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodule-private -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodules-cache-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-cache-path= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-cache-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodules-codegen - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-codegen -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-codegen -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fmodules-codegen -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fmodules-debuginfo - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-debuginfo -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-debuginfo -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fmodules-debuginfo -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fmodules-disable-diagnostic-validation - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-disable-diagnostic-validation -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-disable-diagnostic-validation -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodules-embed-all-files - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fmodules-embed-all-files -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodules-embed-file= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-embed-file= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-embed-file= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fmodules-embed-file= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fmodules-hash-content - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-hash-content -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-hash-content -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fmodules-hash-content -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fmodules-local-submodule-visibility - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-local-submodule-visibility -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-local-submodule-visibility -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fmodules-local-submodule-visibility -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fmodules-prune-after= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-prune-after= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-prune-after= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodules-prune-interval= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-prune-interval= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-prune-interval= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodules-strict-context-hash - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-strict-context-hash -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-strict-context-hash -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fmodules-strict-context-hash -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fmodules-user-build-path - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-user-build-path -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-user-build-path -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodules-validate-once-per-build-session - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-validate-once-per-build-session -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-validate-once-per-build-session -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodules-validate-system-headers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fmodules-validate-system-headers -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodules-validate-system-headers -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodulo-sched - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fmodulo-sched -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fmodulo-sched -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodulo-sched -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fmodulo-sched-allow-regmoves - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fmodulo-sched-allow-regmoves -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fmodulo-sched-allow-regmoves -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fmodulo-sched-allow-regmoves -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fnative-half-arguments-and-returns - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fnative-half-arguments-and-returns -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fnative-half-arguments-and-returns -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fnative-half-arguments-and-returns -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fnative-half-type - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fnative-half-type -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fnative-half-type -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fnative-half-type -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-aapcs-bitfield-width - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-aapcs-bitfield-width -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-aapcs-bitfield-width -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-aggressive-function-elimination - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-aggressive-function-elimination -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-aggressive-function-elimination -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-aggressive-function-elimination -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-align-commons - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-align-commons -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-align-commons -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-align-commons -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-align-jumps - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-align-jumps -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-align-jumps -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-align-jumps -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-align-labels - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-align-labels -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-align-labels -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-align-labels -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-align-loops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-align-loops -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-align-loops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-align-loops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-all-intrinsics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-all-intrinsics -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-all-intrinsics -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-all-intrinsics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-keep-inline-functions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-keep-inline-functions -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-keep-inline-functions -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-keep-inline-functions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-unit-at-a-time - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-unit-at-a-time -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-unit-at-a-time -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-unit-at-a-time -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-apinotes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-apinotes -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-apinotes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-apinotes-modules - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-apinotes-modules -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-apinotes-modules -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-backtrace - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-backtrace -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-backtrace -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-backtrace -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-bitfield-type-align - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-bitfield-type-align -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-bitfield-type-align -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-bitfield-type-align -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-bounds-check - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-bounds-check -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-bounds-check -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-bounds-check -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-branch-count-reg - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-branch-count-reg -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-branch-count-reg -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-branch-count-reg -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-caller-saves - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-caller-saves -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-caller-saves -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-caller-saves -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-check-array-temporaries - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-check-array-temporaries -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-check-array-temporaries -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-check-array-temporaries -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-check-new - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-check-new -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-check-new -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-complete-member-pointers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-complete-member-pointers -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-complete-member-pointers -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-const-strings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-const-strings -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-const-strings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-const-strings -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-convergent-functions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-convergent-functions -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-convergent-functions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-crash-diagnostics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-crash-diagnostics -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -fno-cray-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-cray-pointer -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-cray-pointer -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-cray-pointer -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-cuda-flush-denormals-to-zero - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-cuda-flush-denormals-to-zero -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-cuda-flush-denormals-to-zero -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-cuda-host-device-constexpr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-cuda-host-device-constexpr -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-cuda-host-device-constexpr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-cuda-host-device-constexpr -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-cuda-short-ptr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-cuda-short-ptr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-cuda-short-ptr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-cx-fortran-rules - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-cx-fortran-rules -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-cx-fortran-rules -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-cx-limited-range - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-cx-limited-range -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-cx-limited-range -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-d-lines-as-code - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-d-lines-as-code -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-d-lines-as-code -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-d-lines-as-code -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-d-lines-as-comments - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-d-lines-as-comments -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-d-lines-as-comments -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-d-lines-as-comments -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-debug-pass-manager - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-debug-pass-manager -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-debug-pass-manager -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-debug-pass-manager -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-declspec - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-declspec -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-declspec -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-default-inline - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-default-inline -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-default-inline -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-default-inline -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-deprecated-macro - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-deprecated-macro -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-deprecated-macro -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-deprecated-macro -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-devirtualize - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-devirtualize -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-devirtualize -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-devirtualize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-devirtualize-speculatively - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-devirtualize-speculatively -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-devirtualize-speculatively -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-devirtualize-speculatively -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-diagnostics-use-presumed-location - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-diagnostics-use-presumed-location -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-diagnostics-use-presumed-location -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-diagnostics-use-presumed-location -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-discard-value-names - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-discard-value-names -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-discard-value-names -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -cc1as -fno-dllexport-inlines - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-dllexport-inlines -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-dllexport-inlines -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-dllexport-inlines -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-dollar-ok - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-dollar-ok -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-dollar-ok -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-dollar-ok -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-dump-fortran-optimized - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-dump-fortran-optimized -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-dump-fortran-optimized -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-dump-fortran-optimized -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-dump-fortran-original - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-dump-fortran-original -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-dump-fortran-original -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-dump-fortran-original -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-dump-parse-tree - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-dump-parse-tree -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-dump-parse-tree -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-dump-parse-tree -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-experimental-isel - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-experimental-isel -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-experimental-isel -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-experimental-isel -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-experimental-relative-c++-abi-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-experimental-relative-c++-abi-vtables -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-experimental-relative-c++-abi-vtables -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-external-blas - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-external-blas -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-external-blas -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-external-blas -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-f2c - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-f2c -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-f2c -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-f2c -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-implicit-modules-use-lock - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-implicit-modules-use-lock -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-implicit-modules-use-lock -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-implicit-modules-use-lock -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-fine-grained-bitfield-accesses - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-fine-grained-bitfield-accesses -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-fine-grained-bitfield-accesses -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-finite-math-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-finite-math-only -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-finite-math-only -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-finite-math-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-inline-limit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-inline-limit -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-inline-limit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-inline-limit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-float-store - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-float-store -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-float-store -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-float-store -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-frontend-optimize - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-frontend-optimize -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-frontend-optimize -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-frontend-optimize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-gcse - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-gcse -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-gcse -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-gcse -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-gcse-after-reload - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-gcse-after-reload -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-gcse-after-reload -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-gcse-after-reload -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-gcse-las - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-gcse-las -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-gcse-las -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-gcse-las -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-gcse-sm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-gcse-sm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-gcse-sm -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-gcse-sm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-global-isel - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-global-isel -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-global-isel -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-global-isel -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-gpu-allow-device-init - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-gpu-allow-device-init -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-gpu-allow-device-init -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-gpu-defer-diag - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-gpu-defer-diag -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-gpu-defer-diag -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-gpu-exclude-wrong-side-overloads - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-gpu-exclude-wrong-side-overloads -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-gpu-exclude-wrong-side-overloads -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-gpu-flush-denormals-to-zero - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-gpu-flush-denormals-to-zero -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-gpu-flush-denormals-to-zero -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-gpu-rdc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-gpu-rdc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-gpu-rdc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-gpu-sanitize - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-gpu-sanitize -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-gpu-sanitize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-hip-emit-relocatable - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-hip-emit-relocatable -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-hip-emit-relocatable -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-hip-fp32-correctly-rounded-divide-sqrt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-hip-fp32-correctly-rounded-divide-sqrt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-hip-kernel-arg-name - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-hip-kernel-arg-name -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-hip-kernel-arg-name -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-hip-new-launch-api - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-hip-new-launch-api -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-hip-new-launch-api -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-init-local-zero - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-init-local-zero -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-init-local-zero -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-init-local-zero -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-inline - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-inline -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-inline -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-inline-functions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-inline-functions -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-inline-functions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-inline-functions-called-once - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-inline-functions-called-once -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-inline-functions-called-once -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-inline-functions-called-once -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-inline-small-functions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-inline-small-functions -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-inline-small-functions -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-inline-small-functions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-integer-4-integer-8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-integer-4-integer-8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-integer-4-integer-8 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-integer-4-integer-8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-ipa-cp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-ipa-cp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-ipa-cp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-ipa-cp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-ivopts - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-ivopts -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-ivopts -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-ivopts -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-limit-debug-info - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-limit-debug-info -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -fno-version-loops-for-stride - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-version-loops-for-stride -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-version-loops-for-stride -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-version-loops-for-stride -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-version-loops-for-stride -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-lto-unit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-lto-unit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-lto-unit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-lto-unit -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang_cl -fno-math-builtin -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-math-builtin -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-math-builtin -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-max-identifier-length - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-max-identifier-length -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-max-identifier-length -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-max-identifier-length -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-merge-constants - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-merge-constants -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-merge-constants -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-merge-constants -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-module-maps - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-module-maps -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-module-maps -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-module-maps -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-module-private - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-module-private -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-module-private -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-module-private -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-modules-error-recovery - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-modules-error-recovery -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-modules-error-recovery -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-modules-error-recovery -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-modules-global-index - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-modules-global-index -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-modules-global-index -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-modules-global-index -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-modules-share-filemanager - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-modules-share-filemanager -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-modules-share-filemanager -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-modules-share-filemanager -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-modules-validate-system-headers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-modules-validate-system-headers -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-modules-validate-system-headers -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-modules-validate-system-headers -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-modulo-sched - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-modulo-sched -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-modulo-sched -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-modulo-sched -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-modulo-sched-allow-regmoves - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-modulo-sched-allow-regmoves -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-modulo-sched-allow-regmoves -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-modulo-sched-allow-regmoves -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-offload-implicit-host-device-templates - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-offload-implicit-host-device-templates -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-offload-implicit-host-device-templates -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-offload-via-llvm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-offload-via-llvm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-offload-via-llvm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-openmp-new-driver - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-openmp-new-driver -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-openmp-new-driver -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-openmp-new-driver -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-pack-derived - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-pack-derived -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-pack-derived -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-pack-derived -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-padding-on-unsigned-fixed-point - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-padding-on-unsigned-fixed-point -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-padding-on-unsigned-fixed-point -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-padding-on-unsigned-fixed-point -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-pch-timestamp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-pch-timestamp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-pch-timestamp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-pch-timestamp -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-peel-loops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-peel-loops -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-peel-loops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-peel-loops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-ppc-native-vector-element-order - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-ppc-native-vector-element-order -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-ppc-native-vector-element-order -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-ppc-native-vector-element-order -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-ppc-native-vector-element-order -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-prefetch-loop-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-prefetch-loop-arrays -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-prefetch-loop-arrays -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-prefetch-loop-arrays -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-profile-correction - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-profile-correction -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-profile-correction -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-profile-correction -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-profile-use - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-profile-use -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-profile-use -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-profile-use -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-profile-values - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-profile-values -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-profile-values -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-profile-values -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-range-check - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-range-check -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-range-check -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-range-check -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-real-4-real-10 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-real-4-real-10 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-real-4-real-10 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-real-4-real-10 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-real-4-real-16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-real-4-real-16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-real-4-real-16 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-real-4-real-16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-real-4-real-8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-real-4-real-8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-real-4-real-8 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-real-4-real-8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-real-8-real-10 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-real-8-real-10 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-real-8-real-10 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-real-8-real-10 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-real-8-real-16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-real-8-real-16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-real-8-real-16 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-real-8-real-16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-real-8-real-4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-real-8-real-4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-real-8-real-4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-real-8-real-4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-realloc-lhs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-realloc-lhs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-realloc-lhs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-realloc-lhs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-recovery-ast - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-recovery-ast -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-recovery-ast -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-recovery-ast -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-recovery-ast-type - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-recovery-ast-type -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-recovery-ast-type -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-recovery-ast-type -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-recursive - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-recursive -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-recursive -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-recursive -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-reformat - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-reformat -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-reformat -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-reformat -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-reformat -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-rename-registers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-rename-registers -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-rename-registers -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-rename-registers -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-reorder-blocks - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-reorder-blocks -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-reorder-blocks -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-reorder-blocks -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-repack-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-repack-arrays -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-repack-arrays -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-repack-arrays -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-rtlib-add-rpath - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-rtlib-add-rpath -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-rtlib-add-rpath -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-rtlib-add-rpath -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-address-globals-dead-stripping - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-address-globals-dead-stripping -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-address-outline-instrumentation - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-address-outline-instrumentation -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-address-poison-custom-array-cookie - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-address-poison-custom-array-cookie -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-address-use-after-scope - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-address-use-after-scope -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-address-use-odr-indicator - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-address-use-odr-indicator -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-cfi-canonical-jump-tables - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-cfi-canonical-jump-tables -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-cfi-cross-dso - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-cfi-cross-dso -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-coverage= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-coverage= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-hwaddress-experimental-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-hwaddress-experimental-aliasing -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-ignorelist - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-ignorelist -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-link-c++-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-link-c++-runtime -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-link-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-link-runtime -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-memory-track-origins - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-memory-track-origins -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-memory-use-after-dtor - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-memory-use-after-dtor -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-minimal-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-minimal-runtime -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-recover - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-recover -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-recover= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-recover= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-stats - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-stats -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-thread-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-thread-atomics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-thread-func-entry-exit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-thread-func-entry-exit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-thread-memory-access - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-thread-memory-access -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-trap - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-trap -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-trap= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-trap= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sanitize-undefined-trap-on-error - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fno-sanitize-undefined-trap-on-error -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-schedule-insns - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-schedule-insns -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-schedule-insns -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-schedule-insns -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-schedule-insns2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-schedule-insns2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-schedule-insns2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-schedule-insns2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-second-underscore - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-second-underscore -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-second-underscore -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-second-underscore -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sign-zero - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-sign-zero -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-sign-zero -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-sign-zero -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-signaling-nans - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-signaling-nans -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-signaling-nans -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-signaling-nans -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-signed-wchar - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-signed-wchar -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-signed-wchar -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-signed-wchar -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-single-precision-constant - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-single-precision-constant -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-single-precision-constant -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-single-precision-constant -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-stack-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-stack-arrays -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-stack-arrays -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-stack-arrays -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-stack-arrays -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-strength-reduce - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-strength-reduce -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-strength-reduce -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-strength-reduce -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-sycl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-sycl -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fno-sycl -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-tracer - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-tracer -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-tracer -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-tracer -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-tree-dce - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-tree-dce -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-tree-dce -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-tree-dce -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-tree-ter - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-tree-ter -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-tree-ter -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-tree-ter -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-tree-vrp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-tree-vrp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-tree-vrp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-tree-vrp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-unroll-all-loops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-unroll-all-loops -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-unroll-all-loops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-unroll-all-loops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-unsafe-loop-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-unsafe-loop-optimizations -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-unsafe-loop-optimizations -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-unsafe-loop-optimizations -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-unsigned-char - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-unsigned-char -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-unsigned-char -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-unsigned-char -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-unswitch-loops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-unswitch-loops -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-unswitch-loops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-unswitch-loops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -fno-use-ctor-homing -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-use-ctor-homing -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-use-ctor-homing -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-use-linker-plugin - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-use-linker-plugin -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-use-linker-plugin -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-use-linker-plugin -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-validate-pch - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-validate-pch -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-validate-pch -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-validate-pch -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-variable-expansion-in-unroller - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-variable-expansion-in-unroller -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-variable-expansion-in-unroller -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-variable-expansion-in-unroller -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-vect-cost-model - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-vect-cost-model -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-vect-cost-model -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-vect-cost-model -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-verify-intermediate-code - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-verify-intermediate-code -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -fno-wchar - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fno-wchar -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-wchar -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fno-wchar -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fno-web - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-web -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-web -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-web -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-whole-file - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-whole-file -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-whole-file -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-whole-file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fno-whole-program - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fno-whole-program -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fno-whole-program -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fno-whole-program -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fobjc-arc-cxxlib= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fobjc-arc-cxxlib= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fobjc-arc-cxxlib= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fobjc-arc-cxxlib= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fobjc-dispatch-method= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fobjc-dispatch-method= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fobjc-dispatch-method= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fobjc-dispatch-method= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fobjc-runtime-has-weak - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fobjc-runtime-has-weak -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fobjc-runtime-has-weak -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fobjc-runtime-has-weak -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fobjc-subscripting-legacy-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fobjc-subscripting-legacy-runtime -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fobjc-subscripting-legacy-runtime -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fobjc-subscripting-legacy-runtime -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -foffload-implicit-host-device-templates - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -foffload-implicit-host-device-templates -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -foffload-via-llvm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -foffload-via-llvm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fopenmp-host-ir-file-path - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fopenmp-host-ir-file-path -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fopenmp-host-ir-file-path -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fopenmp-host-ir-file-path -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fopenmp-is-target-device - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fopenmp-is-target-device -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fopenmp-is-target-device -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fopenmp-is-target-device -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fopenmp-new-driver - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fopenmp-new-driver -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fopenmp-new-driver -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fopenmp-new-driver -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fopenmp-targets= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fopenmp-targets= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fopenmp-targets= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -force_cpusubtype_ALL - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -force_cpusubtype_ALL -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -force_cpusubtype_ALL -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -force_cpusubtype_ALL -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -force_flat_namespace - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -force_flat_namespace -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -force_flat_namespace -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -force_flat_namespace -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -force_load - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -force_load -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -force_load -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -force_load -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -foverride-record-layout= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -foverride-record-layout= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -foverride-record-layout= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -foverride-record-layout= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fpack-derived - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fpack-derived -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fpack-derived -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fpack-derived -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fpadding-on-unsigned-fixed-point - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fpadding-on-unsigned-fixed-point -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fpadding-on-unsigned-fixed-point -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fpadding-on-unsigned-fixed-point -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fparse-all-comments - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fparse-all-comments -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fparse-all-comments -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fpatchable-function-entry-offset= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fpatchable-function-entry-offset= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fpatchable-function-entry-offset= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fpatchable-function-entry-offset= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fpeel-loops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fpeel-loops -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fpeel-loops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fpeel-loops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fplugin-arg- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fplugin-arg- -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fplugin-arg- -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fplugin-arg- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fppc-native-vector-element-order - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fppc-native-vector-element-order -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fppc-native-vector-element-order -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fppc-native-vector-element-order -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fppc-native-vector-element-order -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fprebuilt-module-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fprebuilt-module-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fprefetch-loop-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fprefetch-loop-arrays -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fprefetch-loop-arrays -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fprefetch-loop-arrays -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fpreprocess-include-lines - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fpreprocess-include-lines -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fpreprocess-include-lines -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fpreprocess-include-lines -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fpreprocess-include-lines -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fpreserve-vec3-type - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fpreserve-vec3-type -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fpreserve-vec3-type -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fpreserve-vec3-type -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fprofile-correction - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fprofile-correction -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fprofile-correction -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fprofile-correction -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fprofile-instrument= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fprofile-instrument= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fprofile-instrument= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fprofile-instrument= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fprofile-instrument-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fprofile-instrument-path= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fprofile-instrument-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fprofile-instrument-path= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fprofile-instrument-use-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fprofile-instrument-use-path= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fprofile-instrument-use-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fprofile-instrument-use-path= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fprofile-values - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fprofile-values -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fprofile-values -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fprofile-values -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -framework - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -framework -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -framework -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -framework -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -frandomize-layout-seed= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -frandomize-layout-seed= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -frandomize-layout-seed= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -frandomize-layout-seed-file= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -frandomize-layout-seed-file= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -frandomize-layout-seed-file= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -frange-check - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -frange-check -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -frange-check -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -frange-check -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -freal-4-real-10 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -freal-4-real-10 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -freal-4-real-10 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -freal-4-real-10 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -freal-4-real-16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -freal-4-real-16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -freal-4-real-16 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -freal-4-real-16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -freal-4-real-8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -freal-4-real-8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -freal-4-real-8 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -freal-4-real-8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -freal-8-real-10 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -freal-8-real-10 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -freal-8-real-10 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -freal-8-real-10 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -freal-8-real-16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -freal-8-real-16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -freal-8-real-16 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -freal-8-real-16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -freal-8-real-4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -freal-8-real-4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -freal-8-real-4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -freal-8-real-4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -frealloc-lhs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -frealloc-lhs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -frealloc-lhs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -frealloc-lhs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -frecord-marker= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -frecord-marker= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -frecord-marker= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -frecord-marker= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -frecovery-ast - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -frecovery-ast -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -frecovery-ast -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -frecovery-ast -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -frecovery-ast-type - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -frecovery-ast-type -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -frecovery-ast-type -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -frecovery-ast-type -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -frecursive - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -frecursive -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -frecursive -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -frecursive -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -frename-registers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -frename-registers -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -frename-registers -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -frename-registers -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -freorder-blocks - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -freorder-blocks -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -freorder-blocks -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -freorder-blocks -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -frepack-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -frepack-arrays -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -frepack-arrays -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -frepack-arrays -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -frtlib-add-rpath - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -frtlib-add-rpath -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -frtlib-add-rpath -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -frtlib-add-rpath -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-address-field-padding= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-address-field-padding= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-address-globals-dead-stripping - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-address-globals-dead-stripping -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-address-outline-instrumentation - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-address-outline-instrumentation -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-address-poison-custom-array-cookie - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-address-poison-custom-array-cookie -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-address-use-after-scope - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-address-use-after-scope -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-address-use-odr-indicator - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-address-use-odr-indicator -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-cfi-canonical-jump-tables - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-cfi-canonical-jump-tables -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-cfi-cross-dso - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-cfi-cross-dso -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-cfi-icall-generalize-pointers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-cfi-icall-generalize-pointers -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-cfi-icall-experimental-normalize-integers - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-cfi-icall-experimental-normalize-integers -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-coverage= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-coverage= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-coverage-8bit-counters - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-8bit-counters -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-8bit-counters -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-8bit-counters -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-allowlist= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-coverage-allowlist= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-coverage-control-flow - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-control-flow -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-control-flow -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-control-flow -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-ignorelist= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-coverage-ignorelist= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-coverage-indirect-calls - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-indirect-calls -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-indirect-calls -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-indirect-calls -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-inline-8bit-counters - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-inline-8bit-counters -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-inline-8bit-counters -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-inline-8bit-counters -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-inline-bool-flag - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-inline-bool-flag -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-inline-bool-flag -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-inline-bool-flag -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-no-prune - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-no-prune -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-no-prune -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-no-prune -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-pc-table - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-pc-table -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-pc-table -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-pc-table -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-stack-depth - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-stack-depth -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-stack-depth -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-stack-depth -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-trace-bb - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-trace-bb -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-trace-bb -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-trace-bb -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-trace-cmp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-trace-cmp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-trace-cmp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-trace-cmp -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-trace-div - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-trace-div -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-trace-div -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-trace-div -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-trace-gep - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-trace-gep -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-trace-gep -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-trace-gep -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-trace-loads - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-trace-loads -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-trace-loads -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-trace-loads -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-trace-pc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-trace-pc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-trace-pc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-trace-pc -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-trace-pc-guard - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-trace-pc-guard -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-trace-pc-guard -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-trace-pc-guard -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-trace-stores - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-trace-stores -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-trace-stores -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-trace-stores -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-coverage-type= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsanitize-coverage-type= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsanitize-coverage-type= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsanitize-coverage-type= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsanitize-hwaddress-abi= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-hwaddress-abi= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-hwaddress-experimental-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-hwaddress-experimental-aliasing -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-ignorelist= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-ignorelist= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-link-c++-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-link-c++-runtime -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-link-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-link-runtime -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-memory-track-origins - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-memory-track-origins -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-memory-track-origins= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-memory-track-origins= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-memory-use-after-dtor - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-memory-use-after-dtor -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-memtag-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-memtag-mode= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-minimal-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-minimal-runtime -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-recover - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-recover -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-recover= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-recover= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-stats - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-stats -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-system-ignorelist= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-system-ignorelist= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-thread-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-thread-atomics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-thread-func-entry-exit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-thread-func-entry-exit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-thread-memory-access - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-thread-memory-access -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-trap - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-trap -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-trap= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-trap= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-undefined-ignore-overflow-pattern= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-undefined-ignore-overflow-pattern= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-undefined-strip-path-components= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-undefined-strip-path-components= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-undefined-trap-on-error - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-undefined-trap-on-error -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fschedule-insns - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fschedule-insns -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fschedule-insns -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fschedule-insns -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fschedule-insns2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fschedule-insns2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fschedule-insns2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fschedule-insns2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsecond-underscore - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fsecond-underscore -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fsecond-underscore -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsecond-underscore -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fshow-skipped-includes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fshow-skipped-includes -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fshow-skipped-includes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsign-zero - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fsign-zero -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fsign-zero -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsign-zero -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsignaling-nans - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fsignaling-nans -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fsignaling-nans -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsignaling-nans -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsigned-wchar - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsigned-wchar -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsigned-wchar -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsigned-wchar -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsingle-precision-constant - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fsingle-precision-constant -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fsingle-precision-constant -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsingle-precision-constant -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fspv-target-env= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fspv-target-env= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fspv-target-env= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -cc1as -fstack-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fstack-arrays -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fstack-arrays -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fstack-arrays -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fstack-arrays -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fstrength-reduce - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fstrength-reduce -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fstrength-reduce -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fstrength-reduce -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsycl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fsycl -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fsycl -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsycl-is-device - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsycl-is-device -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsycl-is-device -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsycl-is-device -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsycl-is-host - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fsycl-is-host -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fsycl-is-host -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fsycl-is-host -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fsyntax-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -fsystem-module - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsystem-module -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ftabstop - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ftabstop -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ftabstop -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ftabstop -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ftest-module-file-extension= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ftest-module-file-extension= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ftest-module-file-extension= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ftest-module-file-extension= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -ftracer - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ftracer -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ftracer -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ftracer -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ftree-dce - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ftree-dce -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ftree-dce -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ftree-dce -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ftree-ter - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ftree-ter -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ftree-ter -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ftree-ter -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ftree-vrp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ftree-vrp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ftree-vrp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ftree-vrp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ftype-visibility= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ftype-visibility= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ftype-visibility= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -ftype-visibility= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -function-alignment - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -function-alignment -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -function-alignment -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -function-alignment -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -funknown-anytype - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -funknown-anytype -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -funknown-anytype -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -funknown-anytype -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -funroll-all-loops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -funroll-all-loops -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -funroll-all-loops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -funroll-all-loops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -funsafe-loop-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -funsafe-loop-optimizations -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -funsafe-loop-optimizations -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -funsafe-loop-optimizations -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -funswitch-loops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -funswitch-loops -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -funswitch-loops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -funswitch-loops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -funwind-tables= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -funwind-tables= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -funwind-tables= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -funwind-tables= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang_cl -fuse-ctor-homing -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fuse-ctor-homing -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fuse-ctor-homing -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fuse-cuid= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fuse-cuid= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -fuse-cuid= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fuse-linker-plugin - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fuse-linker-plugin -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fuse-linker-plugin -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fuse-linker-plugin -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fuse-register-sized-bitfield-access - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fuse-register-sized-bitfield-access -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fuse-register-sized-bitfield-access -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fuse-register-sized-bitfield-access -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fvariable-expansion-in-unroller - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fvariable-expansion-in-unroller -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fvariable-expansion-in-unroller -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fvariable-expansion-in-unroller -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fvect-cost-model - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fvect-cost-model -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fvect-cost-model -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fvect-cost-model -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fverify-debuginfo-preserve - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fverify-debuginfo-preserve -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fverify-debuginfo-preserve -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fverify-debuginfo-preserve -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fverify-debuginfo-preserve-export= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fverify-debuginfo-preserve-export= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fverify-debuginfo-preserve-export= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fverify-debuginfo-preserve-export= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fverify-intermediate-code - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fverify-intermediate-code -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -fwarn-stack-size= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fwarn-stack-size= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fwarn-stack-size= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fwarn-stack-size= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fwchar-type= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -fwchar-type= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fwchar-type= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -fwchar-type= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -fweb - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fweb -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fweb -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fweb -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fwhole-file - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fwhole-file -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fwhole-file -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fwhole-file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fwhole-program - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fwhole-program -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fwhole-program -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fwhole-program -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -g0 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -g0 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -g0 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -g0 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -g1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -g1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -g1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -g1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -g2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -g2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -g2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -g2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -g3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -g3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -g3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -g3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -g - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -g -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as --gcc-install-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --gcc-install-dir= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --gcc-install-dir= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --gcc-install-dir= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --gcc-toolchain= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --gcc-toolchain= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --gcc-toolchain= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --gcc-toolchain= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --gcc-triple= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --gcc-triple= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --gcc-triple= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --gcc-triple= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gcodeview-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -gcodeview-ghash - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -gcoff - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gcoff -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gcoff -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gcoff -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gcolumn-info - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gcolumn-info -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -gdbx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gdbx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gdbx -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gdbx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gdwarf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gdwarf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gdwarf32 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gdwarf32 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -gdwarf64 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gdwarf64 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gdwarf-2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gdwarf-2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gdwarf-2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gdwarf-2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gdwarf-3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gdwarf-3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gdwarf-3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gdwarf-3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gdwarf-4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gdwarf-4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gdwarf-4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gdwarf-4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gdwarf-5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gdwarf-5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gdwarf-5 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gdwarf-5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gdwarf-aranges - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gdwarf-aranges -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gdwarf-aranges -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gdwarf-aranges -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gembed-source - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -gembed-source -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gembed-source -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gen-cdb-fragment-path - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gen-cdb-fragment-path -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -gen-reproducer - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gen-reproducer -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -gen-reproducer= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gen-reproducer= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -gfull - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gfull -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gfull -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gfull -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ggdb - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ggdb -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ggdb -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ggdb -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ggdb0 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ggdb0 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ggdb0 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ggdb0 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ggdb1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ggdb1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ggdb1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ggdb1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ggdb2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ggdb2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ggdb2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ggdb2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ggdb3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ggdb3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ggdb3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ggdb3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ggnu-pubnames - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ggnu-pubnames -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ggnu-pubnames -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ginline-line-tables - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ginline-line-tables -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -gline-directives-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gline-directives-only -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -gline-tables-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gline-tables-only -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -glldb - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -glldb -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -glldb -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -glldb -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gmlt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gmlt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gmlt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gmlt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gmodules - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gmodules -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gmodules -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gmodules -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gno-codeview-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -gno-codeview-ghash - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gno-codeview-ghash -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -gno-column-info - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -gno-embed-source - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gno-embed-source -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gno-embed-source -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gno-embed-source -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gno-gnu-pubnames - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gno-gnu-pubnames -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gno-gnu-pubnames -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gno-gnu-pubnames -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gno-inline-line-tables - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -gno-modules - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gno-modules -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gno-modules -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gno-modules -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gno-omit-unreferenced-methods - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gno-omit-unreferenced-methods -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -gno-pubnames - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gno-pubnames -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gno-pubnames -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gno-pubnames -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gno-record-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gno-record-command-line -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gno-record-command-line -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gno-record-command-line -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gno-simple-template-names - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gno-simple-template-names -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gno-simple-template-names -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gno-simple-template-names -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gno-split-dwarf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gno-split-dwarf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -gno-strict-dwarf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gno-strict-dwarf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -gno-template-alias - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gno-template-alias -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gno-template-alias -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gno-template-alias -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gomit-unreferenced-methods - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as --gpu-bundle-output - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --gpu-bundle-output -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --gpu-bundle-output -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --gpu-instrument-lib= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --gpu-instrument-lib= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --gpu-instrument-lib= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --gpu-max-threads-per-block= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc --gpu-max-threads-per-block= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --gpu-use-aux-triple-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --gpu-use-aux-triple-only -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --gpu-use-aux-triple-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gpubnames - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -gpubnames -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gpubnames -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gpulibc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -gpulibc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gpulibc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -grecord-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -grecord-command-line -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -grecord-command-line -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -grecord-command-line -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gsce - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gsce -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gsce -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gsce -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gsimple-template-names - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gsimple-template-names -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gsimple-template-names -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gsimple-template-names -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gsimple-template-names= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -gsimple-template-names= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gsimple-template-names= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -gsimple-template-names= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -gsplit-dwarf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gsplit-dwarf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -gsplit-dwarf= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gsplit-dwarf= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -gsrc-hash= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -gsrc-hash= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gsrc-hash= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -gsrc-hash= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -gstabs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gstabs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gstabs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gstabs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gstrict-dwarf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -gtemplate-alias - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -gtemplate-alias -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gtemplate-alias -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gtoggle - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gtoggle -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gtoggle -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gtoggle -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gused - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gused -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gused -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gused -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gvms - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gvms -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gvms -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gvms -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gxcoff - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gxcoff -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gxcoff -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gxcoff -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gz - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gz -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gz -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gz -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -gz= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -gz= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -gz= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -gz= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -header-include-file - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -header-include-file -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -header-include-file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -header-include-file -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -header-include-filtering= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -header-include-filtering= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -header-include-filtering= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -header-include-filtering= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -header-include-format= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -header-include-format= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -header-include-format= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -header-include-format= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -headerpad_max_install_names - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -headerpad_max_install_names -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -headerpad_max_install_names -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -headerpad_max_install_names -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -help -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -cc1as --hip-device-lib= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --hip-device-lib= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --hip-device-lib= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --hip-link - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --hip-link -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --hip-link -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --hip-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --hip-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --hip-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --hip-version= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --hip-version= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --hip-version= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --hipspv-pass-plugin= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --hipspv-pass-plugin= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --hipspv-pass-plugin= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --hipstdpar - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc --hipstdpar -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --hipstdpar-interpose-alloc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc --hipstdpar-interpose-alloc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --hipstdpar-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --hipstdpar-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --hipstdpar-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --hipstdpar-prim-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --hipstdpar-prim-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --hipstdpar-prim-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --hipstdpar-thrust-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --hipstdpar-thrust-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --hipstdpar-thrust-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -hlsl-entry - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -hlsl-entry -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang -cc1as -iapinotes-modules - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -iapinotes-modules -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -iapinotes-modules -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ibuiltininc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -ibuiltininc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -ibuiltininc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ibuiltininc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -idirafter - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -idirafter -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -idirafter -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -iframework - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -iframework -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -iframework -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -iframeworkwithsysroot - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -iframeworkwithsysroot -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -iframeworkwithsysroot -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -imacros - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -imacros -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -imacros -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -image_base - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -image_base -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -image_base -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -image_base -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -imultilib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -imultilib -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -imultilib -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -imultilib -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -include - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -include -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -include -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -include-pch - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -include-pch -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -include-pch -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -init - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -init -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -init -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -init -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -init-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -init-only -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -init-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -init-only -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -inline-asm= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -inline-asm= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -install_name - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -install_name -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -install_name -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -install_name -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -interface-stub-version= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -interface-stub-version= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -interface-stub-version= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -internal-externc-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -internal-externc-isystem -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -internal-externc-isystem -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -internal-externc-isystem -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -internal-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -internal-isystem -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -internal-isystem -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -internal-isystem -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -iprefix - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -iprefix -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -iprefix -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -iquote - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -iquote -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -iquote -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -isysroot - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -isysroot -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -isysroot -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -isystem - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -isystem -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -isystem -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -isystem-after - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -isystem-after -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -isystem-after -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -ivfsoverlay - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -ivfsoverlay -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -ivfsoverlay -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -iwithprefix - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -iwithprefix -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -iwithprefix -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -iwithprefixbefore - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -iwithprefixbefore -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -iwithprefixbefore -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -iwithsysroot - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -iwithsysroot -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -iwithsysroot -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -keep_private_externs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -keep_private_externs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -keep_private_externs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -keep_private_externs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -l - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -l -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -l -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -l -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -lazy_framework - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -lazy_framework -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -lazy_framework -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -lazy_framework -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -lazy_library - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -lazy_library -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -lazy_library -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -lazy_library -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --ld-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --ld-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --ld-path= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --ld-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --libomptarget-amdgcn-bc-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --libomptarget-amdgcn-bc-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --libomptarget-amdgcn-bc-path= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --libomptarget-amdgcn-bc-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --libomptarget-amdgpu-bc-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --libomptarget-amdgpu-bc-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --libomptarget-amdgpu-bc-path= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --libomptarget-amdgpu-bc-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --libomptarget-nvptx-bc-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --libomptarget-nvptx-bc-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --libomptarget-nvptx-bc-path= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --libomptarget-nvptx-bc-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --linker-option= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --linker-option= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --linker-option= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -llvm-verify-each - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -llvm-verify-each -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -llvm-verify-each -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -load - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -load -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -load -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -m16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -m32 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m32 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -m3dnow - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m3dnow -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -m3dnow -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -m3dnow -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -m3dnowa - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m3dnowa -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -m3dnowa -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -m3dnowa -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -m64 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m64 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -m68000 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m68000 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -m68000 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -m68000 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -m68010 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m68010 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -m68010 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -m68010 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -m68020 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m68020 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -m68020 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -m68020 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -m68030 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m68030 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -m68030 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -m68030 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -m68040 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m68040 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -m68040 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -m68040 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -m68060 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m68060 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -m68060 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -m68060 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -m68881 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m68881 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -m68881 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -m68881 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -m80387 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -m80387 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -m80387 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -m80387 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mseses - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mseses -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mseses -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mabi= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mabi= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mabi=ieeelongdouble - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mabi=ieeelongdouble -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mabi=quadword-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mabi=quadword-atomics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mabi=vec-extabi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mabi=vec-extabi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mabicalls - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mabicalls -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mabicalls -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mabicalls -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mabs= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mabs= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mabs= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mabs= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -madx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -madx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -madx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -maes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -maes -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -maes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -main-file-name -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -main-file-name -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -main-file-name -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -maix32 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -maix32 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -maix32 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -maix64 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -maix64 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -maix64 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -maix-shared-lib-tls-model-opt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -maix-shared-lib-tls-model-opt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -maix-shared-lib-tls-model-opt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -maix-shared-lib-tls-model-opt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -maix-small-local-dynamic-tls - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -maix-small-local-dynamic-tls -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -maix-small-local-dynamic-tls -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -maix-small-local-dynamic-tls -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -maix-small-local-exec-tls - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -maix-small-local-exec-tls -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -maix-small-local-exec-tls -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -maix-small-local-exec-tls -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -maix-struct-return - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -maix-struct-return -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -malign-branch= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -malign-branch= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -malign-branch= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -malign-branch-boundary= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -malign-branch-boundary= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -malign-branch-boundary= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -malign-double - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -malign-double -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -malign-functions= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -malign-functions= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -malign-functions= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -malign-functions= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -malign-jumps= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -malign-jumps= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -malign-jumps= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -malign-jumps= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -malign-loops= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -malign-loops= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -malign-loops= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -malign-loops= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -maltivec - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -maltivec -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -maltivec -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -maltivec -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mamdgpu-ieee - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mamdgpu-ieee -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mamdgpu-ieee -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mamdgpu-precise-memory-op - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mamdgpu-precise-memory-op -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mamdgpu-precise-memory-op -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mamx-avx512 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mamx-avx512 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mamx-avx512 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mamx-bf16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mamx-bf16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mamx-bf16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mamx-complex - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mamx-complex -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mamx-complex -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mamx-fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mamx-fp16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mamx-fp16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mamx-fp8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mamx-fp8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mamx-fp8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mamx-int8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mamx-int8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mamx-int8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mamx-movrs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mamx-movrs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mamx-movrs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mamx-tf32 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mamx-tf32 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mamx-tf32 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mamx-tile - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mamx-tile -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mamx-tile -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mamx-transpose - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mamx-transpose -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mamx-transpose -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mannotate-tablejump - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mannotate-tablejump -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mannotate-tablejump -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mappletvos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mappletvos-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mappletvos-version-min= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mappletvos-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mappletvsimulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mappletvsimulator-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mappletvsimulator-version-min= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mappletvsimulator-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mapx-features= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mapx-features= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mapx-features= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mapx-inline-asm-use-gpr32 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mapx-inline-asm-use-gpr32 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mapx-inline-asm-use-gpr32 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mapxf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mapxf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mapxf -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mapxf -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -march= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -march= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -marm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -marm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -marm -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -marm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -marm64x - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -marm64x -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -marm64x -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -masm= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -masm= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -masm= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -massembler-fatal-warnings -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -massembler-fatal-warnings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -massembler-fatal-warnings -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang_cl -massembler-no-warn -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -massembler-no-warn -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -massembler-no-warn -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -matomics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -matomics -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -matomics -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -matomics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx10.1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx10.1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mavx10.1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mavx10.1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx10.1-256 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx10.1-256 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx10.1-256 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx10.1-512 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx10.1-512 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx10.1-512 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx10.2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx10.2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mavx10.2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mavx10.2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx10.2-256 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx10.2-256 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx10.2-256 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx10.2-512 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx10.2-512 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx10.2-512 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512bf16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512bf16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512bf16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512bitalg - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512bitalg -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512bitalg -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512bw - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512bw -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512bw -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512cd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512cd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512cd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512dq - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512dq -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512dq -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512f - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512f -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512f -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512fp16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512fp16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512ifma - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512ifma -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512ifma -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512vbmi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512vbmi -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512vbmi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512vbmi2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512vbmi2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512vbmi2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512vl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512vl -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512vl -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512vnni - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512vnni -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512vnni -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512vp2intersect - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512vp2intersect -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512vp2intersect -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavx512vpopcntdq - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavx512vpopcntdq -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavx512vpopcntdq -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavxifma - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavxifma -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavxifma -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavxneconvert - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavxneconvert -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavxneconvert -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavxvnni - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavxvnni -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavxvnni -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavxvnniint16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavxvnniint16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavxvnniint16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mavxvnniint8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mavxvnniint8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mavxvnniint8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mbackchain - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mbackchain -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mbig-endian - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mbig-endian -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mbig-endian -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mbmi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mbmi -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mbmi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mbmi2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mbmi2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mbmi2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mbranch-likely - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mbranch-likely -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mbranch-likely -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mbranch-protection= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mbranch-protection= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mbranch-protection= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mbranch-protection-pauth-lr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mbranch-protection-pauth-lr -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mbranch-protection-pauth-lr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mbranch-protection-pauth-lr -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mbranch-target-enforce - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mbranch-target-enforce -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mbranch-target-enforce -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mbranch-target-enforce -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mbranches-within-32B-boundaries - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mbranches-within-32B-boundaries -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mbranches-within-32B-boundaries -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mbulk-memory - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mbulk-memory -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mbulk-memory -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mbulk-memory -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mbulk-memory-opt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mbulk-memory-opt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mbulk-memory-opt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mbulk-memory-opt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcabac - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcabac -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mcabac -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mcabac -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcall-indirect-overlong - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcall-indirect-overlong -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mcall-indirect-overlong -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mcall-indirect-overlong -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcf-branch-label-scheme= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mcf-branch-label-scheme= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcheck-zero-division - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcheck-zero-division -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mcheck-zero-division -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mcheck-zero-division -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcldemote - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcldemote -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mcldemote -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mclflushopt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mclflushopt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mclflushopt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mclwb - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mclwb -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mclwb -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mclzero - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mclzero -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mclzero -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcmodel= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mcmodel= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcmpb - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcmpb -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mcmpb -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mcmpb -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcmpccxadd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcmpccxadd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mcmpccxadd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcmse - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mcmse -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mcmse -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcode-object-version= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mcode-object-version= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcompact-branches= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcompact-branches= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mcompact-branches= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mcompact-branches= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mconsole - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mconsole -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mconsole -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mconstant-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mconstant-cfstrings -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mconstant-cfstrings -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mconstant-cfstrings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mconstructor-aliases - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mconstructor-aliases -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcpu= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcpu= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mcpu= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcrbits - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcrbits -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mcrbits -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mcrbits -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcrc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcrc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mcrc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcrc32 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcrc32 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mcrc32 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcumode - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcumode -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mcumode -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mcumode -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcx16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcx16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mcx16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mdaz-ftz - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mdaz-ftz -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mdaz-ftz -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mdebug-pass - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mdebug-pass -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mdebug-pass -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mdebug-pass -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mdefault-build-attributes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mdefault-build-attributes -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mdefault-build-attributes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mdefault-visibility-export-mapping= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mdefault-visibility-export-mapping= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mdirect-move - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mdirect-move -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mdirect-move -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mdirect-move -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mdiv32 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mdiv32 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mdiv32 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mdll - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mdll -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mdll -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mdouble= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mdouble= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mdouble-float - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mdouble-float -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mdouble-float -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mdsp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mdsp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mdsp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mdsp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mdspr2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mdspr2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mdspr2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mdspr2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mdynamic-no-pic - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mdynamic-no-pic -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mdynamic-no-pic -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -meabi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -meabi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mefpu2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mefpu2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mefpu2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mefpu2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -membedded-data - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -membedded-data -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -membedded-data -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -membedded-data -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -menable-experimental-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -menable-experimental-extensions -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -menable-experimental-extensions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -menable-no-infs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -menable-no-infs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -menable-no-infs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -menable-no-infs -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -menable-no-nans - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -menable-no-nans -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -menable-no-nans -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -menable-no-nans -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -menqcmd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -menqcmd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -menqcmd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mevex512 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mevex512 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mevex512 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mexception-handling - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mexception-handling -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mexception-handling -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mexception-handling -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mexec-model= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mexec-model= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mexec-model= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mexec-model= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mexecute-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mexecute-only -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mexecute-only -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mexecute-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mextended-const - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mextended-const -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mextended-const -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mextended-const -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mextern-sdata - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mextern-sdata -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mextern-sdata -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mextern-sdata -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mf16c - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mf16c -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mf16c -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfancy-math-387 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfancy-math-387 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfancy-math-387 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfancy-math-387 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfentry - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mfentry -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfix4300 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfix4300 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfix4300 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfix4300 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfix-and-continue - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfix-and-continue -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfix-and-continue -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfix-and-continue -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfix-cmse-cve-2021-35465 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfix-cmse-cve-2021-35465 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfix-cmse-cve-2021-35465 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfix-cmse-cve-2021-35465 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfix-cortex-a53-835769 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfix-cortex-a53-835769 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfix-cortex-a53-835769 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfix-cortex-a53-835769 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfix-cortex-a57-aes-1742098 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfix-cortex-a57-aes-1742098 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfix-cortex-a57-aes-1742098 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfix-cortex-a57-aes-1742098 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfix-cortex-a72-aes-1655431 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfix-cortex-a72-aes-1655431 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfix-cortex-a72-aes-1655431 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfix-cortex-a72-aes-1655431 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfix-gr712rc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfix-gr712rc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfix-gr712rc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfix-gr712rc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfix-ut700 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfix-ut700 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfix-ut700 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfix-ut700 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfloat128 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfloat128 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfloat128 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfloat128 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfloat-abi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mfloat-abi -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfloat-abi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mfloat-abi -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mfloat-abi= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfloat-abi= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mfloat-abi= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfma - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfma -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mfma -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfma4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfma4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mfma4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfp16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfp16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfp16 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfp16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfp32 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfp32 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfp32 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfp32 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfp64 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfp64 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfp64 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfp64 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfpmath - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mfpmath -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfpmath -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mfpmath -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mfpmath= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfpmath= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mfpmath= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfprnd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfprnd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfprnd -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfprnd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfpu - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfpu -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfpu -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfpu -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfpu= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfpu= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mfpu= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfpxx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfpxx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfpxx -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfpxx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mframe-chain= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mframe-chain= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mframe-chain= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mframe-chain= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mframe-pointer= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mframe-pointer= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mframe-pointer= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mframe-pointer= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mfrecipe - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfrecipe -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mfrecipe -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfsgsbase - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfsgsbase -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mfsgsbase -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfsmuld - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfsmuld -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mfsmuld -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mfsmuld -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfunction-return= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mfunction-return= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mfxsr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mfxsr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mfxsr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mgeneral-regs-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mgeneral-regs-only -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mgeneral-regs-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mgfni - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mgfni -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mgfni -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mginv - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mginv -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mginv -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mginv -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mglibc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mglibc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mglibc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mglibc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mglobal-merge - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mglobal-merge -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mgpopt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mgpopt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mgpopt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mgpopt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mguard= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mguard= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mguard= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mguarded-control-stack - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mguarded-control-stack -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mguarded-control-stack -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mguarded-control-stack -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mhard-float - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mhard-float -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mhard-float -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mhard-quad-float - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mhard-quad-float -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mhard-quad-float -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mhard-quad-float -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mharden-sls= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mharden-sls= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mharden-sls= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mhvx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mhvx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mhvx -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mhvx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mhvx= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mhvx= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mhvx= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mhvx= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mhvx-ieee-fp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mhvx-ieee-fp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mhvx-ieee-fp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mhvx-ieee-fp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mhvx-length= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mhvx-length= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mhvx-length= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mhvx-length= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mhvx-qfloat - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mhvx-qfloat -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mhvx-qfloat -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mhvx-qfloat -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mhreset - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mhreset -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mhreset -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mhtm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mhtm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mhtm -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mhtm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mhwdiv= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mhwdiv= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mhwdiv= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mhwmult= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mhwmult= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mhwmult= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -miamcu - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -miamcu -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -miamcu -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mieee-fp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mieee-fp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mieee-fp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mieee-fp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mieee-rnd-near - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mieee-rnd-near -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mieee-rnd-near -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mieee-rnd-near -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mignore-xcoff-visibility - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mignore-xcoff-visibility -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -migrate - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -migrate -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -migrate -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -migrate -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -no-finalize-removal - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-finalize-removal -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-finalize-removal -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -no-finalize-removal -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -no-ns-alloc-error - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-ns-alloc-error -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-ns-alloc-error -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -no-ns-alloc-error -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mimplicit-float - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mimplicit-float -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mimplicit-float -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mimplicit-it= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mimplicit-it= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mimplicit-it= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -mincremental-linker-compatible -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mindirect-branch-cs-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mindirect-branch-cs-prefix -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mindirect-jump= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mindirect-jump= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mindirect-jump= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mindirect-jump= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -minline-all-stringops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -minline-all-stringops -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -minline-all-stringops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -minline-all-stringops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -minvariant-function-descriptors - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -minvariant-function-descriptors -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -minvariant-function-descriptors -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -minvariant-function-descriptors -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -minvpcid - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -minvpcid -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -minvpcid -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mios-simulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mios-simulator-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mios-simulator-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mios-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mios-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mios-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips16 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips32 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips32 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips32 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips32 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips32r2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips32r2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips32r2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips32r2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips32r3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips32r3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips32r3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips32r3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips32r5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips32r5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips32r5 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips32r5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips32r6 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips32r6 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips32r6 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips32r6 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips5 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips64 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips64 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips64 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips64 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips64r2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips64r2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips64r2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips64r2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips64r3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips64r3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips64r3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips64r3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips64r5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips64r5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips64r5 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips64r5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mips64r6 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mips64r6 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mips64r6 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mips64r6 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -misel - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -misel -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -misel -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -misel -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mkernel - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mkernel -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mkernel -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mkl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mkl -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mkl -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlam-bh - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlam-bh -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mlam-bh -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlamcas - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlamcas -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mlamcas -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlarge-data-threshold= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mlarge-data-threshold= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlasx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlasx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mlasx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mld-seq-sa - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mld-seq-sa -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mld-seq-sa -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mldc1-sdc1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mldc1-sdc1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mldc1-sdc1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mldc1-sdc1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlimit-float-precision - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mlimit-float-precision -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mlimit-float-precision -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mlimit-float-precision -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mlink-bitcode-file - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mlink-bitcode-file -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mlink-bitcode-file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mlink-bitcode-file -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mlink-builtin-bitcode - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mlink-builtin-bitcode -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mlink-builtin-bitcode -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mlink-builtin-bitcode -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mlink-builtin-bitcode-postopt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mlink-builtin-bitcode-postopt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlinker-version= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlinker-version= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mlinker-version= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlittle-endian - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlittle-endian -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mlittle-endian -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlocal-sdata - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlocal-sdata -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mlocal-sdata -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mlocal-sdata -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlong-calls - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlong-calls -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mlong-calls -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlong-double-128 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mlong-double-128 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mlong-double-128 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlong-double-64 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mlong-double-64 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mlong-double-64 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlong-double-80 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mlong-double-80 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mlong-double-80 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlongcall - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlongcall -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mlongcall -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mlongcall -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlr-for-calls-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlr-for-calls-only -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mlr-for-calls-only -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mlr-for-calls-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlsx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlsx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mlsx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlvi-cfi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlvi-cfi -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mlvi-cfi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlvi-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlvi-hardening -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mlvi-hardening -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlwp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlwp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mlwp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mlzcnt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mlzcnt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mlzcnt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmacos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmacos-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mmacos-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmadd4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmadd4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mmadd4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmadd4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -mmapsyms=implicit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmapsyms=implicit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mmapsyms=implicit -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mmark-bti-property - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmark-bti-property -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mmark-bti-property -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmark-bti-property -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmcu= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmcu= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mmcu= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmemops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mmemops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmemops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmfcrf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmfcrf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mmfcrf -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmfcrf -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmfocrf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmfocrf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mmfocrf -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmfocrf -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmicromips - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmicromips -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mmicromips -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmicromips -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmlir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmlir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mmlir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmma - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmma -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mmma -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmma -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmmx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmmx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mmmx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmovbe - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmovbe -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mmovbe -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmovdir64b - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmovdir64b -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mmovdir64b -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmovdiri - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmovdiri -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mmovdiri -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmovrs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmovrs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mmovrs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmpx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmpx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mmpx -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmpx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mms-bitfields - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mms-bitfields -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -mmsa -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmsa -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mmt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmultimemory - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmultimemory -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mmultimemory -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmultimemory -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmultivalue - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmultivalue -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mmultivalue -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmultivalue -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmutable-globals - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmutable-globals -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mmutable-globals -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mmutable-globals -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mmwaitx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mmwaitx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mmwaitx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mnan= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mnan= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mnan= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mnan= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-3dnow - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-3dnow -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-3dnow -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-3dnow -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-3dnowa - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-3dnowa -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-3dnowa -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-3dnowa -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-80387 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-80387 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-80387 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-80387 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-abicalls - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-abicalls -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-abicalls -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-abicalls -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-adx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-adx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-adx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-aes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-aes -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-aes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-altivec - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-altivec -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-altivec -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-altivec -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-amdgpu-ieee - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mno-amdgpu-ieee -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-amdgpu-precise-memory-op - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-amdgpu-precise-memory-op -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-amdgpu-precise-memory-op -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-amx-avx512 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-amx-avx512 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-amx-avx512 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-amx-bf16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-amx-bf16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-amx-bf16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-amx-complex - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-amx-complex -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-amx-complex -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-amx-fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-amx-fp16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-amx-fp16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-amx-fp8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-amx-fp8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-amx-fp8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-amx-int8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-amx-int8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-amx-int8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-amx-movrs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-amx-movrs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-amx-movrs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-amx-tf32 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-amx-tf32 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-amx-tf32 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-amx-tile - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-amx-tile -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-amx-tile -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-amx-transpose - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-amx-transpose -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-amx-transpose -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-annotate-tablejump - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-annotate-tablejump -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-annotate-tablejump -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-apx-features= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-apx-features= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-apx-features= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-apxf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-apxf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-apxf -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-apxf -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-atomics -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-atomics -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-atomics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx10.1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx10.1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-avx10.1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-avx10.1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx10.1-256 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx10.1-256 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx10.1-256 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx10.1-512 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx10.1-512 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx10.1-512 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx10.2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx10.2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-avx10.2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-avx10.2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx10.2-256 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx10.2-256 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx10.2-256 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx10.2-512 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx10.2-512 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx10.2-512 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512bf16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512bf16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512bf16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512bitalg - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512bitalg -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512bitalg -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512bw - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512bw -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512bw -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512cd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512cd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512cd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512dq - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512dq -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512dq -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512f - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512f -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512f -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512fp16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512fp16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512ifma - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512ifma -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512ifma -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512vbmi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512vbmi -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512vbmi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512vbmi2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512vbmi2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512vbmi2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512vl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512vl -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512vl -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512vnni - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512vnni -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512vnni -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512vp2intersect - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512vp2intersect -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512vp2intersect -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avx512vpopcntdq - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avx512vpopcntdq -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avx512vpopcntdq -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avxifma - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avxifma -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avxifma -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avxneconvert - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avxneconvert -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avxneconvert -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avxvnni - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avxvnni -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avxvnni -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avxvnniint16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avxvnniint16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avxvnniint16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-avxvnniint8 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-avxvnniint8 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-avxvnniint8 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-backchain - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mno-backchain -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-bmi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-bmi -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-bmi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-bmi2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-bmi2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-bmi2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-branch-likely - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-branch-likely -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-branch-likely -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-bti-at-return-twice - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-bti-at-return-twice -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-bti-at-return-twice -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-bti-at-return-twice -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-bulk-memory - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-bulk-memory -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-bulk-memory -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-bulk-memory -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-bulk-memory-opt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-bulk-memory-opt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-bulk-memory-opt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-bulk-memory-opt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-call-indirect-overlong - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-call-indirect-overlong -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-call-indirect-overlong -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-call-indirect-overlong -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-check-zero-division - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-check-zero-division -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-check-zero-division -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-check-zero-division -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-cldemote - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-cldemote -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-cldemote -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-clflushopt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-clflushopt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-clflushopt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-clwb - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-clwb -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-clwb -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-clzero - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-clzero -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-clzero -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-cmpb - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-cmpb -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-cmpb -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-cmpb -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-cmpccxadd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-cmpccxadd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-cmpccxadd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-constant-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-constant-cfstrings -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-constant-cfstrings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-constructor-aliases - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mno-constructor-aliases -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-crbits - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-crbits -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-crbits -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-crbits -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-crc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-crc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-crc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-crc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-crc32 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-crc32 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-crc32 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-cumode - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-cumode -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-cumode -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-cumode -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-cx16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-cx16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-cx16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-daz-ftz - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-daz-ftz -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-daz-ftz -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-default-build-attributes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-default-build-attributes -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-default-build-attributes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-div32 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-div32 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-div32 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-dsp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-dsp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-dsp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-dsp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-dspr2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-dspr2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-dspr2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-dspr2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-embedded-data - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-embedded-data -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-embedded-data -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-embedded-data -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-enqcmd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-enqcmd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-enqcmd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-evex512 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-evex512 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-evex512 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-exception-handling - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-exception-handling -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-exception-handling -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-exception-handling -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -mnoexecstack -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mnoexecstack -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mnoexecstack -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mno-execute-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-execute-only -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-execute-only -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-execute-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-extended-const - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-extended-const -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-extended-const -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-extended-const -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-extern-sdata - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-extern-sdata -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-extern-sdata -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-extern-sdata -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-f16c - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-f16c -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-f16c -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fix-cmse-cve-2021-35465 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fix-cmse-cve-2021-35465 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-fix-cmse-cve-2021-35465 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-fix-cmse-cve-2021-35465 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fix-cortex-a53-835769 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fix-cortex-a53-835769 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-fix-cortex-a53-835769 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-fix-cortex-a53-835769 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fix-cortex-a57-aes-1742098 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fix-cortex-a57-aes-1742098 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-fix-cortex-a57-aes-1742098 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-fix-cortex-a57-aes-1742098 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fix-cortex-a72-aes-1655431 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fix-cortex-a72-aes-1655431 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-fix-cortex-a72-aes-1655431 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-fix-cortex-a72-aes-1655431 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-float128 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-float128 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-float128 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-float128 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fma - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fma -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-fma -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fma4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fma4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-fma4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fmv - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mno-fmv -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-fmv -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fp16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-fp16 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-fp16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fp-ret-in-387 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fp-ret-in-387 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-fp-ret-in-387 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-fp-ret-in-387 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fprnd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fprnd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-fprnd -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-fprnd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fpu - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fpu -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-fpu -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-fpu -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-frecipe - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-frecipe -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-frecipe -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fsgsbase - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fsgsbase -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-fsgsbase -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fsmuld - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fsmuld -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-fsmuld -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-fsmuld -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-fxsr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-fxsr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-fxsr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-gather - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-gather -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-gather -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-gfni - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-gfni -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-gfni -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-ginv - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-ginv -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-ginv -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-ginv -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-global-merge - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mno-global-merge -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-gpopt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-gpopt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-gpopt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-gpopt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-hvx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-hvx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-hvx -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-hvx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-hvx-ieee-fp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-hvx-ieee-fp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-hvx-ieee-fp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-hvx-ieee-fp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-hvx-qfloat - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-hvx-qfloat -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-hvx-qfloat -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-hvx-qfloat -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-hreset - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-hreset -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-hreset -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-htm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-htm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-htm -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-htm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-iamcu - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-iamcu -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-iamcu -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-implicit-float - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-implicit-float -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-implicit-float -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-incremental-linker-compatible - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-incremental-linker-compatible -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-incremental-linker-compatible -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-inline-all-stringops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-inline-all-stringops -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-inline-all-stringops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-inline-all-stringops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-invariant-function-descriptors - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-invariant-function-descriptors -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-invariant-function-descriptors -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-invariant-function-descriptors -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-invpcid - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-invpcid -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-invpcid -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-isel - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-isel -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-isel -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-isel -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-kl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-kl -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-kl -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-lam-bh - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-lam-bh -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-lam-bh -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-lamcas - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-lamcas -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-lamcas -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-lasx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-lasx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-lasx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-ld-seq-sa - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-ld-seq-sa -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-ld-seq-sa -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-ldc1-sdc1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-ldc1-sdc1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-ldc1-sdc1 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-ldc1-sdc1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-link-builtin-bitcode-postopt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mno-link-builtin-bitcode-postopt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-local-sdata - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-local-sdata -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-local-sdata -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-local-sdata -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-long-calls - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-long-calls -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-long-calls -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-longcall - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-longcall -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-longcall -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-longcall -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-lsx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-lsx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-lsx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-lvi-cfi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-lvi-cfi -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-lvi-cfi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-lvi-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-lvi-hardening -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-lvi-hardening -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-lwp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-lwp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-lwp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-lzcnt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-lzcnt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-lzcnt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-madd4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-madd4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-madd4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-madd4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-memops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mno-memops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-memops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-mfcrf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-mfcrf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-mfcrf -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-mfcrf -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-mfocrf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-mfocrf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-mfocrf -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-mfocrf -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-micromips - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-micromips -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-micromips -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-micromips -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-mips16 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-mips16 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-mips16 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-mips16 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-mma - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-mma -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-mma -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-mma -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-mmx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-mmx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-mmx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-movbe - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-movbe -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-movbe -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-movdir64b - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-movdir64b -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-movdir64b -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-movdiri - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-movdiri -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-movdiri -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-movrs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-movrs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-movrs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-movt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-movt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-movt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-movt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-mpx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-mpx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-mpx -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-mpx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-ms-bitfields - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-ms-bitfields -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-ms-bitfields -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-msa - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-msa -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-msa -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-msa -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-mt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-mt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-mt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-mt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-multimemory - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-multimemory -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-multimemory -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-multimemory -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-multivalue - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-multivalue -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-multivalue -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-multivalue -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-mutable-globals - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-mutable-globals -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-mutable-globals -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-mutable-globals -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-mwaitx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-mwaitx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-mwaitx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-neg-immediates - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-neg-immediates -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-neg-immediates -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-neg-immediates -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-nontrapping-fptoint - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-nontrapping-fptoint -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-nontrapping-fptoint -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-nontrapping-fptoint -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-nvj - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mno-nvj -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-nvj -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-nvs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mno-nvs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-nvs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-odd-spreg - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-odd-spreg -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-odd-spreg -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-odd-spreg -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-omit-leaf-frame-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-omit-leaf-frame-pointer -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-omit-leaf-frame-pointer -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-outline - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mno-outline -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-outline -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-outline-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mno-outline-atomics -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-outline-atomics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-packed-stack - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mno-packed-stack -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-packets - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mno-packets -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-packets -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-pascal-strings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-pascal-strings -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-pascal-strings -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-pascal-strings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-pclmul - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-pclmul -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-pclmul -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-pconfig - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-pconfig -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-pconfig -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-pcrel - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-pcrel -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-pcrel -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-pcrel -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-pic-data-is-text-relative - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-pic-data-is-text-relative -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-pic-data-is-text-relative -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-pku - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-pku -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-pku -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-popc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-popc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-popc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-popc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-popcnt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-popcnt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-popcnt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-popcntd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-popcntd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-popcntd -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-popcntd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-power10-vector - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-power10-vector -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-power10-vector -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-power10-vector -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-power8-vector - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-power8-vector -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-power8-vector -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-power8-vector -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-power9-vector - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-power9-vector -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-power9-vector -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-power9-vector -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-prefetchi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-prefetchi -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-prefetchi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-prefixed - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-prefixed -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-prefixed -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-prefixed -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-prfchw - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-prfchw -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-prfchw -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-ptwrite - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-ptwrite -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-ptwrite -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-pure-code - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-pure-code -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-pure-code -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-pure-code -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-raoint - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-raoint -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-raoint -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-rdpid - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-rdpid -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-rdpid -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-rdpru - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-rdpru -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-rdpru -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-rdrnd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-rdrnd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-rdrnd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-rdseed - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-rdseed -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-rdseed -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-red-zone - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-red-zone -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-red-zone -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-reference-types - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-reference-types -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-reference-types -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-reference-types -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-regnames - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-regnames -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-regnames -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-relax - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-relax -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-relax -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-relax-all - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-relax-all -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-relax-all -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-relax-pic-calls - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-relax-pic-calls -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-relax-pic-calls -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-relax-pic-calls -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-relaxed-simd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-relaxed-simd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-relaxed-simd -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-relaxed-simd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-restrict-it - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-restrict-it -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-restrict-it -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-restrict-it -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-retpoline - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-retpoline -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-retpoline -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-retpoline-external-thunk - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-retpoline-external-thunk -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-retpoline-external-thunk -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-rtd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-rtd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-rtd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-rtm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-rtm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-rtm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sahf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sahf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-sahf -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-save-restore - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-save-restore -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-save-restore -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-save-restore -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-scalar-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-scalar-strict-align -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-scalar-strict-align -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-scatter - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-scatter -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-scatter -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-serialize - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-serialize -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-serialize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-seses - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-seses -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-seses -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sgx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sgx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-sgx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sha - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sha -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-sha -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sha512 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sha512 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-sha512 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-shstk - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-shstk -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-shstk -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sign-ext - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sign-ext -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-sign-ext -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-sign-ext -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-simd128 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-simd128 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-simd128 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-simd128 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-skip-rax-setup - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mno-skip-rax-setup -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sm3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sm3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-sm3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sm4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sm4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-sm4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-soft-float - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-soft-float -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-soft-float -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-spe - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-spe -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-spe -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-spe -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-speculative-load-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-speculative-load-hardening -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-speculative-load-hardening -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sse - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sse -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-sse -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sse2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sse2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-sse2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sse3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sse3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-sse3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sse4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sse4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-sse4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-sse4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sse4.1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sse4.1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-sse4.1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sse4.2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sse4.2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-sse4.2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-sse4a - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-sse4a -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-sse4a -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-ssse3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-ssse3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-ssse3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-stack-arg-probe - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mno-stack-arg-probe -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-stackrealign - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-stackrealign -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-stackrealign -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-tail-call - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-tail-call -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-tail-call -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-tail-call -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-tbm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-tbm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-tbm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-tgsplit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-tgsplit -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-tgsplit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-tgsplit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-thumb - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-thumb -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-thumb -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-thumb -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-tls-direct-seg-refs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mno-tls-direct-seg-refs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-tocdata - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mno-tocdata -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-tocdata= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mno-tocdata= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-tsxldtrk - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-tsxldtrk -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-tsxldtrk -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -mno-type-check -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-type-check -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mno-type-check -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mno-uintr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-uintr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-uintr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-unaligned-access - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-unaligned-access -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-unaligned-access -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-unaligned-symbols - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-unaligned-symbols -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-unaligned-symbols -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-unsafe-fp-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-unsafe-fp-atomics -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-unsafe-fp-atomics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-usermsr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-usermsr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-usermsr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-v8plus - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-v8plus -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-v8plus -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-v8plus -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-vaes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-vaes -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-vaes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-vector-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-vector-strict-align -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-vector-strict-align -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-vevpu - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-vevpu -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-vevpu -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-vevpu -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-virt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-virt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-virt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-virt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-vis - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-vis -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-vis -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-vis -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-vis2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-vis2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-vis2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-vis2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-vis3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-vis3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-vis3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-vis3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-vpclmulqdq - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-vpclmulqdq -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-vpclmulqdq -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-vsx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-vsx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-vsx -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-vsx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-vx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-vx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-vx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-vzeroupper - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-vzeroupper -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-vzeroupper -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-waitpkg - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-waitpkg -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-waitpkg -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-warn-nonportable-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-warn-nonportable-cfstrings -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-warn-nonportable-cfstrings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-wavefrontsize64 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-wavefrontsize64 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-wavefrontsize64 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-wbnoinvd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-wbnoinvd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-wbnoinvd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-wide-arithmetic - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-wide-arithmetic -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-wide-arithmetic -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-wide-arithmetic -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-widekl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-widekl -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-widekl -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-x87 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-x87 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-x87 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-xcoff-roptr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-xcoff-roptr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-xcoff-roptr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-xgot - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-xgot -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-xgot -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-xgot -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-xop - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-xop -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-xop -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-xsave - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-xsave -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-xsave -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-xsavec - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-xsavec -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-xsavec -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-xsaveopt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-xsaveopt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-xsaveopt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-xsaves - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-xsaves -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mno-xsaves -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-zvector - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-zvector -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-zvector -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-zvector -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mnocrc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mnocrc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mnocrc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mnocrc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-direct-move - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-direct-move -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-direct-move -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-direct-move -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mnontrapping-fptoint - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mnontrapping-fptoint -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mnontrapping-fptoint -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mnontrapping-fptoint -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mnop-mcount - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mnop-mcount -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-paired-vector-memops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-paired-vector-memops -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-paired-vector-memops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-paired-vector-memops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mno-crypto - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mno-crypto -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mno-crypto -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mno-crypto -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mnvj - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mnvj -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mnvj -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mnvs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mnvs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mnvs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -modd-spreg - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -modd-spreg -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -modd-spreg -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -modd-spreg -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -module-dependency-dir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -module-dependency-dir -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -module-dependency-dir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -module-dir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -module-dir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -module-dir -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -module-dir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -module-dir -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -module-file-deps - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -module-file-deps -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -module-file-deps -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -module-file-deps -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -module-file-info - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -module-file-info -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -module-file-info -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -momit-leaf-frame-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -momit-leaf-frame-pointer -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -momit-leaf-frame-pointer -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -moslib= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -moslib= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -moslib= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -moutline - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -moutline -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -moutline -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -moutline-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -moutline-atomics -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -moutline-atomics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpacked-stack - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mpacked-stack -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpackets - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mpackets -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mpackets -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpad-max-prefix-size= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpad-max-prefix-size= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mpad-max-prefix-size= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpaired-vector-memops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpaired-vector-memops -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mpaired-vector-memops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mpaired-vector-memops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpascal-strings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpascal-strings -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mpascal-strings -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mpascal-strings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpclmul - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpclmul -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mpclmul -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpconfig - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpconfig -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mpconfig -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpcrel - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpcrel -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mpcrel -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mpcrel -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpic-data-is-text-relative - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpic-data-is-text-relative -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mpic-data-is-text-relative -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpku - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpku -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mpku -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpopc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpopc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mpopc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mpopc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpopcnt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpopcnt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mpopcnt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpopcntd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpopcntd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mpopcntd -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mpopcntd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpower10-vector - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpower10-vector -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mpower10-vector -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mpower10-vector -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mcrypto - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mcrypto -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mcrypto -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mcrypto -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpower8-vector - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpower8-vector -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mpower8-vector -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mpower8-vector -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpower9-vector - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpower9-vector -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mpower9-vector -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mpower9-vector -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mprefer-vector-width= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mprefer-vector-width= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mprefetchi - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mprefetchi -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mprefetchi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mprefixed - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mprefixed -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mprefixed -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mprefixed -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mprfchw - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mprfchw -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mprfchw -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mprintf-kind= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mprintf-kind= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mprivileged - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mprivileged -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mprivileged -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mprivileged -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mptwrite - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mptwrite -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mptwrite -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mpure-code - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mpure-code -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mpure-code -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mpure-code -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mqdsp6-compat - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mqdsp6-compat -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mraoint - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mraoint -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mraoint -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mrdpid - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mrdpid -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mrdpid -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mrdpru - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mrdpru -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mrdpru -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mrdrnd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mrdrnd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mrdrnd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mrdseed - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mrdseed -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mrdseed -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mreassociate - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mreassociate -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mreassociate -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mreassociate -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mrecip - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mrecip -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mrecip -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mrecip= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mrecip= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mrecord-mcount - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mrecord-mcount -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mred-zone - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mred-zone -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mred-zone -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mreference-types - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mreference-types -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mreference-types -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mreference-types -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mregnames - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mregnames -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mregparm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mregparm -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mregparm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mregparm -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mregparm= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mregparm= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mregparm= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mrelax - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mrelax -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mrelax -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -mrelax-all -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mrelax-pic-calls - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mrelax-pic-calls -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mrelax-pic-calls -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mrelax-pic-calls -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -mrelax-relocations=no -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mrelax-relocations=no -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mrelax-relocations=no -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mrelaxed-simd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mrelaxed-simd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mrelaxed-simd -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mrelaxed-simd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -mrelocation-model -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mrelocation-model -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mrelocation-model -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mrestrict-it - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mrestrict-it -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mrestrict-it -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mrestrict-it -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mretpoline - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mretpoline -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mretpoline -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mretpoline-external-thunk - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mretpoline-external-thunk -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mretpoline-external-thunk -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mrop-protect - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mrop-protect -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mrop-protect -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mrop-protect -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mrtd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mrtd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mrtm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mrtm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mrtm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mrvv-vector-bits= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mrvv-vector-bits= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mrvv-vector-bits= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msahf - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msahf -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msahf -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msave-reg-params - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -msave-reg-params -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msave-restore - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msave-restore -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -msave-restore -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -msave-restore -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -msave-temp-labels -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -msave-temp-labels -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -msave-temp-labels -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mscalar-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mscalar-strict-align -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mscalar-strict-align -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msecure-plt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msecure-plt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -msecure-plt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -msecure-plt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mserialize - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mserialize -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mserialize -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msgx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msgx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msgx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msha - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msha -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msha -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msha512 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msha512 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msha512 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mshstk - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mshstk -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mshstk -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msign-ext - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msign-ext -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -msign-ext -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -msign-ext -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msign-return-address= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -msign-return-address= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msign-return-address-key= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -msign-return-address-key= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -msign-return-address-key= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -msign-return-address-key= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -msim - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msim -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msim -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msimd128 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msimd128 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -msimd128 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -msimd128 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msimd= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msimd= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msimd= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msingle-float - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msingle-float -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msingle-float -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mskip-rax-setup - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mskip-rax-setup -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msm3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msm3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msm3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msm4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msm4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msm4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msmall-data-limit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -msmall-data-limit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -msmall-data-limit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -msmall-data-limit -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -msmall-data-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msmall-data-limit= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msmall-data-limit= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msmall-data-threshold= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msmall-data-threshold= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msmall-data-threshold= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msoft-float - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -msoft-float -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msoft-quad-float - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msoft-quad-float -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -msoft-quad-float -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -msoft-quad-float -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mspe - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mspe -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mspe -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mspe -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mspeculative-load-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mspeculative-load-hardening -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msse - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msse -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msse -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msse2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msse2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msse2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -msse2avx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msse3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msse3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msse3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msse4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msse4 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -msse4 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -msse4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msse4.1 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msse4.1 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msse4.1 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msse4.2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msse4.2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msse4.2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msse4a - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msse4a -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -msse4a -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mssse3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mssse3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mssse3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mstack-alignment= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mstack-alignment= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mstack-arg-probe - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mstack-arg-probe -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mstack-arg-probe -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mstack-probe-size= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mstack-probe-size= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mstack-protector-guard= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mstack-protector-guard= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mstack-protector-guard-offset= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mstack-protector-guard-offset= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mstack-protector-guard-reg= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mstack-protector-guard-reg= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mstack-protector-guard-symbol= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mstack-protector-guard-symbol= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mstackrealign - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mstackrealign -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msve-vector-bits= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -msve-vector-bits= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -msve-vector-bits= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -msve-vector-bits= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -msvr4-struct-return - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -msvr4-struct-return -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mt-migrate-directory - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mt-migrate-directory -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mt-migrate-directory -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mt-migrate-directory -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mtail-call - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mtail-call -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mtail-call -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mtail-call -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtargetos= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mtargetos= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mtargetos= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtbm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mtbm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mtbm -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtgsplit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mtgsplit -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mtgsplit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mtgsplit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mthread-model - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mthread-model -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mthreads - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mthreads -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mthreads -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mthumb - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mthumb -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mthumb -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtls-dialect= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mtls-dialect= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mtls-dialect= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtls-direct-seg-refs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mtls-direct-seg-refs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mtls-direct-seg-refs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtls-size= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mtls-size= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtocdata - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mtocdata -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtocdata= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mtocdata= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mtp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mtp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -mtp -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -mtp= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mtp= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mtp= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mtp= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtsxldtrk - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mtsxldtrk -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mtsxldtrk -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtune= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mtune= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mtune= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtvos-simulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mtvos-simulator-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mtvos-simulator-version-min= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mtvos-simulator-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mtvos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mtvos-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mtvos-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -muclibc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -muclibc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -muclibc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -muclibc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -muintr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -muintr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -muintr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -multi_module - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -multi_module -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -multi_module -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -multi_module -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -multi-lib-config= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -multi-lib-config= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -multi-lib-config= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -multi-lib-config= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -multiply_defined - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -multiply_defined -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -multiply_defined -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -multiply_defined -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -multiply_defined_unused - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -multiply_defined_unused -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -multiply_defined_unused -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -multiply_defined_unused -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -munaligned-access - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -munaligned-access -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -munaligned-access -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -munaligned-symbols - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -munaligned-symbols -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -munaligned-symbols -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -municode - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -municode -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -municode -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -munsafe-fp-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -munsafe-fp-atomics -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -musermsr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -musermsr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -musermsr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv5 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv5 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv5 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv5 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv55 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv55 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv55 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv55 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv60 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv60 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv60 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv60 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv62 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv62 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv62 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv62 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv65 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv65 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv65 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv65 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv66 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv66 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv66 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv66 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv67 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv67 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv67 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv67 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv67t - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv67t -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv67t -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv67t -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv68 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv68 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv68 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv68 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv69 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv69 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv69 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv69 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv71 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv71 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv71 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv71 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv71t - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv71t -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv71t -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv71t -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv73 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv73 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv73 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv73 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mv8plus - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mv8plus -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mv8plus -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mv8plus -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvaes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mvaes -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mvaes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvector-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mvector-strict-align -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mvector-strict-align -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvevpu - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mvevpu -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mvevpu -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mvevpu -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvirt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mvirt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mvirt -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mvirt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvis - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mvis -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mvis -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mvis -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvis2 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mvis2 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mvis2 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mvis2 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvis3 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mvis3 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mvis3 -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mvis3 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvpclmulqdq - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mvpclmulqdq -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mvpclmulqdq -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvscale-max= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mvscale-max= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mvscale-max= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvscale-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -mvscale-min= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mvscale-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvsx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mvsx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mvsx -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mvsx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvx - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mvx -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mvx -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mvzeroupper - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mvzeroupper -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mvzeroupper -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mwaitpkg - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mwaitpkg -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mwaitpkg -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mwarn-nonportable-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mwarn-nonportable-cfstrings -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mwarn-nonportable-cfstrings -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mwatchos-simulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mwatchos-simulator-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mwatchos-simulator-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mwatchos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mwatchos-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mwatchos-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mwatchsimulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mwatchsimulator-version-min= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mwatchsimulator-version-min= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mwatchsimulator-version-min= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mwavefrontsize64 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mwavefrontsize64 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mwavefrontsize64 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mwbnoinvd - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mwbnoinvd -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mwbnoinvd -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mwide-arithmetic - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mwide-arithmetic -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mwide-arithmetic -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mwide-arithmetic -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mwidekl - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mwidekl -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mwidekl -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mwindows - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mwindows -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mwindows -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mx32 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mx32 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -mx87 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mx87 -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mx87 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mxcoff-build-id= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mxcoff-build-id= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mxcoff-build-id= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mxcoff-build-id= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mxcoff-roptr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -mxcoff-roptr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mxgot - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mxgot -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mxgot -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mxgot -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mxop - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mxop -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mxop -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mxsave - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mxsave -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mxsave -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mxsavec - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mxsavec -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mxsavec -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mxsaveopt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mxsaveopt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mxsaveopt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mxsaves - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mxsaves -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -mxsaves -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mzos-hlq-clang= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mzos-hlq-clang= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mzos-hlq-clang= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mzos-hlq-clang= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mzos-hlq-csslib= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mzos-hlq-csslib= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mzos-hlq-csslib= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mzos-hlq-csslib= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mzos-hlq-le= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mzos-hlq-le= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mzos-hlq-le= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mzos-hlq-le= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mzos-sys-include= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mzos-sys-include= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mzos-sys-include= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mzos-sys-include= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -mzvector - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -mzvector -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -mzvector -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -mzvector -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1 -n -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -n -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -n -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -n -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -new-struct-path-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -new-struct-path-tbaa -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -new-struct-path-tbaa -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -new-struct-path-tbaa -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -no_dead_strip_inits_and_terms - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -no_dead_strip_inits_and_terms -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -no_dead_strip_inits_and_terms -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no_dead_strip_inits_and_terms -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -no-canonical-prefixes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -no-canonical-prefixes -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -no-clear-ast-before-backend - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-clear-ast-before-backend -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-clear-ast-before-backend -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -no-clear-ast-before-backend -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -no-code-completion-globals - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-code-completion-globals -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-code-completion-globals -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -no-code-completion-globals -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -no-code-completion-ns-level-decls - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-code-completion-ns-level-decls -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-code-completion-ns-level-decls -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -no-code-completion-ns-level-decls -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --no-cuda-gpu-arch= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-cuda-gpu-arch= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --no-cuda-gpu-arch= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-cuda-include-ptx= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-cuda-include-ptx= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --no-cuda-include-ptx= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-cuda-noopt-device-debug - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-cuda-noopt-device-debug -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --no-cuda-noopt-device-debug -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-cuda-version-check - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-cuda-version-check -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --no-cuda-version-check -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-default-config - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-default-config -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -no-emit-llvm-uselists - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-emit-llvm-uselists -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-emit-llvm-uselists -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -no-emit-llvm-uselists -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -no-enable-noundef-analysis - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-enable-noundef-analysis -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-enable-noundef-analysis -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -no-enable-noundef-analysis -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --no-gpu-bundle-output - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-gpu-bundle-output -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --no-gpu-bundle-output -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -no-hip-rt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -no-hip-rt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -no-hip-rt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -no-implicit-float - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-implicit-float -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-implicit-float -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -no-implicit-float -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -no-integrated-cpp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -no-integrated-cpp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -no-integrated-cpp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-integrated-cpp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-offload-add-rpath - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-offload-add-rpath -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --no-offload-add-rpath -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --no-offload-add-rpath -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-offload-arch= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-offload-arch= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --no-offload-arch= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-offload-compress - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-offload-compress -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --no-offload-compress -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-offload-new-driver - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc --no-offload-new-driver -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -no-pedantic - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -no-pedantic -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -no-pedantic -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-pedantic -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -no-pie - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -no-pie -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -no-pie -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-pie -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -no-pointer-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-pointer-tbaa -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-pointer-tbaa -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -no-pointer-tbaa -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -no-pthread - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-pthread -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-pthread -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -no-round-trip-args - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-round-trip-args -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-round-trip-args -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -no-round-trip-args -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -no-struct-path-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -no-struct-path-tbaa -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -no-struct-path-tbaa -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -no-struct-path-tbaa -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --no-system-header-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --no-system-header-prefix= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --no-system-header-prefix= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --no-wasm-opt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --no-wasm-opt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --no-wasm-opt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nobuiltininc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -nodefaultlibs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nodefaultlibs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -nodriverkitlib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nodriverkitlib -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -nodriverkitlib -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nodriverkitlib -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nofixprebinding - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nofixprebinding -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -nofixprebinding -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nofixprebinding -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nogpuinc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nogpuinc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -nogpuinc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nogpuinc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nogpulib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -nogpulib -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nogpulib -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nogpulibc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -nogpulibc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nogpulibc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nohipwrapperinc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nohipwrapperinc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -nohipwrapperinc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nohipwrapperinc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nolibc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nolibc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -nolibc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nolibc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nomultidefs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nomultidefs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -nomultidefs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nomultidefs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nopie - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nopie -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -nopie -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nopie -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -noprebind - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -noprebind -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -noprebind -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -noprebind -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -noprofilelib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -noprofilelib -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -noprofilelib -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -noprofilelib -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -noseglinkedit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -noseglinkedit -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -noseglinkedit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -noseglinkedit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nostartfiles - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nostartfiles -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -nostartfiles -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nostartfiles -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nostdinc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nostdinc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -nostdinc++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -nostdinc++ -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nostdinc++ -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nostdlib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nostdlib -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -nostdlibinc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nostdlibinc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -nostdlibinc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nostdlibinc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nostdlib++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -nostdlib++ -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -nostdlib++ -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nostdlib++ -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -nostdsysteminc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -nostdsysteminc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -nostdsysteminc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -nostdsysteminc -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --nvptx-arch-tool= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --nvptx-arch-tool= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --nvptx-arch-tool= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -o -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objc-isystem -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-allowlist-dir-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-atomic-property -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-migrate-all -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-migrate-annotation -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-migrate-designated-init -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-migrate-instancetype -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-migrate-literals -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-migrate-ns-macros -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-migrate-property -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-migrate-property-dot-syntax -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-migrate-protocol-conformance -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-migrate-readonly-property -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-migrate-readwrite-property -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-migrate-subscripting -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-ns-nonatomic-iosonly -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcmt-returns-innerpointer-property -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -objcxx-isystem -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -object -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --offload= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --offload-add-rpath -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --offload-arch= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --offload-compress -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --offload-compression-level= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --offload-device-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --offload-host-device -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --offload-host-only -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --offload-link -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --offload-new-driver -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fexperimental-openacc-macro-override= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -fexperimental-openacc-macro-override= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -fexperimental-openacc-macro-override= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -fexperimental-openacc-macro-override= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -opt-record-file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -opt-record-format -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc -opt-record-passes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_dxc --output-asm-variant= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -p - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -p -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -p -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -p -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -pagezero_size - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -pagezero_size -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -pagezero_size -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pagezero_size -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -pass-exit-codes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -pass-exit-codes -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -pass-exit-codes -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pass-exit-codes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -pch-through-hdrstop-create - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -pch-through-hdrstop-create -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pch-through-hdrstop-create -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -pch-through-hdrstop-create -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -pch-through-hdrstop-use - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -pch-through-hdrstop-use -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pch-through-hdrstop-use -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -pch-through-hdrstop-use -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -pch-through-header= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -pch-through-header= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pch-through-header= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -pch-through-header= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -pedantic - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -pedantic -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pedantic -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -pedantic-errors - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -pedantic-errors -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pedantic-errors -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -pg - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -pg -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pg -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -pic-is-pie - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -pic-is-pie -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pic-is-pie -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -pic-is-pie -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -pic-level - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -pic-level -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pic-level -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -pic-level -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -pie - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -pie -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -pie -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pie -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -pipe - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -pipe -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -pipe -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pipe -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -plugin - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -plugin -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -plugin -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -plugin -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -plugin-arg- - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -plugin-arg- -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -plugin-arg- -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -plugin-arg- -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -pointer-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -pointer-tbaa -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pointer-tbaa -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -pointer-tbaa -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -preamble-bytes= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -preamble-bytes= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -preamble-bytes= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -preamble-bytes= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -prebind - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -prebind -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -prebind -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -prebind -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -prebind_all_twolevel_modules - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -prebind_all_twolevel_modules -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -prebind_all_twolevel_modules -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -prebind_all_twolevel_modules -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -preload - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -preload -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -preload -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -preload -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-dependency-directives-minimized-source - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -print-dependency-directives-minimized-source -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -print-dependency-directives-minimized-source -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -print-dependency-directives-minimized-source -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -print-diagnostic-options - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-diagnostic-options -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -print-diagnostic-options -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-effective-triple - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-effective-triple -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -print-effective-triple -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-enabled-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -print-enabled-extensions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-file-name= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-file-name= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -print-file-name= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-ivar-layout - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -print-ivar-layout -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -print-ivar-layout -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-libgcc-file-name - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-libgcc-file-name -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -print-libgcc-file-name -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-multi-directory - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-multi-directory -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -print-multi-directory -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -print-multi-directory -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-multi-flags-experimental - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-multi-flags-experimental -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -print-multi-flags-experimental -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -print-multi-flags-experimental -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-multi-lib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-multi-lib -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -print-multi-lib -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -print-multi-lib -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-multi-os-directory - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-multi-os-directory -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -print-multi-os-directory -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -print-multi-os-directory -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-preamble - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -print-preamble -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -print-preamble -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -print-preamble -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -print-prog-name= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-prog-name= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -print-prog-name= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-resource-dir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-resource-dir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -print-resource-dir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-rocm-search-dirs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-rocm-search-dirs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -print-rocm-search-dirs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-runtime-dir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-runtime-dir -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -print-runtime-dir -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-search-dirs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-search-dirs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -print-search-dirs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-stats - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -print-stats -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -print-stats -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -print-stats -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -print-library-module-manifest-path - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-library-module-manifest-path -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -print-library-module-manifest-path -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-supported-cpus - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -print-supported-cpus -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-supported-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -print-supported-extensions -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-target-triple - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-target-triple -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -print-target-triple -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -print-targets - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -print-targets -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -print-targets -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -private_bundle - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -private_bundle -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -private_bundle -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -private_bundle -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --product-name= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --product-name= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --product-name= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -pthread - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -pthread -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pthread -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -pthreads - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -pthreads -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -pthreads -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -pthreads -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --ptxas-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --ptxas-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --ptxas-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -r - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -r -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -r -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -r -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -rdynamic - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -rdynamic -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -rdynamic -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -rdynamic -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -read_only_relocs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -read_only_relocs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -read_only_relocs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -read_only_relocs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -record-command-line -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -record-command-line -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -record-command-line -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -reexport_framework - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -reexport_framework -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -reexport_framework -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -reexport_framework -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -reexport-l - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -reexport-l -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -reexport-l -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -reexport-l -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -reexport_library - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -reexport_library -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -reexport_library -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -reexport_library -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -regcall4 - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -regcall4 -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -relaxed-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -relaxed-aliasing -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -relaxed-aliasing -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -relaxed-aliasing -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -relocatable-pch - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -relocatable-pch -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -relocatable-pch -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -remap - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -remap -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -remap -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -remap -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -remap-file - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -remap-file -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -remap-file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -remap-file -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -resource-dir - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -resource-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -resource-dir= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -rewrite-legacy-objc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -rewrite-legacy-objc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -rewrite-legacy-objc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -rewrite-legacy-objc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -rewrite-macros - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -rewrite-macros -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -rewrite-macros -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -rewrite-macros -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -rewrite-objc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -rewrite-objc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -rewrite-objc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -rewrite-test - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -rewrite-test -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -rewrite-test -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -rewrite-test -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --rocm-device-lib-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --rocm-device-lib-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --rocm-device-lib-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --rocm-path= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --rocm-path= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --rocm-path= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -round-trip-args - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -round-trip-args -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -round-trip-args -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -round-trip-args -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -rpath - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -rpath -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as --rsp-quoting= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --rsp-quoting= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -rtlib= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -rtlib= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -rtlib= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -s - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -s -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -s -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -s -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-address-destructor= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-address-destructor= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -fsanitize-address-use-after-return= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -fsanitize-address-use-after-return= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -save-stats - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -save-stats -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -save-stats -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -save-stats -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -save-stats= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -save-stats= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -save-stats= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -save-stats= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -save-temps - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -save-temps -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -save-temps -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -save-temps -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -save-temps= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -save-temps= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -save-temps= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -sectalign - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -sectalign -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -sectalign -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -sectalign -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -sectcreate - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -sectcreate -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -sectcreate -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -sectcreate -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -sectobjectsymbols - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -sectobjectsymbols -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -sectobjectsymbols -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -sectobjectsymbols -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -sectorder - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -sectorder -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -sectorder -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -sectorder -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -seg1addr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -seg1addr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -seg1addr -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -seg1addr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -seg_addr_table - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -seg_addr_table -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -seg_addr_table -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -seg_addr_table -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -seg_addr_table_filename - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -seg_addr_table_filename -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -seg_addr_table_filename -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -seg_addr_table_filename -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -segaddr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -segaddr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -segaddr -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -segaddr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -segcreate - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -segcreate -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -segcreate -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -segcreate -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -seglinkedit - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -seglinkedit -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -seglinkedit -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -seglinkedit -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -segprot - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -segprot -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -segprot -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -segprot -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -segs_read_ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -segs_read_ -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -segs_read_ -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -segs_read_ -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -segs_read_only_addr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -segs_read_only_addr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -segs_read_only_addr -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -segs_read_only_addr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -segs_read_write_addr - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -segs_read_write_addr -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -segs_read_write_addr -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -segs_read_write_addr -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -setup-static-analyzer - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -setup-static-analyzer -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -setup-static-analyzer -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -setup-static-analyzer -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -shared - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -shared -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -shared-libgcc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -shared-libgcc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -shared-libgcc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -shared-libgcc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -shared-libsan - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -shared-libsan -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -shared-libsan -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -shared-libsan -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1 -show-encoding -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -show-encoding -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -show-encoding -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -show-encoding -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --show-includes - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --show-includes -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --show-includes -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang --show-includes -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1 -show-inst -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -show-inst -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -show-inst -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -show-inst -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -single_module - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -single_module -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -single_module -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -single_module -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -skip-function-bodies - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -skip-function-bodies -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -skip-function-bodies -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -skip-function-bodies -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -source-date-epoch - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -source-date-epoch -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -source-date-epoch -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -source-date-epoch -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -specs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -specs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -specs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -specs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -specs= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -specs= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -specs= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -specs= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as /spirv - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /spirv -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl /spirv -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: %clang /spirv -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -split-dwarf-file - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -split-dwarf-file -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -split-dwarf-file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -split-dwarf-file -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang_cl -split-dwarf-output -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -split-dwarf-output -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -split-dwarf-output -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -stack-protector - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -stack-protector -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -stack-protector -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -stack-protector -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -stack-protector-buffer-size - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -stack-protector-buffer-size -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -stack-protector-buffer-size -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -stack-protector-buffer-size -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -stack-usage-file - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -stack-usage-file -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -stack-usage-file -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -stack-usage-file -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --start-no-unused-arguments - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --start-no-unused-arguments -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -startfiles - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -startfiles -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -startfiles -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -startfiles -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -static - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -static -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -static-define - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -static-define -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -static-define -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -static-define -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -static-libgcc - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -static-libgcc -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -static-libgcc -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -static-libgcc -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -static-libgfortran - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -static-libgfortran -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -static-libgfortran -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -static-libgfortran -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -static-libsan - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -static-libsan -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -static-libsan -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -static-libsan -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -static-libstdc++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -static-libstdc++ -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -static-libstdc++ -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -static-libstdc++ -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -static-openmp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -static-openmp -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -static-openmp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -static-openmp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -static-pie - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -static-pie -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -static-pie -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -static-pie -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -stats-file= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -stats-file= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -stats-file= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -stats-file= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -stats-file-append - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -stats-file-append -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -stats-file-append -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -stats-file-append -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -std= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -std= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -std= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -std-default= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -std-default= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -std-default= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -std-default= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -stdlib - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -stdlib -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -stdlib= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -stdlib= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -stdlib= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -stdlib++-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -stdlib++-isystem -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -stdlib++-isystem -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -stdlib++-isystem -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -sub_library - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -sub_library -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -sub_library -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -sub_library -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -sub_umbrella - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -sub_umbrella -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -sub_umbrella -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -sub_umbrella -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --sycl-link - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --sycl-link -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --sycl-link -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -sycl-std= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -sycl-std= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --symbol-graph-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --symbol-graph-dir= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --symbol-graph-dir= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -sys-header-deps - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -sys-header-deps -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -sys-header-deps -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -sys-header-deps -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --system-header-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl --system-header-prefix= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --system-header-prefix= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -t - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -t -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -t -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -t -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --target= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --target= -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -target-abi -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -target-abi -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -target-abi -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang_cl -target-cpu -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -target-cpu -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -target-cpu -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang_cl -target-feature -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -target-feature -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -target-feature -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -target - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -target -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -target-linker-version - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -target-linker-version -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -target-linker-version -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -target-linker-version -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as /T - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 /T -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl /T -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_cl -target-sdk-version= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -target-sdk-version= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -target-sdk-version= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -templight-dump - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -templight-dump -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -templight-dump -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -templight-dump -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -test-io - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -test-io -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -test-io -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -test-io -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -test-io -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -time - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -time -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -time -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -time -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -traditional - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -traditional -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -traditional -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -traditional -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -traditional-cpp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -traditional-cpp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -traditional-cpp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -trigraphs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -trigraphs -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -trigraphs -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -trigraphs -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -trim-egraph - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -trim-egraph -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -trim-egraph -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -trim-egraph -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang_cl -triple -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -triple -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -triple -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -triple= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -triple= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -triple= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -triple= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang_cl -tune-cpu -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -tune-cpu -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -tune-cpu -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -twolevel_namespace - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -twolevel_namespace -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -twolevel_namespace -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -twolevel_namespace -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -twolevel_namespace_hints - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -twolevel_namespace_hints -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -twolevel_namespace_hints -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -twolevel_namespace_hints -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -u - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -u -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -u -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -u -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -umbrella - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -umbrella -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -umbrella -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -umbrella -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -undef - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -undef -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -undef -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -undefined - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -undefined -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -undefined -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -undefined -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -unexported_symbols_list - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -unexported_symbols_list -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -unexported_symbols_list -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -unexported_symbols_list -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -Wextra - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Waliasing - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wampersand - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Warray-bounds - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wc-binding-type - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wcharacter-truncation - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wconversion - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wdo-subscript - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wfunction-elimination - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wimplicit-interface - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wimplicit-procedure - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wintrinsic-shadow - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wuse-without-only - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wintrinsics-std - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wline-truncation - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wno-align-commons - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wno-overwrite-recursive - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wno-tabs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wreal-q-constant - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wsurprising - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wunderflow - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wunused-parameter - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wrealloc-lhs - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wrealloc-lhs-all - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wfrontend-loop-interchange - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -Wtarget-lifetime - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -unwindlib= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -unwindlib= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -unwindlib= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -v - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -vectorize-loops - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -vectorize-loops -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -vectorize-loops -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -vectorize-loops -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -vectorize-slp - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -vectorize-slp -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -vectorize-slp -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -vectorize-slp -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -verify - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -verify -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -verify -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -verify -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -verify= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -verify= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -verify= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -verify= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as --verify-debug-info - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --verify-debug-info -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl --verify-debug-info -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc --verify-debug-info -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -verify-ignore-unexpected - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -verify-ignore-unexpected -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -verify-ignore-unexpected -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -verify-ignore-unexpected -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -verify-ignore-unexpected= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -verify-ignore-unexpected= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -verify-ignore-unexpected= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -verify-ignore-unexpected= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -verify-pch - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -verify-pch -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -verify-pch -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang_cl -version -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -version -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -version -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -vfsoverlay - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1as -via-file-asm - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -via-file-asm -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang -cc1as -vtordisp-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_cl -vtordisp-mode= -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -vtordisp-mode= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -vtordisp-mode= -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVis %s
// RUN: not %clang -cc1as -w - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -w -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --warning-suppression-mappings= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc --warning-suppression-mappings= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as --wasm-opt - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 --wasm-opt -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc --wasm-opt -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -weak_framework - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -weak_framework -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -weak_framework -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -weak_library - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -weak_library -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -weak_library -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -weak_reference_mismatches - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -weak_reference_mismatches -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -weak_reference_mismatches -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -weak-l - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -weak-l -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -weak-l -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -whatsloaded - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -whatsloaded -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -whatsloaded -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -why_load - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -why_load -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -why_load -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -whyload - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -whyload -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_dxc -whyload -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -working-directory - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -working-directory -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -working-directory= - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -working-directory= -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -x - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang_dxc -x -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -y - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -y -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -y -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -y -### | FileCheck -check-prefix=DXCOption %s
// RUN: not %clang -cc1as -z - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOption %s
// RUN: not %clang -cc1 -z -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1Option %s
// RUN: not %clang_cl -z -### /c /WX | FileCheck -check-prefix=CLOption %s
// RUN: not %clang_dxc -z -### | FileCheck -check-prefix=DXCOption %s
// CC1AsOption: {{(unknown argument|n?N?o such file or directory)}}
// CC1Option: {{(unknown argument|n?N?o such file or directory)}}
// CLOption: {{(unknown argument ignored in|no such file or directory)}}
// DXCOption: {{(unknown argument|no such file or directory)}}
// DefaultVis: {{(unknown argument|unsupported option|argument unused|no such file or directory)}}
