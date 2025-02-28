// NOTE: This lit test was automatically generated to validate unintentionally exposed arguments to various driver flavours.
// NOTE: To make changes, see llvm-project/clang/utils/generate_unsupported_in_drivermode.py from which it was generated.
// NOTE: Regenerate this Lit test with the following:
// NOTE: python generate_unsupported_in_drivermode.py llvm-project/clang/include/clang/Driver/Options.td --llvm-bin llvm-project/build/bin --llvm-tblgen llvm-tblgen

// RUN: not %clang -cc1as -A -A- -B -C -CC -D -E -EB -EL -Eonly -F -faapcs-bitfield-load -G -G= -H -J -K -L -M -MD -MF -MG -MJ -MM -MMD -MP -MQ -MT -MV -Mach -O -O0 -O4 -O -ObjC -ObjC++ -Ofast -P -Q -Qn -Qunused-arguments -Qy -R -Rpass= -Rpass-analysis= -Rpass-missed= -S -T -U -V -WCL4 -W -Wa, -Wall -Wdeprecated -Wframe-larger-than -Wframe-larger-than= -Winvalid-constexpr -Winvalid-gnu-asm-cast -Wl, -Wlarge-by-value-copy= -Wlarge-by-value-copy -Wlarger-than- -Wlarger-than= -Wno-deprecated -Wno-invalid-constexpr -Wno-nonportable-cfstrings -Wno-rewrite-macros -Wno-system-headers -Wno-write-strings -Wnonportable-cfstrings -Wp, -Wsystem-headers -Wsystem-headers-in-module= -Wundef-prefix= -Wwrite-strings -X -Xanalyzer -Xarch_ -Xarch_device -Xarch_host -Xassembler -Xclang -Xcuda-fatbinary -Xcuda-ptxas -Xflang -Xlinker -Xoffload-linker -Xopenmp-target -Xopenmp-target= -Xpreprocessor -Z -Z-Xlinker-no-demangle -Z-reserved-lib-cckext -Z-reserved-lib-stdc++ -Zlinker-input --CLASSPATH --CLASSPATH= -- -###  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK0 %s

// CC1AsOptionCHECK0: {{(unknown argument).*-A}}
// CC1AsOptionCHECK0: {{(unknown argument).*-A-}}
// CC1AsOptionCHECK0: {{(unknown argument).*-B}}
// CC1AsOptionCHECK0: {{(unknown argument).*-C}}
// CC1AsOptionCHECK0: {{(unknown argument).*-CC}}
// CC1AsOptionCHECK0: {{(unknown argument).*-D}}
// CC1AsOptionCHECK0: {{(unknown argument).*-E}}
// CC1AsOptionCHECK0: {{(unknown argument).*-EB}}
// CC1AsOptionCHECK0: {{(unknown argument).*-EL}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Eonly}}
// CC1AsOptionCHECK0: {{(unknown argument).*-F}}
// CC1AsOptionCHECK0: {{(unknown argument).*-faapcs-bitfield-load}}
// CC1AsOptionCHECK0: {{(unknown argument).*-G}}
// CC1AsOptionCHECK0: {{(unknown argument).*-G=}}
// CC1AsOptionCHECK0: {{(unknown argument).*-H}}
// CC1AsOptionCHECK0: {{(unknown argument).*-J}}
// CC1AsOptionCHECK0: {{(unknown argument).*-K}}
// CC1AsOptionCHECK0: {{(unknown argument).*-L}}
// CC1AsOptionCHECK0: {{(unknown argument).*-M}}
// CC1AsOptionCHECK0: {{(unknown argument).*-MD}}
// CC1AsOptionCHECK0: {{(unknown argument).*-MF}}
// CC1AsOptionCHECK0: {{(unknown argument).*-MG}}
// CC1AsOptionCHECK0: {{(unknown argument).*-MJ}}
// CC1AsOptionCHECK0: {{(unknown argument).*-MM}}
// CC1AsOptionCHECK0: {{(unknown argument).*-MMD}}
// CC1AsOptionCHECK0: {{(unknown argument).*-MP}}
// CC1AsOptionCHECK0: {{(unknown argument).*-MQ}}
// CC1AsOptionCHECK0: {{(unknown argument).*-MT}}
// CC1AsOptionCHECK0: {{(unknown argument).*-MV}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Mach}}
// CC1AsOptionCHECK0: {{(unknown argument).*-O}}
// CC1AsOptionCHECK0: {{(unknown argument).*-O0}}
// CC1AsOptionCHECK0: {{(unknown argument).*-O4}}
// CC1AsOptionCHECK0: {{(unknown argument).*-O}}
// CC1AsOptionCHECK0: {{(unknown argument).*-ObjC}}
// CC1AsOptionCHECK0: {{(unknown argument).*-ObjC\+\+}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Ofast}}
// CC1AsOptionCHECK0: {{(unknown argument).*-P}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Q}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Qn}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Qunused-arguments}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Qy}}
// CC1AsOptionCHECK0: {{(unknown argument).*-R}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Rpass=}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Rpass-analysis=}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Rpass-missed=}}
// CC1AsOptionCHECK0: {{(unknown argument).*-S}}
// CC1AsOptionCHECK0: {{(unknown argument).*-T}}
// CC1AsOptionCHECK0: {{(unknown argument).*-U}}
// CC1AsOptionCHECK0: {{(unknown argument).*-V}}
// CC1AsOptionCHECK0: {{(unknown argument).*-WCL4}}
// CC1AsOptionCHECK0: {{(unknown argument).*-W}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wa,}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wall}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wdeprecated}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wframe-larger-than}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wframe-larger-than=}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Winvalid-constexpr}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Winvalid-gnu-asm-cast}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wl,}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wlarge-by-value-copy=}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wlarge-by-value-copy}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wlarger-than-}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wlarger-than=}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wno-deprecated}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wno-invalid-constexpr}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wno-nonportable-cfstrings}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wno-rewrite-macros}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wno-system-headers}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wno-write-strings}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wnonportable-cfstrings}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wp,}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wsystem-headers}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wsystem-headers-in-module=}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wundef-prefix=}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Wwrite-strings}}
// CC1AsOptionCHECK0: {{(unknown argument).*-X}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xanalyzer}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xarch_}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xarch_device}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xarch_host}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xassembler}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xclang}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xcuda-fatbinary}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xcuda-ptxas}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xflang}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xlinker}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xoffload-linker}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xopenmp-target}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xopenmp-target=}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Xpreprocessor}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Z}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Z-Xlinker-no-demangle}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Z-reserved-lib-cckext}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Z-reserved-lib-stdc\+\+}}
// CC1AsOptionCHECK0: {{(unknown argument).*-Zlinker-input}}
// CC1AsOptionCHECK0: {{(unknown argument).*--CLASSPATH}}
// CC1AsOptionCHECK0: {{(unknown argument).*--CLASSPATH=}}
// CC1AsOptionCHECK0: {{(unknown argument).*--}}
// CC1AsOptionCHECK0: {{(unknown argument).*-###}}
// RUN: not %clang -cc1as -AI -Brepro -Brepro- -Bt -Bt+ -C -D -E -EH -EP -F -FA -FC -FI -FR -FS -FU -Fa -Fd -Fe -Fe: -Fi -Fi: -Fm -Fo -Fo: -Fp -Fp: -Fr -Fx -G1 -G2 -GA -GF -GF- -GH -GL -GL- -GR -GR- -GS -GS- -GT -GX -GX- -GZ -Gd -Ge -Gh -Gm -Gm- -Gr -Gregcall -Gregcall4 -Gs -Gv -Gw -Gw- -Gy -Gy- -Gz -H -HELP -J -JMC -JMC- -LD -LDd -LN -MD -MDd -MP -MT -MTd -O -P -QIfist -QIntel-jcc-erratum -? -Qfast_transcendentals -Qimprecise_fwaits -Qpar -Qpar-report -Qsafe_fp_loads -Qspectre -Qspectre-load -Qspectre-load-cf -Qvec -Qvec- -Qvec-report -RTC -TC -TP -Tc -Tp -U -V -W0 -W1 -W2  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK1 %s

// CC1AsOptionCHECK1: {{(unknown argument).*-AI}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Brepro}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Brepro-}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Bt}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Bt\+}}
// CC1AsOptionCHECK1: {{(unknown argument).*-C}}
// CC1AsOptionCHECK1: {{(unknown argument).*-D}}
// CC1AsOptionCHECK1: {{(unknown argument).*-E}}
// CC1AsOptionCHECK1: {{(unknown argument).*-EH}}
// CC1AsOptionCHECK1: {{(unknown argument).*-EP}}
// CC1AsOptionCHECK1: {{(unknown argument).*-F}}
// CC1AsOptionCHECK1: {{(unknown argument).*-FA}}
// CC1AsOptionCHECK1: {{(unknown argument).*-FC}}
// CC1AsOptionCHECK1: {{(unknown argument).*-FI}}
// CC1AsOptionCHECK1: {{(unknown argument).*-FR}}
// CC1AsOptionCHECK1: {{(unknown argument).*-FS}}
// CC1AsOptionCHECK1: {{(unknown argument).*-FU}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fa}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fd}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fe}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fe:}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fi}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fi:}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fm}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fo}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fo:}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fp}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fp:}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fr}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Fx}}
// CC1AsOptionCHECK1: {{(unknown argument).*-G1}}
// CC1AsOptionCHECK1: {{(unknown argument).*-G2}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GA}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GF}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GF-}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GH}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GL}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GL-}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GR}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GR-}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GS}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GS-}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GT}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GX}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GX-}}
// CC1AsOptionCHECK1: {{(unknown argument).*-GZ}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gd}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Ge}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gh}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gm}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gm-}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gr}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gregcall}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gregcall4}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gs}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gv}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gw}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gw-}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gy}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gy-}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Gz}}
// CC1AsOptionCHECK1: {{(unknown argument).*-H}}
// CC1AsOptionCHECK1: {{(unknown argument).*-HELP}}
// CC1AsOptionCHECK1: {{(unknown argument).*-J}}
// CC1AsOptionCHECK1: {{(unknown argument).*-JMC}}
// CC1AsOptionCHECK1: {{(unknown argument).*-JMC-}}
// CC1AsOptionCHECK1: {{(unknown argument).*-LD}}
// CC1AsOptionCHECK1: {{(unknown argument).*-LDd}}
// CC1AsOptionCHECK1: {{(unknown argument).*-LN}}
// CC1AsOptionCHECK1: {{(unknown argument).*-MD}}
// CC1AsOptionCHECK1: {{(unknown argument).*-MDd}}
// CC1AsOptionCHECK1: {{(unknown argument).*-MP}}
// CC1AsOptionCHECK1: {{(unknown argument).*-MT}}
// CC1AsOptionCHECK1: {{(unknown argument).*-MTd}}
// CC1AsOptionCHECK1: {{(unknown argument).*-O}}
// CC1AsOptionCHECK1: {{(unknown argument).*-P}}
// CC1AsOptionCHECK1: {{(unknown argument).*-QIfist}}
// CC1AsOptionCHECK1: {{(unknown argument).*-QIntel-jcc-erratum}}
// CC1AsOptionCHECK1: {{(unknown argument).*-?}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Qfast_transcendentals}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Qimprecise_fwaits}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Qpar}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Qpar-report}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Qsafe_fp_loads}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Qspectre}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Qspectre-load}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Qspectre-load-cf}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Qvec}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Qvec-}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Qvec-report}}
// CC1AsOptionCHECK1: {{(unknown argument).*-RTC}}
// CC1AsOptionCHECK1: {{(unknown argument).*-TC}}
// CC1AsOptionCHECK1: {{(unknown argument).*-TP}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Tc}}
// CC1AsOptionCHECK1: {{(unknown argument).*-Tp}}
// CC1AsOptionCHECK1: {{(unknown argument).*-U}}
// CC1AsOptionCHECK1: {{(unknown argument).*-V}}
// CC1AsOptionCHECK1: {{(unknown argument).*-W0}}
// CC1AsOptionCHECK1: {{(unknown argument).*-W1}}
// CC1AsOptionCHECK1: {{(unknown argument).*-W2}}
// RUN: not %clang -cc1as -W3 -W4 -WL -WX -WX- -Wall -Wp64 -Wv -X -Y- -Yc -Yd -Yl -Yu -Z7 -ZH:MD5 -ZH:SHA1 -ZH:SHA_256 -ZI -ZW -Za -Zc: -Zc:__STDC__ -Zc:__cplusplus -Zc:alignedNew -Zc:alignedNew- -Zc:auto -Zc:char8_t -Zc:char8_t- -Zc:dllexportInlines -Zc:dllexportInlines- -Zc:forScope -Zc:inline -Zc:rvalueCast -Zc:sizedDealloc -Zc:sizedDealloc- -Zc:strictStrings -Zc:ternary -Zc:threadSafeInit -Zc:threadSafeInit- -Zc:tlsGuards -Zc:tlsGuards- -Zc:trigraphs -Zc:trigraphs- -Zc:twoPhase -Zc:twoPhase- -Zc:wchar_t -Zc:wchar_t- -Ze -Zg -Zi -Zl -Zm -Zo -Zo- -Zp -Zp -Zs -analyze- -arch: -arm64EC -await -await: -bigobj -c -cgthreads -clang: -clr -constexpr: -d1 -d1PP -d1reportAllClassLayout -d2 -d2FastFail -d2Zi+ -diagnostics:caret -diagnostics:classic -diagnostics:column -diasdkdir -doc -errorReport -execution-charset: -experimental: -exportHeader -external: -external:I -external:W0 -external:W1 -external:W2 -external:W3 -external:W4 -external:env: -favor -fno-sanitize-address-vcasan-lib -fp:contract -fp:except -fp:except- -fp:fast -fp:precise -fp:strict  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK2 %s

// CC1AsOptionCHECK2: {{(unknown argument).*-W3}}
// CC1AsOptionCHECK2: {{(unknown argument).*-W4}}
// CC1AsOptionCHECK2: {{(unknown argument).*-WL}}
// CC1AsOptionCHECK2: {{(unknown argument).*-WX}}
// CC1AsOptionCHECK2: {{(unknown argument).*-WX-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Wall}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Wp64}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Wv}}
// CC1AsOptionCHECK2: {{(unknown argument).*-X}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Y-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Yc}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Yd}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Yl}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Yu}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Z7}}
// CC1AsOptionCHECK2: {{(unknown argument).*-ZH:MD5}}
// CC1AsOptionCHECK2: {{(unknown argument).*-ZH:SHA1}}
// CC1AsOptionCHECK2: {{(unknown argument).*-ZH:SHA_256}}
// CC1AsOptionCHECK2: {{(unknown argument).*-ZI}}
// CC1AsOptionCHECK2: {{(unknown argument).*-ZW}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Za}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:__STDC__}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:__cplusplus}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:alignedNew}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:alignedNew-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:auto}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:char8_t}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:char8_t-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:dllexportInlines}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:dllexportInlines-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:forScope}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:inline}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:rvalueCast}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:sizedDealloc}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:sizedDealloc-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:strictStrings}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:ternary}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:threadSafeInit}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:threadSafeInit-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:tlsGuards}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:tlsGuards-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:trigraphs}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:trigraphs-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:twoPhase}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:twoPhase-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:wchar_t}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zc:wchar_t-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Ze}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zg}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zi}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zl}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zm}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zo}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zo-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zp}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zp}}
// CC1AsOptionCHECK2: {{(unknown argument).*-Zs}}
// CC1AsOptionCHECK2: {{(unknown argument).*-analyze-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-arch:}}
// CC1AsOptionCHECK2: {{(unknown argument).*-arm64EC}}
// CC1AsOptionCHECK2: {{(unknown argument).*-await}}
// CC1AsOptionCHECK2: {{(unknown argument).*-await:}}
// CC1AsOptionCHECK2: {{(unknown argument).*-bigobj}}
// CC1AsOptionCHECK2: {{(unknown argument).*-c}}
// CC1AsOptionCHECK2: {{(unknown argument).*-cgthreads}}
// CC1AsOptionCHECK2: {{(unknown argument).*-clang:}}
// CC1AsOptionCHECK2: {{(unknown argument).*-clr}}
// CC1AsOptionCHECK2: {{(unknown argument).*-constexpr:}}
// CC1AsOptionCHECK2: {{(unknown argument).*-d1}}
// CC1AsOptionCHECK2: {{(unknown argument).*-d1PP}}
// CC1AsOptionCHECK2: {{(unknown argument).*-d1reportAllClassLayout}}
// CC1AsOptionCHECK2: {{(unknown argument).*-d2}}
// CC1AsOptionCHECK2: {{(unknown argument).*-d2FastFail}}
// CC1AsOptionCHECK2: {{(unknown argument).*-d2Zi\+}}
// CC1AsOptionCHECK2: {{(unknown argument).*-diagnostics:caret}}
// CC1AsOptionCHECK2: {{(unknown argument).*-diagnostics:classic}}
// CC1AsOptionCHECK2: {{(unknown argument).*-diagnostics:column}}
// CC1AsOptionCHECK2: {{(unknown argument).*-diasdkdir}}
// CC1AsOptionCHECK2: {{(unknown argument).*-doc}}
// CC1AsOptionCHECK2: {{(unknown argument).*-errorReport}}
// CC1AsOptionCHECK2: {{(unknown argument).*-execution-charset:}}
// CC1AsOptionCHECK2: {{(unknown argument).*-experimental:}}
// CC1AsOptionCHECK2: {{(unknown argument).*-exportHeader}}
// CC1AsOptionCHECK2: {{(unknown argument).*-external:}}
// CC1AsOptionCHECK2: {{(unknown argument).*-external:I}}
// CC1AsOptionCHECK2: {{(unknown argument).*-external:W0}}
// CC1AsOptionCHECK2: {{(unknown argument).*-external:W1}}
// CC1AsOptionCHECK2: {{(unknown argument).*-external:W2}}
// CC1AsOptionCHECK2: {{(unknown argument).*-external:W3}}
// CC1AsOptionCHECK2: {{(unknown argument).*-external:W4}}
// CC1AsOptionCHECK2: {{(unknown argument).*-external:env:}}
// CC1AsOptionCHECK2: {{(unknown argument).*-favor}}
// CC1AsOptionCHECK2: {{(unknown argument).*-fno-sanitize-address-vcasan-lib}}
// CC1AsOptionCHECK2: {{(unknown argument).*-fp:contract}}
// CC1AsOptionCHECK2: {{(unknown argument).*-fp:except}}
// CC1AsOptionCHECK2: {{(unknown argument).*-fp:except-}}
// CC1AsOptionCHECK2: {{(unknown argument).*-fp:fast}}
// CC1AsOptionCHECK2: {{(unknown argument).*-fp:precise}}
// CC1AsOptionCHECK2: {{(unknown argument).*-fp:strict}}
// RUN: not %clang -cc1as -fsanitize=address -fsanitize-address-use-after-return -guard: -headerUnit -headerUnit:angle -headerUnit:quote -headerName: -homeparams -hotpatch -imsvc -kernel -kernel- -link -nologo -permissive -permissive- -reference -sdl -sdl- -showFilenames -showFilenames- -showIncludes -showIncludes:user -sourceDependencies -sourceDependencies:directives -source-charset: -std: -translateInclude -tune: -u -utf-8 -validate-charset -validate-charset- -vctoolsdir -vctoolsversion -vd -vmb -vmg -vmm -vms -vmv -volatile:iso -volatile:ms -wd -winsdkdir -winsdkversion -winsysroot --all-warnings --analyze --analyzer-no-default-checks --analyzer-output --assemble --assert --assert= --bootclasspath --bootclasspath= --classpath --classpath= --comments --comments-in-macros --compile --constant-cfstrings --debug --debug= --define-macro --define-macro= --dependencies --dyld-prefix --dyld-prefix= --encoding --encoding= --entry --extdirs --extdirs= --extra-warnings --for-linker --for-linker= --force-link --force-link= --help-hidden --imacros= --include= --include-barrier --include-directory --include-directory= --include-directory-after --include-directory-after= --include-prefix --include-prefix= --include-with-prefix --include-with-prefix= --include-with-prefix-after --include-with-prefix-after= --include-with-prefix-before --include-with-prefix-before= --language --language= --library-directory --library-directory= --mhwdiv  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK3 %s

// CC1AsOptionCHECK3: {{(unknown argument).*-fsanitize=address}}
// CC1AsOptionCHECK3: {{(unknown argument).*-fsanitize-address-use-after-return}}
// CC1AsOptionCHECK3: {{(unknown argument).*-guard:}}
// CC1AsOptionCHECK3: {{(unknown argument).*-headerUnit}}
// CC1AsOptionCHECK3: {{(unknown argument).*-headerUnit:angle}}
// CC1AsOptionCHECK3: {{(unknown argument).*-headerUnit:quote}}
// CC1AsOptionCHECK3: {{(unknown argument).*-headerName:}}
// CC1AsOptionCHECK3: {{(unknown argument).*-homeparams}}
// CC1AsOptionCHECK3: {{(unknown argument).*-hotpatch}}
// CC1AsOptionCHECK3: {{(unknown argument).*-imsvc}}
// CC1AsOptionCHECK3: {{(unknown argument).*-kernel}}
// CC1AsOptionCHECK3: {{(unknown argument).*-kernel-}}
// CC1AsOptionCHECK3: {{(unknown argument).*-link}}
// CC1AsOptionCHECK3: {{(unknown argument).*-nologo}}
// CC1AsOptionCHECK3: {{(unknown argument).*-permissive}}
// CC1AsOptionCHECK3: {{(unknown argument).*-permissive-}}
// CC1AsOptionCHECK3: {{(unknown argument).*-reference}}
// CC1AsOptionCHECK3: {{(unknown argument).*-sdl}}
// CC1AsOptionCHECK3: {{(unknown argument).*-sdl-}}
// CC1AsOptionCHECK3: {{(unknown argument).*-showFilenames}}
// CC1AsOptionCHECK3: {{(unknown argument).*-showFilenames-}}
// CC1AsOptionCHECK3: {{(unknown argument).*-showIncludes}}
// CC1AsOptionCHECK3: {{(unknown argument).*-showIncludes:user}}
// CC1AsOptionCHECK3: {{(unknown argument).*-sourceDependencies}}
// CC1AsOptionCHECK3: {{(unknown argument).*-sourceDependencies:directives}}
// CC1AsOptionCHECK3: {{(unknown argument).*-source-charset:}}
// CC1AsOptionCHECK3: {{(unknown argument).*-std:}}
// CC1AsOptionCHECK3: {{(unknown argument).*-translateInclude}}
// CC1AsOptionCHECK3: {{(unknown argument).*-tune:}}
// CC1AsOptionCHECK3: {{(unknown argument).*-u}}
// CC1AsOptionCHECK3: {{(unknown argument).*-utf-8}}
// CC1AsOptionCHECK3: {{(unknown argument).*-validate-charset}}
// CC1AsOptionCHECK3: {{(unknown argument).*-validate-charset-}}
// CC1AsOptionCHECK3: {{(unknown argument).*-vctoolsdir}}
// CC1AsOptionCHECK3: {{(unknown argument).*-vctoolsversion}}
// CC1AsOptionCHECK3: {{(unknown argument).*-vd}}
// CC1AsOptionCHECK3: {{(unknown argument).*-vmb}}
// CC1AsOptionCHECK3: {{(unknown argument).*-vmg}}
// CC1AsOptionCHECK3: {{(unknown argument).*-vmm}}
// CC1AsOptionCHECK3: {{(unknown argument).*-vms}}
// CC1AsOptionCHECK3: {{(unknown argument).*-vmv}}
// CC1AsOptionCHECK3: {{(unknown argument).*-volatile:iso}}
// CC1AsOptionCHECK3: {{(unknown argument).*-volatile:ms}}
// CC1AsOptionCHECK3: {{(unknown argument).*-wd}}
// CC1AsOptionCHECK3: {{(unknown argument).*-winsdkdir}}
// CC1AsOptionCHECK3: {{(unknown argument).*-winsdkversion}}
// CC1AsOptionCHECK3: {{(unknown argument).*-winsysroot}}
// CC1AsOptionCHECK3: {{(unknown argument).*--all-warnings}}
// CC1AsOptionCHECK3: {{(unknown argument).*--analyze}}
// CC1AsOptionCHECK3: {{(unknown argument).*--analyzer-no-default-checks}}
// CC1AsOptionCHECK3: {{(unknown argument).*--analyzer-output}}
// CC1AsOptionCHECK3: {{(unknown argument).*--assemble}}
// CC1AsOptionCHECK3: {{(unknown argument).*--assert}}
// CC1AsOptionCHECK3: {{(unknown argument).*--assert=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--bootclasspath}}
// CC1AsOptionCHECK3: {{(unknown argument).*--bootclasspath=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--classpath}}
// CC1AsOptionCHECK3: {{(unknown argument).*--classpath=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--comments}}
// CC1AsOptionCHECK3: {{(unknown argument).*--comments-in-macros}}
// CC1AsOptionCHECK3: {{(unknown argument).*--compile}}
// CC1AsOptionCHECK3: {{(unknown argument).*--constant-cfstrings}}
// CC1AsOptionCHECK3: {{(unknown argument).*--debug}}
// CC1AsOptionCHECK3: {{(unknown argument).*--debug=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--define-macro}}
// CC1AsOptionCHECK3: {{(unknown argument).*--define-macro=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--dependencies}}
// CC1AsOptionCHECK3: {{(unknown argument).*--dyld-prefix}}
// CC1AsOptionCHECK3: {{(unknown argument).*--dyld-prefix=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--encoding}}
// CC1AsOptionCHECK3: {{(unknown argument).*--encoding=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--entry}}
// CC1AsOptionCHECK3: {{(unknown argument).*--extdirs}}
// CC1AsOptionCHECK3: {{(unknown argument).*--extdirs=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--extra-warnings}}
// CC1AsOptionCHECK3: {{(unknown argument).*--for-linker}}
// CC1AsOptionCHECK3: {{(unknown argument).*--for-linker=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--force-link}}
// CC1AsOptionCHECK3: {{(unknown argument).*--force-link=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--help-hidden}}
// CC1AsOptionCHECK3: {{(unknown argument).*--imacros=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-barrier}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-directory}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-directory=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-directory-after}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-directory-after=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-prefix}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-prefix=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-with-prefix}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-with-prefix=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-with-prefix-after}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-with-prefix-after=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-with-prefix-before}}
// CC1AsOptionCHECK3: {{(unknown argument).*--include-with-prefix-before=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--language}}
// CC1AsOptionCHECK3: {{(unknown argument).*--language=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--library-directory}}
// CC1AsOptionCHECK3: {{(unknown argument).*--library-directory=}}
// CC1AsOptionCHECK3: {{(unknown argument).*--mhwdiv}}
// RUN: not %clang -cc1as --mhwdiv= --no-line-commands --no-standard-includes --no-standard-libraries --no-undefined --param --param= --precompile --prefix --prefix= --preprocess --print-diagnostic-categories --print-file-name --print-missing-file-dependencies --print-prog-name --profile --resource --resource= --rtlib -serialize-diagnostics --signed-char --std --stdlib --sysroot --sysroot= --target-help --trace-includes --undefine-macro --undefine-macro= --unsigned-char --user-dependencies --verbose --warn- --warn-= --write-dependencies --write-user-dependencies -add-plugin -alias_list -faligned-alloc-unavailable -all_load -allowable_client -faltivec-src-compat= --amdgpu-arch-tool= -cfg-add-implicit-dtors -unoptimized-cfg -analyze -analyze-function -analyze-function= -analyzer-checker -analyzer-checker= -analyzer-checker-help -analyzer-checker-help-alpha -analyzer-checker-help-developer -analyzer-checker-option-help -analyzer-checker-option-help-alpha -analyzer-checker-option-help-developer -analyzer-config -analyzer-config-compatibility-mode -analyzer-config-compatibility-mode= -analyzer-config-help -analyzer-constraints -analyzer-constraints= -analyzer-disable-all-checks -analyzer-disable-checker -analyzer-disable-checker= -analyzer-disable-retry-exhausted -analyzer-display-progress -analyzer-dump-egraph -analyzer-dump-egraph= -analyzer-inline-max-stack-depth -analyzer-inline-max-stack-depth= -analyzer-inlining-mode -analyzer-inlining-mode= -analyzer-list-enabled-checkers -analyzer-max-loop -analyzer-note-analysis-entry-points -analyzer-opt-analyze-headers -analyzer-output -analyzer-output= -analyzer-purge -analyzer-purge= -analyzer-stats -analyzer-viz-egraph-graphviz -analyzer-werror -fnew-alignment -faligned-new -fno-aligned-new -fsched-interblock -ftemplate-depth- -ftree-vectorize -fno-tree-vectorize -fcuda-rdc -ftree-slp-vectorize -fno-tree-slp-vectorize -fterminated-vtables -fno-cuda-rdc --hip-device-lib-path= -grecord-gcc-switches -gno-record-gcc-switches -miphoneos-version-min=  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK4 %s

// CC1AsOptionCHECK4: {{(unknown argument).*--mhwdiv=}}
// CC1AsOptionCHECK4: {{(unknown argument).*--no-line-commands}}
// CC1AsOptionCHECK4: {{(unknown argument).*--no-standard-includes}}
// CC1AsOptionCHECK4: {{(unknown argument).*--no-standard-libraries}}
// CC1AsOptionCHECK4: {{(unknown argument).*--no-undefined}}
// CC1AsOptionCHECK4: {{(unknown argument).*--param}}
// CC1AsOptionCHECK4: {{(unknown argument).*--param=}}
// CC1AsOptionCHECK4: {{(unknown argument).*--precompile}}
// CC1AsOptionCHECK4: {{(unknown argument).*--prefix}}
// CC1AsOptionCHECK4: {{(unknown argument).*--prefix=}}
// CC1AsOptionCHECK4: {{(unknown argument).*--preprocess}}
// CC1AsOptionCHECK4: {{(unknown argument).*--print-diagnostic-categories}}
// CC1AsOptionCHECK4: {{(unknown argument).*--print-file-name}}
// CC1AsOptionCHECK4: {{(unknown argument).*--print-missing-file-dependencies}}
// CC1AsOptionCHECK4: {{(unknown argument).*--print-prog-name}}
// CC1AsOptionCHECK4: {{(unknown argument).*--profile}}
// CC1AsOptionCHECK4: {{(unknown argument).*--resource}}
// CC1AsOptionCHECK4: {{(unknown argument).*--resource=}}
// CC1AsOptionCHECK4: {{(unknown argument).*--rtlib}}
// CC1AsOptionCHECK4: {{(unknown argument).*-serialize-diagnostics}}
// CC1AsOptionCHECK4: {{(unknown argument).*--signed-char}}
// CC1AsOptionCHECK4: {{(unknown argument).*--std}}
// CC1AsOptionCHECK4: {{(unknown argument).*--stdlib}}
// CC1AsOptionCHECK4: {{(unknown argument).*--sysroot}}
// CC1AsOptionCHECK4: {{(unknown argument).*--sysroot=}}
// CC1AsOptionCHECK4: {{(unknown argument).*--target-help}}
// CC1AsOptionCHECK4: {{(unknown argument).*--trace-includes}}
// CC1AsOptionCHECK4: {{(unknown argument).*--undefine-macro}}
// CC1AsOptionCHECK4: {{(unknown argument).*--undefine-macro=}}
// CC1AsOptionCHECK4: {{(unknown argument).*--unsigned-char}}
// CC1AsOptionCHECK4: {{(unknown argument).*--user-dependencies}}
// CC1AsOptionCHECK4: {{(unknown argument).*--verbose}}
// CC1AsOptionCHECK4: {{(unknown argument).*--warn-}}
// CC1AsOptionCHECK4: {{(unknown argument).*--warn-=}}
// CC1AsOptionCHECK4: {{(unknown argument).*--write-dependencies}}
// CC1AsOptionCHECK4: {{(unknown argument).*--write-user-dependencies}}
// CC1AsOptionCHECK4: {{(unknown argument).*-add-plugin}}
// CC1AsOptionCHECK4: {{(unknown argument).*-alias_list}}
// CC1AsOptionCHECK4: {{(unknown argument).*-faligned-alloc-unavailable}}
// CC1AsOptionCHECK4: {{(unknown argument).*-all_load}}
// CC1AsOptionCHECK4: {{(unknown argument).*-allowable_client}}
// CC1AsOptionCHECK4: {{(unknown argument).*-faltivec-src-compat=}}
// CC1AsOptionCHECK4: {{(unknown argument).*--amdgpu-arch-tool=}}
// CC1AsOptionCHECK4: {{(unknown argument).*-cfg-add-implicit-dtors}}
// CC1AsOptionCHECK4: {{(unknown argument).*-unoptimized-cfg}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyze}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyze-function}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyze-function=}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-checker}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-checker=}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-checker-help}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-checker-help-alpha}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-checker-help-developer}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-checker-option-help}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-checker-option-help-alpha}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-checker-option-help-developer}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-config}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-config-compatibility-mode}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-config-compatibility-mode=}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-config-help}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-constraints}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-constraints=}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-disable-all-checks}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-disable-checker}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-disable-checker=}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-disable-retry-exhausted}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-display-progress}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-dump-egraph}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-dump-egraph=}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-inline-max-stack-depth}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-inline-max-stack-depth=}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-inlining-mode}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-inlining-mode=}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-list-enabled-checkers}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-max-loop}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-note-analysis-entry-points}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-opt-analyze-headers}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-output}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-output=}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-purge}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-purge=}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-stats}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-viz-egraph-graphviz}}
// CC1AsOptionCHECK4: {{(unknown argument).*-analyzer-werror}}
// CC1AsOptionCHECK4: {{(unknown argument).*-fnew-alignment}}
// CC1AsOptionCHECK4: {{(unknown argument).*-faligned-new}}
// CC1AsOptionCHECK4: {{(unknown argument).*-fno-aligned-new}}
// CC1AsOptionCHECK4: {{(unknown argument).*-fsched-interblock}}
// CC1AsOptionCHECK4: {{(unknown argument).*-ftemplate-depth-}}
// CC1AsOptionCHECK4: {{(unknown argument).*-ftree-vectorize}}
// CC1AsOptionCHECK4: {{(unknown argument).*-fno-tree-vectorize}}
// CC1AsOptionCHECK4: {{(unknown argument).*-fcuda-rdc}}
// CC1AsOptionCHECK4: {{(unknown argument).*-ftree-slp-vectorize}}
// CC1AsOptionCHECK4: {{(unknown argument).*-fno-tree-slp-vectorize}}
// CC1AsOptionCHECK4: {{(unknown argument).*-fterminated-vtables}}
// CC1AsOptionCHECK4: {{(unknown argument).*-fno-cuda-rdc}}
// CC1AsOptionCHECK4: {{(unknown argument).*--hip-device-lib-path=}}
// CC1AsOptionCHECK4: {{(unknown argument).*-grecord-gcc-switches}}
// CC1AsOptionCHECK4: {{(unknown argument).*-gno-record-gcc-switches}}
// CC1AsOptionCHECK4: {{(unknown argument).*-miphoneos-version-min=}}
// RUN: not %clang -cc1as -miphonesimulator-version-min= -mllvm= -mmacosx-version-min= -nocudainc -nogpulib -nocudalib -print-multiarch --system-header-prefix --no-system-header-prefix -mcpu=help -mtune=help -integrated-as -no-integrated-as -coverage-data-file= -coverage-notes-file= -fopenmp-is-device -fcuda-approx-transcendentals -fno-cuda-approx-transcendentals -Gs -O1 -O2 -Ob0 -Ob1 -Ob2 -Ob3 -Od -Og -Oi -Oi- -Os -Ot -Ox -Oy -Oy- -Qgather- -Qscatter- -Xmicrosoft-visualc-tools-root -Xmicrosoft-visualc-tools-version -Xmicrosoft-windows-sdk-root -Xmicrosoft-windows-sdk-version -Xmicrosoft-windows-sys-root -Qembed_debug -shared-libasan -static-libasan -fslp-vectorize-aggressive -fident -fno-ident -fdiagnostics-color -fno-diagnostics-color -frecord-gcc-switches -fno-record-gcc-switches -fno-slp-vectorize-aggressive -Xclang= -Xparser -Xcompiler -fexpensive-optimizations -fno-expensive-optimizations -fdefer-pop -fno-defer-pop -fextended-identifiers -fno-extended-identifiers -fsanitize-blacklist= -fno-sanitize-blacklist -fhonor-infinites -fno-honor-infinites -findirect-virtual-calls --config -ansi -arch -arch_errors_fatal -arch_only -ast-dump -ast-dump= -ast-dump-all -ast-dump-all= -ast-dump-decl-types -ast-dump-filter -ast-dump-filter= -ast-dump-lookups -ast-list -ast-merge -ast-print -ast-view --autocomplete= -aux-target-cpu -aux-target-feature -aux-triple -b -bind_at_load -building-pch-with-obj -bundle -bundle_loader -c -c-isystem -canonical-prefixes -ccc- -ccc-gcc-name -ccc-install-dir -ccc-print-bindings -ccc-print-phases  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK5 %s

// CC1AsOptionCHECK5: {{(unknown argument).*-miphonesimulator-version-min=}}
// CC1AsOptionCHECK5: {{(unknown argument).*-mllvm=}}
// CC1AsOptionCHECK5: {{(unknown argument).*-mmacosx-version-min=}}
// CC1AsOptionCHECK5: {{(unknown argument).*-nocudainc}}
// CC1AsOptionCHECK5: {{(unknown argument).*-nogpulib}}
// CC1AsOptionCHECK5: {{(unknown argument).*-nocudalib}}
// CC1AsOptionCHECK5: {{(unknown argument).*-print-multiarch}}
// CC1AsOptionCHECK5: {{(unknown argument).*--system-header-prefix}}
// CC1AsOptionCHECK5: {{(unknown argument).*--no-system-header-prefix}}
// CC1AsOptionCHECK5: {{(unknown argument).*-mcpu=help}}
// CC1AsOptionCHECK5: {{(unknown argument).*-mtune=help}}
// CC1AsOptionCHECK5: {{(unknown argument).*-integrated-as}}
// CC1AsOptionCHECK5: {{(unknown argument).*-no-integrated-as}}
// CC1AsOptionCHECK5: {{(unknown argument).*-coverage-data-file=}}
// CC1AsOptionCHECK5: {{(unknown argument).*-coverage-notes-file=}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fopenmp-is-device}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fcuda-approx-transcendentals}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fno-cuda-approx-transcendentals}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Gs}}
// CC1AsOptionCHECK5: {{(unknown argument).*-O1}}
// CC1AsOptionCHECK5: {{(unknown argument).*-O2}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Ob0}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Ob1}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Ob2}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Ob3}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Od}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Og}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Oi}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Oi-}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Os}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Ot}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Ox}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Oy}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Oy-}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Qgather-}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Qscatter-}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Xmicrosoft-visualc-tools-root}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Xmicrosoft-visualc-tools-version}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Xmicrosoft-windows-sdk-root}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Xmicrosoft-windows-sdk-version}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Xmicrosoft-windows-sys-root}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Qembed_debug}}
// CC1AsOptionCHECK5: {{(unknown argument).*-shared-libasan}}
// CC1AsOptionCHECK5: {{(unknown argument).*-static-libasan}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fslp-vectorize-aggressive}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fident}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fno-ident}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fdiagnostics-color}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fno-diagnostics-color}}
// CC1AsOptionCHECK5: {{(unknown argument).*-frecord-gcc-switches}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fno-record-gcc-switches}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fno-slp-vectorize-aggressive}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Xclang=}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Xparser}}
// CC1AsOptionCHECK5: {{(unknown argument).*-Xcompiler}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fexpensive-optimizations}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fno-expensive-optimizations}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fdefer-pop}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fno-defer-pop}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fextended-identifiers}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fno-extended-identifiers}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fsanitize-blacklist=}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fno-sanitize-blacklist}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fhonor-infinites}}
// CC1AsOptionCHECK5: {{(unknown argument).*-fno-honor-infinites}}
// CC1AsOptionCHECK5: {{(unknown argument).*-findirect-virtual-calls}}
// CC1AsOptionCHECK5: {{(unknown argument).*--config}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ansi}}
// CC1AsOptionCHECK5: {{(unknown argument).*-arch}}
// CC1AsOptionCHECK5: {{(unknown argument).*-arch_errors_fatal}}
// CC1AsOptionCHECK5: {{(unknown argument).*-arch_only}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ast-dump}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ast-dump=}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ast-dump-all}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ast-dump-all=}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ast-dump-decl-types}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ast-dump-filter}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ast-dump-filter=}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ast-dump-lookups}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ast-list}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ast-merge}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ast-print}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ast-view}}
// CC1AsOptionCHECK5: {{(unknown argument).*--autocomplete=}}
// CC1AsOptionCHECK5: {{(unknown argument).*-aux-target-cpu}}
// CC1AsOptionCHECK5: {{(unknown argument).*-aux-target-feature}}
// CC1AsOptionCHECK5: {{(unknown argument).*-aux-triple}}
// CC1AsOptionCHECK5: {{(unknown argument).*-b}}
// CC1AsOptionCHECK5: {{(unknown argument).*-bind_at_load}}
// CC1AsOptionCHECK5: {{(unknown argument).*-building-pch-with-obj}}
// CC1AsOptionCHECK5: {{(unknown argument).*-bundle}}
// CC1AsOptionCHECK5: {{(unknown argument).*-bundle_loader}}
// CC1AsOptionCHECK5: {{(unknown argument).*-c}}
// CC1AsOptionCHECK5: {{(unknown argument).*-c-isystem}}
// CC1AsOptionCHECK5: {{(unknown argument).*-canonical-prefixes}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ccc-}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ccc-gcc-name}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ccc-install-dir}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ccc-print-bindings}}
// CC1AsOptionCHECK5: {{(unknown argument).*-ccc-print-phases}}
// RUN: not %clang -cc1as -cfguard -cfguard-no-checks -chain-include -cl-denorms-are-zero -cl-ext= -cl-fast-relaxed-math -cl-finite-math-only -cl-fp32-correctly-rounded-divide-sqrt -cl-kernel-arg-info -cl-mad-enable -cl-no-signed-zeros -cl-no-stdinc -cl-opt-disable -cl-single-precision-constant -cl-std= -cl-strict-aliasing -cl-uniform-work-group-size -cl-unsafe-math-optimizations -clear-ast-before-backend -client_name -code-completion-at -code-completion-at= -code-completion-brief-comments -code-completion-macros -code-completion-patterns -code-completion-with-fixits -combine -compatibility_version -compiler-options-dump -complex-range= --config= --config-system-dir= --config-user-dir= -coverage -coverage-version= -cpp -cpp-precomp --cuda-compile-host-device --cuda-device-only --cuda-feature= --cuda-gpu-arch= --cuda-host-only --cuda-include-ptx= --cuda-noopt-device-debug --cuda-path= --cuda-path-ignore-env -cuid= -current_version -cxx-isystem -fc++-static-destructors -fc++-static-destructors= -dA -dD -dE -dI -dM -d -d -darwin-target-variant -dead_strip -debug-forward-template-params -dependency-dot -dependency-file --dependent-lib= -detailed-preprocessing-record -diagnostic-log-file -serialize-diagnostic-file -disable-O0-optnone -disable-free -disable-lifetime-markers -disable-llvm-optzns -disable-llvm-passes -disable-llvm-verifier -disable-objc-default-synthesize-properties -disable-pragma-debug-crash -disable-red-zone -discard-value-names --driver-mode= -dsym-dir -dump-coverage-mapping -dump-deserialized-decls -dump-raw-tokens -dump-tokens -dumpdir -dumpmachine -dumpspecs -dumpversion -dwarf-explicit-import -dwarf-ext-refs -Fc -Fo -Vd --E -HV -hlsl-no-stdinc --dxv-path= -validator-version -dylib_file -dylinker -dylinker_install_name  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK6 %s

// CC1AsOptionCHECK6: {{(unknown argument).*-cfguard}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cfguard-no-checks}}
// CC1AsOptionCHECK6: {{(unknown argument).*-chain-include}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-denorms-are-zero}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-ext=}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-fast-relaxed-math}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-finite-math-only}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-fp32-correctly-rounded-divide-sqrt}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-kernel-arg-info}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-mad-enable}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-no-signed-zeros}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-no-stdinc}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-opt-disable}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-single-precision-constant}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-std=}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-strict-aliasing}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-uniform-work-group-size}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cl-unsafe-math-optimizations}}
// CC1AsOptionCHECK6: {{(unknown argument).*-clear-ast-before-backend}}
// CC1AsOptionCHECK6: {{(unknown argument).*-client_name}}
// CC1AsOptionCHECK6: {{(unknown argument).*-code-completion-at}}
// CC1AsOptionCHECK6: {{(unknown argument).*-code-completion-at=}}
// CC1AsOptionCHECK6: {{(unknown argument).*-code-completion-brief-comments}}
// CC1AsOptionCHECK6: {{(unknown argument).*-code-completion-macros}}
// CC1AsOptionCHECK6: {{(unknown argument).*-code-completion-patterns}}
// CC1AsOptionCHECK6: {{(unknown argument).*-code-completion-with-fixits}}
// CC1AsOptionCHECK6: {{(unknown argument).*-combine}}
// CC1AsOptionCHECK6: {{(unknown argument).*-compatibility_version}}
// CC1AsOptionCHECK6: {{(unknown argument).*-compiler-options-dump}}
// CC1AsOptionCHECK6: {{(unknown argument).*-complex-range=}}
// CC1AsOptionCHECK6: {{(unknown argument).*--config=}}
// CC1AsOptionCHECK6: {{(unknown argument).*--config-system-dir=}}
// CC1AsOptionCHECK6: {{(unknown argument).*--config-user-dir=}}
// CC1AsOptionCHECK6: {{(unknown argument).*-coverage}}
// CC1AsOptionCHECK6: {{(unknown argument).*-coverage-version=}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cpp}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cpp-precomp}}
// CC1AsOptionCHECK6: {{(unknown argument).*--cuda-compile-host-device}}
// CC1AsOptionCHECK6: {{(unknown argument).*--cuda-device-only}}
// CC1AsOptionCHECK6: {{(unknown argument).*--cuda-feature=}}
// CC1AsOptionCHECK6: {{(unknown argument).*--cuda-gpu-arch=}}
// CC1AsOptionCHECK6: {{(unknown argument).*--cuda-host-only}}
// CC1AsOptionCHECK6: {{(unknown argument).*--cuda-include-ptx=}}
// CC1AsOptionCHECK6: {{(unknown argument).*--cuda-noopt-device-debug}}
// CC1AsOptionCHECK6: {{(unknown argument).*--cuda-path=}}
// CC1AsOptionCHECK6: {{(unknown argument).*--cuda-path-ignore-env}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cuid=}}
// CC1AsOptionCHECK6: {{(unknown argument).*-current_version}}
// CC1AsOptionCHECK6: {{(unknown argument).*-cxx-isystem}}
// CC1AsOptionCHECK6: {{(unknown argument).*-fc\+\+-static-destructors}}
// CC1AsOptionCHECK6: {{(unknown argument).*-fc\+\+-static-destructors=}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dA}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dD}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dE}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dI}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dM}}
// CC1AsOptionCHECK6: {{(unknown argument).*-d}}
// CC1AsOptionCHECK6: {{(unknown argument).*-d}}
// CC1AsOptionCHECK6: {{(unknown argument).*-darwin-target-variant}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dead_strip}}
// CC1AsOptionCHECK6: {{(unknown argument).*-debug-forward-template-params}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dependency-dot}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dependency-file}}
// CC1AsOptionCHECK6: {{(unknown argument).*--dependent-lib=}}
// CC1AsOptionCHECK6: {{(unknown argument).*-detailed-preprocessing-record}}
// CC1AsOptionCHECK6: {{(unknown argument).*-diagnostic-log-file}}
// CC1AsOptionCHECK6: {{(unknown argument).*-serialize-diagnostic-file}}
// CC1AsOptionCHECK6: {{(unknown argument).*-disable-O0-optnone}}
// CC1AsOptionCHECK6: {{(unknown argument).*-disable-free}}
// CC1AsOptionCHECK6: {{(unknown argument).*-disable-lifetime-markers}}
// CC1AsOptionCHECK6: {{(unknown argument).*-disable-llvm-optzns}}
// CC1AsOptionCHECK6: {{(unknown argument).*-disable-llvm-passes}}
// CC1AsOptionCHECK6: {{(unknown argument).*-disable-llvm-verifier}}
// CC1AsOptionCHECK6: {{(unknown argument).*-disable-objc-default-synthesize-properties}}
// CC1AsOptionCHECK6: {{(unknown argument).*-disable-pragma-debug-crash}}
// CC1AsOptionCHECK6: {{(unknown argument).*-disable-red-zone}}
// CC1AsOptionCHECK6: {{(unknown argument).*-discard-value-names}}
// CC1AsOptionCHECK6: {{(unknown argument).*--driver-mode=}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dsym-dir}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dump-coverage-mapping}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dump-deserialized-decls}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dump-raw-tokens}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dump-tokens}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dumpdir}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dumpmachine}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dumpspecs}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dumpversion}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dwarf-explicit-import}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dwarf-ext-refs}}
// CC1AsOptionCHECK6: {{(unknown argument).*-Fc}}
// CC1AsOptionCHECK6: {{(unknown argument).*-Fo}}
// CC1AsOptionCHECK6: {{(unknown argument).*-Vd}}
// CC1AsOptionCHECK6: {{(unknown argument).*--E}}
// CC1AsOptionCHECK6: {{(unknown argument).*-HV}}
// CC1AsOptionCHECK6: {{(unknown argument).*-hlsl-no-stdinc}}
// CC1AsOptionCHECK6: {{(unknown argument).*--dxv-path=}}
// CC1AsOptionCHECK6: {{(unknown argument).*-validator-version}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dylib_file}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dylinker}}
// CC1AsOptionCHECK6: {{(unknown argument).*-dylinker_install_name}}
// RUN: not %clang -cc1as -dynamic -dynamiclib -e -ehcontguard --embed-dir= -emit-ast -emit-cir -emit-codegen-only --emit-extension-symbol-graphs -emit-fir -emit-header-unit -emit-hlfir -emit-html -emit-interface-stubs -emit-llvm -emit-llvm-bc -emit-llvm-only -emit-llvm-uselists -emit-merged-ifs -emit-mlir -emit-module -emit-module-interface -emit-obj -emit-pch --pretty-sgf -emit-pristine-llvm -emit-reduced-module-interface --emit-sgf-symbol-labels-for-testing --emit-static-lib -emit-symbol-graph -enable-16bit-types -enable-noundef-analysis -enable-tlsdesc --end-no-unused-arguments -error-on-deserialized-decl -error-on-deserialized-decl= -exception-model -exception-model= -fexperimental-modules-reduced-bmi -exported_symbols_list -extract-api --extract-api-ignores= -fPIC -fPIE -faapcs-bitfield-width -faarch64-jump-table-hardening -faccess-control -faddress-space-map-mangling= -faddrsig -faggressive-function-elimination -falign-commons -falign-functions -falign-functions= -falign-jumps -falign-jumps= -falign-labels -falign-labels= -falign-loops -falign-loops= -faligned-allocation -faligned-new= -fall-intrinsics -fallow-editor-placeholders -fallow-pch-with-different-modules-cache-path -fallow-pch-with-compiler-errors -fallow-pcm-with-compiler-errors -fallow-unsupported -falternative-parameter-statement -faltivec -fanalyzed-objects-for-unparse -fandroid-pad-segment -fkeep-inline-functions -funit-at-a-time -fansi-escape-codes -fapinotes -fapinotes-modules -fapinotes-swift-version= -fapple-kext -fapple-link-rtlib -fapple-pragma-pack -fapplication-extension -fapply-global-visibility-to-externs -fapprox-func -fasm -fasm-blocks -fassociative-math -fassume-nothrow-exception-dtor -fassume-sane-operator-new -fassume-unique-vtables -fassumptions -fast -fastcp -fastf -fasync-exceptions -fasynchronous-unwind-tables -fauto-import -fauto-profile= -fauto-profile-accurate -fautolink -fautomatic  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK7 %s

// CC1AsOptionCHECK7: {{(unknown argument).*-dynamic}}
// CC1AsOptionCHECK7: {{(unknown argument).*-dynamiclib}}
// CC1AsOptionCHECK7: {{(unknown argument).*-e}}
// CC1AsOptionCHECK7: {{(unknown argument).*-ehcontguard}}
// CC1AsOptionCHECK7: {{(unknown argument).*--embed-dir=}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-ast}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-cir}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-codegen-only}}
// CC1AsOptionCHECK7: {{(unknown argument).*--emit-extension-symbol-graphs}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-fir}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-header-unit}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-hlfir}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-html}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-interface-stubs}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-llvm}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-llvm-bc}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-llvm-only}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-llvm-uselists}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-merged-ifs}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-mlir}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-module}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-module-interface}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-obj}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-pch}}
// CC1AsOptionCHECK7: {{(unknown argument).*--pretty-sgf}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-pristine-llvm}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-reduced-module-interface}}
// CC1AsOptionCHECK7: {{(unknown argument).*--emit-sgf-symbol-labels-for-testing}}
// CC1AsOptionCHECK7: {{(unknown argument).*--emit-static-lib}}
// CC1AsOptionCHECK7: {{(unknown argument).*-emit-symbol-graph}}
// CC1AsOptionCHECK7: {{(unknown argument).*-enable-16bit-types}}
// CC1AsOptionCHECK7: {{(unknown argument).*-enable-noundef-analysis}}
// CC1AsOptionCHECK7: {{(unknown argument).*-enable-tlsdesc}}
// CC1AsOptionCHECK7: {{(unknown argument).*--end-no-unused-arguments}}
// CC1AsOptionCHECK7: {{(unknown argument).*-error-on-deserialized-decl}}
// CC1AsOptionCHECK7: {{(unknown argument).*-error-on-deserialized-decl=}}
// CC1AsOptionCHECK7: {{(unknown argument).*-exception-model}}
// CC1AsOptionCHECK7: {{(unknown argument).*-exception-model=}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fexperimental-modules-reduced-bmi}}
// CC1AsOptionCHECK7: {{(unknown argument).*-exported_symbols_list}}
// CC1AsOptionCHECK7: {{(unknown argument).*-extract-api}}
// CC1AsOptionCHECK7: {{(unknown argument).*--extract-api-ignores=}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fPIC}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fPIE}}
// CC1AsOptionCHECK7: {{(unknown argument).*-faapcs-bitfield-width}}
// CC1AsOptionCHECK7: {{(unknown argument).*-faarch64-jump-table-hardening}}
// CC1AsOptionCHECK7: {{(unknown argument).*-faccess-control}}
// CC1AsOptionCHECK7: {{(unknown argument).*-faddress-space-map-mangling=}}
// CC1AsOptionCHECK7: {{(unknown argument).*-faddrsig}}
// CC1AsOptionCHECK7: {{(unknown argument).*-faggressive-function-elimination}}
// CC1AsOptionCHECK7: {{(unknown argument).*-falign-commons}}
// CC1AsOptionCHECK7: {{(unknown argument).*-falign-functions}}
// CC1AsOptionCHECK7: {{(unknown argument).*-falign-functions=}}
// CC1AsOptionCHECK7: {{(unknown argument).*-falign-jumps}}
// CC1AsOptionCHECK7: {{(unknown argument).*-falign-jumps=}}
// CC1AsOptionCHECK7: {{(unknown argument).*-falign-labels}}
// CC1AsOptionCHECK7: {{(unknown argument).*-falign-labels=}}
// CC1AsOptionCHECK7: {{(unknown argument).*-falign-loops}}
// CC1AsOptionCHECK7: {{(unknown argument).*-falign-loops=}}
// CC1AsOptionCHECK7: {{(unknown argument).*-faligned-allocation}}
// CC1AsOptionCHECK7: {{(unknown argument).*-faligned-new=}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fall-intrinsics}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fallow-editor-placeholders}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fallow-pch-with-different-modules-cache-path}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fallow-pch-with-compiler-errors}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fallow-pcm-with-compiler-errors}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fallow-unsupported}}
// CC1AsOptionCHECK7: {{(unknown argument).*-falternative-parameter-statement}}
// CC1AsOptionCHECK7: {{(unknown argument).*-faltivec}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fanalyzed-objects-for-unparse}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fandroid-pad-segment}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fkeep-inline-functions}}
// CC1AsOptionCHECK7: {{(unknown argument).*-funit-at-a-time}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fansi-escape-codes}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fapinotes}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fapinotes-modules}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fapinotes-swift-version=}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fapple-kext}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fapple-link-rtlib}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fapple-pragma-pack}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fapplication-extension}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fapply-global-visibility-to-externs}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fapprox-func}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fasm}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fasm-blocks}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fassociative-math}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fassume-nothrow-exception-dtor}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fassume-sane-operator-new}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fassume-unique-vtables}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fassumptions}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fast}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fastcp}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fastf}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fasync-exceptions}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fasynchronous-unwind-tables}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fauto-import}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fauto-profile=}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fauto-profile-accurate}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fautolink}}
// CC1AsOptionCHECK7: {{(unknown argument).*-fautomatic}}
// RUN: not %clang -cc1as -fbackslash -fbacktrace -fbasic-block-address-map -fbfloat16-excess-precision= -fbinutils-version= -fblas-matmul-limit= -fblocks -fblocks-runtime-optional -fbootclasspath= -fborland-extensions -fbounds-check -fexperimental-bounds-safety -fbracket-depth -fbracket-depth= -fbranch-count-reg -fbuild-session-file= -fbuild-session-timestamp= -fbuiltin -fbuiltin-headers-in-system-modules -fbuiltin-module-map -fcall-saved-x10 -fcall-saved-x11 -fcall-saved-x12 -fcall-saved-x13 -fcall-saved-x14 -fcall-saved-x15 -fcall-saved-x18 -fcall-saved-x8 -fcall-saved-x9 -fcaller-saves -fcaret-diagnostics -fcaret-diagnostics-max-lines= -fcf-protection -fcf-protection= -fcf-runtime-abi= -fcgl -fchar8_t -fcheck= -fcheck-array-temporaries -fcheck-new -fclang-abi-compat= -fclangir -fclasspath= -fcoarray= -fcodegen-data-generate -fcodegen-data-generate= -fcodegen-data-use -fcodegen-data-use= -fcolor-diagnostics -fcomment-block-commands= -fcommon -fcompatibility-qualified-id-block-type-checking -fcompile-resource= -fcomplete-member-pointers -fcomplex-arithmetic= -fconst-strings -fconstant-cfstrings -fconstant-string-class -fconstant-string-class= -fconstexpr-backtrace-limit= -fconstexpr-depth= -fconstexpr-steps= -fconvergent-functions -fconvert= -fcoro-aligned-allocation -fcoroutines -fcoverage-mapping -fcoverage-prefix-map= -fcrash-diagnostics -fcrash-diagnostics= -fcrash-diagnostics-dir= -fcray-pointer -fcreate-profile -fcs-profile-generate -fcs-profile-generate= -fctor-dtor-return-this -fcuda-allow-variadic-functions -fcuda-flush-denormals-to-zero -fcuda-include-gpubinary -fcuda-is-device -fcuda-short-ptr -fcx-fortran-rules -fcx-limited-range -fc++-abi= -fcxx-exceptions -fcxx-modules -fd-lines-as-code -fd-lines-as-comments -fdata-sections -fdebug-default-version= -fdebug-dump-all -fdebug-dump-parse-tree -fdebug-dump-parse-tree-no-sema -fdebug-dump-parsing-log -fdebug-dump-pft -fdebug-dump-provenance -fdebug-dump-symbols -fdebug-info-for-profiling -fdebug-macro -fdebug-measure-parse-tree  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK8 %s

// CC1AsOptionCHECK8: {{(unknown argument).*-fbackslash}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbacktrace}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbasic-block-address-map}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbfloat16-excess-precision=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbinutils-version=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fblas-matmul-limit=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fblocks}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fblocks-runtime-optional}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbootclasspath=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fborland-extensions}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbounds-check}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fexperimental-bounds-safety}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbracket-depth}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbracket-depth=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbranch-count-reg}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbuild-session-file=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbuild-session-timestamp=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbuiltin}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbuiltin-headers-in-system-modules}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fbuiltin-module-map}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcall-saved-x10}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcall-saved-x11}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcall-saved-x12}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcall-saved-x13}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcall-saved-x14}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcall-saved-x15}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcall-saved-x18}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcall-saved-x8}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcall-saved-x9}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcaller-saves}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcaret-diagnostics}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcaret-diagnostics-max-lines=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcf-protection}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcf-protection=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcf-runtime-abi=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcgl}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fchar8_t}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcheck=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcheck-array-temporaries}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcheck-new}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fclang-abi-compat=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fclangir}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fclasspath=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcoarray=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcodegen-data-generate}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcodegen-data-generate=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcodegen-data-use}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcodegen-data-use=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcolor-diagnostics}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcomment-block-commands=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcommon}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcompatibility-qualified-id-block-type-checking}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcompile-resource=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcomplete-member-pointers}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcomplex-arithmetic=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fconst-strings}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fconstant-cfstrings}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fconstant-string-class}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fconstant-string-class=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fconstexpr-backtrace-limit=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fconstexpr-depth=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fconstexpr-steps=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fconvergent-functions}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fconvert=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcoro-aligned-allocation}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcoroutines}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcoverage-mapping}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcoverage-prefix-map=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcrash-diagnostics}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcrash-diagnostics=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcrash-diagnostics-dir=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcray-pointer}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcreate-profile}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcs-profile-generate}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcs-profile-generate=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fctor-dtor-return-this}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcuda-allow-variadic-functions}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcuda-flush-denormals-to-zero}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcuda-include-gpubinary}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcuda-is-device}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcuda-short-ptr}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcx-fortran-rules}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcx-limited-range}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fc\+\+-abi=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcxx-exceptions}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fcxx-modules}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fd-lines-as-code}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fd-lines-as-comments}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fdata-sections}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fdebug-default-version=}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fdebug-dump-all}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fdebug-dump-parse-tree}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fdebug-dump-parse-tree-no-sema}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fdebug-dump-parsing-log}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fdebug-dump-pft}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fdebug-dump-provenance}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fdebug-dump-symbols}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fdebug-info-for-profiling}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fdebug-macro}}
// CC1AsOptionCHECK8: {{(unknown argument).*-fdebug-measure-parse-tree}}
// RUN: not %clang -cc1as -fdebug-module-writer -fdebug-pass-arguments -fdebug-pass-manager -fdebug-pass-structure -fdebug-pre-fir-tree -fdebug-ranges-base-address -fdebug-types-section -fdebug-unparse -fdebug-unparse-no-sema -fdebug-unparse-with-modules -fdebug-unparse-with-symbols -fdebugger-cast-result-to-id -fdebugger-objc-literal -fdebugger-support -fdeclare-opencl-builtins -fdeclspec -fdefault-calling-conv= -fdefault-double-8 -fdefault-inline -fdefault-integer-8 -fdefault-real-8 -fdefine-target-os-macros -fdelayed-template-parsing -fdelete-null-pointer-checks -fdenormal-fp-math= -fdenormal-fp-math-f32= -fdepfile-entry= -fdeprecated-macro -fdevirtualize -fdevirtualize-speculatively -fdiagnostics-absolute-paths -fdiagnostics-color= -fdiagnostics-fixit-info -fdiagnostics-format -fdiagnostics-format= -fdiagnostics-hotness-threshold= -fdiagnostics-misexpect-tolerance= -fdiagnostics-parseable-fixits -fdiagnostics-print-source-range-info -fdiagnostics-show-category -fdiagnostics-show-category= -fdiagnostics-show-hotness -fdiagnostics-show-line-numbers -fdiagnostics-show-location= -fdiagnostics-show-note-include-stack -fdiagnostics-show-option -fdiagnostics-show-template-tree -fdigraphs -fdirect-access-external-data -fdirectives-only -fdisable-block-signature-string -fdisable-integer-16 -fdisable-integer-2 -fdisable-module-hash -fdisable-real-10 -fdisable-real-3 -fdiscard-value-names -fdollar-ok -fdollars-in-identifiers -fdouble-square-bracket-attributes -fdriver-only -fdump-fortran-optimized -fdump-fortran-original -fdump-parse-tree -fdump-record-layouts -fdump-record-layouts-canonical -fdump-record-layouts-complete -fdump-record-layouts-simple -fdump-vtable-layouts -fdwarf2-cfi-asm -fdwarf-directory-asm -fdwarf-exceptions -felide-constructors -feliminate-unused-debug-symbols -feliminate-unused-debug-types -fembed-bitcode -fembed-bitcode-marker -fembed-offload-object= -femit-all-decls -femulated-tls -fenable-matrix -fencode-extended-block-signature -fencoding= -ferror-limit -ferror-limit= -fescaping-block-tail-calls -fexceptions -fexcess-precision= -fexec-charset= -fexperimental-assignment-tracking= -fexperimental-isel -fexperimental-late-parse-attributes -fexperimental-library -fexperimental-max-bitint-width= -fexperimental-new-constant-interpreter -fexperimental-omit-vtable-rtti -fexperimental-relative-c++-abi-vtables -fexperimental-sanitize-metadata= -fexperimental-sanitize-metadata=atomics -fexperimental-sanitize-metadata=covered  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK9 %s

// CC1AsOptionCHECK9: {{(unknown argument).*-fdebug-module-writer}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebug-pass-arguments}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebug-pass-manager}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebug-pass-structure}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebug-pre-fir-tree}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebug-ranges-base-address}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebug-types-section}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebug-unparse}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebug-unparse-no-sema}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebug-unparse-with-modules}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebug-unparse-with-symbols}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebugger-cast-result-to-id}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebugger-objc-literal}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdebugger-support}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdeclare-opencl-builtins}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdeclspec}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdefault-calling-conv=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdefault-double-8}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdefault-inline}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdefault-integer-8}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdefault-real-8}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdefine-target-os-macros}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdelayed-template-parsing}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdelete-null-pointer-checks}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdenormal-fp-math=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdenormal-fp-math-f32=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdepfile-entry=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdeprecated-macro}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdevirtualize}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdevirtualize-speculatively}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-absolute-paths}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-color=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-fixit-info}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-format}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-format=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-hotness-threshold=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-misexpect-tolerance=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-parseable-fixits}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-print-source-range-info}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-show-category}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-show-category=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-show-hotness}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-show-line-numbers}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-show-location=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-show-note-include-stack}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-show-option}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiagnostics-show-template-tree}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdigraphs}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdirect-access-external-data}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdirectives-only}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdisable-block-signature-string}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdisable-integer-16}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdisable-integer-2}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdisable-module-hash}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdisable-real-10}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdisable-real-3}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdiscard-value-names}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdollar-ok}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdollars-in-identifiers}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdouble-square-bracket-attributes}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdriver-only}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdump-fortran-optimized}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdump-fortran-original}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdump-parse-tree}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdump-record-layouts}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdump-record-layouts-canonical}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdump-record-layouts-complete}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdump-record-layouts-simple}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdump-vtable-layouts}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdwarf2-cfi-asm}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdwarf-directory-asm}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fdwarf-exceptions}}
// CC1AsOptionCHECK9: {{(unknown argument).*-felide-constructors}}
// CC1AsOptionCHECK9: {{(unknown argument).*-feliminate-unused-debug-symbols}}
// CC1AsOptionCHECK9: {{(unknown argument).*-feliminate-unused-debug-types}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fembed-bitcode}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fembed-bitcode-marker}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fembed-offload-object=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-femit-all-decls}}
// CC1AsOptionCHECK9: {{(unknown argument).*-femulated-tls}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fenable-matrix}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fencode-extended-block-signature}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fencoding=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-ferror-limit}}
// CC1AsOptionCHECK9: {{(unknown argument).*-ferror-limit=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fescaping-block-tail-calls}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexceptions}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexcess-precision=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexec-charset=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexperimental-assignment-tracking=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexperimental-isel}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexperimental-late-parse-attributes}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexperimental-library}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexperimental-max-bitint-width=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexperimental-new-constant-interpreter}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexperimental-omit-vtable-rtti}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexperimental-relative-c\+\+-abi-vtables}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexperimental-sanitize-metadata=}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexperimental-sanitize-metadata=atomics}}
// CC1AsOptionCHECK9: {{(unknown argument).*-fexperimental-sanitize-metadata=covered}}
// RUN: not %clang -cc1as -fexperimental-sanitize-metadata=uar -fexperimental-sanitize-metadata-ignorelist= -fexperimental-strict-floating-point -fextdirs= -fextend-arguments= -fextend-variable-liveness -fextend-variable-liveness= -fexternal-blas -fexternc-nounwind -ff2c -ffake-address-space-map -ffast-math -ffat-lto-objects -ffile-compilation-dir= -ffile-prefix-map= -ffile-reproducible -fimplicit-modules-use-lock -ffine-grained-bitfield-accesses -ffinite-loops -ffinite-math-only -finline-limit -ffixed-a0 -ffixed-a1 -ffixed-a2 -ffixed-a3 -ffixed-a4 -ffixed-a5 -ffixed-a6 -ffixed-d0 -ffixed-d1 -ffixed-d2 -ffixed-d3 -ffixed-d4 -ffixed-d5 -ffixed-d6 -ffixed-d7 -ffixed-form -ffixed-g1 -ffixed-g2 -ffixed-g3 -ffixed-g4 -ffixed-g5 -ffixed-g6 -ffixed-g7 -ffixed-i0 -ffixed-i1 -ffixed-i2 -ffixed-i3 -ffixed-i4 -ffixed-i5 -ffixed-l0 -ffixed-l1 -ffixed-l2 -ffixed-l3 -ffixed-l4 -ffixed-l5 -ffixed-l6 -ffixed-l7 -ffixed-line-length= -ffixed-line-length- -ffixed-o0 -ffixed-o1 -ffixed-o2 -ffixed-o3 -ffixed-o4 -ffixed-o5 -ffixed-point -ffixed-r19 -ffixed-r9 -ffixed-x1 -ffixed-x10 -ffixed-x11 -ffixed-x12 -ffixed-x13 -ffixed-x14 -ffixed-x15 -ffixed-x16 -ffixed-x17 -ffixed-x18 -ffixed-x19 -ffixed-x2 -ffixed-x20 -ffixed-x21 -ffixed-x22 -ffixed-x23 -ffixed-x24 -ffixed-x25 -ffixed-x26 -ffixed-x27 -ffixed-x28 -ffixed-x29 -ffixed-x3 -ffixed-x30 -ffixed-x31 -ffixed-x4 -ffixed-x5 -ffixed-x6 -ffixed-x7 -ffixed-x8 -ffixed-x9  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK10 %s

// CC1AsOptionCHECK10: {{(unknown argument).*-fexperimental-sanitize-metadata=uar}}
// CC1AsOptionCHECK10: {{(unknown argument).*-fexperimental-sanitize-metadata-ignorelist=}}
// CC1AsOptionCHECK10: {{(unknown argument).*-fexperimental-strict-floating-point}}
// CC1AsOptionCHECK10: {{(unknown argument).*-fextdirs=}}
// CC1AsOptionCHECK10: {{(unknown argument).*-fextend-arguments=}}
// CC1AsOptionCHECK10: {{(unknown argument).*-fextend-variable-liveness}}
// CC1AsOptionCHECK10: {{(unknown argument).*-fextend-variable-liveness=}}
// CC1AsOptionCHECK10: {{(unknown argument).*-fexternal-blas}}
// CC1AsOptionCHECK10: {{(unknown argument).*-fexternc-nounwind}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ff2c}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffake-address-space-map}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffast-math}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffat-lto-objects}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffile-compilation-dir=}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffile-prefix-map=}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffile-reproducible}}
// CC1AsOptionCHECK10: {{(unknown argument).*-fimplicit-modules-use-lock}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffine-grained-bitfield-accesses}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffinite-loops}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffinite-math-only}}
// CC1AsOptionCHECK10: {{(unknown argument).*-finline-limit}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-a0}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-a1}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-a2}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-a3}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-a4}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-a5}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-a6}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-d0}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-d1}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-d2}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-d3}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-d4}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-d5}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-d6}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-d7}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-form}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-g1}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-g2}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-g3}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-g4}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-g5}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-g6}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-g7}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-i0}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-i1}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-i2}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-i3}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-i4}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-i5}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-l0}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-l1}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-l2}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-l3}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-l4}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-l5}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-l6}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-l7}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-line-length=}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-line-length-}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-o0}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-o1}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-o2}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-o3}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-o4}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-o5}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-point}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-r19}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-r9}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x1}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x10}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x11}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x12}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x13}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x14}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x15}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x16}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x17}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x18}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x19}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x2}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x20}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x21}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x22}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x23}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x24}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x25}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x26}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x27}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x28}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x29}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x3}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x30}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x31}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x4}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x5}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x6}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x7}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x8}}
// CC1AsOptionCHECK10: {{(unknown argument).*-ffixed-x9}}
// RUN: not %clang -cc1as -ffloat16-excess-precision= -ffloat-store -ffor-scope -fforbid-guard-variables -fforce-check-cxx20-modules-input-files -fforce-dwarf-frame -fforce-emit-vtables -fforce-enable-int128 -ffp-contract= -ffp-eval-method= -ffp-exception-behavior= -ffp-model= -ffpe-trap= -ffree-form -ffree-line-length- -ffreestanding -ffriend-injection -ffrontend-optimize -ffuchsia-api-level= -ffunction-attribute-list -ffunction-sections -fgcse -fgcse-after-reload -fgcse-las -fgcse-sm -fget-definition -fget-symbols-sources -fglobal-isel -fgnu -fgnu89-inline -fgnu-inline-asm -fgnu-keywords -fgnu-runtime -fgnuc-version= -fgpu-allow-device-init -fgpu-approx-transcendentals -fgpu-default-stream= -fgpu-defer-diag -fgpu-exclude-wrong-side-overloads -fgpu-flush-denormals-to-zero -fgpu-inline-threshold= -fgpu-rdc -fgpu-sanitize -fhalf-no-semantic-interposition -fheinous-gnu-extensions -fhermetic-module-files -fhip-dump-offload-linker-script -fhip-emit-relocatable -fhip-fp32-correctly-rounded-divide-sqrt -fhip-kernel-arg-name -fhip-new-launch-api -fhlsl-strict-availability -fhonor-infinities -fhonor-nans -fhosted -fignore-exceptions -filelist -fimplement-inlines -fimplicit-module-maps -fimplicit-modules -fimplicit-none -fimplicit-none-ext -fimplicit-templates -finclude-default-header -fincremental-extensions -finit-character= -finit-global-zero -finit-integer= -finit-local-zero -finit-logical= -finit-real= -finline -finline-functions -finline-functions-called-once -finline-hint-functions -finline-limit= -finline-max-stacksize= -finline-small-functions -finput-charset= -finstrument-function-entry-bare -finstrument-functions -finstrument-functions-after-inlining -finteger-4-integer-8 -fintegrated-as -fintegrated-cc1 -fintegrated-objemitter -fintrinsic-modules-path -fipa-cp -fivopts -fix-only-warnings -fix-what-you-can -fixit -fixit= -fixit-recompile -fixit-to-temporary -fjmc -fjump-tables -fkeep-persistent-storage-variables -fkeep-static-consts -fkeep-system-includes  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK11 %s

// CC1AsOptionCHECK11: {{(unknown argument).*-ffloat16-excess-precision=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffloat-store}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffor-scope}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fforbid-guard-variables}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fforce-check-cxx20-modules-input-files}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fforce-dwarf-frame}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fforce-emit-vtables}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fforce-enable-int128}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffp-contract=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffp-eval-method=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffp-exception-behavior=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffp-model=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffpe-trap=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffree-form}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffree-line-length-}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffreestanding}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffriend-injection}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffrontend-optimize}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffuchsia-api-level=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffunction-attribute-list}}
// CC1AsOptionCHECK11: {{(unknown argument).*-ffunction-sections}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgcse}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgcse-after-reload}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgcse-las}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgcse-sm}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fget-definition}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fget-symbols-sources}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fglobal-isel}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgnu}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgnu89-inline}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgnu-inline-asm}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgnu-keywords}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgnu-runtime}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgnuc-version=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgpu-allow-device-init}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgpu-approx-transcendentals}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgpu-default-stream=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgpu-defer-diag}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgpu-exclude-wrong-side-overloads}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgpu-flush-denormals-to-zero}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgpu-inline-threshold=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgpu-rdc}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fgpu-sanitize}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fhalf-no-semantic-interposition}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fheinous-gnu-extensions}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fhermetic-module-files}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fhip-dump-offload-linker-script}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fhip-emit-relocatable}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fhip-fp32-correctly-rounded-divide-sqrt}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fhip-kernel-arg-name}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fhip-new-launch-api}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fhlsl-strict-availability}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fhonor-infinities}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fhonor-nans}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fhosted}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fignore-exceptions}}
// CC1AsOptionCHECK11: {{(unknown argument).*-filelist}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fimplement-inlines}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fimplicit-module-maps}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fimplicit-modules}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fimplicit-none}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fimplicit-none-ext}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fimplicit-templates}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finclude-default-header}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fincremental-extensions}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finit-character=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finit-global-zero}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finit-integer=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finit-local-zero}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finit-logical=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finit-real=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finline}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finline-functions}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finline-functions-called-once}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finline-hint-functions}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finline-limit=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finline-max-stacksize=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finline-small-functions}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finput-charset=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finstrument-function-entry-bare}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finstrument-functions}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finstrument-functions-after-inlining}}
// CC1AsOptionCHECK11: {{(unknown argument).*-finteger-4-integer-8}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fintegrated-as}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fintegrated-cc1}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fintegrated-objemitter}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fintrinsic-modules-path}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fipa-cp}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fivopts}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fix-only-warnings}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fix-what-you-can}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fixit}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fixit=}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fixit-recompile}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fixit-to-temporary}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fjmc}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fjump-tables}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fkeep-persistent-storage-variables}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fkeep-static-consts}}
// CC1AsOptionCHECK11: {{(unknown argument).*-fkeep-system-includes}}
// RUN: not %clang -cc1as -flang-deprecated-no-hlfir -flang-experimental-hlfir -flarge-sizes -flat_namespace -flax-vector-conversions -flax-vector-conversions= -flimit-debug-info -flimited-precision= -flogical-abbreviations -floop-interchange -fversion-loops-for-stride -flto -flto= -flto=auto -flto=jobserver -flto-jobs= -flto-unit -flto-visibility-public-std -fmacro-backtrace-limit= -fmacro-prefix-map= -fmath-errno -fmax-array-constructor= -fmax-errors= -fmax-identifier-length -fmax-stack-var-size= -fmax-subrecord-length= -fmax-tokens= -fmax-type-align= -fcoverage-mcdc -fmcdc-max-conditions= -fmcdc-max-test-vectors= -fmemory-profile -fmemory-profile= -fmemory-profile-use= -fmerge-all-constants -fmerge-constants -fmerge-functions -fmessage-length= -fminimize-whitespace -fmodule-feature -fmodule-file= -fmodule-file-deps -fmodule-file-home-is-cwd -fmodule-format= -fmodule-header -fmodule-header= -fmodule-implementation-of -fmodule-map-file= -fmodule-map-file-home-is-cwd -fmodule-maps -fmodule-name= -fmodule-output -fmodule-output= -fmodule-private -fmodulemap-allow-subdirectory-search -fmodules -fmodules-cache-path= -fmodules-codegen -fmodules-debuginfo -fmodules-decluse -fmodules-disable-diagnostic-validation -fmodules-embed-all-files -fmodules-embed-file= -fmodules-hash-content -fmodules-ignore-macro= -fmodules-local-submodule-visibility -fmodules-prune-after= -fmodules-prune-interval= -fmodules-search-all -fmodules-skip-diagnostic-options -fmodules-skip-header-search-paths -fmodules-strict-context-hash -fmodules-strict-decluse -fmodules-user-build-path -fmodules-validate-input-files-content -fmodules-validate-once-per-build-session -fmodules-validate-system-headers -fmodulo-sched -fmodulo-sched-allow-regmoves -fms-compatibility -fms-compatibility-version= -fms-define-stdc -fms-extensions -fms-hotpatch -fms-kernel -fms-memptr-rep= -fms-omit-default-lib -fms-runtime-lib= -fms-tls-guards -fms-volatile -fmsc-version= -fmudflap -fmudflapth -fmultilib-flag= -fnative-half-arguments-and-returns -fnative-half-type -fnested-functions -fnew-alignment= -fnew-infallible -fnext-runtime  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK12 %s

// CC1AsOptionCHECK12: {{(unknown argument).*-flang-deprecated-no-hlfir}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flang-experimental-hlfir}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flarge-sizes}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flat_namespace}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flax-vector-conversions}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flax-vector-conversions=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flimit-debug-info}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flimited-precision=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flogical-abbreviations}}
// CC1AsOptionCHECK12: {{(unknown argument).*-floop-interchange}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fversion-loops-for-stride}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flto}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flto=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flto=auto}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flto=jobserver}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flto-jobs=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flto-unit}}
// CC1AsOptionCHECK12: {{(unknown argument).*-flto-visibility-public-std}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmacro-backtrace-limit=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmacro-prefix-map=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmath-errno}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmax-array-constructor=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmax-errors=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmax-identifier-length}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmax-stack-var-size=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmax-subrecord-length=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmax-tokens=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmax-type-align=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fcoverage-mcdc}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmcdc-max-conditions=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmcdc-max-test-vectors=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmemory-profile}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmemory-profile=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmemory-profile-use=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmerge-all-constants}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmerge-constants}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmerge-functions}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmessage-length=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fminimize-whitespace}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-feature}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-file=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-file-deps}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-file-home-is-cwd}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-format=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-header}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-header=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-implementation-of}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-map-file=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-map-file-home-is-cwd}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-maps}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-name=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-output}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-output=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodule-private}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodulemap-allow-subdirectory-search}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-cache-path=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-codegen}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-debuginfo}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-decluse}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-disable-diagnostic-validation}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-embed-all-files}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-embed-file=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-hash-content}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-ignore-macro=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-local-submodule-visibility}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-prune-after=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-prune-interval=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-search-all}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-skip-diagnostic-options}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-skip-header-search-paths}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-strict-context-hash}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-strict-decluse}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-user-build-path}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-validate-input-files-content}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-validate-once-per-build-session}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodules-validate-system-headers}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodulo-sched}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmodulo-sched-allow-regmoves}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fms-compatibility}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fms-compatibility-version=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fms-define-stdc}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fms-extensions}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fms-hotpatch}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fms-kernel}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fms-memptr-rep=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fms-omit-default-lib}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fms-runtime-lib=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fms-tls-guards}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fms-volatile}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmsc-version=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmudflap}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmudflapth}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fmultilib-flag=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fnative-half-arguments-and-returns}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fnative-half-type}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fnested-functions}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fnew-alignment=}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fnew-infallible}}
// CC1AsOptionCHECK12: {{(unknown argument).*-fnext-runtime}}
// RUN: not %clang -cc1as -fno-PIC -fno-PIE -fno-aapcs-bitfield-width -fno-aarch64-jump-table-hardening -fno-access-control -fno-addrsig -fno-aggressive-function-elimination -fno-align-commons -fno-align-functions -fno-align-jumps -fno-align-labels -fno-align-loops -fno-aligned-allocation -fno-all-intrinsics -fno-allow-editor-placeholders -fno-altivec -fno-analyzed-objects-for-unparse -fno-android-pad-segment -fno-keep-inline-functions -fno-unit-at-a-time -fno-apinotes -fno-apinotes-modules -fno-apple-pragma-pack -fno-application-extension -fno-approx-func -fno-asm -fno-asm-blocks -fno-associative-math -fno-assume-nothrow-exception-dtor -fno-assume-sane-operator-new -fno-assume-unique-vtables -fno-assumptions -fno-async-exceptions -fno-asynchronous-unwind-tables -fno-auto-import -fno-auto-profile -fno-auto-profile-accurate -fno-autolink -fno-automatic -fno-backslash -fno-backtrace -fno-basic-block-address-map -fno-bitfield-type-align -fno-blocks -fno-borland-extensions -fno-bounds-check -fno-experimental-bounds-safety -fno-branch-count-reg -fno-builtin -fno-builtin- -fno-caller-saves -fno-caret-diagnostics -fno-char8_t -fno-check-array-temporaries -fno-check-new -fno-clangir -fno-color-diagnostics -fno-common -fno-complete-member-pointers -fno-const-strings -fno-constant-cfstrings -fno-convergent-functions -fno-coro-aligned-allocation -fno-coroutines -fno-coverage-mapping -fno-crash-diagnostics -fno-cray-pointer -fno-cuda-flush-denormals-to-zero -fno-cuda-host-device-constexpr -fno-cuda-short-ptr -fno-cx-fortran-rules -fno-cx-limited-range -fno-cxx-exceptions -fno-cxx-modules -fno-d-lines-as-code -fno-d-lines-as-comments -fno-data-sections -fno-debug-info-for-profiling -fno-debug-macro -fno-debug-pass-manager -fno-debug-ranges-base-address -fno-debug-types-section -fno-declspec -fno-default-inline -fno-define-target-os-macros -fno-delayed-template-parsing -fno-delete-null-pointer-checks -fno-deprecated-macro -fno-devirtualize -fno-devirtualize-speculatively -fno-diagnostics-fixit-info -fno-diagnostics-show-hotness -fno-diagnostics-show-line-numbers -fno-diagnostics-show-note-include-stack -fno-diagnostics-show-option -fno-diagnostics-use-presumed-location -fno-digraphs -fno-direct-access-external-data -fno-directives-only -fno-disable-block-signature-string  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK13 %s

// CC1AsOptionCHECK13: {{(unknown argument).*-fno-PIC}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-PIE}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-aapcs-bitfield-width}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-aarch64-jump-table-hardening}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-access-control}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-addrsig}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-aggressive-function-elimination}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-align-commons}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-align-functions}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-align-jumps}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-align-labels}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-align-loops}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-aligned-allocation}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-all-intrinsics}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-allow-editor-placeholders}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-altivec}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-analyzed-objects-for-unparse}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-android-pad-segment}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-keep-inline-functions}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-unit-at-a-time}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-apinotes}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-apinotes-modules}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-apple-pragma-pack}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-application-extension}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-approx-func}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-asm}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-asm-blocks}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-associative-math}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-assume-nothrow-exception-dtor}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-assume-sane-operator-new}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-assume-unique-vtables}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-assumptions}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-async-exceptions}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-asynchronous-unwind-tables}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-auto-import}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-auto-profile}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-auto-profile-accurate}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-autolink}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-automatic}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-backslash}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-backtrace}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-basic-block-address-map}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-bitfield-type-align}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-blocks}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-borland-extensions}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-bounds-check}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-experimental-bounds-safety}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-branch-count-reg}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-builtin}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-builtin-}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-caller-saves}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-caret-diagnostics}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-char8_t}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-check-array-temporaries}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-check-new}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-clangir}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-color-diagnostics}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-common}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-complete-member-pointers}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-const-strings}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-constant-cfstrings}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-convergent-functions}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-coro-aligned-allocation}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-coroutines}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-coverage-mapping}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-crash-diagnostics}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-cray-pointer}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-cuda-flush-denormals-to-zero}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-cuda-host-device-constexpr}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-cuda-short-ptr}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-cx-fortran-rules}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-cx-limited-range}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-cxx-exceptions}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-cxx-modules}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-d-lines-as-code}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-d-lines-as-comments}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-data-sections}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-debug-info-for-profiling}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-debug-macro}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-debug-pass-manager}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-debug-ranges-base-address}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-debug-types-section}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-declspec}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-default-inline}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-define-target-os-macros}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-delayed-template-parsing}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-delete-null-pointer-checks}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-deprecated-macro}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-devirtualize}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-devirtualize-speculatively}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-diagnostics-fixit-info}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-diagnostics-show-hotness}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-diagnostics-show-line-numbers}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-diagnostics-show-note-include-stack}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-diagnostics-show-option}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-diagnostics-use-presumed-location}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-digraphs}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-direct-access-external-data}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-directives-only}}
// CC1AsOptionCHECK13: {{(unknown argument).*-fno-disable-block-signature-string}}
// RUN: not %clang -cc1as -fno-discard-value-names -fno-dllexport-inlines -fno-dollar-ok -fno-dollars-in-identifiers -fno-double-square-bracket-attributes -fno-dump-fortran-optimized -fno-dump-fortran-original -fno-dump-parse-tree -fno-dwarf2-cfi-asm -fno-dwarf-directory-asm -fno-elide-constructors -fno-elide-type -fno-eliminate-unused-debug-symbols -fno-eliminate-unused-debug-types -fno-emit-compact-unwind-non-canonical -fno-emulated-tls -fno-escaping-block-tail-calls -fno-exceptions -fno-experimental-isel -fno-experimental-late-parse-attributes -fno-experimental-library -fno-experimental-omit-vtable-rtti -fno-experimental-relative-c++-abi-vtables -fno-experimental-sanitize-metadata= -fno-external-blas -fno-f2c -fno-fast-math -fno-fat-lto-objects -fno-file-reproducible -fno-implicit-modules-use-lock -fno-fine-grained-bitfield-accesses -fno-finite-loops -fno-finite-math-only -fno-inline-limit -fno-fixed-point -fno-float-store -fno-for-scope -fno-force-dwarf-frame -fno-force-emit-vtables -fno-force-enable-int128 -fno-friend-injection -fno-frontend-optimize -fno-function-attribute-list -fno-function-sections -fno-gcse -fno-gcse-after-reload -fno-gcse-las -fno-gcse-sm -fno-global-isel -fno-gnu -fno-gnu89-inline -fno-gnu-inline-asm -fno-gnu-keywords -fno-gpu-allow-device-init -fno-gpu-approx-transcendentals -fno-gpu-defer-diag -fno-gpu-exclude-wrong-side-overloads -fno-gpu-flush-denormals-to-zero -fno-gpu-rdc -fno-gpu-sanitize -fno-hip-emit-relocatable -fno-hip-fp32-correctly-rounded-divide-sqrt -fno-hip-kernel-arg-name -fno-hip-new-launch-api -fno-honor-infinities -fno-honor-nans -fno-implement-inlines -fno-implicit-module-maps -fno-implicit-modules -fno-implicit-none -fno-implicit-none-ext -fno-implicit-templates -fno-init-global-zero -fno-init-local-zero -fno-inline -fno-inline-functions -fno-inline-functions-called-once -fno-inline-small-functions -fno-integer-4-integer-8 -fno-integrated-as -fno-integrated-cc1 -fno-integrated-objemitter -fno-ipa-cp -fno-ivopts -fno-jmc -fno-jump-tables -fno-keep-persistent-storage-variables -fno-keep-static-consts -fno-keep-system-includes -fno-knr-functions -fno-lax-vector-conversions -fno-limit-debug-info -fno-logical-abbreviations -fno-loop-interchange -fno-version-loops-for-stride -fno-lto -fno-lto-unit -fno-math-errno -fno-max-identifier-length -fno-max-type-align  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK14 %s

// CC1AsOptionCHECK14: {{(unknown argument).*-fno-discard-value-names}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-dllexport-inlines}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-dollar-ok}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-dollars-in-identifiers}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-double-square-bracket-attributes}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-dump-fortran-optimized}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-dump-fortran-original}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-dump-parse-tree}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-dwarf2-cfi-asm}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-dwarf-directory-asm}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-elide-constructors}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-elide-type}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-eliminate-unused-debug-symbols}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-eliminate-unused-debug-types}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-emit-compact-unwind-non-canonical}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-emulated-tls}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-escaping-block-tail-calls}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-exceptions}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-experimental-isel}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-experimental-late-parse-attributes}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-experimental-library}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-experimental-omit-vtable-rtti}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-experimental-relative-c\+\+-abi-vtables}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-experimental-sanitize-metadata=}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-external-blas}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-f2c}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-fast-math}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-fat-lto-objects}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-file-reproducible}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-implicit-modules-use-lock}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-fine-grained-bitfield-accesses}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-finite-loops}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-finite-math-only}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-inline-limit}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-fixed-point}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-float-store}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-for-scope}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-force-dwarf-frame}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-force-emit-vtables}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-force-enable-int128}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-friend-injection}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-frontend-optimize}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-function-attribute-list}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-function-sections}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gcse}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gcse-after-reload}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gcse-las}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gcse-sm}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-global-isel}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gnu}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gnu89-inline}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gnu-inline-asm}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gnu-keywords}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gpu-allow-device-init}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gpu-approx-transcendentals}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gpu-defer-diag}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gpu-exclude-wrong-side-overloads}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gpu-flush-denormals-to-zero}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gpu-rdc}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-gpu-sanitize}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-hip-emit-relocatable}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-hip-fp32-correctly-rounded-divide-sqrt}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-hip-kernel-arg-name}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-hip-new-launch-api}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-honor-infinities}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-honor-nans}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-implement-inlines}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-implicit-module-maps}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-implicit-modules}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-implicit-none}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-implicit-none-ext}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-implicit-templates}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-init-global-zero}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-init-local-zero}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-inline}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-inline-functions}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-inline-functions-called-once}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-inline-small-functions}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-integer-4-integer-8}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-integrated-as}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-integrated-cc1}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-integrated-objemitter}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-ipa-cp}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-ivopts}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-jmc}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-jump-tables}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-keep-persistent-storage-variables}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-keep-static-consts}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-keep-system-includes}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-knr-functions}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-lax-vector-conversions}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-limit-debug-info}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-logical-abbreviations}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-loop-interchange}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-version-loops-for-stride}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-lto}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-lto-unit}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-math-errno}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-max-identifier-length}}
// CC1AsOptionCHECK14: {{(unknown argument).*-fno-max-type-align}}
// RUN: not %clang -cc1as -fno-coverage-mcdc -fno-memory-profile -fno-merge-all-constants -fno-merge-constants -fno-minimize-whitespace -fno-module-file-deps -fno-module-maps -fno-module-private -fno-modulemap-allow-subdirectory-search -fno-modules -fno-modules-check-relocated -fno-modules-decluse -fno-modules-error-recovery -fno-modules-global-index -fno-modules-prune-non-affecting-module-map-files -fno-modules-search-all -fno-modules-share-filemanager -fno-modules-skip-diagnostic-options -fno-modules-skip-header-search-paths -fno-strict-modules-decluse -fno_modules-validate-input-files-content -fno-modules-validate-system-headers -fno-modules-validate-textual-header-includes -fno-modulo-sched -fno-modulo-sched-allow-regmoves -fno-ms-compatibility -fno-ms-extensions -fno-ms-tls-guards -fno-ms-volatile -fno-new-infallible -fno-non-call-exceptions -fno-objc-arc -fno-objc-arc-exceptions -fno-objc-avoid-heapify-local-blocks -fno-objc-convert-messages-to-runtime-calls -fno-objc-encode-cxx-class-template-spec -fno-objc-exceptions -fno-objc-infer-related-result-type -fno-objc-legacy-dispatch -fno-objc-nonfragile-abi -fno-objc-weak -fno-offload-implicit-host-device-templates -fno-offload-lto -fno-offload-uniform-block -fno-offload-via-llvm -fno-omit-frame-pointer -fno-openmp -fno-openmp-assume-teams-oversubscription -fno-openmp-assume-threads-oversubscription -fno-openmp-cuda-mode -fno-openmp-extensions -fno-openmp-new-driver -fno-openmp-optimistic-collapse -fno-openmp-simd -fno-openmp-target-debug -fno-openmp-target-jit -fno-openmp-target-new-runtime -fno-operator-names -fno-optimize-sibling-calls -fno-pack-derived -fno-pack-struct -fno-padding-on-unsigned-fixed-point -fno-pascal-strings -fno-pch-codegen -fno-pch-debuginfo -fno-pch-instantiate-templates -fno-pch-timestamp -fno_pch-validate-input-files-content -fno-peel-loops -fno-permissive -fno-pic -fno-pie -fno-plt -fno-pointer-tbaa -fno-ppc-native-vector-element-order -fno-prebuilt-implicit-modules -fno-prefetch-loop-arrays -fno-preserve-as-comments -fno-printf -fno-profile -fno-profile-arcs -fno-profile-correction -fno-profile-generate -fno-profile-generate-sampling -fno-profile-instr-generate -fno-profile-instr-use -fno-profile-reusedist -fno-profile-sample-accurate -fno-profile-sample-use -fno-profile-use -fno-profile-values -fno-protect-parens -fno-pseudo-probe-for-profiling -fno-ptrauth-auth-traps -fno-ptrauth-calls -fno-ptrauth-elf-got -fno-ptrauth-function-pointer-type-discrimination -fno-ptrauth-indirect-gotos -fno-ptrauth-init-fini -fno-ptrauth-init-fini-address-discrimination  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK15 %s

// CC1AsOptionCHECK15: {{(unknown argument).*-fno-coverage-mcdc}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-memory-profile}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-merge-all-constants}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-merge-constants}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-minimize-whitespace}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-module-file-deps}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-module-maps}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-module-private}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modulemap-allow-subdirectory-search}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modules}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modules-check-relocated}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modules-decluse}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modules-error-recovery}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modules-global-index}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modules-prune-non-affecting-module-map-files}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modules-search-all}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modules-share-filemanager}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modules-skip-diagnostic-options}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modules-skip-header-search-paths}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-strict-modules-decluse}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno_modules-validate-input-files-content}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modules-validate-system-headers}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modules-validate-textual-header-includes}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modulo-sched}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-modulo-sched-allow-regmoves}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-ms-compatibility}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-ms-extensions}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-ms-tls-guards}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-ms-volatile}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-new-infallible}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-non-call-exceptions}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-objc-arc}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-objc-arc-exceptions}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-objc-avoid-heapify-local-blocks}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-objc-convert-messages-to-runtime-calls}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-objc-encode-cxx-class-template-spec}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-objc-exceptions}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-objc-infer-related-result-type}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-objc-legacy-dispatch}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-objc-nonfragile-abi}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-objc-weak}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-offload-implicit-host-device-templates}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-offload-lto}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-offload-uniform-block}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-offload-via-llvm}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-omit-frame-pointer}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-openmp}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-openmp-assume-teams-oversubscription}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-openmp-assume-threads-oversubscription}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-openmp-cuda-mode}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-openmp-extensions}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-openmp-new-driver}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-openmp-optimistic-collapse}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-openmp-simd}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-openmp-target-debug}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-openmp-target-jit}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-openmp-target-new-runtime}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-operator-names}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-optimize-sibling-calls}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-pack-derived}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-pack-struct}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-padding-on-unsigned-fixed-point}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-pascal-strings}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-pch-codegen}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-pch-debuginfo}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-pch-instantiate-templates}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-pch-timestamp}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno_pch-validate-input-files-content}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-peel-loops}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-permissive}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-pic}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-pie}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-plt}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-pointer-tbaa}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-ppc-native-vector-element-order}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-prebuilt-implicit-modules}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-prefetch-loop-arrays}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-preserve-as-comments}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-printf}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-profile}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-profile-arcs}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-profile-correction}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-profile-generate}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-profile-generate-sampling}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-profile-instr-generate}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-profile-instr-use}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-profile-reusedist}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-profile-sample-accurate}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-profile-sample-use}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-profile-use}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-profile-values}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-protect-parens}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-pseudo-probe-for-profiling}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-ptrauth-auth-traps}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-ptrauth-calls}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-ptrauth-elf-got}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-ptrauth-function-pointer-type-discrimination}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-ptrauth-indirect-gotos}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-ptrauth-init-fini}}
// CC1AsOptionCHECK15: {{(unknown argument).*-fno-ptrauth-init-fini-address-discrimination}}
// RUN: not %clang -cc1as -fno-ptrauth-intrinsics -fno-ptrauth-returns -fno-ptrauth-type-info-vtable-pointer-discrimination -fno-ptrauth-vtable-pointer-address-discrimination -fno-ptrauth-vtable-pointer-type-discrimination -fno-range-check -fno-raw-string-literals -fno-real-4-real-10 -fno-real-4-real-16 -fno-real-4-real-8 -fno-real-8-real-10 -fno-real-8-real-16 -fno-real-8-real-4 -fno-realloc-lhs -fno-reciprocal-math -fno-record-command-line -fno-recovery-ast -fno-recovery-ast-type -fno-recursive -fno-reformat -fno-register-global-dtors-with-atexit -fno-regs-graph -fno-rename-registers -fno-reorder-blocks -fno-repack-arrays -fno-retain-subst-template-type-parm-type-ast-nodes -fno-rewrite-imports -fno-rewrite-includes -fno-ripa -fno-ropi -fno-rounding-math -fno-rtlib-add-rpath -fno-rtlib-defaultlib -fno-rtti -fno-rtti-data -fno-rwpi -fno-safe-buffer-usage-suggestions -fno-sanitize= -fno-sanitize-address-globals-dead-stripping -fno-sanitize-address-outline-instrumentation -fno-sanitize-address-poison-custom-array-cookie -fno-sanitize-address-use-after-scope -fno-sanitize-address-use-odr-indicator -fno-sanitize-cfi-canonical-jump-tables -fno-sanitize-cfi-cross-dso -fno-sanitize-coverage= -fno-sanitize-hwaddress-experimental-aliasing -fno-sanitize-ignorelist -fno-sanitize-link-c++-runtime -fno-sanitize-link-runtime -fno-sanitize-memory-param-retval -fno-sanitize-memory-track-origins -fno-sanitize-memory-use-after-dtor -fno-sanitize-merge -fno-sanitize-merge= -fno-sanitize-minimal-runtime -fno-sanitize-recover -fno-sanitize-recover= -fno-sanitize-stable-abi -fno-sanitize-stats -fno-sanitize-thread-atomics -fno-sanitize-thread-func-entry-exit -fno-sanitize-thread-memory-access -fno-sanitize-trap -fno-sanitize-trap= -fno-sanitize-undefined-trap-on-error -fno-save-main-program -fno-save-optimization-record -fno-schedule-insns -fno-schedule-insns2 -fno-second-underscore -fno-see -fno-semantic-interposition -fno-separate-named-sections -fno-short-enums -fno-short-wchar -fno-show-column -fno-show-source-location -fno-sign-zero -fno-signaling-math -fno-signaling-nans -fno-signed-char -fno-signed-wchar -fno-signed-zeros -fno-single-precision-constant -fno-sized-deallocation -fno-skip-odr-check-in-gmf -fno-slp-vectorize -fno-spec-constr-count -fno-spell-checking -fno-split-dwarf-inlining -fno-split-lto-unit -fno-split-machine-functions -fno-split-stack -fno-stack-arrays -fno-stack-check -fno-stack-clash-protection -fno-stack-protector -fno-stack-size-section -fno-standalone-debug  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK16 %s

// CC1AsOptionCHECK16: {{(unknown argument).*-fno-ptrauth-intrinsics}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-ptrauth-returns}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-ptrauth-type-info-vtable-pointer-discrimination}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-ptrauth-vtable-pointer-address-discrimination}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-ptrauth-vtable-pointer-type-discrimination}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-range-check}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-raw-string-literals}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-real-4-real-10}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-real-4-real-16}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-real-4-real-8}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-real-8-real-10}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-real-8-real-16}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-real-8-real-4}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-realloc-lhs}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-reciprocal-math}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-record-command-line}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-recovery-ast}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-recovery-ast-type}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-recursive}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-reformat}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-register-global-dtors-with-atexit}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-regs-graph}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-rename-registers}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-reorder-blocks}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-repack-arrays}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-retain-subst-template-type-parm-type-ast-nodes}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-rewrite-imports}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-rewrite-includes}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-ripa}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-ropi}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-rounding-math}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-rtlib-add-rpath}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-rtlib-defaultlib}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-rtti}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-rtti-data}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-rwpi}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-safe-buffer-usage-suggestions}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize=}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-address-globals-dead-stripping}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-address-outline-instrumentation}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-address-poison-custom-array-cookie}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-address-use-after-scope}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-address-use-odr-indicator}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-cfi-canonical-jump-tables}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-cfi-cross-dso}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-coverage=}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-hwaddress-experimental-aliasing}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-ignorelist}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-link-c\+\+-runtime}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-link-runtime}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-memory-param-retval}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-memory-track-origins}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-memory-use-after-dtor}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-merge}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-merge=}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-minimal-runtime}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-recover}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-recover=}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-stable-abi}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-stats}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-thread-atomics}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-thread-func-entry-exit}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-thread-memory-access}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-trap}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-trap=}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sanitize-undefined-trap-on-error}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-save-main-program}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-save-optimization-record}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-schedule-insns}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-schedule-insns2}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-second-underscore}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-see}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-semantic-interposition}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-separate-named-sections}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-short-enums}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-short-wchar}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-show-column}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-show-source-location}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sign-zero}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-signaling-math}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-signaling-nans}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-signed-char}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-signed-wchar}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-signed-zeros}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-single-precision-constant}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-sized-deallocation}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-skip-odr-check-in-gmf}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-slp-vectorize}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-spec-constr-count}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-spell-checking}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-split-dwarf-inlining}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-split-lto-unit}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-split-machine-functions}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-split-stack}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-stack-arrays}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-stack-check}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-stack-clash-protection}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-stack-protector}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-stack-size-section}}
// CC1AsOptionCHECK16: {{(unknown argument).*-fno-standalone-debug}}
// RUN: not %clang -cc1as -fno-strength-reduce -fno-strict-aliasing -fno-strict-enums -fno-strict-float-cast-overflow -fno-strict-overflow -fno-strict-return -fno-strict-vtable-pointers -fno-struct-path-tbaa -fno-sycl -fno-temp-file -fno-test-coverage -fno-threadsafe-statics -fno-tls-model -fno-tracer -fno-trapping-math -fno-tree-dce -fno-tree-salias -fno-tree-ter -fno-tree-vectorizer-verbose -fno-tree-vrp -fno-trigraphs -fno-underscoring -fno-unified-lto -fno-unique-basic-block-section-names -fno-unique-internal-linkage-names -fno-unique-section-names -fno-unroll-all-loops -fno-unroll-loops -fno-unsafe-loop-optimizations -fno-unsafe-math-optimizations -fno-unsigned -fno-unsigned-char -fno-unswitch-loops -fno-unwind-tables -fno-use-cxa-atexit -fno-use-init-array -fno-use-line-directives -fno-use-linker-plugin -fno-validate-pch -fno-var-tracking -fno-variable-expansion-in-unroller -fno-vect-cost-model -fno-vectorize -fno-verbose-asm -fno-verify-intermediate-code -fno-virtual-function-elimination -fno-visibility-from-dllstorageclass -fno-visibility-inlines-hidden -fno-visibility-inlines-hidden-static-local-var -fno-wchar -fno-web -fno-whole-file -fno-whole-program -fno-whole-program-vtables -fno-working-directory -fno-wrapv -fno-wrapv-pointer -fno-xl-pragma-pack -fno-xor-operator -fno-xray-always-emit-customevents -fno-xray-always-emit-typedevents -fno-xray-function-index -fno-xray-ignore-loops -fno-xray-instrument -fno-xray-link-deps -fno-xray-shared -fno-zero-initialized-in-bss -fno-zos-extensions -fno-zvector -fnon-call-exceptions -fnoopenmp-relocatable-target -fnoopenmp-use-tls -fobjc-abi-version= -fobjc-arc -fobjc-arc-cxxlib= -fobjc-arc-exceptions -fobjc-atdefs -fobjc-avoid-heapify-local-blocks -fobjc-call-cxx-cdtors -fobjc-convert-messages-to-runtime-calls -fobjc-disable-direct-methods-for-testing -fobjc-dispatch-method= -fobjc-encode-cxx-class-template-spec -fobjc-exceptions -fobjc-gc -fobjc-gc-only -fobjc-infer-related-result-type -fobjc-legacy-dispatch -fobjc-link-runtime -fobjc-new-property -fobjc-nonfragile-abi -fobjc-nonfragile-abi-version= -fobjc-runtime= -fobjc-runtime-has-weak -fobjc-sender-dependent-dispatch -fobjc-subscripting-legacy-runtime -fobjc-weak -foffload-implicit-host-device-templates -foffload-lto -foffload-lto=  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK17 %s

// CC1AsOptionCHECK17: {{(unknown argument).*-fno-strength-reduce}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-strict-aliasing}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-strict-enums}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-strict-float-cast-overflow}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-strict-overflow}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-strict-return}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-strict-vtable-pointers}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-struct-path-tbaa}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-sycl}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-temp-file}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-test-coverage}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-threadsafe-statics}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-tls-model}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-tracer}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-trapping-math}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-tree-dce}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-tree-salias}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-tree-ter}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-tree-vectorizer-verbose}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-tree-vrp}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-trigraphs}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-underscoring}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-unified-lto}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-unique-basic-block-section-names}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-unique-internal-linkage-names}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-unique-section-names}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-unroll-all-loops}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-unroll-loops}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-unsafe-loop-optimizations}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-unsafe-math-optimizations}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-unsigned}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-unsigned-char}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-unswitch-loops}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-unwind-tables}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-use-cxa-atexit}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-use-init-array}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-use-line-directives}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-use-linker-plugin}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-validate-pch}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-var-tracking}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-variable-expansion-in-unroller}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-vect-cost-model}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-vectorize}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-verbose-asm}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-verify-intermediate-code}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-virtual-function-elimination}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-visibility-from-dllstorageclass}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-visibility-inlines-hidden}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-visibility-inlines-hidden-static-local-var}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-wchar}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-web}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-whole-file}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-whole-program}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-whole-program-vtables}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-working-directory}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-wrapv}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-wrapv-pointer}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-xl-pragma-pack}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-xor-operator}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-xray-always-emit-customevents}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-xray-always-emit-typedevents}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-xray-function-index}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-xray-ignore-loops}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-xray-instrument}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-xray-link-deps}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-xray-shared}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-zero-initialized-in-bss}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-zos-extensions}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fno-zvector}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fnon-call-exceptions}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fnoopenmp-relocatable-target}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fnoopenmp-use-tls}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-abi-version=}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-arc}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-arc-cxxlib=}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-arc-exceptions}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-atdefs}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-avoid-heapify-local-blocks}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-call-cxx-cdtors}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-convert-messages-to-runtime-calls}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-disable-direct-methods-for-testing}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-dispatch-method=}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-encode-cxx-class-template-spec}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-exceptions}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-gc}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-gc-only}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-infer-related-result-type}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-legacy-dispatch}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-link-runtime}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-new-property}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-nonfragile-abi}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-nonfragile-abi-version=}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-runtime=}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-runtime-has-weak}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-sender-dependent-dispatch}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-subscripting-legacy-runtime}}
// CC1AsOptionCHECK17: {{(unknown argument).*-fobjc-weak}}
// CC1AsOptionCHECK17: {{(unknown argument).*-foffload-implicit-host-device-templates}}
// CC1AsOptionCHECK17: {{(unknown argument).*-foffload-lto}}
// CC1AsOptionCHECK17: {{(unknown argument).*-foffload-lto=}}
// RUN: not %clang -cc1as -foffload-uniform-block -foffload-via-llvm -fomit-frame-pointer -fopenacc -fopenmp -fopenmp= -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state -fopenmp-assume-teams-oversubscription -fopenmp-assume-threads-oversubscription -fopenmp-cuda-blocks-per-sm= -fopenmp-cuda-mode -fopenmp-cuda-number-of-sm= -fopenmp-cuda-teams-reduction-recs-num= -fopenmp-enable-irbuilder -fopenmp-extensions -fopenmp-force-usm -fopenmp-host-ir-file-path -fopenmp-is-target-device -fopenmp-new-driver -fopenmp-offload-mandatory -fopenmp-optimistic-collapse -fopenmp-relocatable-target -fopenmp-simd -fopenmp-target-debug -fopenmp-target-debug= -fopenmp-target-jit -fopenmp-target-new-runtime -fopenmp-targets= -fopenmp-use-tls -fopenmp-version= -foperator-arrow-depth= -foperator-names -foptimization-record-file= -foptimization-record-passes= -foptimize-sibling-calls -force_cpusubtype_ALL -force_flat_namespace -force_load -fforce-addr -forder-file-instrumentation -foutput-class-dir= -foverride-record-layout= -fpack-derived -fpack-struct -fpack-struct= -fpadding-on-unsigned-fixed-point -fparse-all-comments -fpascal-strings -fpass-by-value-is-noalias -fpass-plugin= -fpatchable-function-entry= -fpatchable-function-entry-offset= -fpcc-struct-return -fpch-codegen -fpch-debuginfo -fpch-instantiate-templates -fpch-preprocess -fpch-validate-input-files-content -fpeel-loops -fpermissive -fpic -fpie -fplt -fplugin= -fplugin-arg- -fpointer-tbaa -fppc-native-vector-element-order -fprebuilt-implicit-modules -fprebuilt-module-path= -fprefetch-loop-arrays -fpreprocess-include-lines -fpreserve-as-comments -fprintf -fproc-stat-report -fproc-stat-report= -fprofile -fprofile-arcs -fprofile-continuous -fprofile-correction -fprofile-dir= -fprofile-exclude-files= -fprofile-filter-files= -fprofile-function-groups= -fprofile-generate -fprofile-generate= -fprofile-generate-cold-function-coverage -fprofile-generate-cold-function-coverage= -fprofile-generate-sampling -fprofile-instr-generate -fprofile-instr-generate= -fprofile-instr-use -fprofile-instr-use= -fprofile-instrument= -fprofile-instrument-path= -fprofile-instrument-use-path= -fprofile-list= -fprofile-remapping-file= -fprofile-reusedist -fprofile-sample-accurate  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK18 %s

// CC1AsOptionCHECK18: {{(unknown argument).*-foffload-uniform-block}}
// CC1AsOptionCHECK18: {{(unknown argument).*-foffload-via-llvm}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fomit-frame-pointer}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenacc}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-assume-no-nested-parallelism}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-assume-no-thread-state}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-assume-teams-oversubscription}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-assume-threads-oversubscription}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-cuda-blocks-per-sm=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-cuda-mode}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-cuda-number-of-sm=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-cuda-teams-reduction-recs-num=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-enable-irbuilder}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-extensions}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-force-usm}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-host-ir-file-path}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-is-target-device}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-new-driver}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-offload-mandatory}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-optimistic-collapse}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-relocatable-target}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-simd}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-target-debug}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-target-debug=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-target-jit}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-target-new-runtime}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-targets=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-use-tls}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fopenmp-version=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-foperator-arrow-depth=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-foperator-names}}
// CC1AsOptionCHECK18: {{(unknown argument).*-foptimization-record-file=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-foptimization-record-passes=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-foptimize-sibling-calls}}
// CC1AsOptionCHECK18: {{(unknown argument).*-force_cpusubtype_ALL}}
// CC1AsOptionCHECK18: {{(unknown argument).*-force_flat_namespace}}
// CC1AsOptionCHECK18: {{(unknown argument).*-force_load}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fforce-addr}}
// CC1AsOptionCHECK18: {{(unknown argument).*-forder-file-instrumentation}}
// CC1AsOptionCHECK18: {{(unknown argument).*-foutput-class-dir=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-foverride-record-layout=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpack-derived}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpack-struct}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpack-struct=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpadding-on-unsigned-fixed-point}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fparse-all-comments}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpascal-strings}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpass-by-value-is-noalias}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpass-plugin=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpatchable-function-entry=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpatchable-function-entry-offset=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpcc-struct-return}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpch-codegen}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpch-debuginfo}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpch-instantiate-templates}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpch-preprocess}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpch-validate-input-files-content}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpeel-loops}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpermissive}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpic}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpie}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fplt}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fplugin=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fplugin-arg-}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpointer-tbaa}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fppc-native-vector-element-order}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprebuilt-implicit-modules}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprebuilt-module-path=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprefetch-loop-arrays}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpreprocess-include-lines}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fpreserve-as-comments}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprintf}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fproc-stat-report}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fproc-stat-report=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-arcs}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-continuous}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-correction}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-dir=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-exclude-files=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-filter-files=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-function-groups=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-generate}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-generate=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-generate-cold-function-coverage}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-generate-cold-function-coverage=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-generate-sampling}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-instr-generate}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-instr-generate=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-instr-use}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-instr-use=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-instrument=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-instrument-path=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-instrument-use-path=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-list=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-remapping-file=}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-reusedist}}
// CC1AsOptionCHECK18: {{(unknown argument).*-fprofile-sample-accurate}}
// RUN: not %clang -cc1as -fprofile-sample-use= -fprofile-selected-function-group= -fprofile-update= -fprofile-use -fprofile-use= -fprofile-values -fprotect-parens -fpseudo-probe-for-profiling -fptrauth-auth-traps -fptrauth-calls -fptrauth-elf-got -fptrauth-function-pointer-type-discrimination -fptrauth-indirect-gotos -fptrauth-init-fini -fptrauth-init-fini-address-discrimination -fptrauth-intrinsics -fptrauth-returns -fptrauth-type-info-vtable-pointer-discrimination -fptrauth-vtable-pointer-address-discrimination -fptrauth-vtable-pointer-type-discrimination -framework -frandom-seed= -frandomize-layout-seed= -frandomize-layout-seed-file= -frange-check -fraw-string-literals -freal-4-real-10 -freal-4-real-16 -freal-4-real-8 -freal-8-real-10 -freal-8-real-16 -freal-8-real-4 -frealloc-lhs -freciprocal-math -frecord-command-line -frecord-marker= -frecovery-ast -frecovery-ast-type -frecursive -freg-struct-return -fregister-global-dtors-with-atexit -fregs-graph -frename-registers -freorder-blocks -frepack-arrays -fretain-comments-from-system-headers -fretain-subst-template-type-parm-type-ast-nodes -frewrite-imports -frewrite-includes -fripa -fropi -frounding-math -frtlib-add-rpath -frtlib-defaultlib -frtti -frtti-data -frwpi -fsafe-buffer-usage-suggestions -fsample-profile-use-profi -fsanitize= -fsanitize-address-field-padding= -fsanitize-address-globals-dead-stripping -fsanitize-address-outline-instrumentation -fsanitize-address-poison-custom-array-cookie -fsanitize-address-use-after-scope -fsanitize-address-use-odr-indicator -fsanitize-cfi-canonical-jump-tables -fsanitize-cfi-cross-dso -fsanitize-cfi-icall-generalize-pointers -fsanitize-cfi-icall-experimental-normalize-integers -fsanitize-coverage= -fsanitize-coverage-8bit-counters -fsanitize-coverage-allowlist= -fsanitize-coverage-control-flow -fsanitize-coverage-ignorelist= -fsanitize-coverage-indirect-calls -fsanitize-coverage-inline-8bit-counters -fsanitize-coverage-inline-bool-flag -fsanitize-coverage-no-prune -fsanitize-coverage-pc-table -fsanitize-coverage-stack-depth -fsanitize-coverage-trace-bb -fsanitize-coverage-trace-cmp -fsanitize-coverage-trace-div -fsanitize-coverage-trace-gep -fsanitize-coverage-trace-loads -fsanitize-coverage-trace-pc -fsanitize-coverage-trace-pc-guard -fsanitize-coverage-trace-stores -fsanitize-coverage-type= -fsanitize-hwaddress-abi= -fsanitize-hwaddress-experimental-aliasing -fsanitize-ignorelist= -fsanitize-kcfi-arity -fsanitize-link-c++-runtime -fsanitize-link-runtime -fsanitize-memory-param-retval -fsanitize-memory-track-origins -fsanitize-memory-track-origins= -fsanitize-memory-use-after-dtor  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK19 %s

// CC1AsOptionCHECK19: {{(unknown argument).*-fprofile-sample-use=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fprofile-selected-function-group=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fprofile-update=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fprofile-use}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fprofile-use=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fprofile-values}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fprotect-parens}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fpseudo-probe-for-profiling}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fptrauth-auth-traps}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fptrauth-calls}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fptrauth-elf-got}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fptrauth-function-pointer-type-discrimination}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fptrauth-indirect-gotos}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fptrauth-init-fini}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fptrauth-init-fini-address-discrimination}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fptrauth-intrinsics}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fptrauth-returns}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fptrauth-type-info-vtable-pointer-discrimination}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fptrauth-vtable-pointer-address-discrimination}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fptrauth-vtable-pointer-type-discrimination}}
// CC1AsOptionCHECK19: {{(unknown argument).*-framework}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frandom-seed=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frandomize-layout-seed=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frandomize-layout-seed-file=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frange-check}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fraw-string-literals}}
// CC1AsOptionCHECK19: {{(unknown argument).*-freal-4-real-10}}
// CC1AsOptionCHECK19: {{(unknown argument).*-freal-4-real-16}}
// CC1AsOptionCHECK19: {{(unknown argument).*-freal-4-real-8}}
// CC1AsOptionCHECK19: {{(unknown argument).*-freal-8-real-10}}
// CC1AsOptionCHECK19: {{(unknown argument).*-freal-8-real-16}}
// CC1AsOptionCHECK19: {{(unknown argument).*-freal-8-real-4}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frealloc-lhs}}
// CC1AsOptionCHECK19: {{(unknown argument).*-freciprocal-math}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frecord-command-line}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frecord-marker=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frecovery-ast}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frecovery-ast-type}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frecursive}}
// CC1AsOptionCHECK19: {{(unknown argument).*-freg-struct-return}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fregister-global-dtors-with-atexit}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fregs-graph}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frename-registers}}
// CC1AsOptionCHECK19: {{(unknown argument).*-freorder-blocks}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frepack-arrays}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fretain-comments-from-system-headers}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fretain-subst-template-type-parm-type-ast-nodes}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frewrite-imports}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frewrite-includes}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fripa}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fropi}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frounding-math}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frtlib-add-rpath}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frtlib-defaultlib}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frtti}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frtti-data}}
// CC1AsOptionCHECK19: {{(unknown argument).*-frwpi}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsafe-buffer-usage-suggestions}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsample-profile-use-profi}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-address-field-padding=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-address-globals-dead-stripping}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-address-outline-instrumentation}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-address-poison-custom-array-cookie}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-address-use-after-scope}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-address-use-odr-indicator}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-cfi-canonical-jump-tables}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-cfi-cross-dso}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-cfi-icall-generalize-pointers}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-cfi-icall-experimental-normalize-integers}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-8bit-counters}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-allowlist=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-control-flow}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-ignorelist=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-indirect-calls}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-inline-8bit-counters}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-inline-bool-flag}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-no-prune}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-pc-table}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-stack-depth}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-trace-bb}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-trace-cmp}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-trace-div}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-trace-gep}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-trace-loads}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-trace-pc}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-trace-pc-guard}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-trace-stores}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-coverage-type=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-hwaddress-abi=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-hwaddress-experimental-aliasing}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-ignorelist=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-kcfi-arity}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-link-c\+\+-runtime}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-link-runtime}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-memory-param-retval}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-memory-track-origins}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-memory-track-origins=}}
// CC1AsOptionCHECK19: {{(unknown argument).*-fsanitize-memory-use-after-dtor}}
// RUN: not %clang -cc1as -fsanitize-memtag-mode= -fsanitize-merge -fsanitize-merge= -fsanitize-minimal-runtime -fsanitize-recover -fsanitize-recover= -fsanitize-skip-hot-cutoff= -fsanitize-stable-abi -fsanitize-stats -fsanitize-system-ignorelist= -fsanitize-thread-atomics -fsanitize-thread-func-entry-exit -fsanitize-thread-memory-access -fsanitize-trap -fsanitize-trap= -fsanitize-undefined-ignore-overflow-pattern= -fsanitize-undefined-strip-path-components= -fsanitize-undefined-trap-on-error -fsave-main-program -fsave-optimization-record -fsave-optimization-record= -fschedule-insns -fschedule-insns2 -fsecond-underscore -fsee -fseh-exceptions -fsemantic-interposition -fseparate-named-sections -fshort-enums -fshort-wchar -fshow-column -fshow-overloads= -fshow-skipped-includes -fshow-source-location -fsign-zero -fsignaling-math -fsignaling-nans -fsigned-bitfields -fsigned-char -fsigned-wchar -fsigned-zeros -fsingle-precision-constant -fsized-deallocation -fsjlj-exceptions -fskip-odr-check-in-gmf -fslp-vectorize -fspec-constr-count -fspell-checking -fspell-checking-limit= -fsplit-dwarf-inlining -fsplit-lto-unit -fsplit-machine-functions -fsplit-stack -fspv-target-env= -fstack-arrays -fstack-check -fstack-clash-protection -fstack-protector -fstack-protector-all -fstack-protector-strong -fstack-size-section -fstack-usage -fstandalone-debug -fstrength-reduce -fstrict-aliasing -fstrict-enums -fstrict-flex-arrays= -fstrict-float-cast-overflow -fstrict-overflow -fstrict-return -fstrict-vtable-pointers -fstruct-path-tbaa -fsycl -fsycl-device-only -fsycl-host-only -fsycl-is-device -fsycl-is-host -fsymbol-partition= -fsyntax-only -fsystem-module -ftabstop -ftabstop= -ftemplate-backtrace-limit= -ftemplate-depth= -ftemporal-profile -ftest-coverage -ftest-module-file-extension= -fthin-link-bitcode= -fthinlto-index= -fthreadsafe-statics -ftime-report -ftime-report= -ftime-trace -ftime-trace= -ftime-trace-granularity= -ftime-trace-verbose -ftls-model -ftls-model= -ftracer -ftrap-function=  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK20 %s

// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-memtag-mode=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-merge}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-merge=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-minimal-runtime}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-recover}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-recover=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-skip-hot-cutoff=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-stable-abi}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-stats}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-system-ignorelist=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-thread-atomics}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-thread-func-entry-exit}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-thread-memory-access}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-trap}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-trap=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-undefined-ignore-overflow-pattern=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-undefined-strip-path-components=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsanitize-undefined-trap-on-error}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsave-main-program}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsave-optimization-record}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsave-optimization-record=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fschedule-insns}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fschedule-insns2}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsecond-underscore}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsee}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fseh-exceptions}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsemantic-interposition}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fseparate-named-sections}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fshort-enums}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fshort-wchar}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fshow-column}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fshow-overloads=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fshow-skipped-includes}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fshow-source-location}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsign-zero}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsignaling-math}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsignaling-nans}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsigned-bitfields}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsigned-char}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsigned-wchar}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsigned-zeros}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsingle-precision-constant}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsized-deallocation}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsjlj-exceptions}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fskip-odr-check-in-gmf}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fslp-vectorize}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fspec-constr-count}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fspell-checking}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fspell-checking-limit=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsplit-dwarf-inlining}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsplit-lto-unit}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsplit-machine-functions}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsplit-stack}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fspv-target-env=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstack-arrays}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstack-check}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstack-clash-protection}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstack-protector}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstack-protector-all}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstack-protector-strong}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstack-size-section}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstack-usage}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstandalone-debug}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstrength-reduce}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstrict-aliasing}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstrict-enums}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstrict-flex-arrays=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstrict-float-cast-overflow}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstrict-overflow}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstrict-return}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstrict-vtable-pointers}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fstruct-path-tbaa}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsycl}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsycl-device-only}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsycl-host-only}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsycl-is-device}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsycl-is-host}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsymbol-partition=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsyntax-only}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fsystem-module}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftabstop}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftabstop=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftemplate-backtrace-limit=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftemplate-depth=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftemporal-profile}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftest-coverage}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftest-module-file-extension=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fthin-link-bitcode=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fthinlto-index=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-fthreadsafe-statics}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftime-report}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftime-report=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftime-trace}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftime-trace=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftime-trace-granularity=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftime-trace-verbose}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftls-model}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftls-model=}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftracer}}
// CC1AsOptionCHECK20: {{(unknown argument).*-ftrap-function=}}
// RUN: not %clang -cc1as -ftrapping-math -ftrapv -ftrapv-handler -ftrapv-handler= -ftree-dce -ftree-salias -ftree-ter -ftree-vectorizer-verbose -ftree-vrp -ftrigraphs -ftrivial-auto-var-init= -ftrivial-auto-var-init-max-size= -ftrivial-auto-var-init-stop-after= -ftype-visibility= -function-alignment -funderscoring -funified-lto -funique-basic-block-section-names -funique-internal-linkage-names -funique-section-names -funknown-anytype -funroll-all-loops -funroll-loops -funsafe-loop-optimizations -funsafe-math-optimizations -funsigned -funsigned-bitfields -funsigned-char -funswitch-loops -funwind-tables -funwind-tables= -fuse-cuid= -fuse-cxa-atexit -fuse-init-array -fuse-ld= -fuse-line-directives -fuse-linker-plugin -fuse-lipo= -fuse-register-sized-bitfield-access -fvalidate-ast-input-files-content -fvariable-expansion-in-unroller -fveclib= -fvect-cost-model -fvectorize -fverbose-asm -fverify-debuginfo-preserve -fverify-debuginfo-preserve-export= -fverify-intermediate-code -fvirtual-function-elimination -fvisibility= -fvisibility-dllexport= -fvisibility-externs-dllimport= -fvisibility-externs-nodllstorageclass= -fvisibility-from-dllstorageclass -fvisibility-global-new-delete= -fvisibility-global-new-delete-hidden -fvisibility-inlines-hidden -fvisibility-inlines-hidden-static-local-var -fvisibility-ms-compat -fvisibility-nodllstorageclass= -fwarn-stack-size= -fwasm-exceptions -fwchar-type= -fweb -fwhole-file -fwhole-program -fwhole-program-vtables -fwrapv -fwrapv-pointer -fwritable-strings -fxl-pragma-pack -fxor-operator -fxray-always-emit-customevents -fxray-always-emit-typedevents -fxray-always-instrument= -fxray-attr-list= -fxray-function-groups= -fxray-function-index -fxray-ignore-loops -fxray-instruction-threshold= -fxray-instrument -fxray-instrumentation-bundle= -fxray-link-deps -fxray-modes= -fxray-never-instrument= -fxray-selected-function-group= -fxray-shared -fzero-call-used-regs= -fzero-initialized-in-bss -fzos-extensions -fzvector -g0 -g1 -g2 -g3 -g --gcc-install-dir= --gcc-toolchain= --gcc-triple= -gcodeview-command-line  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK21 %s

// CC1AsOptionCHECK21: {{(unknown argument).*-ftrapping-math}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftrapv}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftrapv-handler}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftrapv-handler=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftree-dce}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftree-salias}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftree-ter}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftree-vectorizer-verbose}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftree-vrp}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftrigraphs}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftrivial-auto-var-init=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftrivial-auto-var-init-max-size=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftrivial-auto-var-init-stop-after=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-ftype-visibility=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-function-alignment}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funderscoring}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funified-lto}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funique-basic-block-section-names}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funique-internal-linkage-names}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funique-section-names}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funknown-anytype}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funroll-all-loops}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funroll-loops}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funsafe-loop-optimizations}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funsafe-math-optimizations}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funsigned}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funsigned-bitfields}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funsigned-char}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funswitch-loops}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funwind-tables}}
// CC1AsOptionCHECK21: {{(unknown argument).*-funwind-tables=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fuse-cuid=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fuse-cxa-atexit}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fuse-init-array}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fuse-ld=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fuse-line-directives}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fuse-linker-plugin}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fuse-lipo=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fuse-register-sized-bitfield-access}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvalidate-ast-input-files-content}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvariable-expansion-in-unroller}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fveclib=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvect-cost-model}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvectorize}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fverbose-asm}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fverify-debuginfo-preserve}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fverify-debuginfo-preserve-export=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fverify-intermediate-code}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvirtual-function-elimination}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvisibility=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvisibility-dllexport=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvisibility-externs-dllimport=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvisibility-externs-nodllstorageclass=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvisibility-from-dllstorageclass}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvisibility-global-new-delete=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvisibility-global-new-delete-hidden}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvisibility-inlines-hidden}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvisibility-inlines-hidden-static-local-var}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvisibility-ms-compat}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fvisibility-nodllstorageclass=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fwarn-stack-size=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fwasm-exceptions}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fwchar-type=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fweb}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fwhole-file}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fwhole-program}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fwhole-program-vtables}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fwrapv}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fwrapv-pointer}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fwritable-strings}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxl-pragma-pack}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxor-operator}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-always-emit-customevents}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-always-emit-typedevents}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-always-instrument=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-attr-list=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-function-groups=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-function-index}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-ignore-loops}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-instruction-threshold=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-instrument}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-instrumentation-bundle=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-link-deps}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-modes=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-never-instrument=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-selected-function-group=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fxray-shared}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fzero-call-used-regs=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fzero-initialized-in-bss}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fzos-extensions}}
// CC1AsOptionCHECK21: {{(unknown argument).*-fzvector}}
// CC1AsOptionCHECK21: {{(unknown argument).*-g0}}
// CC1AsOptionCHECK21: {{(unknown argument).*-g1}}
// CC1AsOptionCHECK21: {{(unknown argument).*-g2}}
// CC1AsOptionCHECK21: {{(unknown argument).*-g3}}
// CC1AsOptionCHECK21: {{(unknown argument).*-g}}
// CC1AsOptionCHECK21: {{(unknown argument).*--gcc-install-dir=}}
// CC1AsOptionCHECK21: {{(unknown argument).*--gcc-toolchain=}}
// CC1AsOptionCHECK21: {{(unknown argument).*--gcc-triple=}}
// CC1AsOptionCHECK21: {{(unknown argument).*-gcodeview-command-line}}
// RUN: not %clang -cc1as -gcodeview-ghash -gcoff -gcolumn-info -gdbx -gdwarf -gdwarf-2 -gdwarf-3 -gdwarf-4 -gdwarf-5 -gdwarf-aranges -gembed-source -gen-cdb-fragment-path -gen-reproducer -gen-reproducer= -gfull -ggdb -ggdb0 -ggdb1 -ggdb2 -ggdb3 -ggnu-pubnames -ginline-line-tables -gline-directives-only -gline-tables-only -glldb -gmlt -gmodules -gno-codeview-command-line -gno-codeview-ghash -gno-column-info -gno-embed-source -gno-gnu-pubnames -gno-inline-line-tables -gno-modules -gno-omit-unreferenced-methods -gno-pubnames -gno-record-command-line -gno-simple-template-names -gno-split-dwarf -gno-strict-dwarf -gno-template-alias -gomit-unreferenced-methods --gpu-bundle-output --gpu-instrument-lib= --gpu-max-threads-per-block= --gpu-use-aux-triple-only -gpubnames -gpulibc -grecord-command-line -gsce -gsimple-template-names -gsimple-template-names= -gsplit-dwarf -gsplit-dwarf= -gsrc-hash= -gstabs -gstrict-dwarf -gtemplate-alias -gtoggle -gused -gvms -gxcoff -gz -gz= -header-include-file -header-include-filtering= -header-include-format= -headerpad_max_install_names --hip-device-lib= --hip-link --hip-path= --hip-version= --hipspv-pass-plugin= --hipstdpar --hipstdpar-interpose-alloc --hipstdpar-path= --hipstdpar-prim-path= --hipstdpar-thrust-path= -hlsl-entry -iapinotes-modules -ibuiltininc -idirafter -iframework -iframeworkwithsysroot -imacros -image_base -import-call-optimization -imultilib -include -include-pch -init -init-only -inline-asm= -install_name -interface-stub-version= -internal-externc-isystem -internal-isystem -iprefix -iquote -isysroot  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK22 %s

// CC1AsOptionCHECK22: {{(unknown argument).*-gcodeview-ghash}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gcoff}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gcolumn-info}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gdbx}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gdwarf}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gdwarf-2}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gdwarf-3}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gdwarf-4}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gdwarf-5}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gdwarf-aranges}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gembed-source}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gen-cdb-fragment-path}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gen-reproducer}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gen-reproducer=}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gfull}}
// CC1AsOptionCHECK22: {{(unknown argument).*-ggdb}}
// CC1AsOptionCHECK22: {{(unknown argument).*-ggdb0}}
// CC1AsOptionCHECK22: {{(unknown argument).*-ggdb1}}
// CC1AsOptionCHECK22: {{(unknown argument).*-ggdb2}}
// CC1AsOptionCHECK22: {{(unknown argument).*-ggdb3}}
// CC1AsOptionCHECK22: {{(unknown argument).*-ggnu-pubnames}}
// CC1AsOptionCHECK22: {{(unknown argument).*-ginline-line-tables}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gline-directives-only}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gline-tables-only}}
// CC1AsOptionCHECK22: {{(unknown argument).*-glldb}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gmlt}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gmodules}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-codeview-command-line}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-codeview-ghash}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-column-info}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-embed-source}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-gnu-pubnames}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-inline-line-tables}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-modules}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-omit-unreferenced-methods}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-pubnames}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-record-command-line}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-simple-template-names}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-split-dwarf}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-strict-dwarf}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gno-template-alias}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gomit-unreferenced-methods}}
// CC1AsOptionCHECK22: {{(unknown argument).*--gpu-bundle-output}}
// CC1AsOptionCHECK22: {{(unknown argument).*--gpu-instrument-lib=}}
// CC1AsOptionCHECK22: {{(unknown argument).*--gpu-max-threads-per-block=}}
// CC1AsOptionCHECK22: {{(unknown argument).*--gpu-use-aux-triple-only}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gpubnames}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gpulibc}}
// CC1AsOptionCHECK22: {{(unknown argument).*-grecord-command-line}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gsce}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gsimple-template-names}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gsimple-template-names=}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gsplit-dwarf}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gsplit-dwarf=}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gsrc-hash=}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gstabs}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gstrict-dwarf}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gtemplate-alias}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gtoggle}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gused}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gvms}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gxcoff}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gz}}
// CC1AsOptionCHECK22: {{(unknown argument).*-gz=}}
// CC1AsOptionCHECK22: {{(unknown argument).*-header-include-file}}
// CC1AsOptionCHECK22: {{(unknown argument).*-header-include-filtering=}}
// CC1AsOptionCHECK22: {{(unknown argument).*-header-include-format=}}
// CC1AsOptionCHECK22: {{(unknown argument).*-headerpad_max_install_names}}
// CC1AsOptionCHECK22: {{(unknown argument).*--hip-device-lib=}}
// CC1AsOptionCHECK22: {{(unknown argument).*--hip-link}}
// CC1AsOptionCHECK22: {{(unknown argument).*--hip-path=}}
// CC1AsOptionCHECK22: {{(unknown argument).*--hip-version=}}
// CC1AsOptionCHECK22: {{(unknown argument).*--hipspv-pass-plugin=}}
// CC1AsOptionCHECK22: {{(unknown argument).*--hipstdpar}}
// CC1AsOptionCHECK22: {{(unknown argument).*--hipstdpar-interpose-alloc}}
// CC1AsOptionCHECK22: {{(unknown argument).*--hipstdpar-path=}}
// CC1AsOptionCHECK22: {{(unknown argument).*--hipstdpar-prim-path=}}
// CC1AsOptionCHECK22: {{(unknown argument).*--hipstdpar-thrust-path=}}
// CC1AsOptionCHECK22: {{(unknown argument).*-hlsl-entry}}
// CC1AsOptionCHECK22: {{(unknown argument).*-iapinotes-modules}}
// CC1AsOptionCHECK22: {{(unknown argument).*-ibuiltininc}}
// CC1AsOptionCHECK22: {{(unknown argument).*-idirafter}}
// CC1AsOptionCHECK22: {{(unknown argument).*-iframework}}
// CC1AsOptionCHECK22: {{(unknown argument).*-iframeworkwithsysroot}}
// CC1AsOptionCHECK22: {{(unknown argument).*-imacros}}
// CC1AsOptionCHECK22: {{(unknown argument).*-image_base}}
// CC1AsOptionCHECK22: {{(unknown argument).*-import-call-optimization}}
// CC1AsOptionCHECK22: {{(unknown argument).*-imultilib}}
// CC1AsOptionCHECK22: {{(unknown argument).*-include}}
// CC1AsOptionCHECK22: {{(unknown argument).*-include-pch}}
// CC1AsOptionCHECK22: {{(unknown argument).*-init}}
// CC1AsOptionCHECK22: {{(unknown argument).*-init-only}}
// CC1AsOptionCHECK22: {{(unknown argument).*-inline-asm=}}
// CC1AsOptionCHECK22: {{(unknown argument).*-install_name}}
// CC1AsOptionCHECK22: {{(unknown argument).*-interface-stub-version=}}
// CC1AsOptionCHECK22: {{(unknown argument).*-internal-externc-isystem}}
// CC1AsOptionCHECK22: {{(unknown argument).*-internal-isystem}}
// CC1AsOptionCHECK22: {{(unknown argument).*-iprefix}}
// CC1AsOptionCHECK22: {{(unknown argument).*-iquote}}
// CC1AsOptionCHECK22: {{(unknown argument).*-isysroot}}
// RUN: not %clang -cc1as -isystem -isystem-after -ivfsoverlay -iwithprefix -iwithprefixbefore -iwithsysroot -keep_private_externs -l -lazy_framework -lazy_library --ld-path= --libomptarget-amdgcn-bc-path= --libomptarget-amdgpu-bc-path= --libomptarget-nvptx-bc-path= --libomptarget-spirv-bc-path= --linker-option= -llvm-verify-each -load -m16 -m32 -m3dnow -m3dnowa -m64 -m68000 -m68010 -m68020 -m68030 -m68040 -m68060 -m68881 -m80387 -mseses -mabi= -mabi=ieeelongdouble -mabi=quadword-atomics -mabi=vec-extabi -mabicalls -mabs= -madx -maes -maix32 -maix64 -maix-shared-lib-tls-model-opt -maix-small-local-dynamic-tls -maix-small-local-exec-tls -maix-struct-return -malign-branch= -malign-branch-boundary= -malign-double -malign-functions= -malign-jumps= -malign-loops= -maltivec -mamdgpu-ieee -mamdgpu-precise-memory-op -mamx-avx512 -mamx-bf16 -mamx-complex -mamx-fp16 -mamx-fp8 -mamx-int8 -mamx-movrs -mamx-tf32 -mamx-tile -mamx-transpose -mannotate-tablejump -mappletvos-version-min= -mappletvsimulator-version-min= -mapx-features= -mapx-inline-asm-use-gpr32 -mapxf -march= -marm -marm64x -masm= -matomics -mavx -mavx10.1 -mavx10.1-256 -mavx10.1-512 -mavx10.2 -mavx10.2-256 -mavx10.2-512 -mavx2 -mavx512bf16 -mavx512bitalg -mavx512bw -mavx512cd -mavx512dq -mavx512f -mavx512fp16 -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vl -mavx512vnni -mavx512vp2intersect -mavx512vpopcntdq -mavxifma -mavxneconvert  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK23 %s

// CC1AsOptionCHECK23: {{(unknown argument).*-isystem}}
// CC1AsOptionCHECK23: {{(unknown argument).*-isystem-after}}
// CC1AsOptionCHECK23: {{(unknown argument).*-ivfsoverlay}}
// CC1AsOptionCHECK23: {{(unknown argument).*-iwithprefix}}
// CC1AsOptionCHECK23: {{(unknown argument).*-iwithprefixbefore}}
// CC1AsOptionCHECK23: {{(unknown argument).*-iwithsysroot}}
// CC1AsOptionCHECK23: {{(unknown argument).*-keep_private_externs}}
// CC1AsOptionCHECK23: {{(unknown argument).*-l}}
// CC1AsOptionCHECK23: {{(unknown argument).*-lazy_framework}}
// CC1AsOptionCHECK23: {{(unknown argument).*-lazy_library}}
// CC1AsOptionCHECK23: {{(unknown argument).*--ld-path=}}
// CC1AsOptionCHECK23: {{(unknown argument).*--libomptarget-amdgcn-bc-path=}}
// CC1AsOptionCHECK23: {{(unknown argument).*--libomptarget-amdgpu-bc-path=}}
// CC1AsOptionCHECK23: {{(unknown argument).*--libomptarget-nvptx-bc-path=}}
// CC1AsOptionCHECK23: {{(unknown argument).*--libomptarget-spirv-bc-path=}}
// CC1AsOptionCHECK23: {{(unknown argument).*--linker-option=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-llvm-verify-each}}
// CC1AsOptionCHECK23: {{(unknown argument).*-load}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m16}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m32}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m3dnow}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m3dnowa}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m64}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m68000}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m68010}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m68020}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m68030}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m68040}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m68060}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m68881}}
// CC1AsOptionCHECK23: {{(unknown argument).*-m80387}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mseses}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mabi=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mabi=ieeelongdouble}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mabi=quadword-atomics}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mabi=vec-extabi}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mabicalls}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mabs=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-madx}}
// CC1AsOptionCHECK23: {{(unknown argument).*-maes}}
// CC1AsOptionCHECK23: {{(unknown argument).*-maix32}}
// CC1AsOptionCHECK23: {{(unknown argument).*-maix64}}
// CC1AsOptionCHECK23: {{(unknown argument).*-maix-shared-lib-tls-model-opt}}
// CC1AsOptionCHECK23: {{(unknown argument).*-maix-small-local-dynamic-tls}}
// CC1AsOptionCHECK23: {{(unknown argument).*-maix-small-local-exec-tls}}
// CC1AsOptionCHECK23: {{(unknown argument).*-maix-struct-return}}
// CC1AsOptionCHECK23: {{(unknown argument).*-malign-branch=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-malign-branch-boundary=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-malign-double}}
// CC1AsOptionCHECK23: {{(unknown argument).*-malign-functions=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-malign-jumps=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-malign-loops=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-maltivec}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mamdgpu-ieee}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mamdgpu-precise-memory-op}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mamx-avx512}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mamx-bf16}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mamx-complex}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mamx-fp16}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mamx-fp8}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mamx-int8}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mamx-movrs}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mamx-tf32}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mamx-tile}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mamx-transpose}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mannotate-tablejump}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mappletvos-version-min=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mappletvsimulator-version-min=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mapx-features=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mapx-inline-asm-use-gpr32}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mapxf}}
// CC1AsOptionCHECK23: {{(unknown argument).*-march=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-marm}}
// CC1AsOptionCHECK23: {{(unknown argument).*-marm64x}}
// CC1AsOptionCHECK23: {{(unknown argument).*-masm=}}
// CC1AsOptionCHECK23: {{(unknown argument).*-matomics}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx10.1}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx10.1-256}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx10.1-512}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx10.2}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx10.2-256}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx10.2-512}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx2}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512bf16}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512bitalg}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512bw}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512cd}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512dq}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512f}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512fp16}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512ifma}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512vbmi}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512vbmi2}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512vl}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512vnni}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512vp2intersect}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavx512vpopcntdq}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavxifma}}
// CC1AsOptionCHECK23: {{(unknown argument).*-mavxneconvert}}
// RUN: not %clang -cc1as -mavxvnni -mavxvnniint16 -mavxvnniint8 -mbackchain -mbig-endian -mbmi -mbmi2 -mbranch-likely -mbranch-protection= -mbranch-protection-pauth-lr -mbranch-target-enforce -mbranches-within-32B-boundaries -mbulk-memory -mbulk-memory-opt -mcabac -mcall-indirect-overlong -mcf-branch-label-scheme= -mcheck-zero-division -mcldemote -mclflushopt -mclwb -mclzero -mcmodel= -mcmpb -mcmpccxadd -mcmse -mcode-object-version= -mcompact-branches= -mconsole -mconstant-cfstrings -mconstructor-aliases -mcpu= -mcrbits -mcrc -mcrc32 -mcumode -mcx16 -mdaz-ftz -mdebug-pass -mdefault-build-attributes -mdefault-visibility-export-mapping= -mdirect-move -mdiv32 -mdll -mdouble= -mdouble-float -mdsp -mdspr2 -mdynamic-no-pic -meabi -mefpu2 -membedded-data -menable-experimental-extensions -menable-no-infs -menable-no-nans -menqcmd -mevex512 -mexception-handling -mexec-model= -mexecute-only -mextended-const -mextern-sdata -mf16c -mfancy-math-387 -mfentry -mfix4300 -mfix-and-continue -mfix-cmse-cve-2021-35465 -mfix-cortex-a53-835769 -mfix-cortex-a57-aes-1742098 -mfix-cortex-a72-aes-1655431 -mfix-gr712rc -mfix-ut700 -mfloat128 -mfloat-abi -mfloat-abi= -mfma -mfma4 -mfp16 -mfp32 -mfp64 -mfpmath -mfpmath= -mfprnd -mfpu -mfpu= -mfpxx -mframe-chain= -mframe-pointer= -mfrecipe -mfsgsbase -mfsmuld -mfunction-return= -mfxsr -mgeneral-regs-only -mgfni -mginv -mglibc -mglobal-merge -mgpopt  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK24 %s

// CC1AsOptionCHECK24: {{(unknown argument).*-mavxvnni}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mavxvnniint16}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mavxvnniint8}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mbackchain}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mbig-endian}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mbmi}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mbmi2}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mbranch-likely}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mbranch-protection=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mbranch-protection-pauth-lr}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mbranch-target-enforce}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mbranches-within-32B-boundaries}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mbulk-memory}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mbulk-memory-opt}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcabac}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcall-indirect-overlong}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcf-branch-label-scheme=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcheck-zero-division}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcldemote}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mclflushopt}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mclwb}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mclzero}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcmodel=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcmpb}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcmpccxadd}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcmse}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcode-object-version=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcompact-branches=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mconsole}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mconstant-cfstrings}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mconstructor-aliases}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcpu=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcrbits}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcrc}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcrc32}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcumode}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mcx16}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mdaz-ftz}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mdebug-pass}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mdefault-build-attributes}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mdefault-visibility-export-mapping=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mdirect-move}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mdiv32}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mdll}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mdouble=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mdouble-float}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mdsp}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mdspr2}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mdynamic-no-pic}}
// CC1AsOptionCHECK24: {{(unknown argument).*-meabi}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mefpu2}}
// CC1AsOptionCHECK24: {{(unknown argument).*-membedded-data}}
// CC1AsOptionCHECK24: {{(unknown argument).*-menable-experimental-extensions}}
// CC1AsOptionCHECK24: {{(unknown argument).*-menable-no-infs}}
// CC1AsOptionCHECK24: {{(unknown argument).*-menable-no-nans}}
// CC1AsOptionCHECK24: {{(unknown argument).*-menqcmd}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mevex512}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mexception-handling}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mexec-model=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mexecute-only}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mextended-const}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mextern-sdata}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mf16c}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfancy-math-387}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfentry}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfix4300}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfix-and-continue}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfix-cmse-cve-2021-35465}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfix-cortex-a53-835769}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfix-cortex-a57-aes-1742098}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfix-cortex-a72-aes-1655431}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfix-gr712rc}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfix-ut700}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfloat128}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfloat-abi}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfloat-abi=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfma}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfma4}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfp16}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfp32}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfp64}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfpmath}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfpmath=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfprnd}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfpu}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfpu=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfpxx}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mframe-chain=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mframe-pointer=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfrecipe}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfsgsbase}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfsmuld}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfunction-return=}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mfxsr}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mgeneral-regs-only}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mgfni}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mginv}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mglibc}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mglobal-merge}}
// CC1AsOptionCHECK24: {{(unknown argument).*-mgpopt}}
// RUN: not %clang -cc1as -mguard= -mguarded-control-stack -mhard-float -mhard-quad-float -mharden-sls= -mhvx -mhvx= -mhvx-ieee-fp -mhvx-length= -mhvx-qfloat -mhreset -mhtm -mhwdiv= -mhwmult= -miamcu -mieee-fp -mieee-rnd-near -mignore-xcoff-visibility -no-finalize-removal -no-ns-alloc-error -mimplicit-float -mimplicit-it= -mindirect-branch-cs-prefix -mindirect-jump= -minline-all-stringops -minvariant-function-descriptors -minvpcid -mios-simulator-version-min= -mios-version-min= -mips1 -mips16 -mips2 -mips3 -mips32 -mips32r2 -mips32r3 -mips32r5 -mips32r6 -mips4 -mips5 -mips64 -mips64r2 -mips64r3 -mips64r5 -mips64r6 -misel -mkernel -mkl -mlam-bh -mlamcas -mlarge-data-threshold= -mlasx -mld-seq-sa -mldc1-sdc1 -mlimit-float-precision -mlink-bitcode-file -mlink-builtin-bitcode -mlink-builtin-bitcode-postopt -mlinker-version= -mlittle-endian -mlocal-sdata -mlong-calls -mlong-double-128 -mlong-double-64 -mlong-double-80 -mlongcall -mlr-for-calls-only -mlsx -mlvi-cfi -mlvi-hardening -mlwp -mlzcnt -mmacos-version-min= -mmadd4 -mmark-bti-property -mmcu= -mmemops -mmfcrf -mmfocrf -mmicromips -mmlir -mmma -mmmx -mmovbe -mmovdir64b -mmovdiri -mmovrs -mmpx -mms-bitfields -mmt -mmultimemory -mmultivalue -mmutable-globals -mmwaitx -mnan= -mno-3dnow -mno-3dnowa -mno-80387 -mno-abicalls -mno-adx  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK25 %s

// CC1AsOptionCHECK25: {{(unknown argument).*-mguard=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mguarded-control-stack}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mhard-float}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mhard-quad-float}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mharden-sls=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mhvx}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mhvx=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mhvx-ieee-fp}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mhvx-length=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mhvx-qfloat}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mhreset}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mhtm}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mhwdiv=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mhwmult=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-miamcu}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mieee-fp}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mieee-rnd-near}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mignore-xcoff-visibility}}
// CC1AsOptionCHECK25: {{(unknown argument).*-no-finalize-removal}}
// CC1AsOptionCHECK25: {{(unknown argument).*-no-ns-alloc-error}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mimplicit-float}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mimplicit-it=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mindirect-branch-cs-prefix}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mindirect-jump=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-minline-all-stringops}}
// CC1AsOptionCHECK25: {{(unknown argument).*-minvariant-function-descriptors}}
// CC1AsOptionCHECK25: {{(unknown argument).*-minvpcid}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mios-simulator-version-min=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mios-version-min=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips1}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips16}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips2}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips3}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips32}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips32r2}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips32r3}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips32r5}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips32r6}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips4}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips5}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips64}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips64r2}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips64r3}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips64r5}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mips64r6}}
// CC1AsOptionCHECK25: {{(unknown argument).*-misel}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mkernel}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mkl}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlam-bh}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlamcas}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlarge-data-threshold=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlasx}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mld-seq-sa}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mldc1-sdc1}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlimit-float-precision}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlink-bitcode-file}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlink-builtin-bitcode}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlink-builtin-bitcode-postopt}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlinker-version=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlittle-endian}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlocal-sdata}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlong-calls}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlong-double-128}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlong-double-64}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlong-double-80}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlongcall}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlr-for-calls-only}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlsx}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlvi-cfi}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlvi-hardening}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlwp}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mlzcnt}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmacos-version-min=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmadd4}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmark-bti-property}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmcu=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmemops}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmfcrf}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmfocrf}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmicromips}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmlir}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmma}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmmx}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmovbe}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmovdir64b}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmovdiri}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmovrs}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmpx}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mms-bitfields}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmt}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmultimemory}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmultivalue}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmutable-globals}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mmwaitx}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mnan=}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mno-3dnow}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mno-3dnowa}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mno-80387}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mno-abicalls}}
// CC1AsOptionCHECK25: {{(unknown argument).*-mno-adx}}
// RUN: not %clang -cc1as -mno-aes -mno-altivec -mno-amdgpu-ieee -mno-amdgpu-precise-memory-op -mno-amx-avx512 -mno-amx-bf16 -mno-amx-complex -mno-amx-fp16 -mno-amx-fp8 -mno-amx-int8 -mno-amx-movrs -mno-amx-tf32 -mno-amx-tile -mno-amx-transpose -mno-annotate-tablejump -mno-apx-features= -mno-apxf -mno-atomics -mno-avx -mno-avx10.1 -mno-avx10.1-256 -mno-avx10.1-512 -mno-avx10.2 -mno-avx2 -mno-avx512bf16 -mno-avx512bitalg -mno-avx512bw -mno-avx512cd -mno-avx512dq -mno-avx512f -mno-avx512fp16 -mno-avx512ifma -mno-avx512vbmi -mno-avx512vbmi2 -mno-avx512vl -mno-avx512vnni -mno-avx512vp2intersect -mno-avx512vpopcntdq -mno-avxifma -mno-avxneconvert -mno-avxvnni -mno-avxvnniint16 -mno-avxvnniint8 -mno-backchain -mno-bmi -mno-bmi2 -mno-branch-likely -mno-bti-at-return-twice -mno-bulk-memory -mno-bulk-memory-opt -mno-call-indirect-overlong -mno-check-zero-division -mno-cldemote -mno-clflushopt -mno-clwb -mno-clzero -mno-cmpb -mno-cmpccxadd -mno-constant-cfstrings -mno-constructor-aliases -mno-crbits -mno-crc -mno-crc32 -mno-cumode -mno-cx16 -mno-daz-ftz -mno-default-build-attributes -mno-div32 -mno-dsp -mno-dspr2 -mno-embedded-data -mno-enqcmd -mno-evex512 -mno-exception-handling -mno-execute-only -mno-extended-const -mno-extern-sdata -mno-f16c -mno-fix-cmse-cve-2021-35465 -mno-fix-cortex-a53-835769 -mno-fix-cortex-a57-aes-1742098 -mno-fix-cortex-a72-aes-1655431 -mno-float128 -mno-fma -mno-fma4 -mno-fmv -mno-fp16 -mno-fp-ret-in-387 -mno-fprnd -mno-fpu -mno-frecipe -mno-fsgsbase -mno-fsmuld -mno-fxsr -mno-gather -mno-gfni -mno-ginv -mno-global-merge -mno-gpopt -mno-hvx  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK26 %s

// CC1AsOptionCHECK26: {{(unknown argument).*-mno-aes}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-altivec}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-amdgpu-ieee}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-amdgpu-precise-memory-op}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-amx-avx512}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-amx-bf16}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-amx-complex}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-amx-fp16}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-amx-fp8}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-amx-int8}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-amx-movrs}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-amx-tf32}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-amx-tile}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-amx-transpose}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-annotate-tablejump}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-apx-features=}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-apxf}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-atomics}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx10.1}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx10.1-256}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx10.1-512}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx10.2}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx2}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512bf16}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512bitalg}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512bw}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512cd}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512dq}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512f}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512fp16}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512ifma}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512vbmi}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512vbmi2}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512vl}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512vnni}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512vp2intersect}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avx512vpopcntdq}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avxifma}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avxneconvert}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avxvnni}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avxvnniint16}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-avxvnniint8}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-backchain}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-bmi}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-bmi2}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-branch-likely}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-bti-at-return-twice}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-bulk-memory}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-bulk-memory-opt}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-call-indirect-overlong}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-check-zero-division}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-cldemote}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-clflushopt}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-clwb}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-clzero}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-cmpb}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-cmpccxadd}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-constant-cfstrings}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-constructor-aliases}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-crbits}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-crc}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-crc32}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-cumode}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-cx16}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-daz-ftz}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-default-build-attributes}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-div32}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-dsp}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-dspr2}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-embedded-data}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-enqcmd}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-evex512}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-exception-handling}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-execute-only}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-extended-const}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-extern-sdata}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-f16c}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fix-cmse-cve-2021-35465}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fix-cortex-a53-835769}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fix-cortex-a57-aes-1742098}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fix-cortex-a72-aes-1655431}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-float128}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fma}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fma4}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fmv}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fp16}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fp-ret-in-387}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fprnd}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fpu}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-frecipe}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fsgsbase}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fsmuld}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-fxsr}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-gather}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-gfni}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-ginv}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-global-merge}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-gpopt}}
// CC1AsOptionCHECK26: {{(unknown argument).*-mno-hvx}}
// RUN: not %clang -cc1as -mno-hvx-ieee-fp -mno-hvx-qfloat -mno-hreset -mno-htm -mno-iamcu -mno-implicit-float -mno-incremental-linker-compatible -mno-inline-all-stringops -mno-invariant-function-descriptors -mno-invpcid -mno-isel -mno-kl -mno-lam-bh -mno-lamcas -mno-lasx -mno-ld-seq-sa -mno-ldc1-sdc1 -mno-link-builtin-bitcode-postopt -mno-local-sdata -mno-long-calls -mno-longcall -mno-lsx -mno-lvi-cfi -mno-lvi-hardening -mno-lwp -mno-lzcnt -mno-madd4 -mno-memops -mno-mfcrf -mno-mfocrf -mno-micromips -mno-mips16 -mno-mma -mno-mmx -mno-movbe -mno-movdir64b -mno-movdiri -mno-movrs -mno-movt -mno-mpx -mno-ms-bitfields -mno-msa -mno-mt -mno-multimemory -mno-multivalue -mno-mutable-globals -mno-mwaitx -mno-neg-immediates -mno-nontrapping-fptoint -mno-nvj -mno-nvs -mno-odd-spreg -mno-omit-leaf-frame-pointer -mno-outline -mno-outline-atomics -mno-packed-stack -mno-packets -mno-pascal-strings -mno-pclmul -mno-pconfig -mno-pcrel -mno-pic-data-is-text-relative -mno-pku -mno-popc -mno-popcnt -mno-popcntd -mno-power10-vector -mno-power8-vector -mno-power9-vector -mno-prefetchi -mno-prefixed -mno-prfchw -mno-ptwrite -mno-pure-code -mno-raoint -mno-rdpid -mno-rdpru -mno-rdrnd -mno-rdseed -mno-red-zone -mno-reference-types -mno-regnames -mno-relax -mno-relax-all -mno-relax-pic-calls -mno-relaxed-simd -mno-restrict-it -mno-retpoline -mno-retpoline-external-thunk -mno-rtd -mno-rtm -mno-sahf -mno-save-restore -mno-scalar-strict-align -mno-scatter -mno-scq -mno-serialize -mno-seses -mno-sgx -mno-sha  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK27 %s

// CC1AsOptionCHECK27: {{(unknown argument).*-mno-hvx-ieee-fp}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-hvx-qfloat}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-hreset}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-htm}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-iamcu}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-implicit-float}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-incremental-linker-compatible}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-inline-all-stringops}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-invariant-function-descriptors}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-invpcid}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-isel}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-kl}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-lam-bh}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-lamcas}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-lasx}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-ld-seq-sa}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-ldc1-sdc1}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-link-builtin-bitcode-postopt}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-local-sdata}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-long-calls}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-longcall}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-lsx}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-lvi-cfi}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-lvi-hardening}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-lwp}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-lzcnt}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-madd4}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-memops}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-mfcrf}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-mfocrf}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-micromips}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-mips16}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-mma}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-mmx}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-movbe}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-movdir64b}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-movdiri}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-movrs}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-movt}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-mpx}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-ms-bitfields}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-msa}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-mt}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-multimemory}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-multivalue}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-mutable-globals}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-mwaitx}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-neg-immediates}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-nontrapping-fptoint}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-nvj}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-nvs}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-odd-spreg}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-omit-leaf-frame-pointer}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-outline}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-outline-atomics}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-packed-stack}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-packets}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-pascal-strings}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-pclmul}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-pconfig}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-pcrel}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-pic-data-is-text-relative}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-pku}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-popc}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-popcnt}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-popcntd}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-power10-vector}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-power8-vector}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-power9-vector}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-prefetchi}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-prefixed}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-prfchw}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-ptwrite}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-pure-code}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-raoint}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-rdpid}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-rdpru}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-rdrnd}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-rdseed}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-red-zone}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-reference-types}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-regnames}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-relax}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-relax-all}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-relax-pic-calls}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-relaxed-simd}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-restrict-it}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-retpoline}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-retpoline-external-thunk}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-rtd}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-rtm}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-sahf}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-save-restore}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-scalar-strict-align}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-scatter}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-scq}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-serialize}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-seses}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-sgx}}
// CC1AsOptionCHECK27: {{(unknown argument).*-mno-sha}}
// RUN: not %clang -cc1as -mno-sha512 -mno-shstk -mno-sign-ext -mno-simd128 -mno-skip-rax-setup -mno-sm3 -mno-sm4 -mno-soft-float -mno-spe -mno-speculative-load-hardening -mno-sse -mno-sse2 -mno-sse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-sse4a -mno-ssse3 -mno-stack-arg-probe -mno-stackrealign -mno-strict-align -mno-tail-call -mno-tbm -mno-tgsplit -mno-thumb -mno-tls-direct-seg-refs -mno-tocdata -mno-tocdata= -mno-tsxldtrk -mno-uintr -mno-unaligned-access -mno-unaligned-symbols -mno-unsafe-fp-atomics -mno-usermsr -mno-v8plus -mno-vaes -mno-vector-strict-align -mno-vevpu -mno-virt -mno-vis -mno-vis2 -mno-vis3 -mno-vpclmulqdq -mno-vsx -mno-vx -mno-vzeroupper -mno-waitpkg -mno-warn-nonportable-cfstrings -mno-wavefrontsize64 -mno-wbnoinvd -mno-wide-arithmetic -mno-widekl -mno-x87 -mno-xcoff-roptr -mno-xgot -mno-xop -mno-xsave -mno-xsavec -mno-xsaveopt -mno-xsaves -mno-zvector -mnocrc -mno-direct-move -mnontrapping-fptoint -mnop-mcount -mno-paired-vector-memops -mno-crypto -mnvj -mnvs -modd-spreg -module-dependency-dir -module-dir -module-file-deps -module-file-info -module-suffix -fmodules-reduced-bmi -momit-leaf-frame-pointer -moslib= -moutline -moutline-atomics -mpacked-stack -mpackets -mpad-max-prefix-size= -mpaired-vector-memops -mpascal-strings -mpclmul -mpconfig -mpcrel -mpic-data-is-text-relative -mpku -mpopc -mpopcnt -mpopcntd -mpower10-vector -mcrypto -mpower8-vector -mpower9-vector -mprefer-vector-width= -mprefetchi -mprefixed  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK28 %s

// CC1AsOptionCHECK28: {{(unknown argument).*-mno-sha512}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-shstk}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-sign-ext}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-simd128}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-skip-rax-setup}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-sm3}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-sm4}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-soft-float}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-spe}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-speculative-load-hardening}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-sse}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-sse2}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-sse3}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-sse4}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-sse4.1}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-sse4.2}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-sse4a}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-ssse3}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-stack-arg-probe}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-stackrealign}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-strict-align}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-tail-call}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-tbm}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-tgsplit}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-thumb}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-tls-direct-seg-refs}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-tocdata}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-tocdata=}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-tsxldtrk}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-uintr}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-unaligned-access}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-unaligned-symbols}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-unsafe-fp-atomics}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-usermsr}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-v8plus}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-vaes}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-vector-strict-align}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-vevpu}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-virt}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-vis}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-vis2}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-vis3}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-vpclmulqdq}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-vsx}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-vx}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-vzeroupper}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-waitpkg}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-warn-nonportable-cfstrings}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-wavefrontsize64}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-wbnoinvd}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-wide-arithmetic}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-widekl}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-x87}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-xcoff-roptr}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-xgot}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-xop}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-xsave}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-xsavec}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-xsaveopt}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-xsaves}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-zvector}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mnocrc}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-direct-move}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mnontrapping-fptoint}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mnop-mcount}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-paired-vector-memops}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mno-crypto}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mnvj}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mnvs}}
// CC1AsOptionCHECK28: {{(unknown argument).*-modd-spreg}}
// CC1AsOptionCHECK28: {{(unknown argument).*-module-dependency-dir}}
// CC1AsOptionCHECK28: {{(unknown argument).*-module-dir}}
// CC1AsOptionCHECK28: {{(unknown argument).*-module-file-deps}}
// CC1AsOptionCHECK28: {{(unknown argument).*-module-file-info}}
// CC1AsOptionCHECK28: {{(unknown argument).*-module-suffix}}
// CC1AsOptionCHECK28: {{(unknown argument).*-fmodules-reduced-bmi}}
// CC1AsOptionCHECK28: {{(unknown argument).*-momit-leaf-frame-pointer}}
// CC1AsOptionCHECK28: {{(unknown argument).*-moslib=}}
// CC1AsOptionCHECK28: {{(unknown argument).*-moutline}}
// CC1AsOptionCHECK28: {{(unknown argument).*-moutline-atomics}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpacked-stack}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpackets}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpad-max-prefix-size=}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpaired-vector-memops}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpascal-strings}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpclmul}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpconfig}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpcrel}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpic-data-is-text-relative}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpku}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpopc}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpopcnt}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpopcntd}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpower10-vector}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mcrypto}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpower8-vector}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mpower9-vector}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mprefer-vector-width=}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mprefetchi}}
// CC1AsOptionCHECK28: {{(unknown argument).*-mprefixed}}
// RUN: not %clang -cc1as -mprfchw -mprintf-kind= -mprivileged -mptwrite -mpure-code -mqdsp6-compat -mraoint -mrdpid -mrdpru -mrdrnd -mrdseed -mreassociate -mrecip -mrecip= -mrecord-mcount -mred-zone -mreference-types -mregnames -mregparm -mregparm= -mrelax -mrelax-pic-calls -mrelaxed-simd -mrestrict-it -mretpoline -mretpoline-external-thunk -mrop-protect -mrtd -mrtm -mrvv-vector-bits= -msahf -msave-reg-params -msave-restore -mscalar-strict-align -mscq -msecure-plt -mserialize -msgx -msha -msha512 -mshstk -msign-ext -msign-return-address= -msign-return-address-key= -msim -msimd128 -msimd= -msingle-float -mskip-rax-setup -msm3 -msm4 -msmall-data-limit -msmall-data-limit= -msmall-data-threshold= -msoft-float -msoft-quad-float -mspe -mspeculative-load-hardening -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -msse4a -mssse3 -mstack-alignment= -mstack-arg-probe -mstack-probe-size= -mstack-protector-guard= -mstack-protector-guard-offset= -mstack-protector-guard-reg= -mstack-protector-guard-symbol= -mstackrealign -mstrict-align -msve-vector-bits= -msvr4-struct-return -mtail-call -mtargetos= -mtbm -mtgsplit -mthread-model -mthreads -mthumb -mtls-dialect= -mtls-direct-seg-refs -mtls-size= -mtocdata -mtocdata= -mtp -mtp= -mtsxldtrk -mtune= -mtvos-simulator-version-min= -mtvos-version-min= -muclibc -muintr -multi_module -multi-lib-config= -multiply_defined  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK29 %s

// CC1AsOptionCHECK29: {{(unknown argument).*-mprfchw}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mprintf-kind=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mprivileged}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mptwrite}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mpure-code}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mqdsp6-compat}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mraoint}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrdpid}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrdpru}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrdrnd}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrdseed}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mreassociate}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrecip}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrecip=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrecord-mcount}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mred-zone}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mreference-types}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mregnames}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mregparm}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mregparm=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrelax}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrelax-pic-calls}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrelaxed-simd}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrestrict-it}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mretpoline}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mretpoline-external-thunk}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrop-protect}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrtd}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrtm}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mrvv-vector-bits=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msahf}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msave-reg-params}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msave-restore}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mscalar-strict-align}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mscq}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msecure-plt}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mserialize}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msgx}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msha}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msha512}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mshstk}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msign-ext}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msign-return-address=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msign-return-address-key=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msim}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msimd128}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msimd=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msingle-float}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mskip-rax-setup}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msm3}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msm4}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msmall-data-limit}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msmall-data-limit=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msmall-data-threshold=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msoft-float}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msoft-quad-float}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mspe}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mspeculative-load-hardening}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msse}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msse2}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msse3}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msse4}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msse4.1}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msse4.2}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msse4a}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mssse3}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mstack-alignment=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mstack-arg-probe}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mstack-probe-size=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mstack-protector-guard=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mstack-protector-guard-offset=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mstack-protector-guard-reg=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mstack-protector-guard-symbol=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mstackrealign}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mstrict-align}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msve-vector-bits=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-msvr4-struct-return}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtail-call}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtargetos=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtbm}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtgsplit}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mthread-model}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mthreads}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mthumb}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtls-dialect=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtls-direct-seg-refs}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtls-size=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtocdata}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtocdata=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtp}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtp=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtsxldtrk}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtune=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtvos-simulator-version-min=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-mtvos-version-min=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-muclibc}}
// CC1AsOptionCHECK29: {{(unknown argument).*-muintr}}
// CC1AsOptionCHECK29: {{(unknown argument).*-multi_module}}
// CC1AsOptionCHECK29: {{(unknown argument).*-multi-lib-config=}}
// CC1AsOptionCHECK29: {{(unknown argument).*-multiply_defined}}
// RUN: not %clang -cc1as -multiply_defined_unused -munaligned-access -munaligned-symbols -municode -munsafe-fp-atomics -musermsr -mv5 -mv55 -mv60 -mv62 -mv65 -mv66 -mv67 -mv67t -mv68 -mv69 -mv71 -mv71t -mv73 -mv75 -mv79 -mv8plus -mvaes -mvector-strict-align -mvevpu -mvirt -mvis -mvis2 -mvis3 -mvpclmulqdq -mvscale-max= -mvscale-min= -mvsx -mvx -mvzeroupper -mwaitpkg -mwarn-nonportable-cfstrings -mwatchos-simulator-version-min= -mwatchos-version-min= -mwatchsimulator-version-min= -mwavefrontsize64 -mwbnoinvd -mwide-arithmetic -mwidekl -mwindows -mx32 -mx87 -mxcoff-build-id= -mxcoff-roptr -mxgot -mxop -mxsave -mxsavec -mxsaveopt -mxsaves -mzos-hlq-clang= -mzos-hlq-csslib= -mzos-hlq-le= -mzos-sys-include= -mzos-target= -mzvector -new-struct-path-tbaa -no_dead_strip_inits_and_terms -no-canonical-prefixes -no-clear-ast-before-backend -no-code-completion-globals -no-code-completion-ns-level-decls -no-cpp-precomp --no-cuda-gpu-arch= --no-cuda-include-ptx= --no-cuda-noopt-device-debug --no-cuda-version-check -fno-c++-static-destructors --no-default-config -no-emit-llvm-uselists -no-enable-noundef-analysis --no-gpu-bundle-output -no-hip-rt -no-implicit-float -no-integrated-cpp --no-offload-add-rpath --no-offload-arch= --no-offload-compress --no-offload-new-driver --no-offloadlib -no-pedantic -no-pie -no-pointer-tbaa -no-pthread -no-round-trip-args -no-struct-path-tbaa --no-system-header-prefix= --no-wasm-opt -nobuiltininc -nocpp -nodefaultlibs -nodriverkitlib -nofixprebinding -nogpuinc -nogpulibc  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK30 %s

// CC1AsOptionCHECK30: {{(unknown argument).*-multiply_defined_unused}}
// CC1AsOptionCHECK30: {{(unknown argument).*-munaligned-access}}
// CC1AsOptionCHECK30: {{(unknown argument).*-munaligned-symbols}}
// CC1AsOptionCHECK30: {{(unknown argument).*-municode}}
// CC1AsOptionCHECK30: {{(unknown argument).*-munsafe-fp-atomics}}
// CC1AsOptionCHECK30: {{(unknown argument).*-musermsr}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv5}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv55}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv60}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv62}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv65}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv66}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv67}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv67t}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv68}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv69}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv71}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv71t}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv73}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv75}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv79}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mv8plus}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvaes}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvector-strict-align}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvevpu}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvirt}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvis}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvis2}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvis3}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvpclmulqdq}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvscale-max=}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvscale-min=}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvsx}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvx}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mvzeroupper}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mwaitpkg}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mwarn-nonportable-cfstrings}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mwatchos-simulator-version-min=}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mwatchos-version-min=}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mwatchsimulator-version-min=}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mwavefrontsize64}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mwbnoinvd}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mwide-arithmetic}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mwidekl}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mwindows}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mx32}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mx87}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mxcoff-build-id=}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mxcoff-roptr}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mxgot}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mxop}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mxsave}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mxsavec}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mxsaveopt}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mxsaves}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mzos-hlq-clang=}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mzos-hlq-csslib=}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mzos-hlq-le=}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mzos-sys-include=}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mzos-target=}}
// CC1AsOptionCHECK30: {{(unknown argument).*-mzvector}}
// CC1AsOptionCHECK30: {{(unknown argument).*-new-struct-path-tbaa}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no_dead_strip_inits_and_terms}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-canonical-prefixes}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-clear-ast-before-backend}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-code-completion-globals}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-code-completion-ns-level-decls}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-cpp-precomp}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-cuda-gpu-arch=}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-cuda-include-ptx=}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-cuda-noopt-device-debug}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-cuda-version-check}}
// CC1AsOptionCHECK30: {{(unknown argument).*-fno-c\+\+-static-destructors}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-default-config}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-emit-llvm-uselists}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-enable-noundef-analysis}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-gpu-bundle-output}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-hip-rt}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-implicit-float}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-integrated-cpp}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-offload-add-rpath}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-offload-arch=}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-offload-compress}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-offload-new-driver}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-offloadlib}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-pedantic}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-pie}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-pointer-tbaa}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-pthread}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-round-trip-args}}
// CC1AsOptionCHECK30: {{(unknown argument).*-no-struct-path-tbaa}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-system-header-prefix=}}
// CC1AsOptionCHECK30: {{(unknown argument).*--no-wasm-opt}}
// CC1AsOptionCHECK30: {{(unknown argument).*-nobuiltininc}}
// CC1AsOptionCHECK30: {{(unknown argument).*-nocpp}}
// CC1AsOptionCHECK30: {{(unknown argument).*-nodefaultlibs}}
// CC1AsOptionCHECK30: {{(unknown argument).*-nodriverkitlib}}
// CC1AsOptionCHECK30: {{(unknown argument).*-nofixprebinding}}
// CC1AsOptionCHECK30: {{(unknown argument).*-nogpuinc}}
// CC1AsOptionCHECK30: {{(unknown argument).*-nogpulibc}}
// RUN: not %clang -cc1as -nohipwrapperinc -nolibc -nomultidefs -nopie -noprebind -noprofilelib -noseglinkedit -nostartfiles -nostdinc -nostdinc++ -nostdlib -nostdlibinc -nostdlib++ -nostdsysteminc --nvptx-arch-tool= -fexperimental-openacc-macro-override -fexperimental-openacc-macro-override= -p -pagezero_size -pass-exit-codes -pch-through-hdrstop-create -pch-through-hdrstop-use -pch-through-header= -pedantic -pedantic-errors -pg -pic-is-pie -pic-level -pie -pipe -plugin -plugin-arg- -pointer-tbaa -preamble-bytes= -prebind -prebind_all_twolevel_modules -preload -print-dependency-directives-minimized-source -print-diagnostic-options -print-effective-triple -print-enabled-extensions -print-file-name= -print-ivar-layout -print-libgcc-file-name -print-multi-directory -print-multi-flags-experimental -print-multi-lib -print-multi-os-directory -print-preamble -print-prog-name= -print-resource-dir -print-rocm-search-dirs -print-runtime-dir -print-search-dirs -print-stats -print-library-module-manifest-path -print-supported-cpus -print-supported-extensions -print-target-triple -print-targets -private_bundle --product-name= -pthread -pthreads --ptxas-path= -r -rdynamic -read_only_relocs -reexport_framework -reexport-l -reexport_library -regcall4 -relaxed-aliasing -relocatable-pch -remap -remap-file -resource-dir -resource-dir= -rewrite-legacy-objc -rewrite-macros -rewrite-objc -rewrite-test --rocm-device-lib-path= --rocm-path= -round-trip-args -rpath --rsp-quoting= -rtlib= -s -fsanitize-address-destructor= -fsanitize-address-use-after-return= -save-stats -save-stats= -save-temps -save-temps= -sectalign -sectcreate -sectobjectsymbols -sectorder -seg1addr  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK31 %s

// CC1AsOptionCHECK31: {{(unknown argument).*-nohipwrapperinc}}
// CC1AsOptionCHECK31: {{(unknown argument).*-nolibc}}
// CC1AsOptionCHECK31: {{(unknown argument).*-nomultidefs}}
// CC1AsOptionCHECK31: {{(unknown argument).*-nopie}}
// CC1AsOptionCHECK31: {{(unknown argument).*-noprebind}}
// CC1AsOptionCHECK31: {{(unknown argument).*-noprofilelib}}
// CC1AsOptionCHECK31: {{(unknown argument).*-noseglinkedit}}
// CC1AsOptionCHECK31: {{(unknown argument).*-nostartfiles}}
// CC1AsOptionCHECK31: {{(unknown argument).*-nostdinc}}
// CC1AsOptionCHECK31: {{(unknown argument).*-nostdinc\+\+}}
// CC1AsOptionCHECK31: {{(unknown argument).*-nostdlib}}
// CC1AsOptionCHECK31: {{(unknown argument).*-nostdlibinc}}
// CC1AsOptionCHECK31: {{(unknown argument).*-nostdlib\+\+}}
// CC1AsOptionCHECK31: {{(unknown argument).*-nostdsysteminc}}
// CC1AsOptionCHECK31: {{(unknown argument).*--nvptx-arch-tool=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-fexperimental-openacc-macro-override}}
// CC1AsOptionCHECK31: {{(unknown argument).*-fexperimental-openacc-macro-override=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-p}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pagezero_size}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pass-exit-codes}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pch-through-hdrstop-create}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pch-through-hdrstop-use}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pch-through-header=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pedantic}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pedantic-errors}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pg}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pic-is-pie}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pic-level}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pie}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pipe}}
// CC1AsOptionCHECK31: {{(unknown argument).*-plugin}}
// CC1AsOptionCHECK31: {{(unknown argument).*-plugin-arg-}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pointer-tbaa}}
// CC1AsOptionCHECK31: {{(unknown argument).*-preamble-bytes=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-prebind}}
// CC1AsOptionCHECK31: {{(unknown argument).*-prebind_all_twolevel_modules}}
// CC1AsOptionCHECK31: {{(unknown argument).*-preload}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-dependency-directives-minimized-source}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-diagnostic-options}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-effective-triple}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-enabled-extensions}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-file-name=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-ivar-layout}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-libgcc-file-name}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-multi-directory}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-multi-flags-experimental}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-multi-lib}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-multi-os-directory}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-preamble}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-prog-name=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-resource-dir}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-rocm-search-dirs}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-runtime-dir}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-search-dirs}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-stats}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-library-module-manifest-path}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-supported-cpus}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-supported-extensions}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-target-triple}}
// CC1AsOptionCHECK31: {{(unknown argument).*-print-targets}}
// CC1AsOptionCHECK31: {{(unknown argument).*-private_bundle}}
// CC1AsOptionCHECK31: {{(unknown argument).*--product-name=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pthread}}
// CC1AsOptionCHECK31: {{(unknown argument).*-pthreads}}
// CC1AsOptionCHECK31: {{(unknown argument).*--ptxas-path=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-r}}
// CC1AsOptionCHECK31: {{(unknown argument).*-rdynamic}}
// CC1AsOptionCHECK31: {{(unknown argument).*-read_only_relocs}}
// CC1AsOptionCHECK31: {{(unknown argument).*-reexport_framework}}
// CC1AsOptionCHECK31: {{(unknown argument).*-reexport-l}}
// CC1AsOptionCHECK31: {{(unknown argument).*-reexport_library}}
// CC1AsOptionCHECK31: {{(unknown argument).*-regcall4}}
// CC1AsOptionCHECK31: {{(unknown argument).*-relaxed-aliasing}}
// CC1AsOptionCHECK31: {{(unknown argument).*-relocatable-pch}}
// CC1AsOptionCHECK31: {{(unknown argument).*-remap}}
// CC1AsOptionCHECK31: {{(unknown argument).*-remap-file}}
// CC1AsOptionCHECK31: {{(unknown argument).*-resource-dir}}
// CC1AsOptionCHECK31: {{(unknown argument).*-resource-dir=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-rewrite-legacy-objc}}
// CC1AsOptionCHECK31: {{(unknown argument).*-rewrite-macros}}
// CC1AsOptionCHECK31: {{(unknown argument).*-rewrite-objc}}
// CC1AsOptionCHECK31: {{(unknown argument).*-rewrite-test}}
// CC1AsOptionCHECK31: {{(unknown argument).*--rocm-device-lib-path=}}
// CC1AsOptionCHECK31: {{(unknown argument).*--rocm-path=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-round-trip-args}}
// CC1AsOptionCHECK31: {{(unknown argument).*-rpath}}
// CC1AsOptionCHECK31: {{(unknown argument).*--rsp-quoting=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-rtlib=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-s}}
// CC1AsOptionCHECK31: {{(unknown argument).*-fsanitize-address-destructor=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-fsanitize-address-use-after-return=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-save-stats}}
// CC1AsOptionCHECK31: {{(unknown argument).*-save-stats=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-save-temps}}
// CC1AsOptionCHECK31: {{(unknown argument).*-save-temps=}}
// CC1AsOptionCHECK31: {{(unknown argument).*-sectalign}}
// CC1AsOptionCHECK31: {{(unknown argument).*-sectcreate}}
// CC1AsOptionCHECK31: {{(unknown argument).*-sectobjectsymbols}}
// CC1AsOptionCHECK31: {{(unknown argument).*-sectorder}}
// CC1AsOptionCHECK31: {{(unknown argument).*-seg1addr}}
// RUN: not %clang -cc1as -seg_addr_table -seg_addr_table_filename -segaddr -segcreate -seglinkedit -segprot -segs_read_ -segs_read_only_addr -segs_read_write_addr -setup-static-analyzer -shared -shared-libgcc -shared-libsan --show-includes -single_module -skip-function-bodies -source-date-epoch -specs -specs= -spirv -split-dwarf-file -stack-protector -stack-protector-buffer-size -stack-usage-file --start-no-unused-arguments -startfiles -static -static-define -static-libclosure -static-libgcc -static-libgfortran -static-libsan -static-libstdc++ -static-openmp -static-pie -stats-file= -stats-file-append -std= -std-default= -stdlib -stdlib= -stdlib++-isystem -sub_library -sub_umbrella --sycl-link -sycl-std= --symbol-graph-dir= -sys-header-deps --system-header-prefix= -t --target= -target -target-linker-version -T -templight-dump -test-io -time -traditional -traditional-cpp -trigraphs -trim-egraph -triple= -twolevel_namespace -twolevel_namespace_hints -u -umbrella -undef -undefined -unexported_symbols_list -Wextra -Waliasing -Wampersand -Warray-bounds -Wc-binding-type -Wcharacter-truncation -Wconversion -Wdo-subscript -Wfunction-elimination -Wimplicit-interface -Wimplicit-procedure -Wintrinsic-shadow -Wuse-without-only -Wintrinsics-std -Wline-truncation -Wno-align-commons -Wno-overwrite-recursive -Wno-tabs -Wreal-q-constant -Wsurprising -Wunderflow -Wunused-parameter -Wrealloc-lhs -Wrealloc-lhs-all -Wfrontend-loop-interchange -Wtarget-lifetime -unwindlib= -v -vectorize-loops -vectorize-slp -verify  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK32 %s

// CC1AsOptionCHECK32: {{(unknown argument).*-seg_addr_table}}
// CC1AsOptionCHECK32: {{(unknown argument).*-seg_addr_table_filename}}
// CC1AsOptionCHECK32: {{(unknown argument).*-segaddr}}
// CC1AsOptionCHECK32: {{(unknown argument).*-segcreate}}
// CC1AsOptionCHECK32: {{(unknown argument).*-seglinkedit}}
// CC1AsOptionCHECK32: {{(unknown argument).*-segprot}}
// CC1AsOptionCHECK32: {{(unknown argument).*-segs_read_}}
// CC1AsOptionCHECK32: {{(unknown argument).*-segs_read_only_addr}}
// CC1AsOptionCHECK32: {{(unknown argument).*-segs_read_write_addr}}
// CC1AsOptionCHECK32: {{(unknown argument).*-setup-static-analyzer}}
// CC1AsOptionCHECK32: {{(unknown argument).*-shared}}
// CC1AsOptionCHECK32: {{(unknown argument).*-shared-libgcc}}
// CC1AsOptionCHECK32: {{(unknown argument).*-shared-libsan}}
// CC1AsOptionCHECK32: {{(unknown argument).*--show-includes}}
// CC1AsOptionCHECK32: {{(unknown argument).*-single_module}}
// CC1AsOptionCHECK32: {{(unknown argument).*-skip-function-bodies}}
// CC1AsOptionCHECK32: {{(unknown argument).*-source-date-epoch}}
// CC1AsOptionCHECK32: {{(unknown argument).*-specs}}
// CC1AsOptionCHECK32: {{(unknown argument).*-specs=}}
// CC1AsOptionCHECK32: {{(unknown argument).*-spirv}}
// CC1AsOptionCHECK32: {{(unknown argument).*-split-dwarf-file}}
// CC1AsOptionCHECK32: {{(unknown argument).*-stack-protector}}
// CC1AsOptionCHECK32: {{(unknown argument).*-stack-protector-buffer-size}}
// CC1AsOptionCHECK32: {{(unknown argument).*-stack-usage-file}}
// CC1AsOptionCHECK32: {{(unknown argument).*--start-no-unused-arguments}}
// CC1AsOptionCHECK32: {{(unknown argument).*-startfiles}}
// CC1AsOptionCHECK32: {{(unknown argument).*-static}}
// CC1AsOptionCHECK32: {{(unknown argument).*-static-define}}
// CC1AsOptionCHECK32: {{(unknown argument).*-static-libclosure}}
// CC1AsOptionCHECK32: {{(unknown argument).*-static-libgcc}}
// CC1AsOptionCHECK32: {{(unknown argument).*-static-libgfortran}}
// CC1AsOptionCHECK32: {{(unknown argument).*-static-libsan}}
// CC1AsOptionCHECK32: {{(unknown argument).*-static-libstdc\+\+}}
// CC1AsOptionCHECK32: {{(unknown argument).*-static-openmp}}
// CC1AsOptionCHECK32: {{(unknown argument).*-static-pie}}
// CC1AsOptionCHECK32: {{(unknown argument).*-stats-file=}}
// CC1AsOptionCHECK32: {{(unknown argument).*-stats-file-append}}
// CC1AsOptionCHECK32: {{(unknown argument).*-std=}}
// CC1AsOptionCHECK32: {{(unknown argument).*-std-default=}}
// CC1AsOptionCHECK32: {{(unknown argument).*-stdlib}}
// CC1AsOptionCHECK32: {{(unknown argument).*-stdlib=}}
// CC1AsOptionCHECK32: {{(unknown argument).*-stdlib\+\+-isystem}}
// CC1AsOptionCHECK32: {{(unknown argument).*-sub_library}}
// CC1AsOptionCHECK32: {{(unknown argument).*-sub_umbrella}}
// CC1AsOptionCHECK32: {{(unknown argument).*--sycl-link}}
// CC1AsOptionCHECK32: {{(unknown argument).*-sycl-std=}}
// CC1AsOptionCHECK32: {{(unknown argument).*--symbol-graph-dir=}}
// CC1AsOptionCHECK32: {{(unknown argument).*-sys-header-deps}}
// CC1AsOptionCHECK32: {{(unknown argument).*--system-header-prefix=}}
// CC1AsOptionCHECK32: {{(unknown argument).*-t}}
// CC1AsOptionCHECK32: {{(unknown argument).*--target=}}
// CC1AsOptionCHECK32: {{(unknown argument).*-target}}
// CC1AsOptionCHECK32: {{(unknown argument).*-target-linker-version}}
// CC1AsOptionCHECK32: {{(unknown argument).*-T}}
// CC1AsOptionCHECK32: {{(unknown argument).*-templight-dump}}
// CC1AsOptionCHECK32: {{(unknown argument).*-test-io}}
// CC1AsOptionCHECK32: {{(unknown argument).*-time}}
// CC1AsOptionCHECK32: {{(unknown argument).*-traditional}}
// CC1AsOptionCHECK32: {{(unknown argument).*-traditional-cpp}}
// CC1AsOptionCHECK32: {{(unknown argument).*-trigraphs}}
// CC1AsOptionCHECK32: {{(unknown argument).*-trim-egraph}}
// CC1AsOptionCHECK32: {{(unknown argument).*-triple=}}
// CC1AsOptionCHECK32: {{(unknown argument).*-twolevel_namespace}}
// CC1AsOptionCHECK32: {{(unknown argument).*-twolevel_namespace_hints}}
// CC1AsOptionCHECK32: {{(unknown argument).*-u}}
// CC1AsOptionCHECK32: {{(unknown argument).*-umbrella}}
// CC1AsOptionCHECK32: {{(unknown argument).*-undef}}
// CC1AsOptionCHECK32: {{(unknown argument).*-undefined}}
// CC1AsOptionCHECK32: {{(unknown argument).*-unexported_symbols_list}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wextra}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Waliasing}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wampersand}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Warray-bounds}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wc-binding-type}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wcharacter-truncation}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wconversion}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wdo-subscript}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wfunction-elimination}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wimplicit-interface}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wimplicit-procedure}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wintrinsic-shadow}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wuse-without-only}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wintrinsics-std}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wline-truncation}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wno-align-commons}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wno-overwrite-recursive}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wno-tabs}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wreal-q-constant}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wsurprising}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wunderflow}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wunused-parameter}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wrealloc-lhs}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wrealloc-lhs-all}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wfrontend-loop-interchange}}
// CC1AsOptionCHECK32: {{(unknown argument).*-Wtarget-lifetime}}
// CC1AsOptionCHECK32: {{(unknown argument).*-unwindlib=}}
// CC1AsOptionCHECK32: {{(unknown argument).*-v}}
// CC1AsOptionCHECK32: {{(unknown argument).*-vectorize-loops}}
// CC1AsOptionCHECK32: {{(unknown argument).*-vectorize-slp}}
// CC1AsOptionCHECK32: {{(unknown argument).*-verify}}
// RUN: not %clang -cc1as -verify= --verify-debug-info -verify-ignore-unexpected -verify-ignore-unexpected= -verify-pch -vfsoverlay -via-file-asm -vtordisp-mode= --warning-suppression-mappings= --wasm-opt -weak_framework -weak_library -weak_reference_mismatches -weak-l -whatsloaded -why_load -whyload -working-directory -working-directory= -x -y -z  - < /dev/null 2>&1 | FileCheck -check-prefix=CC1AsOptionCHECK33 %s

// CC1AsOptionCHECK33: {{(unknown argument).*-verify=}}
// CC1AsOptionCHECK33: {{(unknown argument).*--verify-debug-info}}
// CC1AsOptionCHECK33: {{(unknown argument).*-verify-ignore-unexpected}}
// CC1AsOptionCHECK33: {{(unknown argument).*-verify-ignore-unexpected=}}
// CC1AsOptionCHECK33: {{(unknown argument).*-verify-pch}}
// CC1AsOptionCHECK33: {{(unknown argument).*-vfsoverlay}}
// CC1AsOptionCHECK33: {{(unknown argument).*-via-file-asm}}
// CC1AsOptionCHECK33: {{(unknown argument).*-vtordisp-mode=}}
// CC1AsOptionCHECK33: {{(unknown argument).*--warning-suppression-mappings=}}
// CC1AsOptionCHECK33: {{(unknown argument).*--wasm-opt}}
// CC1AsOptionCHECK33: {{(unknown argument).*-weak_framework}}
// CC1AsOptionCHECK33: {{(unknown argument).*-weak_library}}
// CC1AsOptionCHECK33: {{(unknown argument).*-weak_reference_mismatches}}
// CC1AsOptionCHECK33: {{(unknown argument).*-weak-l}}
// CC1AsOptionCHECK33: {{(unknown argument).*-whatsloaded}}
// CC1AsOptionCHECK33: {{(unknown argument).*-why_load}}
// CC1AsOptionCHECK33: {{(unknown argument).*-whyload}}
// CC1AsOptionCHECK33: {{(unknown argument).*-working-directory}}
// CC1AsOptionCHECK33: {{(unknown argument).*-working-directory=}}
// CC1AsOptionCHECK33: {{(unknown argument).*-x}}
// CC1AsOptionCHECK33: {{(unknown argument).*-y}}
// CC1AsOptionCHECK33: {{(unknown argument).*-z}}
// RUN: not %clang -cc1 -A -A- -B -EB -EL -G -G= -J -K -L -M -MD -MF -MJ -MM -MMD -Mach -Q -Qunused-arguments -T -V -X -Xanalyzer -Xarch_ -Xarch_device -Xarch_host -Xassembler -Xclang -Xcuda-fatbinary -Xcuda-ptxas -Xflang -Xlinker -Xoffload-linker -Xopenmp-target -Xopenmp-target= -Xpreprocessor -Z -Z-Xlinker-no-demangle -Z-reserved-lib-cckext -Z-reserved-lib-stdc++ -Zlinker-input --CLASSPATH --CLASSPATH= -- -### -AI -Brepro -Brepro- -Bt -Bt+ -EH -EP -G1 -G2 -GA -GF -GF- -GH -GL -GL- -GR -GR- -GS -GS- -GT -GX -GX- -GZ -Gd -Ge -Gh -Gm -Gm- -Gr -Gregcall -Gregcall4 -Gs -Gv -Gw -Gw- -Gy -Gy- -Gz -HELP -J -JMC -JMC- -LD -LDd -LN -MD -MDd -QIfist -QIntel-jcc-erratum -? -Qfast_transcendentals -Qimprecise_fwaits -Qpar -Qpar-report -Qsafe_fp_loads  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK0 %s

// CC1OptionCHECK0: {{(unknown argument).*-A}}
// CC1OptionCHECK0: {{(unknown argument).*-A-}}
// CC1OptionCHECK0: {{(unknown argument).*-B}}
// CC1OptionCHECK0: {{(unknown argument).*-EB}}
// CC1OptionCHECK0: {{(unknown argument).*-EL}}
// CC1OptionCHECK0: {{(unknown argument).*-G}}
// CC1OptionCHECK0: {{(unknown argument).*-G=}}
// CC1OptionCHECK0: {{(unknown argument).*-J}}
// CC1OptionCHECK0: {{(unknown argument).*-K}}
// CC1OptionCHECK0: {{(unknown argument).*-L}}
// CC1OptionCHECK0: {{(unknown argument).*-M}}
// CC1OptionCHECK0: {{(unknown argument).*-MD}}
// CC1OptionCHECK0: {{(unknown argument).*-MF}}
// CC1OptionCHECK0: {{(unknown argument).*-MJ}}
// CC1OptionCHECK0: {{(unknown argument).*-MM}}
// CC1OptionCHECK0: {{(unknown argument).*-MMD}}
// CC1OptionCHECK0: {{(unknown argument).*-Mach}}
// CC1OptionCHECK0: {{(unknown argument).*-Q}}
// CC1OptionCHECK0: {{(unknown argument).*-Qunused-arguments}}
// CC1OptionCHECK0: {{(unknown argument).*-T}}
// CC1OptionCHECK0: {{(unknown argument).*-V}}
// CC1OptionCHECK0: {{(unknown argument).*-X}}
// CC1OptionCHECK0: {{(unknown argument).*-Xanalyzer}}
// CC1OptionCHECK0: {{(unknown argument).*-Xarch_}}
// CC1OptionCHECK0: {{(unknown argument).*-Xarch_device}}
// CC1OptionCHECK0: {{(unknown argument).*-Xarch_host}}
// CC1OptionCHECK0: {{(unknown argument).*-Xassembler}}
// CC1OptionCHECK0: {{(unknown argument).*-Xclang}}
// CC1OptionCHECK0: {{(unknown argument).*-Xcuda-fatbinary}}
// CC1OptionCHECK0: {{(unknown argument).*-Xcuda-ptxas}}
// CC1OptionCHECK0: {{(unknown argument).*-Xflang}}
// CC1OptionCHECK0: {{(unknown argument).*-Xlinker}}
// CC1OptionCHECK0: {{(unknown argument).*-Xoffload-linker}}
// CC1OptionCHECK0: {{(unknown argument).*-Xopenmp-target}}
// CC1OptionCHECK0: {{(unknown argument).*-Xopenmp-target=}}
// CC1OptionCHECK0: {{(unknown argument).*-Xpreprocessor}}
// CC1OptionCHECK0: {{(unknown argument).*-Z}}
// CC1OptionCHECK0: {{(unknown argument).*-Z-Xlinker-no-demangle}}
// CC1OptionCHECK0: {{(unknown argument).*-Z-reserved-lib-cckext}}
// CC1OptionCHECK0: {{(unknown argument).*-Z-reserved-lib-stdc\+\+}}
// CC1OptionCHECK0: {{(unknown argument).*-Zlinker-input}}
// CC1OptionCHECK0: {{(unknown argument).*--CLASSPATH}}
// CC1OptionCHECK0: {{(unknown argument).*--CLASSPATH=}}
// CC1OptionCHECK0: {{(unknown argument).*--}}
// CC1OptionCHECK0: {{(unknown argument).*-###}}
// CC1OptionCHECK0: {{(unknown argument).*-AI}}
// CC1OptionCHECK0: {{(unknown argument).*-Brepro}}
// CC1OptionCHECK0: {{(unknown argument).*-Brepro-}}
// CC1OptionCHECK0: {{(unknown argument).*-Bt}}
// CC1OptionCHECK0: {{(unknown argument).*-Bt\+}}
// CC1OptionCHECK0: {{(unknown argument).*-EH}}
// CC1OptionCHECK0: {{(unknown argument).*-EP}}
// CC1OptionCHECK0: {{(unknown argument).*-G1}}
// CC1OptionCHECK0: {{(unknown argument).*-G2}}
// CC1OptionCHECK0: {{(unknown argument).*-GA}}
// CC1OptionCHECK0: {{(unknown argument).*-GF}}
// CC1OptionCHECK0: {{(unknown argument).*-GF-}}
// CC1OptionCHECK0: {{(unknown argument).*-GH}}
// CC1OptionCHECK0: {{(unknown argument).*-GL}}
// CC1OptionCHECK0: {{(unknown argument).*-GL-}}
// CC1OptionCHECK0: {{(unknown argument).*-GR}}
// CC1OptionCHECK0: {{(unknown argument).*-GR-}}
// CC1OptionCHECK0: {{(unknown argument).*-GS}}
// CC1OptionCHECK0: {{(unknown argument).*-GS-}}
// CC1OptionCHECK0: {{(unknown argument).*-GT}}
// CC1OptionCHECK0: {{(unknown argument).*-GX}}
// CC1OptionCHECK0: {{(unknown argument).*-GX-}}
// CC1OptionCHECK0: {{(unknown argument).*-GZ}}
// CC1OptionCHECK0: {{(unknown argument).*-Gd}}
// CC1OptionCHECK0: {{(unknown argument).*-Ge}}
// CC1OptionCHECK0: {{(unknown argument).*-Gh}}
// CC1OptionCHECK0: {{(unknown argument).*-Gm}}
// CC1OptionCHECK0: {{(unknown argument).*-Gm-}}
// CC1OptionCHECK0: {{(unknown argument).*-Gr}}
// CC1OptionCHECK0: {{(unknown argument).*-Gregcall}}
// CC1OptionCHECK0: {{(unknown argument).*-Gregcall4}}
// CC1OptionCHECK0: {{(unknown argument).*-Gs}}
// CC1OptionCHECK0: {{(unknown argument).*-Gv}}
// CC1OptionCHECK0: {{(unknown argument).*-Gw}}
// CC1OptionCHECK0: {{(unknown argument).*-Gw-}}
// CC1OptionCHECK0: {{(unknown argument).*-Gy}}
// CC1OptionCHECK0: {{(unknown argument).*-Gy-}}
// CC1OptionCHECK0: {{(unknown argument).*-Gz}}
// CC1OptionCHECK0: {{(unknown argument).*-HELP}}
// CC1OptionCHECK0: {{(unknown argument).*-J}}
// CC1OptionCHECK0: {{(unknown argument).*-JMC}}
// CC1OptionCHECK0: {{(unknown argument).*-JMC-}}
// CC1OptionCHECK0: {{(unknown argument).*-LD}}
// CC1OptionCHECK0: {{(unknown argument).*-LDd}}
// CC1OptionCHECK0: {{(unknown argument).*-LN}}
// CC1OptionCHECK0: {{(unknown argument).*-MD}}
// CC1OptionCHECK0: {{(unknown argument).*-MDd}}
// CC1OptionCHECK0: {{(unknown argument).*-QIfist}}
// CC1OptionCHECK0: {{(unknown argument).*-QIntel-jcc-erratum}}
// CC1OptionCHECK0: {{(unknown argument).*-?}}
// CC1OptionCHECK0: {{(unknown argument).*-Qfast_transcendentals}}
// CC1OptionCHECK0: {{(unknown argument).*-Qimprecise_fwaits}}
// CC1OptionCHECK0: {{(unknown argument).*-Qpar}}
// CC1OptionCHECK0: {{(unknown argument).*-Qpar-report}}
// CC1OptionCHECK0: {{(unknown argument).*-Qsafe_fp_loads}}
// RUN: not %clang -cc1 -Qspectre -Qspectre-load -Qspectre-load-cf -Qvec -Qvec- -Qvec-report -TC -TP -Tc -Tp -V -X -Y- -Yc -Yd -Yl -Yu -Z7 -ZH:MD5 -ZH:SHA1 -ZH:SHA_256 -ZI -ZW -Za -Zc: -Zc:__STDC__ -Zc:__cplusplus -Zc:alignedNew -Zc:alignedNew- -Zc:auto -Zc:char8_t -Zc:char8_t- -Zc:dllexportInlines -Zc:dllexportInlines- -Zc:forScope -Zc:inline -Zc:rvalueCast -Zc:sizedDealloc -Zc:sizedDealloc- -Zc:strictStrings -Zc:ternary -Zc:threadSafeInit -Zc:threadSafeInit- -Zc:tlsGuards -Zc:tlsGuards- -Zc:trigraphs -Zc:trigraphs- -Zc:twoPhase -Zc:twoPhase- -Zc:wchar_t -Zc:wchar_t- -Ze -Zg -Zi -Zl -Zm -Zo -Zo- -Zp -Zp -Zs -analyze- -arch: -arm64EC -await -await: -bigobj -c -cgthreads -clang: -clr -constexpr: -d1 -d1PP -d1reportAllClassLayout -d2 -d2FastFail -d2Zi+ -diagnostics:caret -diagnostics:classic -diagnostics:column -diasdkdir -doc -errorReport -execution-charset: -experimental: -exportHeader -external: -external:I -external:W0 -external:W1 -external:W2 -external:W3 -external:W4 -external:env: -favor -fno-sanitize-address-vcasan-lib -fp:contract -fp:except -fp:except-  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK1 %s

// CC1OptionCHECK1: {{(unknown argument).*-Qspectre}}
// CC1OptionCHECK1: {{(unknown argument).*-Qspectre-load}}
// CC1OptionCHECK1: {{(unknown argument).*-Qspectre-load-cf}}
// CC1OptionCHECK1: {{(unknown argument).*-Qvec}}
// CC1OptionCHECK1: {{(unknown argument).*-Qvec-}}
// CC1OptionCHECK1: {{(unknown argument).*-Qvec-report}}
// CC1OptionCHECK1: {{(unknown argument).*-TC}}
// CC1OptionCHECK1: {{(unknown argument).*-TP}}
// CC1OptionCHECK1: {{(unknown argument).*-Tc}}
// CC1OptionCHECK1: {{(unknown argument).*-Tp}}
// CC1OptionCHECK1: {{(unknown argument).*-V}}
// CC1OptionCHECK1: {{(unknown argument).*-X}}
// CC1OptionCHECK1: {{(unknown argument).*-Y-}}
// CC1OptionCHECK1: {{(unknown argument).*-Yc}}
// CC1OptionCHECK1: {{(unknown argument).*-Yd}}
// CC1OptionCHECK1: {{(unknown argument).*-Yl}}
// CC1OptionCHECK1: {{(unknown argument).*-Yu}}
// CC1OptionCHECK1: {{(unknown argument).*-Z7}}
// CC1OptionCHECK1: {{(unknown argument).*-ZH:MD5}}
// CC1OptionCHECK1: {{(unknown argument).*-ZH:SHA1}}
// CC1OptionCHECK1: {{(unknown argument).*-ZH:SHA_256}}
// CC1OptionCHECK1: {{(unknown argument).*-ZI}}
// CC1OptionCHECK1: {{(unknown argument).*-ZW}}
// CC1OptionCHECK1: {{(unknown argument).*-Za}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:__STDC__}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:__cplusplus}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:alignedNew}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:alignedNew-}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:auto}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:char8_t}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:char8_t-}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:dllexportInlines}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:dllexportInlines-}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:forScope}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:inline}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:rvalueCast}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:sizedDealloc}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:sizedDealloc-}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:strictStrings}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:ternary}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:threadSafeInit}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:threadSafeInit-}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:tlsGuards}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:tlsGuards-}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:trigraphs}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:trigraphs-}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:twoPhase}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:twoPhase-}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:wchar_t}}
// CC1OptionCHECK1: {{(unknown argument).*-Zc:wchar_t-}}
// CC1OptionCHECK1: {{(unknown argument).*-Ze}}
// CC1OptionCHECK1: {{(unknown argument).*-Zg}}
// CC1OptionCHECK1: {{(unknown argument).*-Zi}}
// CC1OptionCHECK1: {{(unknown argument).*-Zl}}
// CC1OptionCHECK1: {{(unknown argument).*-Zm}}
// CC1OptionCHECK1: {{(unknown argument).*-Zo}}
// CC1OptionCHECK1: {{(unknown argument).*-Zo-}}
// CC1OptionCHECK1: {{(unknown argument).*-Zp}}
// CC1OptionCHECK1: {{(unknown argument).*-Zp}}
// CC1OptionCHECK1: {{(unknown argument).*-Zs}}
// CC1OptionCHECK1: {{(unknown argument).*-analyze-}}
// CC1OptionCHECK1: {{(unknown argument).*-arch:}}
// CC1OptionCHECK1: {{(unknown argument).*-arm64EC}}
// CC1OptionCHECK1: {{(unknown argument).*-await}}
// CC1OptionCHECK1: {{(unknown argument).*-await:}}
// CC1OptionCHECK1: {{(unknown argument).*-bigobj}}
// CC1OptionCHECK1: {{(unknown argument).*-c}}
// CC1OptionCHECK1: {{(unknown argument).*-cgthreads}}
// CC1OptionCHECK1: {{(unknown argument).*-clang:}}
// CC1OptionCHECK1: {{(unknown argument).*-clr}}
// CC1OptionCHECK1: {{(unknown argument).*-constexpr:}}
// CC1OptionCHECK1: {{(unknown argument).*-d1}}
// CC1OptionCHECK1: {{(unknown argument).*-d1PP}}
// CC1OptionCHECK1: {{(unknown argument).*-d1reportAllClassLayout}}
// CC1OptionCHECK1: {{(unknown argument).*-d2}}
// CC1OptionCHECK1: {{(unknown argument).*-d2FastFail}}
// CC1OptionCHECK1: {{(unknown argument).*-d2Zi\+}}
// CC1OptionCHECK1: {{(unknown argument).*-diagnostics:caret}}
// CC1OptionCHECK1: {{(unknown argument).*-diagnostics:classic}}
// CC1OptionCHECK1: {{(unknown argument).*-diagnostics:column}}
// CC1OptionCHECK1: {{(unknown argument).*-diasdkdir}}
// CC1OptionCHECK1: {{(unknown argument).*-doc}}
// CC1OptionCHECK1: {{(unknown argument).*-errorReport}}
// CC1OptionCHECK1: {{(unknown argument).*-execution-charset:}}
// CC1OptionCHECK1: {{(unknown argument).*-experimental:}}
// CC1OptionCHECK1: {{(unknown argument).*-exportHeader}}
// CC1OptionCHECK1: {{(unknown argument).*-external:}}
// CC1OptionCHECK1: {{(unknown argument).*-external:I}}
// CC1OptionCHECK1: {{(unknown argument).*-external:W0}}
// CC1OptionCHECK1: {{(unknown argument).*-external:W1}}
// CC1OptionCHECK1: {{(unknown argument).*-external:W2}}
// CC1OptionCHECK1: {{(unknown argument).*-external:W3}}
// CC1OptionCHECK1: {{(unknown argument).*-external:W4}}
// CC1OptionCHECK1: {{(unknown argument).*-external:env:}}
// CC1OptionCHECK1: {{(unknown argument).*-favor}}
// CC1OptionCHECK1: {{(unknown argument).*-fno-sanitize-address-vcasan-lib}}
// CC1OptionCHECK1: {{(unknown argument).*-fp:contract}}
// CC1OptionCHECK1: {{(unknown argument).*-fp:except}}
// CC1OptionCHECK1: {{(unknown argument).*-fp:except-}}
// RUN: not %clang -cc1 -fp:fast -fp:precise -fp:strict -fsanitize-address-use-after-return -guard: -headerUnit -headerUnit:angle -headerUnit:quote -headerName: -homeparams -hotpatch -imsvc -kernel -kernel- -link -nologo -permissive -permissive- -reference -sdl -sdl- -showFilenames -showFilenames- -showIncludes -showIncludes:user -sourceDependencies -sourceDependencies:directives -source-charset: -std: -translateInclude -tune: -u -utf-8 -validate-charset -validate-charset- -vctoolsdir -vctoolsversion -vd -vmb -vmg -vmm -vms -vmv -volatile:iso -volatile:ms -wd -winsdkdir -winsdkversion -winsysroot --all-warnings --analyzer-no-default-checks --assemble --assert --assert= --bootclasspath --bootclasspath= --classpath --classpath= --comments --comments-in-macros --compile --constant-cfstrings --debug --debug= --define-macro --define-macro= --dependencies --dyld-prefix --dyld-prefix= --encoding --encoding= --entry --extdirs --extdirs= --extra-warnings --for-linker --for-linker= --force-link --force-link= --help-hidden --language --language= --library-directory --library-directory= --mhwdiv --mhwdiv= --no-line-commands --no-standard-includes --no-standard-libraries --no-undefined --param --param= --precompile --prefix --prefix= --preprocess --print-diagnostic-categories --print-file-name --print-missing-file-dependencies --print-prog-name  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK2 %s

// CC1OptionCHECK2: {{(unknown argument).*-fp:fast}}
// CC1OptionCHECK2: {{(unknown argument).*-fp:precise}}
// CC1OptionCHECK2: {{(unknown argument).*-fp:strict}}
// CC1OptionCHECK2: {{(unknown argument).*-fsanitize-address-use-after-return}}
// CC1OptionCHECK2: {{(unknown argument).*-guard:}}
// CC1OptionCHECK2: {{(unknown argument).*-headerUnit}}
// CC1OptionCHECK2: {{(unknown argument).*-headerUnit:angle}}
// CC1OptionCHECK2: {{(unknown argument).*-headerUnit:quote}}
// CC1OptionCHECK2: {{(unknown argument).*-headerName:}}
// CC1OptionCHECK2: {{(unknown argument).*-homeparams}}
// CC1OptionCHECK2: {{(unknown argument).*-hotpatch}}
// CC1OptionCHECK2: {{(unknown argument).*-imsvc}}
// CC1OptionCHECK2: {{(unknown argument).*-kernel}}
// CC1OptionCHECK2: {{(unknown argument).*-kernel-}}
// CC1OptionCHECK2: {{(unknown argument).*-link}}
// CC1OptionCHECK2: {{(unknown argument).*-nologo}}
// CC1OptionCHECK2: {{(unknown argument).*-permissive}}
// CC1OptionCHECK2: {{(unknown argument).*-permissive-}}
// CC1OptionCHECK2: {{(unknown argument).*-reference}}
// CC1OptionCHECK2: {{(unknown argument).*-sdl}}
// CC1OptionCHECK2: {{(unknown argument).*-sdl-}}
// CC1OptionCHECK2: {{(unknown argument).*-showFilenames}}
// CC1OptionCHECK2: {{(unknown argument).*-showFilenames-}}
// CC1OptionCHECK2: {{(unknown argument).*-showIncludes}}
// CC1OptionCHECK2: {{(unknown argument).*-showIncludes:user}}
// CC1OptionCHECK2: {{(unknown argument).*-sourceDependencies}}
// CC1OptionCHECK2: {{(unknown argument).*-sourceDependencies:directives}}
// CC1OptionCHECK2: {{(unknown argument).*-source-charset:}}
// CC1OptionCHECK2: {{(unknown argument).*-std:}}
// CC1OptionCHECK2: {{(unknown argument).*-translateInclude}}
// CC1OptionCHECK2: {{(unknown argument).*-tune:}}
// CC1OptionCHECK2: {{(unknown argument).*-u}}
// CC1OptionCHECK2: {{(unknown argument).*-utf-8}}
// CC1OptionCHECK2: {{(unknown argument).*-validate-charset}}
// CC1OptionCHECK2: {{(unknown argument).*-validate-charset-}}
// CC1OptionCHECK2: {{(unknown argument).*-vctoolsdir}}
// CC1OptionCHECK2: {{(unknown argument).*-vctoolsversion}}
// CC1OptionCHECK2: {{(unknown argument).*-vd}}
// CC1OptionCHECK2: {{(unknown argument).*-vmb}}
// CC1OptionCHECK2: {{(unknown argument).*-vmg}}
// CC1OptionCHECK2: {{(unknown argument).*-vmm}}
// CC1OptionCHECK2: {{(unknown argument).*-vms}}
// CC1OptionCHECK2: {{(unknown argument).*-vmv}}
// CC1OptionCHECK2: {{(unknown argument).*-volatile:iso}}
// CC1OptionCHECK2: {{(unknown argument).*-volatile:ms}}
// CC1OptionCHECK2: {{(unknown argument).*-wd}}
// CC1OptionCHECK2: {{(unknown argument).*-winsdkdir}}
// CC1OptionCHECK2: {{(unknown argument).*-winsdkversion}}
// CC1OptionCHECK2: {{(unknown argument).*-winsysroot}}
// CC1OptionCHECK2: {{(unknown argument).*--all-warnings}}
// CC1OptionCHECK2: {{(unknown argument).*--analyzer-no-default-checks}}
// CC1OptionCHECK2: {{(unknown argument).*--assemble}}
// CC1OptionCHECK2: {{(unknown argument).*--assert}}
// CC1OptionCHECK2: {{(unknown argument).*--assert=}}
// CC1OptionCHECK2: {{(unknown argument).*--bootclasspath}}
// CC1OptionCHECK2: {{(unknown argument).*--bootclasspath=}}
// CC1OptionCHECK2: {{(unknown argument).*--classpath}}
// CC1OptionCHECK2: {{(unknown argument).*--classpath=}}
// CC1OptionCHECK2: {{(unknown argument).*--comments}}
// CC1OptionCHECK2: {{(unknown argument).*--comments-in-macros}}
// CC1OptionCHECK2: {{(unknown argument).*--compile}}
// CC1OptionCHECK2: {{(unknown argument).*--constant-cfstrings}}
// CC1OptionCHECK2: {{(unknown argument).*--debug}}
// CC1OptionCHECK2: {{(unknown argument).*--debug=}}
// CC1OptionCHECK2: {{(unknown argument).*--define-macro}}
// CC1OptionCHECK2: {{(unknown argument).*--define-macro=}}
// CC1OptionCHECK2: {{(unknown argument).*--dependencies}}
// CC1OptionCHECK2: {{(unknown argument).*--dyld-prefix}}
// CC1OptionCHECK2: {{(unknown argument).*--dyld-prefix=}}
// CC1OptionCHECK2: {{(unknown argument).*--encoding}}
// CC1OptionCHECK2: {{(unknown argument).*--encoding=}}
// CC1OptionCHECK2: {{(unknown argument).*--entry}}
// CC1OptionCHECK2: {{(unknown argument).*--extdirs}}
// CC1OptionCHECK2: {{(unknown argument).*--extdirs=}}
// CC1OptionCHECK2: {{(unknown argument).*--extra-warnings}}
// CC1OptionCHECK2: {{(unknown argument).*--for-linker}}
// CC1OptionCHECK2: {{(unknown argument).*--for-linker=}}
// CC1OptionCHECK2: {{(unknown argument).*--force-link}}
// CC1OptionCHECK2: {{(unknown argument).*--force-link=}}
// CC1OptionCHECK2: {{(unknown argument).*--help-hidden}}
// CC1OptionCHECK2: {{(unknown argument).*--language}}
// CC1OptionCHECK2: {{(unknown argument).*--language=}}
// CC1OptionCHECK2: {{(unknown argument).*--library-directory}}
// CC1OptionCHECK2: {{(unknown argument).*--library-directory=}}
// CC1OptionCHECK2: {{(unknown argument).*--mhwdiv}}
// CC1OptionCHECK2: {{(unknown argument).*--mhwdiv=}}
// CC1OptionCHECK2: {{(unknown argument).*--no-line-commands}}
// CC1OptionCHECK2: {{(unknown argument).*--no-standard-includes}}
// CC1OptionCHECK2: {{(unknown argument).*--no-standard-libraries}}
// CC1OptionCHECK2: {{(unknown argument).*--no-undefined}}
// CC1OptionCHECK2: {{(unknown argument).*--param}}
// CC1OptionCHECK2: {{(unknown argument).*--param=}}
// CC1OptionCHECK2: {{(unknown argument).*--precompile}}
// CC1OptionCHECK2: {{(unknown argument).*--prefix}}
// CC1OptionCHECK2: {{(unknown argument).*--prefix=}}
// CC1OptionCHECK2: {{(unknown argument).*--preprocess}}
// CC1OptionCHECK2: {{(unknown argument).*--print-diagnostic-categories}}
// CC1OptionCHECK2: {{(unknown argument).*--print-file-name}}
// CC1OptionCHECK2: {{(unknown argument).*--print-missing-file-dependencies}}
// CC1OptionCHECK2: {{(unknown argument).*--print-prog-name}}
// RUN: not %clang -cc1 --profile --resource --resource= --rtlib -serialize-diagnostics --signed-char --std --stdlib --sysroot --sysroot= --target-help --trace-includes --undefine-macro --undefine-macro= --unsigned-char --user-dependencies --verbose --warn- --warn-= --write-dependencies --write-user-dependencies -alias_list -all_load -allowable_client --amdgpu-arch-tool= -fnew-alignment -faligned-new -fno-aligned-new -fsched-interblock -ftemplate-depth- -ftree-vectorize -fno-tree-vectorize -fcuda-rdc -ftree-slp-vectorize -fno-tree-slp-vectorize -fterminated-vtables -fno-cuda-rdc --hip-device-lib-path= -grecord-gcc-switches -gno-record-gcc-switches -miphoneos-version-min= -miphonesimulator-version-min= -mllvm= -mmacosx-version-min= -nocudainc -nocudalib -print-multiarch --system-header-prefix --no-system-header-prefix -integrated-as -fcuda-approx-transcendentals -fno-cuda-approx-transcendentals -Gs -Qgather- -Qscatter- -Xmicrosoft-visualc-tools-root -Xmicrosoft-visualc-tools-version -Xmicrosoft-windows-sdk-root -Xmicrosoft-windows-sdk-version -Xmicrosoft-windows-sys-root -Qembed_debug -shared-libasan -static-libasan -fslp-vectorize-aggressive -fdiagnostics-color -fno-diagnostics-color -frecord-gcc-switches -fno-record-gcc-switches -fno-slp-vectorize-aggressive -Xclang= -Xparser -Xcompiler -fexpensive-optimizations -fno-expensive-optimizations -fdefer-pop -fno-defer-pop -fextended-identifiers -fno-extended-identifiers -fhonor-infinites -fno-honor-infinites -findirect-virtual-calls --config -ansi -arch -arch_errors_fatal -arch_only --autocomplete= -b -bind_at_load -bundle -bundle_loader -c -canonical-prefixes -ccc- -ccc-gcc-name -ccc-install-dir -ccc-print-bindings -ccc-print-phases -cl-denorms-are-zero -cl-no-stdinc  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK3 %s

// CC1OptionCHECK3: {{(unknown argument).*--profile}}
// CC1OptionCHECK3: {{(unknown argument).*--resource}}
// CC1OptionCHECK3: {{(unknown argument).*--resource=}}
// CC1OptionCHECK3: {{(unknown argument).*--rtlib}}
// CC1OptionCHECK3: {{(unknown argument).*-serialize-diagnostics}}
// CC1OptionCHECK3: {{(unknown argument).*--signed-char}}
// CC1OptionCHECK3: {{(unknown argument).*--std}}
// CC1OptionCHECK3: {{(unknown argument).*--stdlib}}
// CC1OptionCHECK3: {{(unknown argument).*--sysroot}}
// CC1OptionCHECK3: {{(unknown argument).*--sysroot=}}
// CC1OptionCHECK3: {{(unknown argument).*--target-help}}
// CC1OptionCHECK3: {{(unknown argument).*--trace-includes}}
// CC1OptionCHECK3: {{(unknown argument).*--undefine-macro}}
// CC1OptionCHECK3: {{(unknown argument).*--undefine-macro=}}
// CC1OptionCHECK3: {{(unknown argument).*--unsigned-char}}
// CC1OptionCHECK3: {{(unknown argument).*--user-dependencies}}
// CC1OptionCHECK3: {{(unknown argument).*--verbose}}
// CC1OptionCHECK3: {{(unknown argument).*--warn-}}
// CC1OptionCHECK3: {{(unknown argument).*--warn-=}}
// CC1OptionCHECK3: {{(unknown argument).*--write-dependencies}}
// CC1OptionCHECK3: {{(unknown argument).*--write-user-dependencies}}
// CC1OptionCHECK3: {{(unknown argument).*-alias_list}}
// CC1OptionCHECK3: {{(unknown argument).*-all_load}}
// CC1OptionCHECK3: {{(unknown argument).*-allowable_client}}
// CC1OptionCHECK3: {{(unknown argument).*--amdgpu-arch-tool=}}
// CC1OptionCHECK3: {{(unknown argument).*-fnew-alignment}}
// CC1OptionCHECK3: {{(unknown argument).*-faligned-new}}
// CC1OptionCHECK3: {{(unknown argument).*-fno-aligned-new}}
// CC1OptionCHECK3: {{(unknown argument).*-fsched-interblock}}
// CC1OptionCHECK3: {{(unknown argument).*-ftemplate-depth-}}
// CC1OptionCHECK3: {{(unknown argument).*-ftree-vectorize}}
// CC1OptionCHECK3: {{(unknown argument).*-fno-tree-vectorize}}
// CC1OptionCHECK3: {{(unknown argument).*-fcuda-rdc}}
// CC1OptionCHECK3: {{(unknown argument).*-ftree-slp-vectorize}}
// CC1OptionCHECK3: {{(unknown argument).*-fno-tree-slp-vectorize}}
// CC1OptionCHECK3: {{(unknown argument).*-fterminated-vtables}}
// CC1OptionCHECK3: {{(unknown argument).*-fno-cuda-rdc}}
// CC1OptionCHECK3: {{(unknown argument).*--hip-device-lib-path=}}
// CC1OptionCHECK3: {{(unknown argument).*-grecord-gcc-switches}}
// CC1OptionCHECK3: {{(unknown argument).*-gno-record-gcc-switches}}
// CC1OptionCHECK3: {{(unknown argument).*-miphoneos-version-min=}}
// CC1OptionCHECK3: {{(unknown argument).*-miphonesimulator-version-min=}}
// CC1OptionCHECK3: {{(unknown argument).*-mllvm=}}
// CC1OptionCHECK3: {{(unknown argument).*-mmacosx-version-min=}}
// CC1OptionCHECK3: {{(unknown argument).*-nocudainc}}
// CC1OptionCHECK3: {{(unknown argument).*-nocudalib}}
// CC1OptionCHECK3: {{(unknown argument).*-print-multiarch}}
// CC1OptionCHECK3: {{(unknown argument).*--system-header-prefix}}
// CC1OptionCHECK3: {{(unknown argument).*--no-system-header-prefix}}
// CC1OptionCHECK3: {{(unknown argument).*-integrated-as}}
// CC1OptionCHECK3: {{(unknown argument).*-fcuda-approx-transcendentals}}
// CC1OptionCHECK3: {{(unknown argument).*-fno-cuda-approx-transcendentals}}
// CC1OptionCHECK3: {{(unknown argument).*-Gs}}
// CC1OptionCHECK3: {{(unknown argument).*-Qgather-}}
// CC1OptionCHECK3: {{(unknown argument).*-Qscatter-}}
// CC1OptionCHECK3: {{(unknown argument).*-Xmicrosoft-visualc-tools-root}}
// CC1OptionCHECK3: {{(unknown argument).*-Xmicrosoft-visualc-tools-version}}
// CC1OptionCHECK3: {{(unknown argument).*-Xmicrosoft-windows-sdk-root}}
// CC1OptionCHECK3: {{(unknown argument).*-Xmicrosoft-windows-sdk-version}}
// CC1OptionCHECK3: {{(unknown argument).*-Xmicrosoft-windows-sys-root}}
// CC1OptionCHECK3: {{(unknown argument).*-Qembed_debug}}
// CC1OptionCHECK3: {{(unknown argument).*-shared-libasan}}
// CC1OptionCHECK3: {{(unknown argument).*-static-libasan}}
// CC1OptionCHECK3: {{(unknown argument).*-fslp-vectorize-aggressive}}
// CC1OptionCHECK3: {{(unknown argument).*-fdiagnostics-color}}
// CC1OptionCHECK3: {{(unknown argument).*-fno-diagnostics-color}}
// CC1OptionCHECK3: {{(unknown argument).*-frecord-gcc-switches}}
// CC1OptionCHECK3: {{(unknown argument).*-fno-record-gcc-switches}}
// CC1OptionCHECK3: {{(unknown argument).*-fno-slp-vectorize-aggressive}}
// CC1OptionCHECK3: {{(unknown argument).*-Xclang=}}
// CC1OptionCHECK3: {{(unknown argument).*-Xparser}}
// CC1OptionCHECK3: {{(unknown argument).*-Xcompiler}}
// CC1OptionCHECK3: {{(unknown argument).*-fexpensive-optimizations}}
// CC1OptionCHECK3: {{(unknown argument).*-fno-expensive-optimizations}}
// CC1OptionCHECK3: {{(unknown argument).*-fdefer-pop}}
// CC1OptionCHECK3: {{(unknown argument).*-fno-defer-pop}}
// CC1OptionCHECK3: {{(unknown argument).*-fextended-identifiers}}
// CC1OptionCHECK3: {{(unknown argument).*-fno-extended-identifiers}}
// CC1OptionCHECK3: {{(unknown argument).*-fhonor-infinites}}
// CC1OptionCHECK3: {{(unknown argument).*-fno-honor-infinites}}
// CC1OptionCHECK3: {{(unknown argument).*-findirect-virtual-calls}}
// CC1OptionCHECK3: {{(unknown argument).*--config}}
// CC1OptionCHECK3: {{(unknown argument).*-ansi}}
// CC1OptionCHECK3: {{(unknown argument).*-arch}}
// CC1OptionCHECK3: {{(unknown argument).*-arch_errors_fatal}}
// CC1OptionCHECK3: {{(unknown argument).*-arch_only}}
// CC1OptionCHECK3: {{(unknown argument).*--autocomplete=}}
// CC1OptionCHECK3: {{(unknown argument).*-b}}
// CC1OptionCHECK3: {{(unknown argument).*-bind_at_load}}
// CC1OptionCHECK3: {{(unknown argument).*-bundle}}
// CC1OptionCHECK3: {{(unknown argument).*-bundle_loader}}
// CC1OptionCHECK3: {{(unknown argument).*-c}}
// CC1OptionCHECK3: {{(unknown argument).*-canonical-prefixes}}
// CC1OptionCHECK3: {{(unknown argument).*-ccc-}}
// CC1OptionCHECK3: {{(unknown argument).*-ccc-gcc-name}}
// CC1OptionCHECK3: {{(unknown argument).*-ccc-install-dir}}
// CC1OptionCHECK3: {{(unknown argument).*-ccc-print-bindings}}
// CC1OptionCHECK3: {{(unknown argument).*-ccc-print-phases}}
// CC1OptionCHECK3: {{(unknown argument).*-cl-denorms-are-zero}}
// CC1OptionCHECK3: {{(unknown argument).*-cl-no-stdinc}}
// RUN: not %clang -cc1 -client_name -combine -compatibility_version --config= --config-system-dir= --config-user-dir= -coverage -cpp -cpp-precomp --cuda-compile-host-device --cuda-device-only --cuda-feature= --cuda-gpu-arch= --cuda-host-only --cuda-include-ptx= --cuda-noopt-device-debug --cuda-path= --cuda-path-ignore-env -current_version -fc++-static-destructors -dA -d -d -darwin-target-variant -dead_strip --defsym --driver-mode= -dsym-dir -dumpmachine -dumpspecs -dumpversion -dwarf-debug-producer -Vd -HV -hlsl-no-stdinc --dxv-path= -dylib_file -dylinker -dylinker_install_name -dynamic -dynamiclib -e -emit-ast -emit-fir -emit-hlfir -emit-mlir -emit-pristine-llvm --emit-static-lib -enable-16bit-types --end-no-unused-arguments -exported_symbols_list -fPIC -fPIE -faccess-control -faggressive-function-elimination -falign-commons -falign-functions -falign-functions= -falign-jumps -falign-jumps= -falign-labels -falign-labels= -falign-loops -faligned-new= -fall-intrinsics -fallow-unsupported -falternative-parameter-statement -faltivec -fanalyzed-objects-for-unparse -fandroid-pad-segment -fkeep-inline-functions -funit-at-a-time -fapple-link-rtlib -fasm -fassociative-math -fassume-sane-operator-new -fassume-unique-vtables -fassumptions -fast -fastcp -fastf -fasynchronous-unwind-tables -fauto-import -fauto-profile= -fauto-profile-accurate -fautolink -fautomatic -fbackslash -fbacktrace -fblas-matmul-limit= -fbootclasspath= -fbounds-check -fbracket-depth= -fbranch-count-reg -fbuild-session-file= -fbuiltin -fbuiltin-module-map -fcall-saved-x10 -fcall-saved-x11 -fcall-saved-x12  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK4 %s

// CC1OptionCHECK4: {{(unknown argument).*-client_name}}
// CC1OptionCHECK4: {{(unknown argument).*-combine}}
// CC1OptionCHECK4: {{(unknown argument).*-compatibility_version}}
// CC1OptionCHECK4: {{(unknown argument).*--config=}}
// CC1OptionCHECK4: {{(unknown argument).*--config-system-dir=}}
// CC1OptionCHECK4: {{(unknown argument).*--config-user-dir=}}
// CC1OptionCHECK4: {{(unknown argument).*-coverage}}
// CC1OptionCHECK4: {{(unknown argument).*-cpp}}
// CC1OptionCHECK4: {{(unknown argument).*-cpp-precomp}}
// CC1OptionCHECK4: {{(unknown argument).*--cuda-compile-host-device}}
// CC1OptionCHECK4: {{(unknown argument).*--cuda-device-only}}
// CC1OptionCHECK4: {{(unknown argument).*--cuda-feature=}}
// CC1OptionCHECK4: {{(unknown argument).*--cuda-gpu-arch=}}
// CC1OptionCHECK4: {{(unknown argument).*--cuda-host-only}}
// CC1OptionCHECK4: {{(unknown argument).*--cuda-include-ptx=}}
// CC1OptionCHECK4: {{(unknown argument).*--cuda-noopt-device-debug}}
// CC1OptionCHECK4: {{(unknown argument).*--cuda-path=}}
// CC1OptionCHECK4: {{(unknown argument).*--cuda-path-ignore-env}}
// CC1OptionCHECK4: {{(unknown argument).*-current_version}}
// CC1OptionCHECK4: {{(unknown argument).*-fc\+\+-static-destructors}}
// CC1OptionCHECK4: {{(unknown argument).*-dA}}
// CC1OptionCHECK4: {{(unknown argument).*-d}}
// CC1OptionCHECK4: {{(unknown argument).*-d}}
// CC1OptionCHECK4: {{(unknown argument).*-darwin-target-variant}}
// CC1OptionCHECK4: {{(unknown argument).*-dead_strip}}
// CC1OptionCHECK4: {{(unknown argument).*--defsym}}
// CC1OptionCHECK4: {{(unknown argument).*--driver-mode=}}
// CC1OptionCHECK4: {{(unknown argument).*-dsym-dir}}
// CC1OptionCHECK4: {{(unknown argument).*-dumpmachine}}
// CC1OptionCHECK4: {{(unknown argument).*-dumpspecs}}
// CC1OptionCHECK4: {{(unknown argument).*-dumpversion}}
// CC1OptionCHECK4: {{(unknown argument).*-dwarf-debug-producer}}
// CC1OptionCHECK4: {{(unknown argument).*-Vd}}
// CC1OptionCHECK4: {{(unknown argument).*-HV}}
// CC1OptionCHECK4: {{(unknown argument).*-hlsl-no-stdinc}}
// CC1OptionCHECK4: {{(unknown argument).*--dxv-path=}}
// CC1OptionCHECK4: {{(unknown argument).*-dylib_file}}
// CC1OptionCHECK4: {{(unknown argument).*-dylinker}}
// CC1OptionCHECK4: {{(unknown argument).*-dylinker_install_name}}
// CC1OptionCHECK4: {{(unknown argument).*-dynamic}}
// CC1OptionCHECK4: {{(unknown argument).*-dynamiclib}}
// CC1OptionCHECK4: {{(unknown argument).*-e}}
// CC1OptionCHECK4: {{(unknown argument).*-emit-ast}}
// CC1OptionCHECK4: {{(unknown argument).*-emit-fir}}
// CC1OptionCHECK4: {{(unknown argument).*-emit-hlfir}}
// CC1OptionCHECK4: {{(unknown argument).*-emit-mlir}}
// CC1OptionCHECK4: {{(unknown argument).*-emit-pristine-llvm}}
// CC1OptionCHECK4: {{(unknown argument).*--emit-static-lib}}
// CC1OptionCHECK4: {{(unknown argument).*-enable-16bit-types}}
// CC1OptionCHECK4: {{(unknown argument).*--end-no-unused-arguments}}
// CC1OptionCHECK4: {{(unknown argument).*-exported_symbols_list}}
// CC1OptionCHECK4: {{(unknown argument).*-fPIC}}
// CC1OptionCHECK4: {{(unknown argument).*-fPIE}}
// CC1OptionCHECK4: {{(unknown argument).*-faccess-control}}
// CC1OptionCHECK4: {{(unknown argument).*-faggressive-function-elimination}}
// CC1OptionCHECK4: {{(unknown argument).*-falign-commons}}
// CC1OptionCHECK4: {{(unknown argument).*-falign-functions}}
// CC1OptionCHECK4: {{(unknown argument).*-falign-functions=}}
// CC1OptionCHECK4: {{(unknown argument).*-falign-jumps}}
// CC1OptionCHECK4: {{(unknown argument).*-falign-jumps=}}
// CC1OptionCHECK4: {{(unknown argument).*-falign-labels}}
// CC1OptionCHECK4: {{(unknown argument).*-falign-labels=}}
// CC1OptionCHECK4: {{(unknown argument).*-falign-loops}}
// CC1OptionCHECK4: {{(unknown argument).*-faligned-new=}}
// CC1OptionCHECK4: {{(unknown argument).*-fall-intrinsics}}
// CC1OptionCHECK4: {{(unknown argument).*-fallow-unsupported}}
// CC1OptionCHECK4: {{(unknown argument).*-falternative-parameter-statement}}
// CC1OptionCHECK4: {{(unknown argument).*-faltivec}}
// CC1OptionCHECK4: {{(unknown argument).*-fanalyzed-objects-for-unparse}}
// CC1OptionCHECK4: {{(unknown argument).*-fandroid-pad-segment}}
// CC1OptionCHECK4: {{(unknown argument).*-fkeep-inline-functions}}
// CC1OptionCHECK4: {{(unknown argument).*-funit-at-a-time}}
// CC1OptionCHECK4: {{(unknown argument).*-fapple-link-rtlib}}
// CC1OptionCHECK4: {{(unknown argument).*-fasm}}
// CC1OptionCHECK4: {{(unknown argument).*-fassociative-math}}
// CC1OptionCHECK4: {{(unknown argument).*-fassume-sane-operator-new}}
// CC1OptionCHECK4: {{(unknown argument).*-fassume-unique-vtables}}
// CC1OptionCHECK4: {{(unknown argument).*-fassumptions}}
// CC1OptionCHECK4: {{(unknown argument).*-fast}}
// CC1OptionCHECK4: {{(unknown argument).*-fastcp}}
// CC1OptionCHECK4: {{(unknown argument).*-fastf}}
// CC1OptionCHECK4: {{(unknown argument).*-fasynchronous-unwind-tables}}
// CC1OptionCHECK4: {{(unknown argument).*-fauto-import}}
// CC1OptionCHECK4: {{(unknown argument).*-fauto-profile=}}
// CC1OptionCHECK4: {{(unknown argument).*-fauto-profile-accurate}}
// CC1OptionCHECK4: {{(unknown argument).*-fautolink}}
// CC1OptionCHECK4: {{(unknown argument).*-fautomatic}}
// CC1OptionCHECK4: {{(unknown argument).*-fbackslash}}
// CC1OptionCHECK4: {{(unknown argument).*-fbacktrace}}
// CC1OptionCHECK4: {{(unknown argument).*-fblas-matmul-limit=}}
// CC1OptionCHECK4: {{(unknown argument).*-fbootclasspath=}}
// CC1OptionCHECK4: {{(unknown argument).*-fbounds-check}}
// CC1OptionCHECK4: {{(unknown argument).*-fbracket-depth=}}
// CC1OptionCHECK4: {{(unknown argument).*-fbranch-count-reg}}
// CC1OptionCHECK4: {{(unknown argument).*-fbuild-session-file=}}
// CC1OptionCHECK4: {{(unknown argument).*-fbuiltin}}
// CC1OptionCHECK4: {{(unknown argument).*-fbuiltin-module-map}}
// CC1OptionCHECK4: {{(unknown argument).*-fcall-saved-x10}}
// CC1OptionCHECK4: {{(unknown argument).*-fcall-saved-x11}}
// CC1OptionCHECK4: {{(unknown argument).*-fcall-saved-x12}}
// RUN: not %clang -cc1 -fcall-saved-x13 -fcall-saved-x14 -fcall-saved-x15 -fcall-saved-x18 -fcall-saved-x8 -fcall-saved-x9 -fcaller-saves -fcaret-diagnostics -fcgl -fcheck= -fcheck-array-temporaries -fclasspath= -fcoarray= -fcodegen-data-generate -fcodegen-data-generate= -fcodegen-data-use -fcodegen-data-use= -fcompile-resource= -fconstant-cfstrings -fconstant-string-class= -fconvert= -fcrash-diagnostics -fcrash-diagnostics= -fcrash-diagnostics-dir= -fcray-pointer -fcreate-profile -fcs-profile-generate -fcs-profile-generate= -fcuda-flush-denormals-to-zero -fcxx-modules -fd-lines-as-code -fd-lines-as-comments -fdebug-default-version= -fdebug-dump-all -fdebug-dump-parse-tree -fdebug-dump-parse-tree-no-sema -fdebug-dump-parsing-log -fdebug-dump-pft -fdebug-dump-provenance -fdebug-dump-symbols -fdebug-macro -fdebug-measure-parse-tree -fdebug-module-writer -fdebug-pass-arguments -fdebug-pass-structure -fdebug-pre-fir-tree -fdebug-types-section -fdebug-unparse -fdebug-unparse-no-sema -fdebug-unparse-with-modules -fdebug-unparse-with-symbols -fdefault-double-8 -fdefault-inline -fdefault-integer-8 -fdefault-real-8 -fdelete-null-pointer-checks -fdevirtualize -fdevirtualize-speculatively -fdiagnostics-color= -fdiagnostics-fixit-info -fdiagnostics-format= -fdiagnostics-show-category= -fdiagnostics-show-line-numbers -fdiagnostics-show-location= -fdiagnostics-show-option -fdisable-integer-16 -fdisable-integer-2 -fdisable-real-10 -fdisable-real-3 -fdiscard-value-names -fdollar-ok -fdouble-square-bracket-attributes -fdriver-only -fdump-fortran-optimized -fdump-fortran-original -fdump-parse-tree -fdwarf2-cfi-asm -fdwarf-directory-asm -fdwarf-exceptions -felide-constructors -feliminate-unused-debug-symbols -feliminate-unused-debug-types -fembed-bitcode -fembed-bitcode-marker -fencoding= -ferror-limit= -fescaping-block-tail-calls -fexcess-precision= -fexec-charset= -fexperimental-isel -fextdirs= -fexternal-blas -ff2c -ffile-compilation-dir= -ffile-prefix-map= -finline-limit -ffixed-a0 -ffixed-a1 -ffixed-a2 -ffixed-a3  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK5 %s

// CC1OptionCHECK5: {{(unknown argument).*-fcall-saved-x13}}
// CC1OptionCHECK5: {{(unknown argument).*-fcall-saved-x14}}
// CC1OptionCHECK5: {{(unknown argument).*-fcall-saved-x15}}
// CC1OptionCHECK5: {{(unknown argument).*-fcall-saved-x18}}
// CC1OptionCHECK5: {{(unknown argument).*-fcall-saved-x8}}
// CC1OptionCHECK5: {{(unknown argument).*-fcall-saved-x9}}
// CC1OptionCHECK5: {{(unknown argument).*-fcaller-saves}}
// CC1OptionCHECK5: {{(unknown argument).*-fcaret-diagnostics}}
// CC1OptionCHECK5: {{(unknown argument).*-fcgl}}
// CC1OptionCHECK5: {{(unknown argument).*-fcheck=}}
// CC1OptionCHECK5: {{(unknown argument).*-fcheck-array-temporaries}}
// CC1OptionCHECK5: {{(unknown argument).*-fclasspath=}}
// CC1OptionCHECK5: {{(unknown argument).*-fcoarray=}}
// CC1OptionCHECK5: {{(unknown argument).*-fcodegen-data-generate}}
// CC1OptionCHECK5: {{(unknown argument).*-fcodegen-data-generate=}}
// CC1OptionCHECK5: {{(unknown argument).*-fcodegen-data-use}}
// CC1OptionCHECK5: {{(unknown argument).*-fcodegen-data-use=}}
// CC1OptionCHECK5: {{(unknown argument).*-fcompile-resource=}}
// CC1OptionCHECK5: {{(unknown argument).*-fconstant-cfstrings}}
// CC1OptionCHECK5: {{(unknown argument).*-fconstant-string-class=}}
// CC1OptionCHECK5: {{(unknown argument).*-fconvert=}}
// CC1OptionCHECK5: {{(unknown argument).*-fcrash-diagnostics}}
// CC1OptionCHECK5: {{(unknown argument).*-fcrash-diagnostics=}}
// CC1OptionCHECK5: {{(unknown argument).*-fcrash-diagnostics-dir=}}
// CC1OptionCHECK5: {{(unknown argument).*-fcray-pointer}}
// CC1OptionCHECK5: {{(unknown argument).*-fcreate-profile}}
// CC1OptionCHECK5: {{(unknown argument).*-fcs-profile-generate}}
// CC1OptionCHECK5: {{(unknown argument).*-fcs-profile-generate=}}
// CC1OptionCHECK5: {{(unknown argument).*-fcuda-flush-denormals-to-zero}}
// CC1OptionCHECK5: {{(unknown argument).*-fcxx-modules}}
// CC1OptionCHECK5: {{(unknown argument).*-fd-lines-as-code}}
// CC1OptionCHECK5: {{(unknown argument).*-fd-lines-as-comments}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-default-version=}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-dump-all}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-dump-parse-tree}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-dump-parse-tree-no-sema}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-dump-parsing-log}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-dump-pft}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-dump-provenance}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-dump-symbols}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-macro}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-measure-parse-tree}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-module-writer}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-pass-arguments}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-pass-structure}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-pre-fir-tree}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-types-section}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-unparse}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-unparse-no-sema}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-unparse-with-modules}}
// CC1OptionCHECK5: {{(unknown argument).*-fdebug-unparse-with-symbols}}
// CC1OptionCHECK5: {{(unknown argument).*-fdefault-double-8}}
// CC1OptionCHECK5: {{(unknown argument).*-fdefault-inline}}
// CC1OptionCHECK5: {{(unknown argument).*-fdefault-integer-8}}
// CC1OptionCHECK5: {{(unknown argument).*-fdefault-real-8}}
// CC1OptionCHECK5: {{(unknown argument).*-fdelete-null-pointer-checks}}
// CC1OptionCHECK5: {{(unknown argument).*-fdevirtualize}}
// CC1OptionCHECK5: {{(unknown argument).*-fdevirtualize-speculatively}}
// CC1OptionCHECK5: {{(unknown argument).*-fdiagnostics-color=}}
// CC1OptionCHECK5: {{(unknown argument).*-fdiagnostics-fixit-info}}
// CC1OptionCHECK5: {{(unknown argument).*-fdiagnostics-format=}}
// CC1OptionCHECK5: {{(unknown argument).*-fdiagnostics-show-category=}}
// CC1OptionCHECK5: {{(unknown argument).*-fdiagnostics-show-line-numbers}}
// CC1OptionCHECK5: {{(unknown argument).*-fdiagnostics-show-location=}}
// CC1OptionCHECK5: {{(unknown argument).*-fdiagnostics-show-option}}
// CC1OptionCHECK5: {{(unknown argument).*-fdisable-integer-16}}
// CC1OptionCHECK5: {{(unknown argument).*-fdisable-integer-2}}
// CC1OptionCHECK5: {{(unknown argument).*-fdisable-real-10}}
// CC1OptionCHECK5: {{(unknown argument).*-fdisable-real-3}}
// CC1OptionCHECK5: {{(unknown argument).*-fdiscard-value-names}}
// CC1OptionCHECK5: {{(unknown argument).*-fdollar-ok}}
// CC1OptionCHECK5: {{(unknown argument).*-fdouble-square-bracket-attributes}}
// CC1OptionCHECK5: {{(unknown argument).*-fdriver-only}}
// CC1OptionCHECK5: {{(unknown argument).*-fdump-fortran-optimized}}
// CC1OptionCHECK5: {{(unknown argument).*-fdump-fortran-original}}
// CC1OptionCHECK5: {{(unknown argument).*-fdump-parse-tree}}
// CC1OptionCHECK5: {{(unknown argument).*-fdwarf2-cfi-asm}}
// CC1OptionCHECK5: {{(unknown argument).*-fdwarf-directory-asm}}
// CC1OptionCHECK5: {{(unknown argument).*-fdwarf-exceptions}}
// CC1OptionCHECK5: {{(unknown argument).*-felide-constructors}}
// CC1OptionCHECK5: {{(unknown argument).*-feliminate-unused-debug-symbols}}
// CC1OptionCHECK5: {{(unknown argument).*-feliminate-unused-debug-types}}
// CC1OptionCHECK5: {{(unknown argument).*-fembed-bitcode}}
// CC1OptionCHECK5: {{(unknown argument).*-fembed-bitcode-marker}}
// CC1OptionCHECK5: {{(unknown argument).*-fencoding=}}
// CC1OptionCHECK5: {{(unknown argument).*-ferror-limit=}}
// CC1OptionCHECK5: {{(unknown argument).*-fescaping-block-tail-calls}}
// CC1OptionCHECK5: {{(unknown argument).*-fexcess-precision=}}
// CC1OptionCHECK5: {{(unknown argument).*-fexec-charset=}}
// CC1OptionCHECK5: {{(unknown argument).*-fexperimental-isel}}
// CC1OptionCHECK5: {{(unknown argument).*-fextdirs=}}
// CC1OptionCHECK5: {{(unknown argument).*-fexternal-blas}}
// CC1OptionCHECK5: {{(unknown argument).*-ff2c}}
// CC1OptionCHECK5: {{(unknown argument).*-ffile-compilation-dir=}}
// CC1OptionCHECK5: {{(unknown argument).*-ffile-prefix-map=}}
// CC1OptionCHECK5: {{(unknown argument).*-finline-limit}}
// CC1OptionCHECK5: {{(unknown argument).*-ffixed-a0}}
// CC1OptionCHECK5: {{(unknown argument).*-ffixed-a1}}
// CC1OptionCHECK5: {{(unknown argument).*-ffixed-a2}}
// CC1OptionCHECK5: {{(unknown argument).*-ffixed-a3}}
// RUN: not %clang -cc1 -ffixed-a4 -ffixed-a5 -ffixed-a6 -ffixed-d0 -ffixed-d1 -ffixed-d2 -ffixed-d3 -ffixed-d4 -ffixed-d5 -ffixed-d6 -ffixed-d7 -ffixed-form -ffixed-g1 -ffixed-g2 -ffixed-g3 -ffixed-g4 -ffixed-g5 -ffixed-g6 -ffixed-g7 -ffixed-i0 -ffixed-i1 -ffixed-i2 -ffixed-i3 -ffixed-i4 -ffixed-i5 -ffixed-l0 -ffixed-l1 -ffixed-l2 -ffixed-l3 -ffixed-l4 -ffixed-l5 -ffixed-l6 -ffixed-l7 -ffixed-line-length= -ffixed-line-length- -ffixed-o0 -ffixed-o1 -ffixed-o2 -ffixed-o3 -ffixed-o4 -ffixed-o5 -ffixed-r19 -ffixed-r9 -ffixed-x1 -ffixed-x10 -ffixed-x11 -ffixed-x12 -ffixed-x13 -ffixed-x14 -ffixed-x15 -ffixed-x16 -ffixed-x17 -ffixed-x18 -ffixed-x19 -ffixed-x2 -ffixed-x20 -ffixed-x21 -ffixed-x22 -ffixed-x23 -ffixed-x24 -ffixed-x25 -ffixed-x26 -ffixed-x27 -ffixed-x28 -ffixed-x29 -ffixed-x3 -ffixed-x30 -ffixed-x31 -ffixed-x4 -ffixed-x5 -ffixed-x6 -ffixed-x7 -ffixed-x8 -ffixed-x9 -ffloat-store -ffor-scope -ffp-model= -ffpe-trap= -ffree-form -ffree-line-length- -ffriend-injection -ffrontend-optimize -ffunction-attribute-list -fgcse -fgcse-after-reload -fgcse-las -fgcse-sm -fget-definition -fget-symbols-sources -fglobal-isel -fgnu -fgnu-inline-asm -fgnu-runtime -fgpu-flush-denormals-to-zero -fgpu-inline-threshold= -fgpu-sanitize -fheinous-gnu-extensions -fhermetic-module-files -fhip-dump-offload-linker-script -fhip-emit-relocatable  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK6 %s

// CC1OptionCHECK6: {{(unknown argument).*-ffixed-a4}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-a5}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-a6}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-d0}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-d1}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-d2}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-d3}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-d4}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-d5}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-d6}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-d7}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-form}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-g1}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-g2}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-g3}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-g4}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-g5}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-g6}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-g7}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-i0}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-i1}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-i2}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-i3}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-i4}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-i5}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-l0}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-l1}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-l2}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-l3}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-l4}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-l5}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-l6}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-l7}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-line-length=}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-line-length-}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-o0}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-o1}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-o2}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-o3}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-o4}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-o5}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-r19}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-r9}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x1}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x10}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x11}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x12}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x13}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x14}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x15}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x16}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x17}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x18}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x19}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x2}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x20}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x21}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x22}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x23}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x24}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x25}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x26}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x27}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x28}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x29}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x3}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x30}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x31}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x4}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x5}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x6}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x7}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x8}}
// CC1OptionCHECK6: {{(unknown argument).*-ffixed-x9}}
// CC1OptionCHECK6: {{(unknown argument).*-ffloat-store}}
// CC1OptionCHECK6: {{(unknown argument).*-ffor-scope}}
// CC1OptionCHECK6: {{(unknown argument).*-ffp-model=}}
// CC1OptionCHECK6: {{(unknown argument).*-ffpe-trap=}}
// CC1OptionCHECK6: {{(unknown argument).*-ffree-form}}
// CC1OptionCHECK6: {{(unknown argument).*-ffree-line-length-}}
// CC1OptionCHECK6: {{(unknown argument).*-ffriend-injection}}
// CC1OptionCHECK6: {{(unknown argument).*-ffrontend-optimize}}
// CC1OptionCHECK6: {{(unknown argument).*-ffunction-attribute-list}}
// CC1OptionCHECK6: {{(unknown argument).*-fgcse}}
// CC1OptionCHECK6: {{(unknown argument).*-fgcse-after-reload}}
// CC1OptionCHECK6: {{(unknown argument).*-fgcse-las}}
// CC1OptionCHECK6: {{(unknown argument).*-fgcse-sm}}
// CC1OptionCHECK6: {{(unknown argument).*-fget-definition}}
// CC1OptionCHECK6: {{(unknown argument).*-fget-symbols-sources}}
// CC1OptionCHECK6: {{(unknown argument).*-fglobal-isel}}
// CC1OptionCHECK6: {{(unknown argument).*-fgnu}}
// CC1OptionCHECK6: {{(unknown argument).*-fgnu-inline-asm}}
// CC1OptionCHECK6: {{(unknown argument).*-fgnu-runtime}}
// CC1OptionCHECK6: {{(unknown argument).*-fgpu-flush-denormals-to-zero}}
// CC1OptionCHECK6: {{(unknown argument).*-fgpu-inline-threshold=}}
// CC1OptionCHECK6: {{(unknown argument).*-fgpu-sanitize}}
// CC1OptionCHECK6: {{(unknown argument).*-fheinous-gnu-extensions}}
// CC1OptionCHECK6: {{(unknown argument).*-fhermetic-module-files}}
// CC1OptionCHECK6: {{(unknown argument).*-fhip-dump-offload-linker-script}}
// CC1OptionCHECK6: {{(unknown argument).*-fhip-emit-relocatable}}
// RUN: not %clang -cc1 -fhip-fp32-correctly-rounded-divide-sqrt -fhonor-infinities -fhonor-nans -fhosted -filelist -filetype -fimplement-inlines -fimplicit-modules -fimplicit-none -fimplicit-none-ext -fimplicit-templates -finit-character= -finit-global-zero -finit-integer= -finit-local-zero -finit-logical= -finit-real= -finline -finline-functions-called-once -finline-limit= -finline-small-functions -finput-charset= -finteger-4-integer-8 -fintegrated-cc1 -fintegrated-objemitter -fintrinsic-modules-path -fipa-cp -fivopts -fjump-tables -flang-deprecated-no-hlfir -flang-experimental-hlfir -flarge-sizes -flat_namespace -flax-vector-conversions -flimit-debug-info -flimited-precision= -flogical-abbreviations -fversion-loops-for-stride -fmax-array-constructor= -fmax-errors= -fmax-identifier-length -fmax-stack-var-size= -fmax-subrecord-length= -fmerge-constants -fmodule-file-deps -fmodule-header -fmodule-header= -fmodule-maps -fmodule-private -fmodules-validate-input-files-content -fmodulo-sched -fmodulo-sched-allow-regmoves -fms-omit-default-lib -fms-runtime-lib= -fms-tls-guards -fmsc-version= -fmudflap -fmudflapth -fmultilib-flag= -fnested-functions -fnext-runtime -fno-PIC -fno-PIE -fno-aarch64-jump-table-hardening -fno-addrsig -fno-aggressive-function-elimination -fno-align-commons -fno-align-functions -fno-align-jumps -fno-align-labels -fno-align-loops -fno-all-intrinsics -fno-allow-editor-placeholders -fno-altivec -fno-analyzed-objects-for-unparse -fno-android-pad-segment -fno-keep-inline-functions -fno-unit-at-a-time -fno-apple-pragma-pack -fno-application-extension -fno-asm -fno-asm-blocks -fno-associative-math -fno-assume-nothrow-exception-dtor -fno-async-exceptions -fno-asynchronous-unwind-tables -fno-auto-profile -fno-auto-profile-accurate -fno-automatic -fno-backslash -fno-backtrace -fno-basic-block-address-map -fno-blocks -fno-borland-extensions -fno-bounds-check -fno-branch-count-reg -fno-caller-saves -fno-check-array-temporaries -fno-color-diagnostics -fno-complete-member-pointers  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK7 %s

// CC1OptionCHECK7: {{(unknown argument).*-fhip-fp32-correctly-rounded-divide-sqrt}}
// CC1OptionCHECK7: {{(unknown argument).*-fhonor-infinities}}
// CC1OptionCHECK7: {{(unknown argument).*-fhonor-nans}}
// CC1OptionCHECK7: {{(unknown argument).*-fhosted}}
// CC1OptionCHECK7: {{(unknown argument).*-filelist}}
// CC1OptionCHECK7: {{(unknown argument).*-filetype}}
// CC1OptionCHECK7: {{(unknown argument).*-fimplement-inlines}}
// CC1OptionCHECK7: {{(unknown argument).*-fimplicit-modules}}
// CC1OptionCHECK7: {{(unknown argument).*-fimplicit-none}}
// CC1OptionCHECK7: {{(unknown argument).*-fimplicit-none-ext}}
// CC1OptionCHECK7: {{(unknown argument).*-fimplicit-templates}}
// CC1OptionCHECK7: {{(unknown argument).*-finit-character=}}
// CC1OptionCHECK7: {{(unknown argument).*-finit-global-zero}}
// CC1OptionCHECK7: {{(unknown argument).*-finit-integer=}}
// CC1OptionCHECK7: {{(unknown argument).*-finit-local-zero}}
// CC1OptionCHECK7: {{(unknown argument).*-finit-logical=}}
// CC1OptionCHECK7: {{(unknown argument).*-finit-real=}}
// CC1OptionCHECK7: {{(unknown argument).*-finline}}
// CC1OptionCHECK7: {{(unknown argument).*-finline-functions-called-once}}
// CC1OptionCHECK7: {{(unknown argument).*-finline-limit=}}
// CC1OptionCHECK7: {{(unknown argument).*-finline-small-functions}}
// CC1OptionCHECK7: {{(unknown argument).*-finput-charset=}}
// CC1OptionCHECK7: {{(unknown argument).*-finteger-4-integer-8}}
// CC1OptionCHECK7: {{(unknown argument).*-fintegrated-cc1}}
// CC1OptionCHECK7: {{(unknown argument).*-fintegrated-objemitter}}
// CC1OptionCHECK7: {{(unknown argument).*-fintrinsic-modules-path}}
// CC1OptionCHECK7: {{(unknown argument).*-fipa-cp}}
// CC1OptionCHECK7: {{(unknown argument).*-fivopts}}
// CC1OptionCHECK7: {{(unknown argument).*-fjump-tables}}
// CC1OptionCHECK7: {{(unknown argument).*-flang-deprecated-no-hlfir}}
// CC1OptionCHECK7: {{(unknown argument).*-flang-experimental-hlfir}}
// CC1OptionCHECK7: {{(unknown argument).*-flarge-sizes}}
// CC1OptionCHECK7: {{(unknown argument).*-flat_namespace}}
// CC1OptionCHECK7: {{(unknown argument).*-flax-vector-conversions}}
// CC1OptionCHECK7: {{(unknown argument).*-flimit-debug-info}}
// CC1OptionCHECK7: {{(unknown argument).*-flimited-precision=}}
// CC1OptionCHECK7: {{(unknown argument).*-flogical-abbreviations}}
// CC1OptionCHECK7: {{(unknown argument).*-fversion-loops-for-stride}}
// CC1OptionCHECK7: {{(unknown argument).*-fmax-array-constructor=}}
// CC1OptionCHECK7: {{(unknown argument).*-fmax-errors=}}
// CC1OptionCHECK7: {{(unknown argument).*-fmax-identifier-length}}
// CC1OptionCHECK7: {{(unknown argument).*-fmax-stack-var-size=}}
// CC1OptionCHECK7: {{(unknown argument).*-fmax-subrecord-length=}}
// CC1OptionCHECK7: {{(unknown argument).*-fmerge-constants}}
// CC1OptionCHECK7: {{(unknown argument).*-fmodule-file-deps}}
// CC1OptionCHECK7: {{(unknown argument).*-fmodule-header}}
// CC1OptionCHECK7: {{(unknown argument).*-fmodule-header=}}
// CC1OptionCHECK7: {{(unknown argument).*-fmodule-maps}}
// CC1OptionCHECK7: {{(unknown argument).*-fmodule-private}}
// CC1OptionCHECK7: {{(unknown argument).*-fmodules-validate-input-files-content}}
// CC1OptionCHECK7: {{(unknown argument).*-fmodulo-sched}}
// CC1OptionCHECK7: {{(unknown argument).*-fmodulo-sched-allow-regmoves}}
// CC1OptionCHECK7: {{(unknown argument).*-fms-omit-default-lib}}
// CC1OptionCHECK7: {{(unknown argument).*-fms-runtime-lib=}}
// CC1OptionCHECK7: {{(unknown argument).*-fms-tls-guards}}
// CC1OptionCHECK7: {{(unknown argument).*-fmsc-version=}}
// CC1OptionCHECK7: {{(unknown argument).*-fmudflap}}
// CC1OptionCHECK7: {{(unknown argument).*-fmudflapth}}
// CC1OptionCHECK7: {{(unknown argument).*-fmultilib-flag=}}
// CC1OptionCHECK7: {{(unknown argument).*-fnested-functions}}
// CC1OptionCHECK7: {{(unknown argument).*-fnext-runtime}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-PIC}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-PIE}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-aarch64-jump-table-hardening}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-addrsig}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-aggressive-function-elimination}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-align-commons}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-align-functions}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-align-jumps}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-align-labels}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-align-loops}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-all-intrinsics}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-allow-editor-placeholders}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-altivec}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-analyzed-objects-for-unparse}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-android-pad-segment}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-keep-inline-functions}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-unit-at-a-time}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-apple-pragma-pack}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-application-extension}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-asm}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-asm-blocks}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-associative-math}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-assume-nothrow-exception-dtor}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-async-exceptions}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-asynchronous-unwind-tables}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-auto-profile}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-auto-profile-accurate}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-automatic}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-backslash}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-backtrace}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-basic-block-address-map}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-blocks}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-borland-extensions}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-bounds-check}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-branch-count-reg}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-caller-saves}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-check-array-temporaries}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-color-diagnostics}}
// CC1OptionCHECK7: {{(unknown argument).*-fno-complete-member-pointers}}
// RUN: not %clang -cc1 -fno-coro-aligned-allocation -fno-coroutines -fno-coverage-mapping -fno-crash-diagnostics -fno-cray-pointer -fno-cuda-flush-denormals-to-zero -fno-cuda-short-ptr -fno-cxx-exceptions -fno-d-lines-as-code -fno-d-lines-as-comments -fno-data-sections -fno-debug-info-for-profiling -fno-debug-macro -fno-debug-ranges-base-address -fno-debug-types-section -fno-default-inline -fno-delayed-template-parsing -fno-devirtualize -fno-devirtualize-speculatively -fno-diagnostics-show-hotness -fno-directives-only -fno-discard-value-names -fno-dollar-ok -fno-double-square-bracket-attributes -fno-dump-fortran-optimized -fno-dump-fortran-original -fno-dump-parse-tree -fno-dwarf2-cfi-asm -fno-eliminate-unused-debug-symbols -fno-emit-compact-unwind-non-canonical -fno-emulated-tls -fno-exceptions -fno-experimental-isel -fno-experimental-library -fno-external-blas -fno-f2c -fno-finite-math-only -fno-inline-limit -fno-fixed-point -fno-float-store -fno-for-scope -fno-force-dwarf-frame -fno-force-emit-vtables -fno-force-enable-int128 -fno-friend-injection -fno-frontend-optimize -fno-function-attribute-list -fno-function-sections -fno-gcse -fno-gcse-after-reload -fno-gcse-las -fno-gcse-sm -fno-global-isel -fno-gnu -fno-gnu89-inline -fno-gpu-allow-device-init -fno-gpu-approx-transcendentals -fno-gpu-defer-diag -fno-gpu-exclude-wrong-side-overloads -fno-gpu-flush-denormals-to-zero -fno-gpu-rdc -fno-gpu-sanitize -fno-hip-emit-relocatable -fno-hip-kernel-arg-name -fno-hip-new-launch-api -fno-honor-infinities -fno-honor-nans -fno-implement-inlines -fno-implicit-module-maps -fno-implicit-none -fno-implicit-none-ext -fno-implicit-templates -fno-init-global-zero -fno-init-local-zero -fno-inline-functions-called-once -fno-inline-small-functions -fno-integer-4-integer-8 -fno-integrated-cc1 -fno-integrated-objemitter -fno-ipa-cp -fno-ivopts -fno-jmc -fno-keep-persistent-storage-variables -fno-keep-static-consts -fno-keep-system-includes -fno-lax-vector-conversions -fno-limit-debug-info -fno-logical-abbreviations -fno-version-loops-for-stride -fno-math-errno -fno-max-identifier-length -fno-max-type-align -fno-coverage-mcdc -fno-memory-profile -fno-merge-all-constants -fno-merge-constants -fno-minimize-whitespace -fno-module-file-deps -fno-module-maps -fno-module-private  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK8 %s

// CC1OptionCHECK8: {{(unknown argument).*-fno-coro-aligned-allocation}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-coroutines}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-coverage-mapping}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-crash-diagnostics}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-cray-pointer}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-cuda-flush-denormals-to-zero}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-cuda-short-ptr}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-cxx-exceptions}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-d-lines-as-code}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-d-lines-as-comments}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-data-sections}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-debug-info-for-profiling}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-debug-macro}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-debug-ranges-base-address}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-debug-types-section}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-default-inline}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-delayed-template-parsing}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-devirtualize}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-devirtualize-speculatively}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-diagnostics-show-hotness}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-directives-only}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-discard-value-names}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-dollar-ok}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-double-square-bracket-attributes}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-dump-fortran-optimized}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-dump-fortran-original}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-dump-parse-tree}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-dwarf2-cfi-asm}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-eliminate-unused-debug-symbols}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-emit-compact-unwind-non-canonical}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-emulated-tls}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-exceptions}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-experimental-isel}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-experimental-library}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-external-blas}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-f2c}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-finite-math-only}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-inline-limit}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-fixed-point}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-float-store}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-for-scope}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-force-dwarf-frame}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-force-emit-vtables}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-force-enable-int128}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-friend-injection}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-frontend-optimize}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-function-attribute-list}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-function-sections}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gcse}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gcse-after-reload}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gcse-las}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gcse-sm}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-global-isel}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gnu}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gnu89-inline}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gpu-allow-device-init}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gpu-approx-transcendentals}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gpu-defer-diag}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gpu-exclude-wrong-side-overloads}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gpu-flush-denormals-to-zero}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gpu-rdc}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-gpu-sanitize}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-hip-emit-relocatable}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-hip-kernel-arg-name}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-hip-new-launch-api}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-honor-infinities}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-honor-nans}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-implement-inlines}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-implicit-module-maps}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-implicit-none}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-implicit-none-ext}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-implicit-templates}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-init-global-zero}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-init-local-zero}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-inline-functions-called-once}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-inline-small-functions}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-integer-4-integer-8}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-integrated-cc1}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-integrated-objemitter}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-ipa-cp}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-ivopts}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-jmc}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-keep-persistent-storage-variables}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-keep-static-consts}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-keep-system-includes}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-lax-vector-conversions}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-limit-debug-info}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-logical-abbreviations}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-version-loops-for-stride}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-math-errno}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-max-identifier-length}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-max-type-align}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-coverage-mcdc}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-memory-profile}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-merge-all-constants}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-merge-constants}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-minimize-whitespace}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-module-file-deps}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-module-maps}}
// CC1OptionCHECK8: {{(unknown argument).*-fno-module-private}}
// RUN: not %clang -cc1 -fno-modules -fno-modules-decluse -fno-strict-modules-decluse -fno_modules-validate-input-files-content -fno-modules-validate-system-headers -fno-modulo-sched -fno-modulo-sched-allow-regmoves -fno-ms-compatibility -fno-ms-extensions -fno-ms-volatile -fno-non-call-exceptions -fno-objc-arc -fno-objc-arc-exceptions -fno-objc-encode-cxx-class-template-spec -fno-objc-exceptions -fno-objc-legacy-dispatch -fno-objc-nonfragile-abi -fno-offload-implicit-host-device-templates -fno-offload-lto -fno-offload-via-llvm -fno-omit-frame-pointer -fno-openmp-cuda-mode -fno-openmp-new-driver -fno-openmp-optimistic-collapse -fno-openmp-target-jit -fno-pack-derived -fno-pack-struct -fno-pascal-strings -fno-pch-codegen -fno-pch-debuginfo -fno_pch-validate-input-files-content -fno-peel-loops -fno-permissive -fno-pic -fno-pie -fno-pointer-tbaa -fno-ppc-native-vector-element-order -fno-prefetch-loop-arrays -fno-printf -fno-profile -fno-profile-arcs -fno-profile-correction -fno-profile-generate -fno-profile-generate-sampling -fno-profile-instr-generate -fno-profile-instr-use -fno-profile-reusedist -fno-profile-sample-accurate -fno-profile-sample-use -fno-profile-use -fno-profile-values -fno-protect-parens -fno-ptrauth-auth-traps -fno-ptrauth-calls -fno-ptrauth-elf-got -fno-ptrauth-function-pointer-type-discrimination -fno-ptrauth-indirect-gotos -fno-ptrauth-init-fini -fno-ptrauth-init-fini-address-discrimination -fno-ptrauth-intrinsics -fno-ptrauth-returns -fno-ptrauth-type-info-vtable-pointer-discrimination -fno-ptrauth-vtable-pointer-address-discrimination -fno-ptrauth-vtable-pointer-type-discrimination -fno-range-check -fno-real-4-real-10 -fno-real-4-real-16 -fno-real-4-real-8 -fno-real-8-real-10 -fno-real-8-real-16 -fno-real-8-real-4 -fno-realloc-lhs -fno-record-command-line -fno-recursive -fno-reformat -fno-register-global-dtors-with-atexit -fno-regs-graph -fno-rename-registers -fno-reorder-blocks -fno-repack-arrays -fno-retain-subst-template-type-parm-type-ast-nodes -fno-rewrite-imports -fno-rewrite-includes -fno-ripa -fno-rtlib-add-rpath -fno-rtlib-defaultlib -fno-safe-buffer-usage-suggestions -fno-save-main-program -fno-save-optimization-record -fno-schedule-insns -fno-schedule-insns2 -fno-second-underscore -fno-see -fno-semantic-interposition -fno-separate-named-sections -fno-short-enums -fno-short-wchar -fno-sign-zero -fno-signaling-math -fno-signaling-nans  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK9 %s

// CC1OptionCHECK9: {{(unknown argument).*-fno-modules}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-modules-decluse}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-strict-modules-decluse}}
// CC1OptionCHECK9: {{(unknown argument).*-fno_modules-validate-input-files-content}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-modules-validate-system-headers}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-modulo-sched}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-modulo-sched-allow-regmoves}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ms-compatibility}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ms-extensions}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ms-volatile}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-non-call-exceptions}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-objc-arc}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-objc-arc-exceptions}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-objc-encode-cxx-class-template-spec}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-objc-exceptions}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-objc-legacy-dispatch}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-objc-nonfragile-abi}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-offload-implicit-host-device-templates}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-offload-lto}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-offload-via-llvm}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-omit-frame-pointer}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-openmp-cuda-mode}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-openmp-new-driver}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-openmp-optimistic-collapse}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-openmp-target-jit}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-pack-derived}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-pack-struct}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-pascal-strings}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-pch-codegen}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-pch-debuginfo}}
// CC1OptionCHECK9: {{(unknown argument).*-fno_pch-validate-input-files-content}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-peel-loops}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-permissive}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-pic}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-pie}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-pointer-tbaa}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ppc-native-vector-element-order}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-prefetch-loop-arrays}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-printf}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-profile}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-profile-arcs}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-profile-correction}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-profile-generate}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-profile-generate-sampling}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-profile-instr-generate}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-profile-instr-use}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-profile-reusedist}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-profile-sample-accurate}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-profile-sample-use}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-profile-use}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-profile-values}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-protect-parens}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ptrauth-auth-traps}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ptrauth-calls}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ptrauth-elf-got}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ptrauth-function-pointer-type-discrimination}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ptrauth-indirect-gotos}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ptrauth-init-fini}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ptrauth-init-fini-address-discrimination}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ptrauth-intrinsics}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ptrauth-returns}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ptrauth-type-info-vtable-pointer-discrimination}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ptrauth-vtable-pointer-address-discrimination}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ptrauth-vtable-pointer-type-discrimination}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-range-check}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-real-4-real-10}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-real-4-real-16}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-real-4-real-8}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-real-8-real-10}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-real-8-real-16}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-real-8-real-4}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-realloc-lhs}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-record-command-line}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-recursive}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-reformat}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-register-global-dtors-with-atexit}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-regs-graph}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-rename-registers}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-reorder-blocks}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-repack-arrays}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-retain-subst-template-type-parm-type-ast-nodes}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-rewrite-imports}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-rewrite-includes}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-ripa}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-rtlib-add-rpath}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-rtlib-defaultlib}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-safe-buffer-usage-suggestions}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-save-main-program}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-save-optimization-record}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-schedule-insns}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-schedule-insns2}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-second-underscore}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-see}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-semantic-interposition}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-separate-named-sections}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-short-enums}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-short-wchar}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-sign-zero}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-signaling-math}}
// CC1OptionCHECK9: {{(unknown argument).*-fno-signaling-nans}}
// RUN: not %clang -cc1 -fno-single-precision-constant -fno-slp-vectorize -fno-spec-constr-count -fno-split-dwarf-inlining -fno-split-lto-unit -fno-split-machine-functions -fno-split-stack -fno-stack-arrays -fno-stack-check -fno-stack-clash-protection -fno-stack-protector -fno-stack-size-section -fno-standalone-debug -fno-strength-reduce -fno-strict-aliasing -fno-strict-enums -fno-strict-overflow -fno-strict-vtable-pointers -fno-struct-path-tbaa -fno-sycl -fno-test-coverage -fno-tls-model -fno-tracer -fno-trapping-math -fno-tree-dce -fno-tree-salias -fno-tree-ter -fno-tree-vectorizer-verbose -fno-tree-vrp -fno-underscoring -fno-unique-basic-block-section-names -fno-unique-internal-linkage-names -fno-unroll-all-loops -fno-unsafe-loop-optimizations -fno-unsafe-math-optimizations -fno-unsigned -fno-unsigned-char -fno-unswitch-loops -fno-unwind-tables -fno-use-line-directives -fno-use-linker-plugin -fno-var-tracking -fno-variable-expansion-in-unroller -fno-vect-cost-model -fno-vectorize -fno-verify-intermediate-code -fno-virtual-function-elimination -fno-visibility-from-dllstorageclass -fno-visibility-inlines-hidden -fno-web -fno-whole-file -fno-whole-program -fno-whole-program-vtables -fno-working-directory -fno-wrapv -fno-wrapv-pointer -fno-xl-pragma-pack -fno-xor-operator -fno-xray-always-emit-customevents -fno-xray-always-emit-typedevents -fno-xray-ignore-loops -fno-xray-instrument -fno-xray-link-deps -fno-xray-shared -fno-zvector -fnon-call-exceptions -fobjc-abi-version= -fobjc-atdefs -fobjc-call-cxx-cdtors -fobjc-convert-messages-to-runtime-calls -fobjc-infer-related-result-type -fobjc-legacy-dispatch -fobjc-link-runtime -fobjc-new-property -fobjc-nonfragile-abi -fobjc-nonfragile-abi-version= -fobjc-sender-dependent-dispatch -foffload-lto -foffload-lto= -fomit-frame-pointer -fopenmp-new-driver -fopenmp-target-jit -fopenmp-use-tls -foperator-names -foptimization-record-file= -foptimization-record-passes= -foptimize-sibling-calls -force_cpusubtype_ALL -force_flat_namespace -force_load -fforce-addr -foutput-class-dir= -fpack-derived -fpack-struct -fpch-preprocess -fpch-validate-input-files-content -fpeel-loops -fpermissive -fpic -fpie  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK10 %s

// CC1OptionCHECK10: {{(unknown argument).*-fno-single-precision-constant}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-slp-vectorize}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-spec-constr-count}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-split-dwarf-inlining}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-split-lto-unit}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-split-machine-functions}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-split-stack}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-stack-arrays}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-stack-check}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-stack-clash-protection}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-stack-protector}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-stack-size-section}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-standalone-debug}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-strength-reduce}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-strict-aliasing}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-strict-enums}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-strict-overflow}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-strict-vtable-pointers}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-struct-path-tbaa}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-sycl}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-test-coverage}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-tls-model}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-tracer}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-trapping-math}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-tree-dce}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-tree-salias}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-tree-ter}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-tree-vectorizer-verbose}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-tree-vrp}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-underscoring}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-unique-basic-block-section-names}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-unique-internal-linkage-names}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-unroll-all-loops}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-unsafe-loop-optimizations}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-unsafe-math-optimizations}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-unsigned}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-unsigned-char}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-unswitch-loops}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-unwind-tables}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-use-line-directives}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-use-linker-plugin}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-var-tracking}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-variable-expansion-in-unroller}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-vect-cost-model}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-vectorize}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-verify-intermediate-code}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-virtual-function-elimination}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-visibility-from-dllstorageclass}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-visibility-inlines-hidden}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-web}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-whole-file}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-whole-program}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-whole-program-vtables}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-working-directory}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-wrapv}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-wrapv-pointer}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-xl-pragma-pack}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-xor-operator}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-xray-always-emit-customevents}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-xray-always-emit-typedevents}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-xray-ignore-loops}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-xray-instrument}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-xray-link-deps}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-xray-shared}}
// CC1OptionCHECK10: {{(unknown argument).*-fno-zvector}}
// CC1OptionCHECK10: {{(unknown argument).*-fnon-call-exceptions}}
// CC1OptionCHECK10: {{(unknown argument).*-fobjc-abi-version=}}
// CC1OptionCHECK10: {{(unknown argument).*-fobjc-atdefs}}
// CC1OptionCHECK10: {{(unknown argument).*-fobjc-call-cxx-cdtors}}
// CC1OptionCHECK10: {{(unknown argument).*-fobjc-convert-messages-to-runtime-calls}}
// CC1OptionCHECK10: {{(unknown argument).*-fobjc-infer-related-result-type}}
// CC1OptionCHECK10: {{(unknown argument).*-fobjc-legacy-dispatch}}
// CC1OptionCHECK10: {{(unknown argument).*-fobjc-link-runtime}}
// CC1OptionCHECK10: {{(unknown argument).*-fobjc-new-property}}
// CC1OptionCHECK10: {{(unknown argument).*-fobjc-nonfragile-abi}}
// CC1OptionCHECK10: {{(unknown argument).*-fobjc-nonfragile-abi-version=}}
// CC1OptionCHECK10: {{(unknown argument).*-fobjc-sender-dependent-dispatch}}
// CC1OptionCHECK10: {{(unknown argument).*-foffload-lto}}
// CC1OptionCHECK10: {{(unknown argument).*-foffload-lto=}}
// CC1OptionCHECK10: {{(unknown argument).*-fomit-frame-pointer}}
// CC1OptionCHECK10: {{(unknown argument).*-fopenmp-new-driver}}
// CC1OptionCHECK10: {{(unknown argument).*-fopenmp-target-jit}}
// CC1OptionCHECK10: {{(unknown argument).*-fopenmp-use-tls}}
// CC1OptionCHECK10: {{(unknown argument).*-foperator-names}}
// CC1OptionCHECK10: {{(unknown argument).*-foptimization-record-file=}}
// CC1OptionCHECK10: {{(unknown argument).*-foptimization-record-passes=}}
// CC1OptionCHECK10: {{(unknown argument).*-foptimize-sibling-calls}}
// CC1OptionCHECK10: {{(unknown argument).*-force_cpusubtype_ALL}}
// CC1OptionCHECK10: {{(unknown argument).*-force_flat_namespace}}
// CC1OptionCHECK10: {{(unknown argument).*-force_load}}
// CC1OptionCHECK10: {{(unknown argument).*-fforce-addr}}
// CC1OptionCHECK10: {{(unknown argument).*-foutput-class-dir=}}
// CC1OptionCHECK10: {{(unknown argument).*-fpack-derived}}
// CC1OptionCHECK10: {{(unknown argument).*-fpack-struct}}
// CC1OptionCHECK10: {{(unknown argument).*-fpch-preprocess}}
// CC1OptionCHECK10: {{(unknown argument).*-fpch-validate-input-files-content}}
// CC1OptionCHECK10: {{(unknown argument).*-fpeel-loops}}
// CC1OptionCHECK10: {{(unknown argument).*-fpermissive}}
// CC1OptionCHECK10: {{(unknown argument).*-fpic}}
// CC1OptionCHECK10: {{(unknown argument).*-fpie}}
// RUN: not %clang -cc1 -fplt -fplugin= -fplugin-arg- -fpointer-tbaa -fppc-native-vector-element-order -fprefetch-loop-arrays -fpreprocess-include-lines -fpreserve-as-comments -fprintf -fproc-stat-report -fproc-stat-report= -fprofile -fprofile-arcs -fprofile-correction -fprofile-dir= -fprofile-generate -fprofile-generate= -fprofile-generate-cold-function-coverage -fprofile-generate-cold-function-coverage= -fprofile-generate-sampling -fprofile-instr-generate -fprofile-instr-generate= -fprofile-instr-use -fprofile-instr-use= -fprofile-reusedist -fprofile-use -fprofile-use= -fprofile-values -framework -frandom-seed= -frange-check -freal-4-real-10 -freal-4-real-16 -freal-4-real-8 -freal-8-real-10 -freal-8-real-16 -freal-8-real-4 -frealloc-lhs -frecord-command-line -frecord-marker= -frecursive -fregs-graph -frename-registers -freorder-blocks -frepack-arrays -fripa -frtlib-add-rpath -frtlib-defaultlib -frtti -frtti-data -fsave-main-program -fsave-optimization-record -fsave-optimization-record= -fschedule-insns -fschedule-insns2 -fsecond-underscore -fsee -fseh-exceptions -fshort-wchar -fshow-column -fshow-source-location -fsign-zero -fsignaling-math -fsignaling-nans -fsigned-bitfields -fsigned-char -fsingle-precision-constant -fsjlj-exceptions -fslp-vectorize -fspec-constr-count -fspell-checking -fspv-target-env= -fstack-arrays -fstack-check -fstack-protector -fstack-protector-all -fstack-protector-strong -fstack-usage -fstandalone-debug -fstrength-reduce -fstrict-aliasing -fstrict-float-cast-overflow -fstrict-overflow -fstrict-return -fstruct-path-tbaa -fsycl -fsycl-device-only -fsycl-host-only -ftabstop= -ftemporal-profile -ftest-coverage -fthreadsafe-statics -ftime-trace -ftls-model -ftracer -ftrapping-math -ftrapv-handler= -ftree-dce -ftree-salias -ftree-ter  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK11 %s

// CC1OptionCHECK11: {{(unknown argument).*-fplt}}
// CC1OptionCHECK11: {{(unknown argument).*-fplugin=}}
// CC1OptionCHECK11: {{(unknown argument).*-fplugin-arg-}}
// CC1OptionCHECK11: {{(unknown argument).*-fpointer-tbaa}}
// CC1OptionCHECK11: {{(unknown argument).*-fppc-native-vector-element-order}}
// CC1OptionCHECK11: {{(unknown argument).*-fprefetch-loop-arrays}}
// CC1OptionCHECK11: {{(unknown argument).*-fpreprocess-include-lines}}
// CC1OptionCHECK11: {{(unknown argument).*-fpreserve-as-comments}}
// CC1OptionCHECK11: {{(unknown argument).*-fprintf}}
// CC1OptionCHECK11: {{(unknown argument).*-fproc-stat-report}}
// CC1OptionCHECK11: {{(unknown argument).*-fproc-stat-report=}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-arcs}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-correction}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-dir=}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-generate}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-generate=}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-generate-cold-function-coverage}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-generate-cold-function-coverage=}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-generate-sampling}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-instr-generate}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-instr-generate=}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-instr-use}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-instr-use=}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-reusedist}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-use}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-use=}}
// CC1OptionCHECK11: {{(unknown argument).*-fprofile-values}}
// CC1OptionCHECK11: {{(unknown argument).*-framework}}
// CC1OptionCHECK11: {{(unknown argument).*-frandom-seed=}}
// CC1OptionCHECK11: {{(unknown argument).*-frange-check}}
// CC1OptionCHECK11: {{(unknown argument).*-freal-4-real-10}}
// CC1OptionCHECK11: {{(unknown argument).*-freal-4-real-16}}
// CC1OptionCHECK11: {{(unknown argument).*-freal-4-real-8}}
// CC1OptionCHECK11: {{(unknown argument).*-freal-8-real-10}}
// CC1OptionCHECK11: {{(unknown argument).*-freal-8-real-16}}
// CC1OptionCHECK11: {{(unknown argument).*-freal-8-real-4}}
// CC1OptionCHECK11: {{(unknown argument).*-frealloc-lhs}}
// CC1OptionCHECK11: {{(unknown argument).*-frecord-command-line}}
// CC1OptionCHECK11: {{(unknown argument).*-frecord-marker=}}
// CC1OptionCHECK11: {{(unknown argument).*-frecursive}}
// CC1OptionCHECK11: {{(unknown argument).*-fregs-graph}}
// CC1OptionCHECK11: {{(unknown argument).*-frename-registers}}
// CC1OptionCHECK11: {{(unknown argument).*-freorder-blocks}}
// CC1OptionCHECK11: {{(unknown argument).*-frepack-arrays}}
// CC1OptionCHECK11: {{(unknown argument).*-fripa}}
// CC1OptionCHECK11: {{(unknown argument).*-frtlib-add-rpath}}
// CC1OptionCHECK11: {{(unknown argument).*-frtlib-defaultlib}}
// CC1OptionCHECK11: {{(unknown argument).*-frtti}}
// CC1OptionCHECK11: {{(unknown argument).*-frtti-data}}
// CC1OptionCHECK11: {{(unknown argument).*-fsave-main-program}}
// CC1OptionCHECK11: {{(unknown argument).*-fsave-optimization-record}}
// CC1OptionCHECK11: {{(unknown argument).*-fsave-optimization-record=}}
// CC1OptionCHECK11: {{(unknown argument).*-fschedule-insns}}
// CC1OptionCHECK11: {{(unknown argument).*-fschedule-insns2}}
// CC1OptionCHECK11: {{(unknown argument).*-fsecond-underscore}}
// CC1OptionCHECK11: {{(unknown argument).*-fsee}}
// CC1OptionCHECK11: {{(unknown argument).*-fseh-exceptions}}
// CC1OptionCHECK11: {{(unknown argument).*-fshort-wchar}}
// CC1OptionCHECK11: {{(unknown argument).*-fshow-column}}
// CC1OptionCHECK11: {{(unknown argument).*-fshow-source-location}}
// CC1OptionCHECK11: {{(unknown argument).*-fsign-zero}}
// CC1OptionCHECK11: {{(unknown argument).*-fsignaling-math}}
// CC1OptionCHECK11: {{(unknown argument).*-fsignaling-nans}}
// CC1OptionCHECK11: {{(unknown argument).*-fsigned-bitfields}}
// CC1OptionCHECK11: {{(unknown argument).*-fsigned-char}}
// CC1OptionCHECK11: {{(unknown argument).*-fsingle-precision-constant}}
// CC1OptionCHECK11: {{(unknown argument).*-fsjlj-exceptions}}
// CC1OptionCHECK11: {{(unknown argument).*-fslp-vectorize}}
// CC1OptionCHECK11: {{(unknown argument).*-fspec-constr-count}}
// CC1OptionCHECK11: {{(unknown argument).*-fspell-checking}}
// CC1OptionCHECK11: {{(unknown argument).*-fspv-target-env=}}
// CC1OptionCHECK11: {{(unknown argument).*-fstack-arrays}}
// CC1OptionCHECK11: {{(unknown argument).*-fstack-check}}
// CC1OptionCHECK11: {{(unknown argument).*-fstack-protector}}
// CC1OptionCHECK11: {{(unknown argument).*-fstack-protector-all}}
// CC1OptionCHECK11: {{(unknown argument).*-fstack-protector-strong}}
// CC1OptionCHECK11: {{(unknown argument).*-fstack-usage}}
// CC1OptionCHECK11: {{(unknown argument).*-fstandalone-debug}}
// CC1OptionCHECK11: {{(unknown argument).*-fstrength-reduce}}
// CC1OptionCHECK11: {{(unknown argument).*-fstrict-aliasing}}
// CC1OptionCHECK11: {{(unknown argument).*-fstrict-float-cast-overflow}}
// CC1OptionCHECK11: {{(unknown argument).*-fstrict-overflow}}
// CC1OptionCHECK11: {{(unknown argument).*-fstrict-return}}
// CC1OptionCHECK11: {{(unknown argument).*-fstruct-path-tbaa}}
// CC1OptionCHECK11: {{(unknown argument).*-fsycl}}
// CC1OptionCHECK11: {{(unknown argument).*-fsycl-device-only}}
// CC1OptionCHECK11: {{(unknown argument).*-fsycl-host-only}}
// CC1OptionCHECK11: {{(unknown argument).*-ftabstop=}}
// CC1OptionCHECK11: {{(unknown argument).*-ftemporal-profile}}
// CC1OptionCHECK11: {{(unknown argument).*-ftest-coverage}}
// CC1OptionCHECK11: {{(unknown argument).*-fthreadsafe-statics}}
// CC1OptionCHECK11: {{(unknown argument).*-ftime-trace}}
// CC1OptionCHECK11: {{(unknown argument).*-ftls-model}}
// CC1OptionCHECK11: {{(unknown argument).*-ftracer}}
// CC1OptionCHECK11: {{(unknown argument).*-ftrapping-math}}
// CC1OptionCHECK11: {{(unknown argument).*-ftrapv-handler=}}
// CC1OptionCHECK11: {{(unknown argument).*-ftree-dce}}
// CC1OptionCHECK11: {{(unknown argument).*-ftree-salias}}
// CC1OptionCHECK11: {{(unknown argument).*-ftree-ter}}
// RUN: not %clang -cc1 -ftree-vectorizer-verbose -ftree-vrp -funderscoring -funique-section-names -funroll-all-loops -funsafe-loop-optimizations -funsigned -funsigned-bitfields -funsigned-char -funswitch-loops -funwind-tables -fuse-cuid= -fuse-cxa-atexit -fuse-init-array -fuse-ld= -fuse-linker-plugin -fuse-lipo= -fvariable-expansion-in-unroller -fvect-cost-model -fvectorize -fverbose-asm -fverify-intermediate-code -fvisibility-global-new-delete-hidden -fvisibility-ms-compat -fwasm-exceptions -fweb -fwhole-file -fwhole-program -fxor-operator -fxray-function-index -fxray-link-deps -fzero-initialized-in-bss -g0 -g1 -g2 -g3 -g --gcc-install-dir= --gcc-toolchain= --gcc-triple= -gcoff -gcolumn-info -gdbx -gdwarf -gdwarf-2 -gdwarf-3 -gdwarf-4 -gdwarf-5 -gdwarf-aranges -gen-cdb-fragment-path -gen-reproducer -gen-reproducer= -gfull -ggdb -ggdb0 -ggdb1 -ggdb2 -ggdb3 -ginline-line-tables -gline-directives-only -gline-tables-only -glldb -gmlt -gmodules -gno-codeview-ghash -gno-embed-source -gno-gnu-pubnames -gno-modules -gno-omit-unreferenced-methods -gno-pubnames -gno-record-command-line -gno-simple-template-names -gno-split-dwarf -gno-strict-dwarf -gno-template-alias --gpu-bundle-output --gpu-instrument-lib= --gpu-use-aux-triple-only -grecord-command-line -gsce -gsimple-template-names -gsplit-dwarf -gsplit-dwarf= -gstabs -gtoggle -gused -gvms -gxcoff -gz -gz= -headerpad_max_install_names --hip-device-lib= --hip-link --hip-path= --hip-version= --hipspv-pass-plugin= --hipstdpar-path= --hipstdpar-prim-path= --hipstdpar-thrust-path= -ibuiltininc  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK12 %s

// CC1OptionCHECK12: {{(unknown argument).*-ftree-vectorizer-verbose}}
// CC1OptionCHECK12: {{(unknown argument).*-ftree-vrp}}
// CC1OptionCHECK12: {{(unknown argument).*-funderscoring}}
// CC1OptionCHECK12: {{(unknown argument).*-funique-section-names}}
// CC1OptionCHECK12: {{(unknown argument).*-funroll-all-loops}}
// CC1OptionCHECK12: {{(unknown argument).*-funsafe-loop-optimizations}}
// CC1OptionCHECK12: {{(unknown argument).*-funsigned}}
// CC1OptionCHECK12: {{(unknown argument).*-funsigned-bitfields}}
// CC1OptionCHECK12: {{(unknown argument).*-funsigned-char}}
// CC1OptionCHECK12: {{(unknown argument).*-funswitch-loops}}
// CC1OptionCHECK12: {{(unknown argument).*-funwind-tables}}
// CC1OptionCHECK12: {{(unknown argument).*-fuse-cuid=}}
// CC1OptionCHECK12: {{(unknown argument).*-fuse-cxa-atexit}}
// CC1OptionCHECK12: {{(unknown argument).*-fuse-init-array}}
// CC1OptionCHECK12: {{(unknown argument).*-fuse-ld=}}
// CC1OptionCHECK12: {{(unknown argument).*-fuse-linker-plugin}}
// CC1OptionCHECK12: {{(unknown argument).*-fuse-lipo=}}
// CC1OptionCHECK12: {{(unknown argument).*-fvariable-expansion-in-unroller}}
// CC1OptionCHECK12: {{(unknown argument).*-fvect-cost-model}}
// CC1OptionCHECK12: {{(unknown argument).*-fvectorize}}
// CC1OptionCHECK12: {{(unknown argument).*-fverbose-asm}}
// CC1OptionCHECK12: {{(unknown argument).*-fverify-intermediate-code}}
// CC1OptionCHECK12: {{(unknown argument).*-fvisibility-global-new-delete-hidden}}
// CC1OptionCHECK12: {{(unknown argument).*-fvisibility-ms-compat}}
// CC1OptionCHECK12: {{(unknown argument).*-fwasm-exceptions}}
// CC1OptionCHECK12: {{(unknown argument).*-fweb}}
// CC1OptionCHECK12: {{(unknown argument).*-fwhole-file}}
// CC1OptionCHECK12: {{(unknown argument).*-fwhole-program}}
// CC1OptionCHECK12: {{(unknown argument).*-fxor-operator}}
// CC1OptionCHECK12: {{(unknown argument).*-fxray-function-index}}
// CC1OptionCHECK12: {{(unknown argument).*-fxray-link-deps}}
// CC1OptionCHECK12: {{(unknown argument).*-fzero-initialized-in-bss}}
// CC1OptionCHECK12: {{(unknown argument).*-g0}}
// CC1OptionCHECK12: {{(unknown argument).*-g1}}
// CC1OptionCHECK12: {{(unknown argument).*-g2}}
// CC1OptionCHECK12: {{(unknown argument).*-g3}}
// CC1OptionCHECK12: {{(unknown argument).*-g}}
// CC1OptionCHECK12: {{(unknown argument).*--gcc-install-dir=}}
// CC1OptionCHECK12: {{(unknown argument).*--gcc-toolchain=}}
// CC1OptionCHECK12: {{(unknown argument).*--gcc-triple=}}
// CC1OptionCHECK12: {{(unknown argument).*-gcoff}}
// CC1OptionCHECK12: {{(unknown argument).*-gcolumn-info}}
// CC1OptionCHECK12: {{(unknown argument).*-gdbx}}
// CC1OptionCHECK12: {{(unknown argument).*-gdwarf}}
// CC1OptionCHECK12: {{(unknown argument).*-gdwarf-2}}
// CC1OptionCHECK12: {{(unknown argument).*-gdwarf-3}}
// CC1OptionCHECK12: {{(unknown argument).*-gdwarf-4}}
// CC1OptionCHECK12: {{(unknown argument).*-gdwarf-5}}
// CC1OptionCHECK12: {{(unknown argument).*-gdwarf-aranges}}
// CC1OptionCHECK12: {{(unknown argument).*-gen-cdb-fragment-path}}
// CC1OptionCHECK12: {{(unknown argument).*-gen-reproducer}}
// CC1OptionCHECK12: {{(unknown argument).*-gen-reproducer=}}
// CC1OptionCHECK12: {{(unknown argument).*-gfull}}
// CC1OptionCHECK12: {{(unknown argument).*-ggdb}}
// CC1OptionCHECK12: {{(unknown argument).*-ggdb0}}
// CC1OptionCHECK12: {{(unknown argument).*-ggdb1}}
// CC1OptionCHECK12: {{(unknown argument).*-ggdb2}}
// CC1OptionCHECK12: {{(unknown argument).*-ggdb3}}
// CC1OptionCHECK12: {{(unknown argument).*-ginline-line-tables}}
// CC1OptionCHECK12: {{(unknown argument).*-gline-directives-only}}
// CC1OptionCHECK12: {{(unknown argument).*-gline-tables-only}}
// CC1OptionCHECK12: {{(unknown argument).*-glldb}}
// CC1OptionCHECK12: {{(unknown argument).*-gmlt}}
// CC1OptionCHECK12: {{(unknown argument).*-gmodules}}
// CC1OptionCHECK12: {{(unknown argument).*-gno-codeview-ghash}}
// CC1OptionCHECK12: {{(unknown argument).*-gno-embed-source}}
// CC1OptionCHECK12: {{(unknown argument).*-gno-gnu-pubnames}}
// CC1OptionCHECK12: {{(unknown argument).*-gno-modules}}
// CC1OptionCHECK12: {{(unknown argument).*-gno-omit-unreferenced-methods}}
// CC1OptionCHECK12: {{(unknown argument).*-gno-pubnames}}
// CC1OptionCHECK12: {{(unknown argument).*-gno-record-command-line}}
// CC1OptionCHECK12: {{(unknown argument).*-gno-simple-template-names}}
// CC1OptionCHECK12: {{(unknown argument).*-gno-split-dwarf}}
// CC1OptionCHECK12: {{(unknown argument).*-gno-strict-dwarf}}
// CC1OptionCHECK12: {{(unknown argument).*-gno-template-alias}}
// CC1OptionCHECK12: {{(unknown argument).*--gpu-bundle-output}}
// CC1OptionCHECK12: {{(unknown argument).*--gpu-instrument-lib=}}
// CC1OptionCHECK12: {{(unknown argument).*--gpu-use-aux-triple-only}}
// CC1OptionCHECK12: {{(unknown argument).*-grecord-command-line}}
// CC1OptionCHECK12: {{(unknown argument).*-gsce}}
// CC1OptionCHECK12: {{(unknown argument).*-gsimple-template-names}}
// CC1OptionCHECK12: {{(unknown argument).*-gsplit-dwarf}}
// CC1OptionCHECK12: {{(unknown argument).*-gsplit-dwarf=}}
// CC1OptionCHECK12: {{(unknown argument).*-gstabs}}
// CC1OptionCHECK12: {{(unknown argument).*-gtoggle}}
// CC1OptionCHECK12: {{(unknown argument).*-gused}}
// CC1OptionCHECK12: {{(unknown argument).*-gvms}}
// CC1OptionCHECK12: {{(unknown argument).*-gxcoff}}
// CC1OptionCHECK12: {{(unknown argument).*-gz}}
// CC1OptionCHECK12: {{(unknown argument).*-gz=}}
// CC1OptionCHECK12: {{(unknown argument).*-headerpad_max_install_names}}
// CC1OptionCHECK12: {{(unknown argument).*--hip-device-lib=}}
// CC1OptionCHECK12: {{(unknown argument).*--hip-link}}
// CC1OptionCHECK12: {{(unknown argument).*--hip-path=}}
// CC1OptionCHECK12: {{(unknown argument).*--hip-version=}}
// CC1OptionCHECK12: {{(unknown argument).*--hipspv-pass-plugin=}}
// CC1OptionCHECK12: {{(unknown argument).*--hipstdpar-path=}}
// CC1OptionCHECK12: {{(unknown argument).*--hipstdpar-prim-path=}}
// CC1OptionCHECK12: {{(unknown argument).*--hipstdpar-thrust-path=}}
// CC1OptionCHECK12: {{(unknown argument).*-ibuiltininc}}
// RUN: not %clang -cc1 -image_base -imultilib -init -install_name -keep_private_externs -l -lazy_framework -lazy_library --ld-path= --libomptarget-amdgcn-bc-path= --libomptarget-amdgpu-bc-path= --libomptarget-nvptx-bc-path= --libomptarget-spirv-bc-path= -m16 -m32 -m3dnow -m3dnowa -m64 -m68000 -m68010 -m68020 -m68030 -m68040 -m68060 -m68881 -m80387 -mseses -mabicalls -mabs= -madx -maes -maix32 -maix64 -maix-shared-lib-tls-model-opt -maix-small-local-dynamic-tls -maix-small-local-exec-tls -malign-branch= -malign-branch-boundary= -malign-functions= -malign-jumps= -malign-loops= -maltivec -mamdgpu-ieee -mamdgpu-precise-memory-op -mamx-avx512 -mamx-bf16 -mamx-complex -mamx-fp16 -mamx-fp8 -mamx-int8 -mamx-movrs -mamx-tf32 -mamx-tile -mamx-transpose -mannotate-tablejump -mappletvos-version-min= -mappletvsimulator-version-min= -mapx-features= -mapx-inline-asm-use-gpr32 -mapxf -march= -marm -marm64x -masm= -matomics -mavx -mavx10.1 -mavx10.1-256 -mavx10.1-512 -mavx10.2 -mavx10.2-256 -mavx10.2-512 -mavx2 -mavx512bf16 -mavx512bitalg -mavx512bw -mavx512cd -mavx512dq -mavx512f -mavx512fp16 -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vl -mavx512vnni -mavx512vp2intersect -mavx512vpopcntdq -mavxifma -mavxneconvert -mavxvnni -mavxvnniint16 -mavxvnniint8 -mbig-endian -mbmi -mbmi2 -mbranch-likely -mbranch-protection= -mbranches-within-32B-boundaries -mbulk-memory -mbulk-memory-opt  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK13 %s

// CC1OptionCHECK13: {{(unknown argument).*-image_base}}
// CC1OptionCHECK13: {{(unknown argument).*-imultilib}}
// CC1OptionCHECK13: {{(unknown argument).*-init}}
// CC1OptionCHECK13: {{(unknown argument).*-install_name}}
// CC1OptionCHECK13: {{(unknown argument).*-keep_private_externs}}
// CC1OptionCHECK13: {{(unknown argument).*-l}}
// CC1OptionCHECK13: {{(unknown argument).*-lazy_framework}}
// CC1OptionCHECK13: {{(unknown argument).*-lazy_library}}
// CC1OptionCHECK13: {{(unknown argument).*--ld-path=}}
// CC1OptionCHECK13: {{(unknown argument).*--libomptarget-amdgcn-bc-path=}}
// CC1OptionCHECK13: {{(unknown argument).*--libomptarget-amdgpu-bc-path=}}
// CC1OptionCHECK13: {{(unknown argument).*--libomptarget-nvptx-bc-path=}}
// CC1OptionCHECK13: {{(unknown argument).*--libomptarget-spirv-bc-path=}}
// CC1OptionCHECK13: {{(unknown argument).*-m16}}
// CC1OptionCHECK13: {{(unknown argument).*-m32}}
// CC1OptionCHECK13: {{(unknown argument).*-m3dnow}}
// CC1OptionCHECK13: {{(unknown argument).*-m3dnowa}}
// CC1OptionCHECK13: {{(unknown argument).*-m64}}
// CC1OptionCHECK13: {{(unknown argument).*-m68000}}
// CC1OptionCHECK13: {{(unknown argument).*-m68010}}
// CC1OptionCHECK13: {{(unknown argument).*-m68020}}
// CC1OptionCHECK13: {{(unknown argument).*-m68030}}
// CC1OptionCHECK13: {{(unknown argument).*-m68040}}
// CC1OptionCHECK13: {{(unknown argument).*-m68060}}
// CC1OptionCHECK13: {{(unknown argument).*-m68881}}
// CC1OptionCHECK13: {{(unknown argument).*-m80387}}
// CC1OptionCHECK13: {{(unknown argument).*-mseses}}
// CC1OptionCHECK13: {{(unknown argument).*-mabicalls}}
// CC1OptionCHECK13: {{(unknown argument).*-mabs=}}
// CC1OptionCHECK13: {{(unknown argument).*-madx}}
// CC1OptionCHECK13: {{(unknown argument).*-maes}}
// CC1OptionCHECK13: {{(unknown argument).*-maix32}}
// CC1OptionCHECK13: {{(unknown argument).*-maix64}}
// CC1OptionCHECK13: {{(unknown argument).*-maix-shared-lib-tls-model-opt}}
// CC1OptionCHECK13: {{(unknown argument).*-maix-small-local-dynamic-tls}}
// CC1OptionCHECK13: {{(unknown argument).*-maix-small-local-exec-tls}}
// CC1OptionCHECK13: {{(unknown argument).*-malign-branch=}}
// CC1OptionCHECK13: {{(unknown argument).*-malign-branch-boundary=}}
// CC1OptionCHECK13: {{(unknown argument).*-malign-functions=}}
// CC1OptionCHECK13: {{(unknown argument).*-malign-jumps=}}
// CC1OptionCHECK13: {{(unknown argument).*-malign-loops=}}
// CC1OptionCHECK13: {{(unknown argument).*-maltivec}}
// CC1OptionCHECK13: {{(unknown argument).*-mamdgpu-ieee}}
// CC1OptionCHECK13: {{(unknown argument).*-mamdgpu-precise-memory-op}}
// CC1OptionCHECK13: {{(unknown argument).*-mamx-avx512}}
// CC1OptionCHECK13: {{(unknown argument).*-mamx-bf16}}
// CC1OptionCHECK13: {{(unknown argument).*-mamx-complex}}
// CC1OptionCHECK13: {{(unknown argument).*-mamx-fp16}}
// CC1OptionCHECK13: {{(unknown argument).*-mamx-fp8}}
// CC1OptionCHECK13: {{(unknown argument).*-mamx-int8}}
// CC1OptionCHECK13: {{(unknown argument).*-mamx-movrs}}
// CC1OptionCHECK13: {{(unknown argument).*-mamx-tf32}}
// CC1OptionCHECK13: {{(unknown argument).*-mamx-tile}}
// CC1OptionCHECK13: {{(unknown argument).*-mamx-transpose}}
// CC1OptionCHECK13: {{(unknown argument).*-mannotate-tablejump}}
// CC1OptionCHECK13: {{(unknown argument).*-mappletvos-version-min=}}
// CC1OptionCHECK13: {{(unknown argument).*-mappletvsimulator-version-min=}}
// CC1OptionCHECK13: {{(unknown argument).*-mapx-features=}}
// CC1OptionCHECK13: {{(unknown argument).*-mapx-inline-asm-use-gpr32}}
// CC1OptionCHECK13: {{(unknown argument).*-mapxf}}
// CC1OptionCHECK13: {{(unknown argument).*-march=}}
// CC1OptionCHECK13: {{(unknown argument).*-marm}}
// CC1OptionCHECK13: {{(unknown argument).*-marm64x}}
// CC1OptionCHECK13: {{(unknown argument).*-masm=}}
// CC1OptionCHECK13: {{(unknown argument).*-matomics}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx10.1}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx10.1-256}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx10.1-512}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx10.2}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx10.2-256}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx10.2-512}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx2}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512bf16}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512bitalg}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512bw}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512cd}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512dq}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512f}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512fp16}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512ifma}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512vbmi}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512vbmi2}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512vl}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512vnni}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512vp2intersect}}
// CC1OptionCHECK13: {{(unknown argument).*-mavx512vpopcntdq}}
// CC1OptionCHECK13: {{(unknown argument).*-mavxifma}}
// CC1OptionCHECK13: {{(unknown argument).*-mavxneconvert}}
// CC1OptionCHECK13: {{(unknown argument).*-mavxvnni}}
// CC1OptionCHECK13: {{(unknown argument).*-mavxvnniint16}}
// CC1OptionCHECK13: {{(unknown argument).*-mavxvnniint8}}
// CC1OptionCHECK13: {{(unknown argument).*-mbig-endian}}
// CC1OptionCHECK13: {{(unknown argument).*-mbmi}}
// CC1OptionCHECK13: {{(unknown argument).*-mbmi2}}
// CC1OptionCHECK13: {{(unknown argument).*-mbranch-likely}}
// CC1OptionCHECK13: {{(unknown argument).*-mbranch-protection=}}
// CC1OptionCHECK13: {{(unknown argument).*-mbranches-within-32B-boundaries}}
// CC1OptionCHECK13: {{(unknown argument).*-mbulk-memory}}
// CC1OptionCHECK13: {{(unknown argument).*-mbulk-memory-opt}}
// RUN: not %clang -cc1 -mcabac -mcall-indirect-overlong -mcheck-zero-division -mcldemote -mclflushopt -mclwb -mclzero -mcmpb -mcmpccxadd -mcompact-branches= -mconsole -mconstant-cfstrings -mcpu= -mcrbits -mcrc -mcrc32 -mcumode -mcx16 -mdaz-ftz -mdefault-build-attributes -mdirect-move -mdiv32 -mdll -mdouble-float -mdsp -mdspr2 -mdynamic-no-pic -mefpu2 -membedded-data -menable-experimental-extensions -menqcmd -mevex512 -mexception-handling -mexec-model= -mexecute-only -mextended-const -mextern-sdata -mf16c -mfancy-math-387 -mfix4300 -mfix-and-continue -mfix-cmse-cve-2021-35465 -mfix-cortex-a53-835769 -mfix-cortex-a57-aes-1742098 -mfix-cortex-a72-aes-1655431 -mfix-gr712rc -mfix-ut700 -mfloat128 -mfloat-abi= -mfma -mfma4 -mfp16 -mfp32 -mfp64 -mfpmath= -mfprnd -mfpu -mfpu= -mfpxx -mframe-chain= -mfrecipe -mfsgsbase -mfsmuld -mfxsr -mgeneral-regs-only -mgfni -mginv -mglibc -mgpopt -mguard= -mhard-float -mhard-quad-float -mharden-sls= -mhvx -mhvx= -mhvx-ieee-fp -mhvx-length= -mhvx-qfloat -mhreset -mhtm -mhwdiv= -mhwmult= -miamcu -mieee-fp -mieee-rnd-near -mimplicit-float -mimplicit-it= -mindirect-jump= -minline-all-stringops -minvariant-function-descriptors -minvpcid -mios-simulator-version-min= -mios-version-min= -mips1 -mips16 -mips2 -mips3 -mips32 -mips32r2 -mips32r3  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK14 %s

// CC1OptionCHECK14: {{(unknown argument).*-mcabac}}
// CC1OptionCHECK14: {{(unknown argument).*-mcall-indirect-overlong}}
// CC1OptionCHECK14: {{(unknown argument).*-mcheck-zero-division}}
// CC1OptionCHECK14: {{(unknown argument).*-mcldemote}}
// CC1OptionCHECK14: {{(unknown argument).*-mclflushopt}}
// CC1OptionCHECK14: {{(unknown argument).*-mclwb}}
// CC1OptionCHECK14: {{(unknown argument).*-mclzero}}
// CC1OptionCHECK14: {{(unknown argument).*-mcmpb}}
// CC1OptionCHECK14: {{(unknown argument).*-mcmpccxadd}}
// CC1OptionCHECK14: {{(unknown argument).*-mcompact-branches=}}
// CC1OptionCHECK14: {{(unknown argument).*-mconsole}}
// CC1OptionCHECK14: {{(unknown argument).*-mconstant-cfstrings}}
// CC1OptionCHECK14: {{(unknown argument).*-mcpu=}}
// CC1OptionCHECK14: {{(unknown argument).*-mcrbits}}
// CC1OptionCHECK14: {{(unknown argument).*-mcrc}}
// CC1OptionCHECK14: {{(unknown argument).*-mcrc32}}
// CC1OptionCHECK14: {{(unknown argument).*-mcumode}}
// CC1OptionCHECK14: {{(unknown argument).*-mcx16}}
// CC1OptionCHECK14: {{(unknown argument).*-mdaz-ftz}}
// CC1OptionCHECK14: {{(unknown argument).*-mdefault-build-attributes}}
// CC1OptionCHECK14: {{(unknown argument).*-mdirect-move}}
// CC1OptionCHECK14: {{(unknown argument).*-mdiv32}}
// CC1OptionCHECK14: {{(unknown argument).*-mdll}}
// CC1OptionCHECK14: {{(unknown argument).*-mdouble-float}}
// CC1OptionCHECK14: {{(unknown argument).*-mdsp}}
// CC1OptionCHECK14: {{(unknown argument).*-mdspr2}}
// CC1OptionCHECK14: {{(unknown argument).*-mdynamic-no-pic}}
// CC1OptionCHECK14: {{(unknown argument).*-mefpu2}}
// CC1OptionCHECK14: {{(unknown argument).*-membedded-data}}
// CC1OptionCHECK14: {{(unknown argument).*-menable-experimental-extensions}}
// CC1OptionCHECK14: {{(unknown argument).*-menqcmd}}
// CC1OptionCHECK14: {{(unknown argument).*-mevex512}}
// CC1OptionCHECK14: {{(unknown argument).*-mexception-handling}}
// CC1OptionCHECK14: {{(unknown argument).*-mexec-model=}}
// CC1OptionCHECK14: {{(unknown argument).*-mexecute-only}}
// CC1OptionCHECK14: {{(unknown argument).*-mextended-const}}
// CC1OptionCHECK14: {{(unknown argument).*-mextern-sdata}}
// CC1OptionCHECK14: {{(unknown argument).*-mf16c}}
// CC1OptionCHECK14: {{(unknown argument).*-mfancy-math-387}}
// CC1OptionCHECK14: {{(unknown argument).*-mfix4300}}
// CC1OptionCHECK14: {{(unknown argument).*-mfix-and-continue}}
// CC1OptionCHECK14: {{(unknown argument).*-mfix-cmse-cve-2021-35465}}
// CC1OptionCHECK14: {{(unknown argument).*-mfix-cortex-a53-835769}}
// CC1OptionCHECK14: {{(unknown argument).*-mfix-cortex-a57-aes-1742098}}
// CC1OptionCHECK14: {{(unknown argument).*-mfix-cortex-a72-aes-1655431}}
// CC1OptionCHECK14: {{(unknown argument).*-mfix-gr712rc}}
// CC1OptionCHECK14: {{(unknown argument).*-mfix-ut700}}
// CC1OptionCHECK14: {{(unknown argument).*-mfloat128}}
// CC1OptionCHECK14: {{(unknown argument).*-mfloat-abi=}}
// CC1OptionCHECK14: {{(unknown argument).*-mfma}}
// CC1OptionCHECK14: {{(unknown argument).*-mfma4}}
// CC1OptionCHECK14: {{(unknown argument).*-mfp16}}
// CC1OptionCHECK14: {{(unknown argument).*-mfp32}}
// CC1OptionCHECK14: {{(unknown argument).*-mfp64}}
// CC1OptionCHECK14: {{(unknown argument).*-mfpmath=}}
// CC1OptionCHECK14: {{(unknown argument).*-mfprnd}}
// CC1OptionCHECK14: {{(unknown argument).*-mfpu}}
// CC1OptionCHECK14: {{(unknown argument).*-mfpu=}}
// CC1OptionCHECK14: {{(unknown argument).*-mfpxx}}
// CC1OptionCHECK14: {{(unknown argument).*-mframe-chain=}}
// CC1OptionCHECK14: {{(unknown argument).*-mfrecipe}}
// CC1OptionCHECK14: {{(unknown argument).*-mfsgsbase}}
// CC1OptionCHECK14: {{(unknown argument).*-mfsmuld}}
// CC1OptionCHECK14: {{(unknown argument).*-mfxsr}}
// CC1OptionCHECK14: {{(unknown argument).*-mgeneral-regs-only}}
// CC1OptionCHECK14: {{(unknown argument).*-mgfni}}
// CC1OptionCHECK14: {{(unknown argument).*-mginv}}
// CC1OptionCHECK14: {{(unknown argument).*-mglibc}}
// CC1OptionCHECK14: {{(unknown argument).*-mgpopt}}
// CC1OptionCHECK14: {{(unknown argument).*-mguard=}}
// CC1OptionCHECK14: {{(unknown argument).*-mhard-float}}
// CC1OptionCHECK14: {{(unknown argument).*-mhard-quad-float}}
// CC1OptionCHECK14: {{(unknown argument).*-mharden-sls=}}
// CC1OptionCHECK14: {{(unknown argument).*-mhvx}}
// CC1OptionCHECK14: {{(unknown argument).*-mhvx=}}
// CC1OptionCHECK14: {{(unknown argument).*-mhvx-ieee-fp}}
// CC1OptionCHECK14: {{(unknown argument).*-mhvx-length=}}
// CC1OptionCHECK14: {{(unknown argument).*-mhvx-qfloat}}
// CC1OptionCHECK14: {{(unknown argument).*-mhreset}}
// CC1OptionCHECK14: {{(unknown argument).*-mhtm}}
// CC1OptionCHECK14: {{(unknown argument).*-mhwdiv=}}
// CC1OptionCHECK14: {{(unknown argument).*-mhwmult=}}
// CC1OptionCHECK14: {{(unknown argument).*-miamcu}}
// CC1OptionCHECK14: {{(unknown argument).*-mieee-fp}}
// CC1OptionCHECK14: {{(unknown argument).*-mieee-rnd-near}}
// CC1OptionCHECK14: {{(unknown argument).*-mimplicit-float}}
// CC1OptionCHECK14: {{(unknown argument).*-mimplicit-it=}}
// CC1OptionCHECK14: {{(unknown argument).*-mindirect-jump=}}
// CC1OptionCHECK14: {{(unknown argument).*-minline-all-stringops}}
// CC1OptionCHECK14: {{(unknown argument).*-minvariant-function-descriptors}}
// CC1OptionCHECK14: {{(unknown argument).*-minvpcid}}
// CC1OptionCHECK14: {{(unknown argument).*-mios-simulator-version-min=}}
// CC1OptionCHECK14: {{(unknown argument).*-mios-version-min=}}
// CC1OptionCHECK14: {{(unknown argument).*-mips1}}
// CC1OptionCHECK14: {{(unknown argument).*-mips16}}
// CC1OptionCHECK14: {{(unknown argument).*-mips2}}
// CC1OptionCHECK14: {{(unknown argument).*-mips3}}
// CC1OptionCHECK14: {{(unknown argument).*-mips32}}
// CC1OptionCHECK14: {{(unknown argument).*-mips32r2}}
// CC1OptionCHECK14: {{(unknown argument).*-mips32r3}}
// RUN: not %clang -cc1 -mips32r5 -mips32r6 -mips4 -mips5 -mips64 -mips64r2 -mips64r3 -mips64r5 -mips64r6 -misel -mkernel -mkl -mlam-bh -mlamcas -mlasx -mld-seq-sa -mldc1-sdc1 -mlinker-version= -mlittle-endian -mlocal-sdata -mlong-calls -mlongcall -mlr-for-calls-only -mlsx -mlvi-cfi -mlvi-hardening -mlwp -mlzcnt -mmacos-version-min= -mmadd4 -mmark-bti-property -mmcu= -mmfcrf -mmfocrf -mmicromips -mmlir -mmma -mmmx -mmovbe -mmovdir64b -mmovdiri -mmovrs -mmpx -mmt -mmultimemory -mmultivalue -mmutable-globals -mmwaitx -mnan= -mno-3dnow -mno-3dnowa -mno-80387 -mno-abicalls -mno-adx -mno-aes -mno-altivec -mno-amdgpu-precise-memory-op -mno-amx-avx512 -mno-amx-bf16 -mno-amx-complex -mno-amx-fp16 -mno-amx-fp8 -mno-amx-int8 -mno-amx-movrs -mno-amx-tf32 -mno-amx-tile -mno-amx-transpose -mno-annotate-tablejump -mno-apx-features= -mno-apxf -mno-atomics -mno-avx -mno-avx10.1 -mno-avx10.1-256 -mno-avx10.1-512 -mno-avx10.2 -mno-avx2 -mno-avx512bf16 -mno-avx512bitalg -mno-avx512bw -mno-avx512cd -mno-avx512dq -mno-avx512f -mno-avx512fp16 -mno-avx512ifma -mno-avx512vbmi -mno-avx512vbmi2 -mno-avx512vl -mno-avx512vnni -mno-avx512vp2intersect -mno-avx512vpopcntdq -mno-avxifma -mno-avxneconvert -mno-avxvnni -mno-avxvnniint16 -mno-avxvnniint8 -mno-bmi -mno-bmi2 -mno-branch-likely -mno-bti-at-return-twice  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK15 %s

// CC1OptionCHECK15: {{(unknown argument).*-mips32r5}}
// CC1OptionCHECK15: {{(unknown argument).*-mips32r6}}
// CC1OptionCHECK15: {{(unknown argument).*-mips4}}
// CC1OptionCHECK15: {{(unknown argument).*-mips5}}
// CC1OptionCHECK15: {{(unknown argument).*-mips64}}
// CC1OptionCHECK15: {{(unknown argument).*-mips64r2}}
// CC1OptionCHECK15: {{(unknown argument).*-mips64r3}}
// CC1OptionCHECK15: {{(unknown argument).*-mips64r5}}
// CC1OptionCHECK15: {{(unknown argument).*-mips64r6}}
// CC1OptionCHECK15: {{(unknown argument).*-misel}}
// CC1OptionCHECK15: {{(unknown argument).*-mkernel}}
// CC1OptionCHECK15: {{(unknown argument).*-mkl}}
// CC1OptionCHECK15: {{(unknown argument).*-mlam-bh}}
// CC1OptionCHECK15: {{(unknown argument).*-mlamcas}}
// CC1OptionCHECK15: {{(unknown argument).*-mlasx}}
// CC1OptionCHECK15: {{(unknown argument).*-mld-seq-sa}}
// CC1OptionCHECK15: {{(unknown argument).*-mldc1-sdc1}}
// CC1OptionCHECK15: {{(unknown argument).*-mlinker-version=}}
// CC1OptionCHECK15: {{(unknown argument).*-mlittle-endian}}
// CC1OptionCHECK15: {{(unknown argument).*-mlocal-sdata}}
// CC1OptionCHECK15: {{(unknown argument).*-mlong-calls}}
// CC1OptionCHECK15: {{(unknown argument).*-mlongcall}}
// CC1OptionCHECK15: {{(unknown argument).*-mlr-for-calls-only}}
// CC1OptionCHECK15: {{(unknown argument).*-mlsx}}
// CC1OptionCHECK15: {{(unknown argument).*-mlvi-cfi}}
// CC1OptionCHECK15: {{(unknown argument).*-mlvi-hardening}}
// CC1OptionCHECK15: {{(unknown argument).*-mlwp}}
// CC1OptionCHECK15: {{(unknown argument).*-mlzcnt}}
// CC1OptionCHECK15: {{(unknown argument).*-mmacos-version-min=}}
// CC1OptionCHECK15: {{(unknown argument).*-mmadd4}}
// CC1OptionCHECK15: {{(unknown argument).*-mmark-bti-property}}
// CC1OptionCHECK15: {{(unknown argument).*-mmcu=}}
// CC1OptionCHECK15: {{(unknown argument).*-mmfcrf}}
// CC1OptionCHECK15: {{(unknown argument).*-mmfocrf}}
// CC1OptionCHECK15: {{(unknown argument).*-mmicromips}}
// CC1OptionCHECK15: {{(unknown argument).*-mmlir}}
// CC1OptionCHECK15: {{(unknown argument).*-mmma}}
// CC1OptionCHECK15: {{(unknown argument).*-mmmx}}
// CC1OptionCHECK15: {{(unknown argument).*-mmovbe}}
// CC1OptionCHECK15: {{(unknown argument).*-mmovdir64b}}
// CC1OptionCHECK15: {{(unknown argument).*-mmovdiri}}
// CC1OptionCHECK15: {{(unknown argument).*-mmovrs}}
// CC1OptionCHECK15: {{(unknown argument).*-mmpx}}
// CC1OptionCHECK15: {{(unknown argument).*-mmt}}
// CC1OptionCHECK15: {{(unknown argument).*-mmultimemory}}
// CC1OptionCHECK15: {{(unknown argument).*-mmultivalue}}
// CC1OptionCHECK15: {{(unknown argument).*-mmutable-globals}}
// CC1OptionCHECK15: {{(unknown argument).*-mmwaitx}}
// CC1OptionCHECK15: {{(unknown argument).*-mnan=}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-3dnow}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-3dnowa}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-80387}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-abicalls}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-adx}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-aes}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-altivec}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-amdgpu-precise-memory-op}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-amx-avx512}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-amx-bf16}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-amx-complex}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-amx-fp16}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-amx-fp8}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-amx-int8}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-amx-movrs}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-amx-tf32}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-amx-tile}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-amx-transpose}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-annotate-tablejump}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-apx-features=}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-apxf}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-atomics}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx10.1}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx10.1-256}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx10.1-512}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx10.2}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx2}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512bf16}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512bitalg}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512bw}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512cd}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512dq}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512f}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512fp16}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512ifma}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512vbmi}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512vbmi2}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512vl}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512vnni}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512vp2intersect}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avx512vpopcntdq}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avxifma}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avxneconvert}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avxvnni}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avxvnniint16}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-avxvnniint8}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-bmi}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-bmi2}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-branch-likely}}
// CC1OptionCHECK15: {{(unknown argument).*-mno-bti-at-return-twice}}
// RUN: not %clang -cc1 -mno-bulk-memory -mno-bulk-memory-opt -mno-call-indirect-overlong -mno-check-zero-division -mno-cldemote -mno-clflushopt -mno-clwb -mno-clzero -mno-cmpb -mno-cmpccxadd -mno-constant-cfstrings -mno-crbits -mno-crc -mno-crc32 -mno-cumode -mno-cx16 -mno-daz-ftz -mno-default-build-attributes -mno-div32 -mno-dsp -mno-dspr2 -mno-embedded-data -mno-enqcmd -mno-evex512 -mno-exception-handling -mno-execute-only -mno-extended-const -mno-extern-sdata -mno-f16c -mno-fix-cmse-cve-2021-35465 -mno-fix-cortex-a53-835769 -mno-fix-cortex-a57-aes-1742098 -mno-fix-cortex-a72-aes-1655431 -mno-float128 -mno-fma -mno-fma4 -mno-fp16 -mno-fp-ret-in-387 -mno-fprnd -mno-fpu -mno-frecipe -mno-fsgsbase -mno-fsmuld -mno-fxsr -mno-gather -mno-gfni -mno-ginv -mno-gpopt -mno-hvx -mno-hvx-ieee-fp -mno-hvx-qfloat -mno-hreset -mno-htm -mno-iamcu -mno-implicit-float -mno-incremental-linker-compatible -mno-inline-all-stringops -mno-invariant-function-descriptors -mno-invpcid -mno-isel -mno-kl -mno-lam-bh -mno-lamcas -mno-lasx -mno-ld-seq-sa -mno-ldc1-sdc1 -mno-local-sdata -mno-long-calls -mno-longcall -mno-lsx -mno-lvi-cfi -mno-lvi-hardening -mno-lwp -mno-lzcnt -mno-madd4 -mno-mfcrf -mno-mfocrf -mno-micromips -mno-mips16 -mno-mma -mno-mmx -mno-movbe -mno-movdir64b -mno-movdiri -mno-movrs -mno-movt -mno-mpx -mno-ms-bitfields -mno-msa -mno-mt -mno-multimemory -mno-multivalue -mno-mutable-globals -mno-mwaitx -mno-neg-immediates -mno-nontrapping-fptoint -mno-odd-spreg -mno-omit-leaf-frame-pointer -mno-pascal-strings -mno-pclmul  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK16 %s

// CC1OptionCHECK16: {{(unknown argument).*-mno-bulk-memory}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-bulk-memory-opt}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-call-indirect-overlong}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-check-zero-division}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-cldemote}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-clflushopt}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-clwb}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-clzero}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-cmpb}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-cmpccxadd}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-constant-cfstrings}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-crbits}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-crc}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-crc32}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-cumode}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-cx16}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-daz-ftz}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-default-build-attributes}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-div32}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-dsp}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-dspr2}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-embedded-data}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-enqcmd}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-evex512}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-exception-handling}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-execute-only}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-extended-const}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-extern-sdata}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-f16c}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fix-cmse-cve-2021-35465}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fix-cortex-a53-835769}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fix-cortex-a57-aes-1742098}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fix-cortex-a72-aes-1655431}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-float128}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fma}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fma4}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fp16}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fp-ret-in-387}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fprnd}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fpu}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-frecipe}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fsgsbase}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fsmuld}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-fxsr}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-gather}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-gfni}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-ginv}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-gpopt}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-hvx}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-hvx-ieee-fp}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-hvx-qfloat}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-hreset}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-htm}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-iamcu}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-implicit-float}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-incremental-linker-compatible}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-inline-all-stringops}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-invariant-function-descriptors}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-invpcid}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-isel}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-kl}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-lam-bh}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-lamcas}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-lasx}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-ld-seq-sa}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-ldc1-sdc1}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-local-sdata}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-long-calls}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-longcall}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-lsx}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-lvi-cfi}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-lvi-hardening}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-lwp}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-lzcnt}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-madd4}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-mfcrf}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-mfocrf}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-micromips}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-mips16}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-mma}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-mmx}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-movbe}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-movdir64b}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-movdiri}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-movrs}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-movt}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-mpx}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-ms-bitfields}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-msa}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-mt}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-multimemory}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-multivalue}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-mutable-globals}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-mwaitx}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-neg-immediates}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-nontrapping-fptoint}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-odd-spreg}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-omit-leaf-frame-pointer}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-pascal-strings}}
// CC1OptionCHECK16: {{(unknown argument).*-mno-pclmul}}
// RUN: not %clang -cc1 -mno-pconfig -mno-pcrel -mno-pic-data-is-text-relative -mno-pku -mno-popc -mno-popcnt -mno-popcntd -mno-power10-vector -mno-power8-vector -mno-power9-vector -mno-prefetchi -mno-prefixed -mno-prfchw -mno-ptwrite -mno-pure-code -mno-raoint -mno-rdpid -mno-rdpru -mno-rdrnd -mno-rdseed -mno-red-zone -mno-reference-types -mno-regnames -mno-relax -mno-relax-all -mno-relax-pic-calls -mno-relaxed-simd -mno-restrict-it -mno-retpoline -mno-retpoline-external-thunk -mno-rtd -mno-rtm -mno-sahf -mno-save-restore -mno-scalar-strict-align -mno-scatter -mno-scq -mno-serialize -mno-seses -mno-sgx -mno-sha -mno-sha512 -mno-shstk -mno-sign-ext -mno-simd128 -mno-sm3 -mno-sm4 -mno-soft-float -mno-spe -mno-speculative-load-hardening -mno-sse -mno-sse2 -mno-sse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-sse4a -mno-ssse3 -mno-stackrealign -mno-strict-align -mno-tail-call -mno-tbm -mno-tgsplit -mno-thumb -mno-tsxldtrk -mno-uintr -mno-unaligned-access -mno-unaligned-symbols -mno-unsafe-fp-atomics -mno-usermsr -mno-v8plus -mno-vaes -mno-vector-strict-align -mno-vevpu -mno-virt -mno-vis -mno-vis2 -mno-vis3 -mno-vpclmulqdq -mno-vsx -mno-vx -mno-vzeroupper -mno-waitpkg -mno-warn-nonportable-cfstrings -mno-wavefrontsize64 -mno-wbnoinvd -mno-wide-arithmetic -mno-widekl -mno-x87 -mno-xcoff-roptr -mno-xgot -mno-xop -mno-xsave -mno-xsavec -mno-xsaveopt -mno-xsaves -mno-zvector -mnocrc -mno-direct-move -mnontrapping-fptoint  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK17 %s

// CC1OptionCHECK17: {{(unknown argument).*-mno-pconfig}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-pcrel}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-pic-data-is-text-relative}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-pku}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-popc}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-popcnt}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-popcntd}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-power10-vector}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-power8-vector}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-power9-vector}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-prefetchi}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-prefixed}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-prfchw}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-ptwrite}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-pure-code}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-raoint}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-rdpid}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-rdpru}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-rdrnd}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-rdseed}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-red-zone}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-reference-types}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-regnames}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-relax}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-relax-all}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-relax-pic-calls}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-relaxed-simd}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-restrict-it}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-retpoline}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-retpoline-external-thunk}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-rtd}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-rtm}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sahf}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-save-restore}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-scalar-strict-align}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-scatter}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-scq}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-serialize}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-seses}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sgx}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sha}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sha512}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-shstk}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sign-ext}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-simd128}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sm3}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sm4}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-soft-float}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-spe}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-speculative-load-hardening}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sse}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sse2}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sse3}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sse4}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sse4.1}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sse4.2}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-sse4a}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-ssse3}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-stackrealign}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-strict-align}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-tail-call}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-tbm}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-tgsplit}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-thumb}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-tsxldtrk}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-uintr}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-unaligned-access}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-unaligned-symbols}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-unsafe-fp-atomics}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-usermsr}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-v8plus}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-vaes}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-vector-strict-align}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-vevpu}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-virt}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-vis}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-vis2}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-vis3}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-vpclmulqdq}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-vsx}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-vx}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-vzeroupper}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-waitpkg}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-warn-nonportable-cfstrings}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-wavefrontsize64}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-wbnoinvd}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-wide-arithmetic}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-widekl}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-x87}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-xcoff-roptr}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-xgot}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-xop}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-xsave}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-xsavec}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-xsaveopt}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-xsaves}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-zvector}}
// CC1OptionCHECK17: {{(unknown argument).*-mnocrc}}
// CC1OptionCHECK17: {{(unknown argument).*-mno-direct-move}}
// CC1OptionCHECK17: {{(unknown argument).*-mnontrapping-fptoint}}
// RUN: not %clang -cc1 -mno-paired-vector-memops -mno-crypto -modd-spreg -module-dir -module-suffix -momit-leaf-frame-pointer -moslib= -mpad-max-prefix-size= -mpaired-vector-memops -mpascal-strings -mpclmul -mpconfig -mpcrel -mpic-data-is-text-relative -mpku -mpopc -mpopcnt -mpopcntd -mpower10-vector -mcrypto -mpower8-vector -mpower9-vector -mprefetchi -mprefixed -mprfchw -mprivileged -mptwrite -mpure-code -mraoint -mrdpid -mrdpru -mrdrnd -mrdseed -mrecip -mred-zone -mreference-types -mregparm= -mrelax -mrelax-pic-calls -mrelaxed-simd -mrestrict-it -mretpoline -mretpoline-external-thunk -mrop-protect -mrtm -mrvv-vector-bits= -msahf -msave-restore -mscalar-strict-align -mscq -msecure-plt -mserialize -msgx -msha -msha512 -mshstk -msign-ext -msim -msimd128 -msimd= -msingle-float -msm3 -msm4 -msmall-data-limit= -msmall-data-threshold= -msoft-quad-float -mspe -msse -msse2 -msse3 -msse4 -msse4.1 -msse4.2 -msse4a -mssse3 -mstack-arg-probe -mstrict-align -msve-vector-bits= -mtail-call -mtargetos= -mtbm -mtgsplit -mthreads -mthumb -mtls-dialect= -mtls-direct-seg-refs -mtp= -mtsxldtrk -mtune= -mtvos-simulator-version-min= -mtvos-version-min= -muclibc -muintr -multi_module -multi-lib-config= -multiply_defined -multiply_defined_unused -munaligned-access -munaligned-symbols -municode  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK18 %s

// CC1OptionCHECK18: {{(unknown argument).*-mno-paired-vector-memops}}
// CC1OptionCHECK18: {{(unknown argument).*-mno-crypto}}
// CC1OptionCHECK18: {{(unknown argument).*-modd-spreg}}
// CC1OptionCHECK18: {{(unknown argument).*-module-dir}}
// CC1OptionCHECK18: {{(unknown argument).*-module-suffix}}
// CC1OptionCHECK18: {{(unknown argument).*-momit-leaf-frame-pointer}}
// CC1OptionCHECK18: {{(unknown argument).*-moslib=}}
// CC1OptionCHECK18: {{(unknown argument).*-mpad-max-prefix-size=}}
// CC1OptionCHECK18: {{(unknown argument).*-mpaired-vector-memops}}
// CC1OptionCHECK18: {{(unknown argument).*-mpascal-strings}}
// CC1OptionCHECK18: {{(unknown argument).*-mpclmul}}
// CC1OptionCHECK18: {{(unknown argument).*-mpconfig}}
// CC1OptionCHECK18: {{(unknown argument).*-mpcrel}}
// CC1OptionCHECK18: {{(unknown argument).*-mpic-data-is-text-relative}}
// CC1OptionCHECK18: {{(unknown argument).*-mpku}}
// CC1OptionCHECK18: {{(unknown argument).*-mpopc}}
// CC1OptionCHECK18: {{(unknown argument).*-mpopcnt}}
// CC1OptionCHECK18: {{(unknown argument).*-mpopcntd}}
// CC1OptionCHECK18: {{(unknown argument).*-mpower10-vector}}
// CC1OptionCHECK18: {{(unknown argument).*-mcrypto}}
// CC1OptionCHECK18: {{(unknown argument).*-mpower8-vector}}
// CC1OptionCHECK18: {{(unknown argument).*-mpower9-vector}}
// CC1OptionCHECK18: {{(unknown argument).*-mprefetchi}}
// CC1OptionCHECK18: {{(unknown argument).*-mprefixed}}
// CC1OptionCHECK18: {{(unknown argument).*-mprfchw}}
// CC1OptionCHECK18: {{(unknown argument).*-mprivileged}}
// CC1OptionCHECK18: {{(unknown argument).*-mptwrite}}
// CC1OptionCHECK18: {{(unknown argument).*-mpure-code}}
// CC1OptionCHECK18: {{(unknown argument).*-mraoint}}
// CC1OptionCHECK18: {{(unknown argument).*-mrdpid}}
// CC1OptionCHECK18: {{(unknown argument).*-mrdpru}}
// CC1OptionCHECK18: {{(unknown argument).*-mrdrnd}}
// CC1OptionCHECK18: {{(unknown argument).*-mrdseed}}
// CC1OptionCHECK18: {{(unknown argument).*-mrecip}}
// CC1OptionCHECK18: {{(unknown argument).*-mred-zone}}
// CC1OptionCHECK18: {{(unknown argument).*-mreference-types}}
// CC1OptionCHECK18: {{(unknown argument).*-mregparm=}}
// CC1OptionCHECK18: {{(unknown argument).*-mrelax}}
// CC1OptionCHECK18: {{(unknown argument).*-mrelax-pic-calls}}
// CC1OptionCHECK18: {{(unknown argument).*-mrelaxed-simd}}
// CC1OptionCHECK18: {{(unknown argument).*-mrestrict-it}}
// CC1OptionCHECK18: {{(unknown argument).*-mretpoline}}
// CC1OptionCHECK18: {{(unknown argument).*-mretpoline-external-thunk}}
// CC1OptionCHECK18: {{(unknown argument).*-mrop-protect}}
// CC1OptionCHECK18: {{(unknown argument).*-mrtm}}
// CC1OptionCHECK18: {{(unknown argument).*-mrvv-vector-bits=}}
// CC1OptionCHECK18: {{(unknown argument).*-msahf}}
// CC1OptionCHECK18: {{(unknown argument).*-msave-restore}}
// CC1OptionCHECK18: {{(unknown argument).*-mscalar-strict-align}}
// CC1OptionCHECK18: {{(unknown argument).*-mscq}}
// CC1OptionCHECK18: {{(unknown argument).*-msecure-plt}}
// CC1OptionCHECK18: {{(unknown argument).*-mserialize}}
// CC1OptionCHECK18: {{(unknown argument).*-msgx}}
// CC1OptionCHECK18: {{(unknown argument).*-msha}}
// CC1OptionCHECK18: {{(unknown argument).*-msha512}}
// CC1OptionCHECK18: {{(unknown argument).*-mshstk}}
// CC1OptionCHECK18: {{(unknown argument).*-msign-ext}}
// CC1OptionCHECK18: {{(unknown argument).*-msim}}
// CC1OptionCHECK18: {{(unknown argument).*-msimd128}}
// CC1OptionCHECK18: {{(unknown argument).*-msimd=}}
// CC1OptionCHECK18: {{(unknown argument).*-msingle-float}}
// CC1OptionCHECK18: {{(unknown argument).*-msm3}}
// CC1OptionCHECK18: {{(unknown argument).*-msm4}}
// CC1OptionCHECK18: {{(unknown argument).*-msmall-data-limit=}}
// CC1OptionCHECK18: {{(unknown argument).*-msmall-data-threshold=}}
// CC1OptionCHECK18: {{(unknown argument).*-msoft-quad-float}}
// CC1OptionCHECK18: {{(unknown argument).*-mspe}}
// CC1OptionCHECK18: {{(unknown argument).*-msse}}
// CC1OptionCHECK18: {{(unknown argument).*-msse2}}
// CC1OptionCHECK18: {{(unknown argument).*-msse3}}
// CC1OptionCHECK18: {{(unknown argument).*-msse4}}
// CC1OptionCHECK18: {{(unknown argument).*-msse4.1}}
// CC1OptionCHECK18: {{(unknown argument).*-msse4.2}}
// CC1OptionCHECK18: {{(unknown argument).*-msse4a}}
// CC1OptionCHECK18: {{(unknown argument).*-mssse3}}
// CC1OptionCHECK18: {{(unknown argument).*-mstack-arg-probe}}
// CC1OptionCHECK18: {{(unknown argument).*-mstrict-align}}
// CC1OptionCHECK18: {{(unknown argument).*-msve-vector-bits=}}
// CC1OptionCHECK18: {{(unknown argument).*-mtail-call}}
// CC1OptionCHECK18: {{(unknown argument).*-mtargetos=}}
// CC1OptionCHECK18: {{(unknown argument).*-mtbm}}
// CC1OptionCHECK18: {{(unknown argument).*-mtgsplit}}
// CC1OptionCHECK18: {{(unknown argument).*-mthreads}}
// CC1OptionCHECK18: {{(unknown argument).*-mthumb}}
// CC1OptionCHECK18: {{(unknown argument).*-mtls-dialect=}}
// CC1OptionCHECK18: {{(unknown argument).*-mtls-direct-seg-refs}}
// CC1OptionCHECK18: {{(unknown argument).*-mtp=}}
// CC1OptionCHECK18: {{(unknown argument).*-mtsxldtrk}}
// CC1OptionCHECK18: {{(unknown argument).*-mtune=}}
// CC1OptionCHECK18: {{(unknown argument).*-mtvos-simulator-version-min=}}
// CC1OptionCHECK18: {{(unknown argument).*-mtvos-version-min=}}
// CC1OptionCHECK18: {{(unknown argument).*-muclibc}}
// CC1OptionCHECK18: {{(unknown argument).*-muintr}}
// CC1OptionCHECK18: {{(unknown argument).*-multi_module}}
// CC1OptionCHECK18: {{(unknown argument).*-multi-lib-config=}}
// CC1OptionCHECK18: {{(unknown argument).*-multiply_defined}}
// CC1OptionCHECK18: {{(unknown argument).*-multiply_defined_unused}}
// CC1OptionCHECK18: {{(unknown argument).*-munaligned-access}}
// CC1OptionCHECK18: {{(unknown argument).*-munaligned-symbols}}
// CC1OptionCHECK18: {{(unknown argument).*-municode}}
// RUN: not %clang -cc1 -musermsr -mv5 -mv55 -mv60 -mv62 -mv65 -mv66 -mv67 -mv67t -mv68 -mv69 -mv71 -mv71t -mv73 -mv75 -mv79 -mv8plus -mvaes -mvector-strict-align -mvevpu -mvirt -mvis -mvis2 -mvis3 -mvpclmulqdq -mvsx -mvx -mvzeroupper -mwaitpkg -mwarn-nonportable-cfstrings -mwatchos-simulator-version-min= -mwatchos-version-min= -mwatchsimulator-version-min= -mwavefrontsize64 -mwbnoinvd -mwide-arithmetic -mwidekl -mwindows -mx32 -mx87 -mxcoff-build-id= -mxgot -mxop -mxsave -mxsavec -mxsaveopt -mxsaves -mzos-hlq-clang= -mzos-hlq-csslib= -mzos-hlq-le= -mzos-sys-include= -mzvector -n -no_dead_strip_inits_and_terms -no-canonical-prefixes -no-cpp-precomp --no-cuda-gpu-arch= --no-cuda-include-ptx= --no-cuda-noopt-device-debug --no-cuda-version-check -fno-c++-static-destructors --no-default-config --no-gpu-bundle-output -no-hip-rt -no-integrated-cpp --no-offload-add-rpath --no-offload-arch= --no-offload-compress -no-pedantic -no-pie --no-wasm-opt -nocpp -nodefaultlibs -nodriverkitlib -nofixprebinding -nogpuinc -nohipwrapperinc -nolibc -nomultidefs -nopie -noprebind -noprofilelib -noseglinkedit -nostartfiles -nostdinc -nostdlib -nostdlibinc -nostdlib++ --nvptx-arch-tool= -fexperimental-openacc-macro-override= -p -pagezero_size -pass-exit-codes -pie -pipe -prebind -prebind_all_twolevel_modules -preload -print-diagnostic-options -print-effective-triple  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK19 %s

// CC1OptionCHECK19: {{(unknown argument).*-musermsr}}
// CC1OptionCHECK19: {{(unknown argument).*-mv5}}
// CC1OptionCHECK19: {{(unknown argument).*-mv55}}
// CC1OptionCHECK19: {{(unknown argument).*-mv60}}
// CC1OptionCHECK19: {{(unknown argument).*-mv62}}
// CC1OptionCHECK19: {{(unknown argument).*-mv65}}
// CC1OptionCHECK19: {{(unknown argument).*-mv66}}
// CC1OptionCHECK19: {{(unknown argument).*-mv67}}
// CC1OptionCHECK19: {{(unknown argument).*-mv67t}}
// CC1OptionCHECK19: {{(unknown argument).*-mv68}}
// CC1OptionCHECK19: {{(unknown argument).*-mv69}}
// CC1OptionCHECK19: {{(unknown argument).*-mv71}}
// CC1OptionCHECK19: {{(unknown argument).*-mv71t}}
// CC1OptionCHECK19: {{(unknown argument).*-mv73}}
// CC1OptionCHECK19: {{(unknown argument).*-mv75}}
// CC1OptionCHECK19: {{(unknown argument).*-mv79}}
// CC1OptionCHECK19: {{(unknown argument).*-mv8plus}}
// CC1OptionCHECK19: {{(unknown argument).*-mvaes}}
// CC1OptionCHECK19: {{(unknown argument).*-mvector-strict-align}}
// CC1OptionCHECK19: {{(unknown argument).*-mvevpu}}
// CC1OptionCHECK19: {{(unknown argument).*-mvirt}}
// CC1OptionCHECK19: {{(unknown argument).*-mvis}}
// CC1OptionCHECK19: {{(unknown argument).*-mvis2}}
// CC1OptionCHECK19: {{(unknown argument).*-mvis3}}
// CC1OptionCHECK19: {{(unknown argument).*-mvpclmulqdq}}
// CC1OptionCHECK19: {{(unknown argument).*-mvsx}}
// CC1OptionCHECK19: {{(unknown argument).*-mvx}}
// CC1OptionCHECK19: {{(unknown argument).*-mvzeroupper}}
// CC1OptionCHECK19: {{(unknown argument).*-mwaitpkg}}
// CC1OptionCHECK19: {{(unknown argument).*-mwarn-nonportable-cfstrings}}
// CC1OptionCHECK19: {{(unknown argument).*-mwatchos-simulator-version-min=}}
// CC1OptionCHECK19: {{(unknown argument).*-mwatchos-version-min=}}
// CC1OptionCHECK19: {{(unknown argument).*-mwatchsimulator-version-min=}}
// CC1OptionCHECK19: {{(unknown argument).*-mwavefrontsize64}}
// CC1OptionCHECK19: {{(unknown argument).*-mwbnoinvd}}
// CC1OptionCHECK19: {{(unknown argument).*-mwide-arithmetic}}
// CC1OptionCHECK19: {{(unknown argument).*-mwidekl}}
// CC1OptionCHECK19: {{(unknown argument).*-mwindows}}
// CC1OptionCHECK19: {{(unknown argument).*-mx32}}
// CC1OptionCHECK19: {{(unknown argument).*-mx87}}
// CC1OptionCHECK19: {{(unknown argument).*-mxcoff-build-id=}}
// CC1OptionCHECK19: {{(unknown argument).*-mxgot}}
// CC1OptionCHECK19: {{(unknown argument).*-mxop}}
// CC1OptionCHECK19: {{(unknown argument).*-mxsave}}
// CC1OptionCHECK19: {{(unknown argument).*-mxsavec}}
// CC1OptionCHECK19: {{(unknown argument).*-mxsaveopt}}
// CC1OptionCHECK19: {{(unknown argument).*-mxsaves}}
// CC1OptionCHECK19: {{(unknown argument).*-mzos-hlq-clang=}}
// CC1OptionCHECK19: {{(unknown argument).*-mzos-hlq-csslib=}}
// CC1OptionCHECK19: {{(unknown argument).*-mzos-hlq-le=}}
// CC1OptionCHECK19: {{(unknown argument).*-mzos-sys-include=}}
// CC1OptionCHECK19: {{(unknown argument).*-mzvector}}
// CC1OptionCHECK19: {{(unknown argument).*-n}}
// CC1OptionCHECK19: {{(unknown argument).*-no_dead_strip_inits_and_terms}}
// CC1OptionCHECK19: {{(unknown argument).*-no-canonical-prefixes}}
// CC1OptionCHECK19: {{(unknown argument).*-no-cpp-precomp}}
// CC1OptionCHECK19: {{(unknown argument).*--no-cuda-gpu-arch=}}
// CC1OptionCHECK19: {{(unknown argument).*--no-cuda-include-ptx=}}
// CC1OptionCHECK19: {{(unknown argument).*--no-cuda-noopt-device-debug}}
// CC1OptionCHECK19: {{(unknown argument).*--no-cuda-version-check}}
// CC1OptionCHECK19: {{(unknown argument).*-fno-c\+\+-static-destructors}}
// CC1OptionCHECK19: {{(unknown argument).*--no-default-config}}
// CC1OptionCHECK19: {{(unknown argument).*--no-gpu-bundle-output}}
// CC1OptionCHECK19: {{(unknown argument).*-no-hip-rt}}
// CC1OptionCHECK19: {{(unknown argument).*-no-integrated-cpp}}
// CC1OptionCHECK19: {{(unknown argument).*--no-offload-add-rpath}}
// CC1OptionCHECK19: {{(unknown argument).*--no-offload-arch=}}
// CC1OptionCHECK19: {{(unknown argument).*--no-offload-compress}}
// CC1OptionCHECK19: {{(unknown argument).*-no-pedantic}}
// CC1OptionCHECK19: {{(unknown argument).*-no-pie}}
// CC1OptionCHECK19: {{(unknown argument).*--no-wasm-opt}}
// CC1OptionCHECK19: {{(unknown argument).*-nocpp}}
// CC1OptionCHECK19: {{(unknown argument).*-nodefaultlibs}}
// CC1OptionCHECK19: {{(unknown argument).*-nodriverkitlib}}
// CC1OptionCHECK19: {{(unknown argument).*-nofixprebinding}}
// CC1OptionCHECK19: {{(unknown argument).*-nogpuinc}}
// CC1OptionCHECK19: {{(unknown argument).*-nohipwrapperinc}}
// CC1OptionCHECK19: {{(unknown argument).*-nolibc}}
// CC1OptionCHECK19: {{(unknown argument).*-nomultidefs}}
// CC1OptionCHECK19: {{(unknown argument).*-nopie}}
// CC1OptionCHECK19: {{(unknown argument).*-noprebind}}
// CC1OptionCHECK19: {{(unknown argument).*-noprofilelib}}
// CC1OptionCHECK19: {{(unknown argument).*-noseglinkedit}}
// CC1OptionCHECK19: {{(unknown argument).*-nostartfiles}}
// CC1OptionCHECK19: {{(unknown argument).*-nostdinc}}
// CC1OptionCHECK19: {{(unknown argument).*-nostdlib}}
// CC1OptionCHECK19: {{(unknown argument).*-nostdlibinc}}
// CC1OptionCHECK19: {{(unknown argument).*-nostdlib\+\+}}
// CC1OptionCHECK19: {{(unknown argument).*--nvptx-arch-tool=}}
// CC1OptionCHECK19: {{(unknown argument).*-fexperimental-openacc-macro-override=}}
// CC1OptionCHECK19: {{(unknown argument).*-p}}
// CC1OptionCHECK19: {{(unknown argument).*-pagezero_size}}
// CC1OptionCHECK19: {{(unknown argument).*-pass-exit-codes}}
// CC1OptionCHECK19: {{(unknown argument).*-pie}}
// CC1OptionCHECK19: {{(unknown argument).*-pipe}}
// CC1OptionCHECK19: {{(unknown argument).*-prebind}}
// CC1OptionCHECK19: {{(unknown argument).*-prebind_all_twolevel_modules}}
// CC1OptionCHECK19: {{(unknown argument).*-preload}}
// CC1OptionCHECK19: {{(unknown argument).*-print-diagnostic-options}}
// CC1OptionCHECK19: {{(unknown argument).*-print-effective-triple}}
// RUN: not %clang -cc1 -print-file-name= -print-libgcc-file-name -print-multi-directory -print-multi-flags-experimental -print-multi-lib -print-multi-os-directory -print-prog-name= -print-resource-dir -print-rocm-search-dirs -print-runtime-dir -print-search-dirs -print-library-module-manifest-path -print-target-triple -print-targets -private_bundle -pthreads --ptxas-path= -r -rdynamic -read_only_relocs -reexport_framework -reexport-l -reexport_library -remap -resource-dir= -rewrite-legacy-objc --rocm-device-lib-path= --rocm-path= -rpath --rsp-quoting= -rtlib= -s -save-stats -save-stats= -save-temps -sectalign -sectcreate -sectobjectsymbols -sectorder -seg1addr -seg_addr_table -seg_addr_table_filename -segaddr -segcreate -seglinkedit -segprot -segs_read_ -segs_read_only_addr -segs_read_write_addr -shared -shared-libgcc -shared-libsan -show-encoding -show-inst -single_module -specs -specs= -spirv --start-no-unused-arguments -startfiles -static -static-libgcc -static-libgfortran -static-libsan -static-libstdc++ -static-openmp -static-pie -std-default= -stdlib -stdlib++-isystem -sub_library -sub_umbrella --sycl-link -t --target= -target -T -test-io -time -traditional -trigraphs -twolevel_namespace -twolevel_namespace_hints -u -umbrella -undefined -unexported_symbols_list --verify-debug-info -via-file-asm --wasm-opt -weak_framework -weak_library -weak_reference_mismatches -weak-l -whatsloaded -why_load -whyload -y -z  -x c++ - < /dev/null 2>&1 | FileCheck -check-prefix=CC1OptionCHECK20 %s

// CC1OptionCHECK20: {{(unknown argument).*-print-file-name=}}
// CC1OptionCHECK20: {{(unknown argument).*-print-libgcc-file-name}}
// CC1OptionCHECK20: {{(unknown argument).*-print-multi-directory}}
// CC1OptionCHECK20: {{(unknown argument).*-print-multi-flags-experimental}}
// CC1OptionCHECK20: {{(unknown argument).*-print-multi-lib}}
// CC1OptionCHECK20: {{(unknown argument).*-print-multi-os-directory}}
// CC1OptionCHECK20: {{(unknown argument).*-print-prog-name=}}
// CC1OptionCHECK20: {{(unknown argument).*-print-resource-dir}}
// CC1OptionCHECK20: {{(unknown argument).*-print-rocm-search-dirs}}
// CC1OptionCHECK20: {{(unknown argument).*-print-runtime-dir}}
// CC1OptionCHECK20: {{(unknown argument).*-print-search-dirs}}
// CC1OptionCHECK20: {{(unknown argument).*-print-library-module-manifest-path}}
// CC1OptionCHECK20: {{(unknown argument).*-print-target-triple}}
// CC1OptionCHECK20: {{(unknown argument).*-print-targets}}
// CC1OptionCHECK20: {{(unknown argument).*-private_bundle}}
// CC1OptionCHECK20: {{(unknown argument).*-pthreads}}
// CC1OptionCHECK20: {{(unknown argument).*--ptxas-path=}}
// CC1OptionCHECK20: {{(unknown argument).*-r}}
// CC1OptionCHECK20: {{(unknown argument).*-rdynamic}}
// CC1OptionCHECK20: {{(unknown argument).*-read_only_relocs}}
// CC1OptionCHECK20: {{(unknown argument).*-reexport_framework}}
// CC1OptionCHECK20: {{(unknown argument).*-reexport-l}}
// CC1OptionCHECK20: {{(unknown argument).*-reexport_library}}
// CC1OptionCHECK20: {{(unknown argument).*-remap}}
// CC1OptionCHECK20: {{(unknown argument).*-resource-dir=}}
// CC1OptionCHECK20: {{(unknown argument).*-rewrite-legacy-objc}}
// CC1OptionCHECK20: {{(unknown argument).*--rocm-device-lib-path=}}
// CC1OptionCHECK20: {{(unknown argument).*--rocm-path=}}
// CC1OptionCHECK20: {{(unknown argument).*-rpath}}
// CC1OptionCHECK20: {{(unknown argument).*--rsp-quoting=}}
// CC1OptionCHECK20: {{(unknown argument).*-rtlib=}}
// CC1OptionCHECK20: {{(unknown argument).*-s}}
// CC1OptionCHECK20: {{(unknown argument).*-save-stats}}
// CC1OptionCHECK20: {{(unknown argument).*-save-stats=}}
// CC1OptionCHECK20: {{(unknown argument).*-save-temps}}
// CC1OptionCHECK20: {{(unknown argument).*-sectalign}}
// CC1OptionCHECK20: {{(unknown argument).*-sectcreate}}
// CC1OptionCHECK20: {{(unknown argument).*-sectobjectsymbols}}
// CC1OptionCHECK20: {{(unknown argument).*-sectorder}}
// CC1OptionCHECK20: {{(unknown argument).*-seg1addr}}
// CC1OptionCHECK20: {{(unknown argument).*-seg_addr_table}}
// CC1OptionCHECK20: {{(unknown argument).*-seg_addr_table_filename}}
// CC1OptionCHECK20: {{(unknown argument).*-segaddr}}
// CC1OptionCHECK20: {{(unknown argument).*-segcreate}}
// CC1OptionCHECK20: {{(unknown argument).*-seglinkedit}}
// CC1OptionCHECK20: {{(unknown argument).*-segprot}}
// CC1OptionCHECK20: {{(unknown argument).*-segs_read_}}
// CC1OptionCHECK20: {{(unknown argument).*-segs_read_only_addr}}
// CC1OptionCHECK20: {{(unknown argument).*-segs_read_write_addr}}
// CC1OptionCHECK20: {{(unknown argument).*-shared}}
// CC1OptionCHECK20: {{(unknown argument).*-shared-libgcc}}
// CC1OptionCHECK20: {{(unknown argument).*-shared-libsan}}
// CC1OptionCHECK20: {{(unknown argument).*-show-encoding}}
// CC1OptionCHECK20: {{(unknown argument).*-show-inst}}
// CC1OptionCHECK20: {{(unknown argument).*-single_module}}
// CC1OptionCHECK20: {{(unknown argument).*-specs}}
// CC1OptionCHECK20: {{(unknown argument).*-specs=}}
// CC1OptionCHECK20: {{(unknown argument).*-spirv}}
// CC1OptionCHECK20: {{(unknown argument).*--start-no-unused-arguments}}
// CC1OptionCHECK20: {{(unknown argument).*-startfiles}}
// CC1OptionCHECK20: {{(unknown argument).*-static}}
// CC1OptionCHECK20: {{(unknown argument).*-static-libgcc}}
// CC1OptionCHECK20: {{(unknown argument).*-static-libgfortran}}
// CC1OptionCHECK20: {{(unknown argument).*-static-libsan}}
// CC1OptionCHECK20: {{(unknown argument).*-static-libstdc\+\+}}
// CC1OptionCHECK20: {{(unknown argument).*-static-openmp}}
// CC1OptionCHECK20: {{(unknown argument).*-static-pie}}
// CC1OptionCHECK20: {{(unknown argument).*-std-default=}}
// CC1OptionCHECK20: {{(unknown argument).*-stdlib}}
// CC1OptionCHECK20: {{(unknown argument).*-stdlib\+\+-isystem}}
// CC1OptionCHECK20: {{(unknown argument).*-sub_library}}
// CC1OptionCHECK20: {{(unknown argument).*-sub_umbrella}}
// CC1OptionCHECK20: {{(unknown argument).*--sycl-link}}
// CC1OptionCHECK20: {{(unknown argument).*-t}}
// CC1OptionCHECK20: {{(unknown argument).*--target=}}
// CC1OptionCHECK20: {{(unknown argument).*-target}}
// CC1OptionCHECK20: {{(unknown argument).*-T}}
// CC1OptionCHECK20: {{(unknown argument).*-test-io}}
// CC1OptionCHECK20: {{(unknown argument).*-time}}
// CC1OptionCHECK20: {{(unknown argument).*-traditional}}
// CC1OptionCHECK20: {{(unknown argument).*-trigraphs}}
// CC1OptionCHECK20: {{(unknown argument).*-twolevel_namespace}}
// CC1OptionCHECK20: {{(unknown argument).*-twolevel_namespace_hints}}
// CC1OptionCHECK20: {{(unknown argument).*-u}}
// CC1OptionCHECK20: {{(unknown argument).*-umbrella}}
// CC1OptionCHECK20: {{(unknown argument).*-undefined}}
// CC1OptionCHECK20: {{(unknown argument).*-unexported_symbols_list}}
// CC1OptionCHECK20: {{(unknown argument).*--verify-debug-info}}
// CC1OptionCHECK20: {{(unknown argument).*-via-file-asm}}
// CC1OptionCHECK20: {{(unknown argument).*--wasm-opt}}
// CC1OptionCHECK20: {{(unknown argument).*-weak_framework}}
// CC1OptionCHECK20: {{(unknown argument).*-weak_library}}
// CC1OptionCHECK20: {{(unknown argument).*-weak_reference_mismatches}}
// CC1OptionCHECK20: {{(unknown argument).*-weak-l}}
// CC1OptionCHECK20: {{(unknown argument).*-whatsloaded}}
// CC1OptionCHECK20: {{(unknown argument).*-why_load}}
// CC1OptionCHECK20: {{(unknown argument).*-whyload}}
// CC1OptionCHECK20: {{(unknown argument).*-y}}
// CC1OptionCHECK20: {{(unknown argument).*-z}}
// RUN: not %clang_cl -A -A- -B -CC -EB -EL -Eonly -K -L -M -MF -MG -MJ -MM -MMD -MQ -MV -Mach -Q -Qn -Qy -R -Rpass= -Rpass-analysis= -Rpass-missed= -S -T -Xanalyzer -Xarch_ -Xarch_device -Xarch_host -Xassembler -Xcuda-fatbinary -Xoffload-linker -Xopenmp-target -Xopenmp-target= -Xpreprocessor -Z -Z-Xlinker-no-demangle -Z-reserved-lib-cckext -Z-reserved-lib-stdc++ -Zlinker-input --CLASSPATH --CLASSPATH= --all-warnings --analyzer-no-default-checks --analyzer-output --assemble --assert --assert= --bootclasspath --bootclasspath= --classpath --classpath= --comments --comments-in-macros --compile --constant-cfstrings --debug --debug= --define-macro --define-macro= --dependencies --dyld-prefix --dyld-prefix= --encoding --encoding= --entry --extdirs --extdirs= --extra-warnings --for-linker --for-linker= --force-link --force-link= --help-hidden --imacros= --include= --include-barrier --include-directory --include-directory= --include-directory-after --include-directory-after= --include-prefix --include-prefix= --include-with-prefix --include-with-prefix= --include-with-prefix-after --include-with-prefix-after= --include-with-prefix-before --include-with-prefix-before= --language --language= --library-directory --library-directory= --mhwdiv --no-line-commands --no-standard-includes --no-standard-libraries --no-undefined  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK0 %s

// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-A}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-A-}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-B}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-CC}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-EB}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-EL}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Eonly}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-K}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-L}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-M}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-MF}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-MG}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-MJ}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-MM}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-MMD}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-MQ}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-MV}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Mach}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Q}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Qn}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Qy}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-R}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Rpass=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Rpass-analysis=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Rpass-missed=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-S}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-T}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Xanalyzer}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Xarch_}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Xarch_device}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Xarch_host}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Xassembler}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Xcuda-fatbinary}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Xoffload-linker}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Xopenmp-target}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Xopenmp-target=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Xpreprocessor}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Z}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Z-Xlinker-no-demangle}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Z-reserved-lib-cckext}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Z-reserved-lib-stdc\+\+}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*-Zlinker-input}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--CLASSPATH}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--CLASSPATH=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--all-warnings}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--analyzer-no-default-checks}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--analyzer-output}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--assemble}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--assert}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--assert=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--bootclasspath}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--bootclasspath=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--classpath}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--classpath=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--comments}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--comments-in-macros}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--compile}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--constant-cfstrings}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--debug}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--debug=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--define-macro}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--define-macro=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--dependencies}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--dyld-prefix}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--dyld-prefix=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--encoding}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--encoding=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--entry}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--extdirs}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--extdirs=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--extra-warnings}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--for-linker}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--for-linker=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--force-link}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--force-link=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--help-hidden}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--imacros=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-barrier}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-directory}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-directory=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-directory-after}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-directory-after=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-prefix}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-prefix=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-with-prefix}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-with-prefix=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-with-prefix-after}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-with-prefix-after=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-with-prefix-before}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--include-with-prefix-before=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--language}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--language=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--library-directory}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--library-directory=}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--mhwdiv}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--no-line-commands}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--no-standard-includes}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--no-standard-libraries}}
// CLOptionCHECK0: {{(unknown argument ignored in clang-cl).*--no-undefined}}
// RUN: not %clang_cl --param --param= --prefix --prefix= --preprocess --print-diagnostic-categories --print-file-name --print-missing-file-dependencies --print-prog-name --profile --resource --resource= --rtlib -serialize-diagnostics --signed-char --std --stdlib --sysroot --sysroot= --target-help --trace-includes --undefine-macro --undefine-macro= --unsigned-char --user-dependencies --verbose -add-plugin -alias_list -faligned-alloc-unavailable -all_load -allowable_client -faltivec-src-compat= -cfg-add-implicit-dtors -unoptimized-cfg -analyze-function -analyze-function= -analyzer-checker -analyzer-checker= -analyzer-checker-help -analyzer-checker-help-alpha -analyzer-checker-help-developer -analyzer-checker-option-help -analyzer-checker-option-help-alpha -analyzer-checker-option-help-developer -analyzer-config -analyzer-config-compatibility-mode -analyzer-config-compatibility-mode= -analyzer-config-help -analyzer-constraints -analyzer-constraints= -analyzer-disable-all-checks -analyzer-disable-checker -analyzer-disable-checker= -analyzer-disable-retry-exhausted -analyzer-display-progress -analyzer-dump-egraph -analyzer-dump-egraph= -analyzer-inline-max-stack-depth -analyzer-inline-max-stack-depth= -analyzer-inlining-mode -analyzer-inlining-mode= -analyzer-list-enabled-checkers -analyzer-max-loop -analyzer-note-analysis-entry-points -analyzer-opt-analyze-headers -analyzer-output -analyzer-output= -analyzer-purge -analyzer-purge= -analyzer-stats -analyzer-viz-egraph-graphviz -analyzer-werror -fnew-alignment -faligned-new -fno-aligned-new -fsched-interblock -ftemplate-depth- -ftree-vectorize -fno-tree-vectorize -ftree-slp-vectorize -fno-tree-slp-vectorize -fterminated-vtables -grecord-gcc-switches -gno-record-gcc-switches -miphonesimulator-version-min= -nocudainc -nogpulib -nocudalib -print-multiarch --system-header-prefix --no-system-header-prefix -integrated-as -no-integrated-as -coverage-data-file= -coverage-notes-file= -fopenmp-is-device -fcuda-approx-transcendentals -fno-cuda-approx-transcendentals -Xmicrosoft-visualc-tools-root -Xmicrosoft-visualc-tools-version  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK1 %s

// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--param}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--param=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--prefix}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--prefix=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--preprocess}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--print-diagnostic-categories}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--print-file-name}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--print-missing-file-dependencies}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--print-prog-name}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--profile}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--resource}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--resource=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--rtlib}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-serialize-diagnostics}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--signed-char}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--std}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--stdlib}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--sysroot}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--sysroot=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--target-help}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--trace-includes}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--undefine-macro}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--undefine-macro=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--unsigned-char}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--user-dependencies}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--verbose}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-add-plugin}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-alias_list}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-faligned-alloc-unavailable}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-all_load}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-allowable_client}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-faltivec-src-compat=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-cfg-add-implicit-dtors}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-unoptimized-cfg}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyze-function}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyze-function=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-checker}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-checker=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-checker-help}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-checker-help-alpha}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-checker-help-developer}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-checker-option-help}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-checker-option-help-alpha}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-checker-option-help-developer}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-config}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-config-compatibility-mode}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-config-compatibility-mode=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-config-help}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-constraints}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-constraints=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-disable-all-checks}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-disable-checker}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-disable-checker=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-disable-retry-exhausted}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-display-progress}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-dump-egraph}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-dump-egraph=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-inline-max-stack-depth}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-inline-max-stack-depth=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-inlining-mode}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-inlining-mode=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-list-enabled-checkers}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-max-loop}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-note-analysis-entry-points}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-opt-analyze-headers}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-output}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-output=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-purge}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-purge=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-stats}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-viz-egraph-graphviz}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-analyzer-werror}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-fnew-alignment}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-faligned-new}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-fno-aligned-new}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-fsched-interblock}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-ftemplate-depth-}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-ftree-vectorize}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-fno-tree-vectorize}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-ftree-slp-vectorize}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-fno-tree-slp-vectorize}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-fterminated-vtables}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-grecord-gcc-switches}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-gno-record-gcc-switches}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-miphonesimulator-version-min=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-nocudainc}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-nogpulib}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-nocudalib}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-print-multiarch}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--system-header-prefix}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*--no-system-header-prefix}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-integrated-as}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-no-integrated-as}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-coverage-data-file=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-coverage-notes-file=}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-fopenmp-is-device}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-fcuda-approx-transcendentals}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-fno-cuda-approx-transcendentals}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-Xmicrosoft-visualc-tools-root}}
// CLOptionCHECK1: {{(unknown argument ignored in clang-cl).*-Xmicrosoft-visualc-tools-version}}
// RUN: not %clang_cl -Xmicrosoft-windows-sdk-root -Xmicrosoft-windows-sdk-version -Xmicrosoft-windows-sys-root -Qembed_debug -shared-libasan -static-libasan -fslp-vectorize-aggressive -frecord-gcc-switches -fno-record-gcc-switches -fno-slp-vectorize-aggressive -Xparser -Xcompiler -fexpensive-optimizations -fno-expensive-optimizations -fdefer-pop -fno-defer-pop -fextended-identifiers -fno-extended-identifiers -fhonor-infinites -fno-honor-infinites -findirect-virtual-calls --config -ansi -arch -arch_errors_fatal -arch_only -as-secure-log-file -ast-dump -ast-dump= -ast-dump-all -ast-dump-all= -ast-dump-decl-types -ast-dump-filter -ast-dump-filter= -ast-dump-lookups -ast-list -ast-merge -ast-print -ast-view --autocomplete= -aux-target-cpu -aux-target-feature -aux-triple -b -bind_at_load -building-pch-with-obj -bundle -bundle_loader -c-isystem -ccc- -ccc-gcc-name -cfguard -cfguard-no-checks -chain-include -cl-denorms-are-zero -cl-ext= -cl-fast-relaxed-math -cl-finite-math-only -cl-fp32-correctly-rounded-divide-sqrt -cl-kernel-arg-info -cl-mad-enable -cl-no-signed-zeros -cl-no-stdinc -cl-opt-disable -cl-single-precision-constant -cl-std= -cl-strict-aliasing -cl-uniform-work-group-size -cl-unsafe-math-optimizations -clear-ast-before-backend -client_name -code-completion-at -code-completion-at= -code-completion-brief-comments -code-completion-macros -code-completion-patterns -code-completion-with-fixits -combine -compatibility_version -compiler-options-dump -complex-range= -compress-debug-sections -compress-debug-sections= -coverage-version= -cpp -cpp-precomp --crel -current_version -cxx-isystem -fc++-static-destructors -fc++-static-destructors= -dA -dD -dE -dI -dM -d -d -darwin-target-variant-sdk-version= -darwin-target-variant-triple  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK2 %s

// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-Xmicrosoft-windows-sdk-root}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-Xmicrosoft-windows-sdk-version}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-Xmicrosoft-windows-sys-root}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-Qembed_debug}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-shared-libasan}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-static-libasan}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fslp-vectorize-aggressive}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-frecord-gcc-switches}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fno-record-gcc-switches}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fno-slp-vectorize-aggressive}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-Xparser}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-Xcompiler}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fexpensive-optimizations}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fno-expensive-optimizations}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fdefer-pop}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fno-defer-pop}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fextended-identifiers}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fno-extended-identifiers}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fhonor-infinites}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fno-honor-infinites}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-findirect-virtual-calls}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*--config}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ansi}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-arch}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-arch_errors_fatal}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-arch_only}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-as-secure-log-file}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ast-dump}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ast-dump=}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ast-dump-all}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ast-dump-all=}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ast-dump-decl-types}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ast-dump-filter}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ast-dump-filter=}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ast-dump-lookups}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ast-list}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ast-merge}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ast-print}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ast-view}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*--autocomplete=}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-aux-target-cpu}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-aux-target-feature}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-aux-triple}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-b}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-bind_at_load}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-building-pch-with-obj}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-bundle}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-bundle_loader}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-c-isystem}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ccc-}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-ccc-gcc-name}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cfguard}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cfguard-no-checks}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-chain-include}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-denorms-are-zero}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-ext=}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-fast-relaxed-math}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-finite-math-only}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-fp32-correctly-rounded-divide-sqrt}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-kernel-arg-info}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-mad-enable}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-no-signed-zeros}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-no-stdinc}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-opt-disable}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-single-precision-constant}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-std=}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-strict-aliasing}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-uniform-work-group-size}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cl-unsafe-math-optimizations}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-clear-ast-before-backend}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-client_name}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-code-completion-at}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-code-completion-at=}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-code-completion-brief-comments}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-code-completion-macros}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-code-completion-patterns}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-code-completion-with-fixits}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-combine}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-compatibility_version}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-compiler-options-dump}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-complex-range=}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-compress-debug-sections}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-compress-debug-sections=}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-coverage-version=}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cpp}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cpp-precomp}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*--crel}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-current_version}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-cxx-isystem}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fc\+\+-static-destructors}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-fc\+\+-static-destructors=}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-dA}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-dD}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-dE}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-dI}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-dM}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-d}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-d}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-darwin-target-variant-sdk-version=}}
// CLOptionCHECK2: {{(unknown argument ignored in clang-cl).*-darwin-target-variant-triple}}
// RUN: not %clang_cl -dead_strip -debug-forward-template-params -debug-info-kind= -debug-info-macro -debugger-tuning= -default-function-attr --defsym -dependency-dot -dependency-file --dependent-lib= -detailed-preprocessing-record -diagnostic-log-file -serialize-diagnostic-file -disable-O0-optnone -disable-free -disable-lifetime-markers -disable-llvm-optzns -disable-llvm-passes -disable-llvm-verifier -disable-objc-default-synthesize-properties -disable-pragma-debug-crash -disable-red-zone -discard-value-names -dsym-dir -dump-coverage-mapping -dump-deserialized-decls -dump-raw-tokens -dump-tokens -dumpdir -dumpmachine -dumpspecs -dumpversion -dwarf-debug-flags -dwarf-debug-producer -dwarf-explicit-import -dwarf-ext-refs -dwarf-version= -Vd -HV -hlsl-no-stdinc --dxv-path= -validator-version -dylib_file -dylinker -dylinker_install_name -dynamic -dynamiclib -e -ehcontguard --embed-dir= -emit-cir -emit-codegen-only --emit-extension-symbol-graphs -emit-fir -emit-header-unit -emit-hlfir -emit-html -emit-interface-stubs -emit-llvm -emit-llvm-bc -emit-llvm-only -emit-llvm-uselists -emit-merged-ifs -emit-mlir -emit-module -emit-module-interface -emit-obj -emit-pch --pretty-sgf -emit-pristine-llvm -emit-reduced-module-interface --emit-sgf-symbol-labels-for-testing --emit-static-lib -emit-symbol-graph -enable-16bit-types -enable-noundef-analysis -enable-tlsdesc -error-on-deserialized-decl -error-on-deserialized-decl= -exception-model -exception-model= -fexperimental-modules-reduced-bmi -exported_symbols_list -extract-api --extract-api-ignores= -fPIC -fPIE -faarch64-jump-table-hardening -faccess-control -faddress-space-map-mangling= -faggressive-function-elimination -falign-commons -falign-functions -falign-functions= -falign-jumps -falign-jumps= -falign-labels -falign-labels= -falign-loops -falign-loops=  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK3 %s

// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dead_strip}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-debug-forward-template-params}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-debug-info-kind=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-debug-info-macro}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-debugger-tuning=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-default-function-attr}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*--defsym}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dependency-dot}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dependency-file}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*--dependent-lib=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-detailed-preprocessing-record}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-diagnostic-log-file}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-serialize-diagnostic-file}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-disable-O0-optnone}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-disable-free}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-disable-lifetime-markers}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-disable-llvm-optzns}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-disable-llvm-passes}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-disable-llvm-verifier}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-disable-objc-default-synthesize-properties}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-disable-pragma-debug-crash}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-disable-red-zone}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-discard-value-names}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dsym-dir}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dump-coverage-mapping}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dump-deserialized-decls}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dump-raw-tokens}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dump-tokens}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dumpdir}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dumpmachine}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dumpspecs}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dumpversion}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dwarf-debug-flags}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dwarf-debug-producer}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dwarf-explicit-import}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dwarf-ext-refs}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dwarf-version=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-Vd}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-HV}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-hlsl-no-stdinc}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*--dxv-path=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-validator-version}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dylib_file}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dylinker}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dylinker_install_name}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dynamic}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-dynamiclib}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-e}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-ehcontguard}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*--embed-dir=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-cir}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-codegen-only}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*--emit-extension-symbol-graphs}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-fir}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-header-unit}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-hlfir}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-html}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-interface-stubs}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-llvm}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-llvm-bc}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-llvm-only}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-llvm-uselists}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-merged-ifs}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-mlir}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-module}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-module-interface}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-obj}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-pch}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*--pretty-sgf}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-pristine-llvm}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-reduced-module-interface}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*--emit-sgf-symbol-labels-for-testing}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*--emit-static-lib}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-emit-symbol-graph}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-enable-16bit-types}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-enable-noundef-analysis}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-enable-tlsdesc}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-error-on-deserialized-decl}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-error-on-deserialized-decl=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-exception-model}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-exception-model=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-fexperimental-modules-reduced-bmi}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-exported_symbols_list}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-extract-api}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*--extract-api-ignores=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-fPIC}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-fPIE}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-faarch64-jump-table-hardening}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-faccess-control}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-faddress-space-map-mangling=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-faggressive-function-elimination}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-falign-commons}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-falign-functions}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-falign-functions=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-falign-jumps}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-falign-jumps=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-falign-labels}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-falign-labels=}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-falign-loops}}
// CLOptionCHECK3: {{(unknown argument ignored in clang-cl).*-falign-loops=}}
// RUN: not %clang_cl -faligned-allocation -faligned-new= -fall-intrinsics -fallow-editor-placeholders -fallow-pch-with-different-modules-cache-path -fallow-pch-with-compiler-errors -fallow-pcm-with-compiler-errors -fallow-unsupported -falternative-parameter-statement -faltivec -fanalyzed-objects-for-unparse -fandroid-pad-segment -fkeep-inline-functions -funit-at-a-time -fapinotes -fapinotes-modules -fapinotes-swift-version= -fapple-kext -fapple-link-rtlib -fapple-pragma-pack -fapplication-extension -fapply-global-visibility-to-externs -fapprox-func -fasm -fasm-blocks -fassociative-math -fassume-nothrow-exception-dtor -fassume-sane-operator-new -fassumptions -fast -fastcp -fastf -fasync-exceptions -fasynchronous-unwind-tables -fauto-import -fauto-profile= -fauto-profile-accurate -fautolink -fautomatic -fbackslash -fbacktrace -fbasic-block-address-map -fbasic-block-sections= -fbfloat16-excess-precision= -fbinutils-version= -fblas-matmul-limit= -fblocks-runtime-optional -fbootclasspath= -fborland-extensions -fbounds-check -fexperimental-bounds-safety -fbracket-depth -fbranch-count-reg -fbuild-session-file= -fbuild-session-timestamp= -fbuiltin-headers-in-system-modules -fbuiltin-module-map -fcaller-saves -fcaret-diagnostics -fcf-runtime-abi= -fcgl -fchar8_t -fcheck= -fcheck-array-temporaries -fcheck-new -fclang-abi-compat= -fclangir -fclasspath= -fcoarray= -fcomment-block-commands= -fcompatibility-qualified-id-block-type-checking -fcompile-resource= -fcomplex-arithmetic= -fconst-strings -fconstant-cfstrings -fconstant-string-class -fconstant-string-class= -fconstexpr-backtrace-limit= -fconstexpr-depth= -fconstexpr-steps= -fconvergent-functions -fconvert= -fcoro-aligned-allocation -fcoroutines -fcoverage-prefix-map= -fcray-pointer -fcreate-profile -fctor-dtor-return-this -fcuda-allow-variadic-functions -fcuda-include-gpubinary -fcuda-is-device -fcx-fortran-rules -fcx-limited-range -fc++-abi= -fcxx-exceptions -fcxx-modules -fd-lines-as-code -fd-lines-as-comments -fdata-sections -fdebug-default-version=  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK4 %s

// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-faligned-allocation}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-faligned-new=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fall-intrinsics}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fallow-editor-placeholders}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fallow-pch-with-different-modules-cache-path}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fallow-pch-with-compiler-errors}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fallow-pcm-with-compiler-errors}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fallow-unsupported}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-falternative-parameter-statement}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-faltivec}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fanalyzed-objects-for-unparse}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fandroid-pad-segment}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fkeep-inline-functions}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-funit-at-a-time}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fapinotes}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fapinotes-modules}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fapinotes-swift-version=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fapple-kext}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fapple-link-rtlib}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fapple-pragma-pack}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fapplication-extension}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fapply-global-visibility-to-externs}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fapprox-func}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fasm}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fasm-blocks}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fassociative-math}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fassume-nothrow-exception-dtor}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fassume-sane-operator-new}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fassumptions}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fast}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fastcp}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fastf}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fasync-exceptions}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fasynchronous-unwind-tables}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fauto-import}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fauto-profile=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fauto-profile-accurate}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fautolink}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fautomatic}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbackslash}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbacktrace}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbasic-block-address-map}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbasic-block-sections=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbfloat16-excess-precision=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbinutils-version=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fblas-matmul-limit=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fblocks-runtime-optional}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbootclasspath=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fborland-extensions}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbounds-check}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fexperimental-bounds-safety}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbracket-depth}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbranch-count-reg}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbuild-session-file=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbuild-session-timestamp=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbuiltin-headers-in-system-modules}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fbuiltin-module-map}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcaller-saves}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcaret-diagnostics}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcf-runtime-abi=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcgl}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fchar8_t}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcheck=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcheck-array-temporaries}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcheck-new}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fclang-abi-compat=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fclangir}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fclasspath=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcoarray=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcomment-block-commands=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcompatibility-qualified-id-block-type-checking}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcompile-resource=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcomplex-arithmetic=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fconst-strings}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fconstant-cfstrings}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fconstant-string-class}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fconstant-string-class=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fconstexpr-backtrace-limit=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fconstexpr-depth=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fconstexpr-steps=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fconvergent-functions}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fconvert=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcoro-aligned-allocation}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcoroutines}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcoverage-prefix-map=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcray-pointer}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcreate-profile}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fctor-dtor-return-this}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcuda-allow-variadic-functions}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcuda-include-gpubinary}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcuda-is-device}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcx-fortran-rules}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcx-limited-range}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fc\+\+-abi=}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcxx-exceptions}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fcxx-modules}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fd-lines-as-code}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fd-lines-as-comments}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fdata-sections}}
// CLOptionCHECK4: {{(unknown argument ignored in clang-cl).*-fdebug-default-version=}}
// RUN: not %clang_cl -fdebug-dump-all -fdebug-dump-parse-tree -fdebug-dump-parse-tree-no-sema -fdebug-dump-parsing-log -fdebug-dump-pft -fdebug-dump-provenance -fdebug-dump-symbols -fdebug-info-for-profiling -fdebug-measure-parse-tree -fdebug-module-writer -fdebug-pass-arguments -fdebug-pass-manager -fdebug-pass-structure -fdebug-pre-fir-tree -fdebug-prefix-map= -fdebug-ranges-base-address -fdebug-types-section -fdebug-unparse -fdebug-unparse-no-sema -fdebug-unparse-with-modules -fdebug-unparse-with-symbols -fdebugger-cast-result-to-id -fdebugger-objc-literal -fdebugger-support -fdeclare-opencl-builtins -fdeclspec -fdefault-calling-conv= -fdefault-double-8 -fdefault-inline -fdefault-integer-8 -fdefault-real-8 -fdefine-target-os-macros -fdenormal-fp-math= -fdenormal-fp-math-f32= -fdepfile-entry= -fdeprecated-macro -fdevirtualize -fdevirtualize-speculatively -fdiagnostics-fixit-info -fdiagnostics-format -fdiagnostics-format= -fdiagnostics-hotness-threshold= -fdiagnostics-misexpect-tolerance= -fdiagnostics-print-source-range-info -fdiagnostics-show-category -fdiagnostics-show-category= -fdiagnostics-show-hotness -fdiagnostics-show-line-numbers -fdiagnostics-show-location= -fdiagnostics-show-note-include-stack -fdiagnostics-show-option -fdiagnostics-show-template-tree -fdigraphs -fdirect-access-external-data -fdirectives-only -fdisable-block-signature-string -fdisable-integer-16 -fdisable-integer-2 -fdisable-module-hash -fdisable-real-10 -fdisable-real-3 -fdiscard-value-names -fdollar-ok -fdollars-in-identifiers -fdouble-square-bracket-attributes -fdump-fortran-optimized -fdump-fortran-original -fdump-parse-tree -fdump-record-layouts -fdump-record-layouts-canonical -fdump-record-layouts-complete -fdump-record-layouts-simple -fdump-vtable-layouts -fdwarf2-cfi-asm -fdwarf-directory-asm -fdwarf-exceptions -felide-constructors -feliminate-unused-debug-symbols -fembed-bitcode -fembed-bitcode= -fembed-bitcode-marker -fembed-offload-object= -femit-all-decls -femit-compact-unwind-non-canonical -femit-dwarf-unwind= -femulated-tls -fenable-matrix -fencode-extended-block-signature -fencoding= -ferror-limit -fescaping-block-tail-calls -fexceptions -fexec-charset= -fexperimental-assignment-tracking= -fexperimental-isel -fexperimental-late-parse-attributes -fexperimental-max-bitint-width= -fexperimental-new-constant-interpreter -fexperimental-omit-vtable-rtti -fexperimental-relative-c++-abi-vtables  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK5 %s

// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-dump-all}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-dump-parse-tree}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-dump-parse-tree-no-sema}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-dump-parsing-log}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-dump-pft}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-dump-provenance}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-dump-symbols}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-info-for-profiling}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-measure-parse-tree}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-module-writer}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-pass-arguments}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-pass-manager}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-pass-structure}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-pre-fir-tree}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-prefix-map=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-ranges-base-address}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-types-section}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-unparse}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-unparse-no-sema}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-unparse-with-modules}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebug-unparse-with-symbols}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebugger-cast-result-to-id}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebugger-objc-literal}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdebugger-support}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdeclare-opencl-builtins}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdeclspec}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdefault-calling-conv=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdefault-double-8}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdefault-inline}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdefault-integer-8}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdefault-real-8}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdefine-target-os-macros}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdenormal-fp-math=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdenormal-fp-math-f32=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdepfile-entry=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdeprecated-macro}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdevirtualize}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdevirtualize-speculatively}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-fixit-info}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-format}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-format=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-hotness-threshold=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-misexpect-tolerance=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-print-source-range-info}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-show-category}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-show-category=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-show-hotness}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-show-line-numbers}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-show-location=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-show-note-include-stack}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-show-option}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiagnostics-show-template-tree}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdigraphs}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdirect-access-external-data}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdirectives-only}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdisable-block-signature-string}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdisable-integer-16}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdisable-integer-2}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdisable-module-hash}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdisable-real-10}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdisable-real-3}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdiscard-value-names}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdollar-ok}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdollars-in-identifiers}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdouble-square-bracket-attributes}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdump-fortran-optimized}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdump-fortran-original}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdump-parse-tree}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdump-record-layouts}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdump-record-layouts-canonical}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdump-record-layouts-complete}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdump-record-layouts-simple}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdump-vtable-layouts}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdwarf2-cfi-asm}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdwarf-directory-asm}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fdwarf-exceptions}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-felide-constructors}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-feliminate-unused-debug-symbols}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fembed-bitcode}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fembed-bitcode=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fembed-bitcode-marker}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fembed-offload-object=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-femit-all-decls}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-femit-compact-unwind-non-canonical}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-femit-dwarf-unwind=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-femulated-tls}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fenable-matrix}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fencode-extended-block-signature}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fencoding=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-ferror-limit}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fescaping-block-tail-calls}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fexceptions}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fexec-charset=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fexperimental-assignment-tracking=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fexperimental-isel}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fexperimental-late-parse-attributes}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fexperimental-max-bitint-width=}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fexperimental-new-constant-interpreter}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fexperimental-omit-vtable-rtti}}
// CLOptionCHECK5: {{(unknown argument ignored in clang-cl).*-fexperimental-relative-c\+\+-abi-vtables}}
// RUN: not %clang_cl -fexperimental-strict-floating-point -fextdirs= -fextend-arguments= -fextend-variable-liveness -fextend-variable-liveness= -fexternal-blas -fexternc-nounwind -ff2c -ffake-address-space-map -ffast-math -ffat-lto-objects -ffile-prefix-map= -fimplicit-modules-use-lock -ffine-grained-bitfield-accesses -ffinite-loops -ffinite-math-only -finline-limit -ffixed-form -ffixed-line-length= -ffixed-line-length- -ffixed-point -ffixed-r19 -ffloat16-excess-precision= -ffloat-store -ffor-scope -fforbid-guard-variables -fforce-check-cxx20-modules-input-files -fforce-dwarf-frame -fforce-enable-int128 -ffp-contract= -ffp-eval-method= -ffp-exception-behavior= -ffp-model= -ffpe-trap= -ffree-form -ffree-line-length- -ffreestanding -ffriend-injection -ffrontend-optimize -ffunction-attribute-list -ffunction-sections -fgcse -fgcse-after-reload -fgcse-las -fgcse-sm -fget-definition -fget-symbols-sources -fglobal-isel -fgnu -fgnu89-inline -fgnu-inline-asm -fgnu-keywords -fgnu-runtime -fgpu-approx-transcendentals -fhalf-no-semantic-interposition -fheinous-gnu-extensions -fhermetic-module-files -fhlsl-strict-availability -fhonor-infinities -fhonor-nans -fhosted -fignore-exceptions -filelist -filetype -fimplement-inlines -fimplicit-none -fimplicit-none-ext -fimplicit-templates -finclude-default-header -fincremental-extensions -finit-character= -finit-global-zero -finit-integer= -finit-local-zero -finit-logical= -finit-real= -finline -finline-functions -finline-functions-called-once -finline-hint-functions -finline-limit= -finline-small-functions -finput-charset= -finstrument-function-entry-bare -finstrument-functions -finstrument-functions-after-inlining -finteger-4-integer-8 -fintegrated-as -fintrinsic-modules-path -fipa-cp -fivopts -fix-only-warnings -fix-what-you-can -fixit -fixit= -fixit-recompile -fixit-to-temporary -fjmc -fjump-tables -fkeep-persistent-storage-variables  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK6 %s

// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fexperimental-strict-floating-point}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fextdirs=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fextend-arguments=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fextend-variable-liveness}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fextend-variable-liveness=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fexternal-blas}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fexternc-nounwind}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ff2c}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffake-address-space-map}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffast-math}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffat-lto-objects}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffile-prefix-map=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fimplicit-modules-use-lock}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffine-grained-bitfield-accesses}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffinite-loops}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffinite-math-only}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finline-limit}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffixed-form}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffixed-line-length=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffixed-line-length-}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffixed-point}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffixed-r19}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffloat16-excess-precision=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffloat-store}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffor-scope}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fforbid-guard-variables}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fforce-check-cxx20-modules-input-files}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fforce-dwarf-frame}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fforce-enable-int128}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffp-contract=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffp-eval-method=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffp-exception-behavior=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffp-model=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffpe-trap=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffree-form}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffree-line-length-}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffreestanding}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffriend-injection}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffrontend-optimize}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffunction-attribute-list}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-ffunction-sections}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fgcse}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fgcse-after-reload}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fgcse-las}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fgcse-sm}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fget-definition}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fget-symbols-sources}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fglobal-isel}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fgnu}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fgnu89-inline}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fgnu-inline-asm}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fgnu-keywords}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fgnu-runtime}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fgpu-approx-transcendentals}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fhalf-no-semantic-interposition}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fheinous-gnu-extensions}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fhermetic-module-files}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fhlsl-strict-availability}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fhonor-infinities}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fhonor-nans}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fhosted}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fignore-exceptions}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-filelist}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-filetype}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fimplement-inlines}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fimplicit-none}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fimplicit-none-ext}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fimplicit-templates}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finclude-default-header}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fincremental-extensions}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finit-character=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finit-global-zero}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finit-integer=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finit-local-zero}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finit-logical=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finit-real=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finline}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finline-functions}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finline-functions-called-once}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finline-hint-functions}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finline-limit=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finline-small-functions}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finput-charset=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finstrument-function-entry-bare}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finstrument-functions}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finstrument-functions-after-inlining}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-finteger-4-integer-8}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fintegrated-as}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fintrinsic-modules-path}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fipa-cp}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fivopts}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fix-only-warnings}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fix-what-you-can}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fixit}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fixit=}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fixit-recompile}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fixit-to-temporary}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fjmc}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fjump-tables}}
// CLOptionCHECK6: {{(unknown argument ignored in clang-cl).*-fkeep-persistent-storage-variables}}
// RUN: not %clang_cl -fkeep-static-consts -fkeep-system-includes -flang-deprecated-no-hlfir -flang-experimental-hlfir -flarge-sizes -flat_namespace -flax-vector-conversions -flax-vector-conversions= -flimited-precision= -flogical-abbreviations -floop-interchange -fversion-loops-for-stride -flto-jobs= -flto-unit -flto-visibility-public-std -fmacro-prefix-map= -fmath-errno -fmax-array-constructor= -fmax-errors= -fmax-identifier-length -fmax-stack-var-size= -fmax-subrecord-length= -fmax-tokens= -fmax-type-align= -fmcdc-max-conditions= -fmcdc-max-test-vectors= -fmemory-profile -fmemory-profile= -fmerge-constants -fmerge-functions -fmessage-length= -fminimize-whitespace -fmodule-feature -fmodule-file-deps -fmodule-file-home-is-cwd -fmodule-format= -fmodule-map-file-home-is-cwd -fmodule-private -fmodulemap-allow-subdirectory-search -fmodules-cache-path= -fmodules-codegen -fmodules-debuginfo -fmodules-disable-diagnostic-validation -fmodules-embed-file= -fmodules-hash-content -fmodules-local-submodule-visibility -fmodules-prune-after= -fmodules-prune-interval= -fmodules-skip-diagnostic-options -fmodules-skip-header-search-paths -fmodules-strict-context-hash -fmodules-user-build-path -fmodules-validate-input-files-content -fmodules-validate-once-per-build-session -fmodules-validate-system-headers -fmodulo-sched -fmodulo-sched-allow-regmoves -fms-kernel -fms-memptr-rep= -fms-tls-guards -fms-volatile -fmudflap -fmudflapth -fmultilib-flag= -fnative-half-arguments-and-returns -fnative-half-type -fnested-functions -fnew-alignment= -fnew-infallible -fnext-runtime -fno-PIC -fno-PIE -fno-aarch64-jump-table-hardening -fno-access-control -fno-aggressive-function-elimination -fno-align-commons -fno-align-functions -fno-align-jumps -fno-align-labels -fno-align-loops -fno-aligned-allocation -fno-all-intrinsics -fno-allow-editor-placeholders -fno-altivec -fno-analyzed-objects-for-unparse -fno-android-pad-segment -fno-keep-inline-functions -fno-unit-at-a-time -fno-apinotes -fno-apinotes-modules -fno-apple-pragma-pack -fno-application-extension -fno-approx-func -fno-asm -fno-asm-blocks -fno-associative-math -fno-assume-nothrow-exception-dtor -fno-assume-sane-operator-new -fno-assumptions -fno-async-exceptions  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK7 %s

// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fkeep-static-consts}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fkeep-system-includes}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-flang-deprecated-no-hlfir}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-flang-experimental-hlfir}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-flarge-sizes}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-flat_namespace}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-flax-vector-conversions}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-flax-vector-conversions=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-flimited-precision=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-flogical-abbreviations}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-floop-interchange}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fversion-loops-for-stride}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-flto-jobs=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-flto-unit}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-flto-visibility-public-std}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmacro-prefix-map=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmath-errno}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmax-array-constructor=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmax-errors=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmax-identifier-length}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmax-stack-var-size=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmax-subrecord-length=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmax-tokens=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmax-type-align=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmcdc-max-conditions=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmcdc-max-test-vectors=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmemory-profile}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmemory-profile=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmerge-constants}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmerge-functions}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmessage-length=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fminimize-whitespace}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodule-feature}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodule-file-deps}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodule-file-home-is-cwd}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodule-format=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodule-map-file-home-is-cwd}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodule-private}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodulemap-allow-subdirectory-search}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-cache-path=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-codegen}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-debuginfo}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-disable-diagnostic-validation}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-embed-file=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-hash-content}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-local-submodule-visibility}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-prune-after=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-prune-interval=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-skip-diagnostic-options}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-skip-header-search-paths}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-strict-context-hash}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-user-build-path}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-validate-input-files-content}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-validate-once-per-build-session}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodules-validate-system-headers}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodulo-sched}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmodulo-sched-allow-regmoves}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fms-kernel}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fms-memptr-rep=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fms-tls-guards}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fms-volatile}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmudflap}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmudflapth}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fmultilib-flag=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fnative-half-arguments-and-returns}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fnative-half-type}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fnested-functions}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fnew-alignment=}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fnew-infallible}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fnext-runtime}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-PIC}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-PIE}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-aarch64-jump-table-hardening}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-access-control}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-aggressive-function-elimination}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-align-commons}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-align-functions}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-align-jumps}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-align-labels}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-align-loops}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-aligned-allocation}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-all-intrinsics}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-allow-editor-placeholders}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-altivec}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-analyzed-objects-for-unparse}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-android-pad-segment}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-keep-inline-functions}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-unit-at-a-time}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-apinotes}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-apinotes-modules}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-apple-pragma-pack}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-application-extension}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-approx-func}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-asm}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-asm-blocks}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-associative-math}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-assume-nothrow-exception-dtor}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-assume-sane-operator-new}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-assumptions}}
// CLOptionCHECK7: {{(unknown argument ignored in clang-cl).*-fno-async-exceptions}}
// RUN: not %clang_cl -fno-asynchronous-unwind-tables -fno-auto-import -fno-auto-profile -fno-auto-profile-accurate -fno-autolink -fno-automatic -fno-backslash -fno-backtrace -fno-basic-block-address-map -fno-bitfield-type-align -fno-borland-extensions -fno-bounds-check -fno-experimental-bounds-safety -fno-branch-count-reg -fno-caller-saves -fno-caret-diagnostics -fno-char8_t -fno-check-array-temporaries -fno-check-new -fno-clangir -fno-common -fno-const-strings -fno-constant-cfstrings -fno-convergent-functions -fno-coro-aligned-allocation -fno-coroutines -fno-cray-pointer -fno-cuda-host-device-constexpr -fno-cx-fortran-rules -fno-cx-limited-range -fno-cxx-exceptions -fno-cxx-modules -fno-d-lines-as-code -fno-d-lines-as-comments -fno-data-sections -fno-debug-info-for-profiling -fno-debug-pass-manager -fno-debug-ranges-base-address -fno-debug-types-section -fno-declspec -fno-default-inline -fno-define-target-os-macros -fno-deprecated-macro -fno-devirtualize -fno-devirtualize-speculatively -fno-diagnostics-fixit-info -fno-diagnostics-show-hotness -fno-diagnostics-show-line-numbers -fno-diagnostics-show-note-include-stack -fno-diagnostics-show-option -fno-diagnostics-use-presumed-location -fno-digraphs -fno-direct-access-external-data -fno-directives-only -fno-disable-block-signature-string -fno-discard-value-names -fno-dllexport-inlines -fno-dollar-ok -fno-dollars-in-identifiers -fno-double-square-bracket-attributes -fno-dump-fortran-optimized -fno-dump-fortran-original -fno-dump-parse-tree -fno-dwarf2-cfi-asm -fno-dwarf-directory-asm -fno-elide-constructors -fno-elide-type -fno-eliminate-unused-debug-symbols -fno-emit-compact-unwind-non-canonical -fno-emulated-tls -fno-escaping-block-tail-calls -fno-exceptions -fno-experimental-isel -fno-experimental-late-parse-attributes -fno-experimental-library -fno-experimental-omit-vtable-rtti -fno-experimental-relative-c++-abi-vtables -fno-external-blas -fno-f2c -fno-fast-math -fno-fat-lto-objects -fno-implicit-modules-use-lock -fno-fine-grained-bitfield-accesses -fno-finite-loops -fno-finite-math-only -fno-inline-limit -fno-fixed-point -fno-float-store -fno-for-scope -fno-force-dwarf-frame -fno-force-enable-int128 -fno-friend-injection -fno-frontend-optimize -fno-function-attribute-list -fno-function-sections -fno-gcse -fno-gcse-after-reload -fno-gcse-las -fno-gcse-sm -fno-global-isel  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK8 %s

// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-asynchronous-unwind-tables}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-auto-import}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-auto-profile}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-auto-profile-accurate}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-autolink}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-automatic}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-backslash}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-backtrace}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-basic-block-address-map}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-bitfield-type-align}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-borland-extensions}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-bounds-check}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-experimental-bounds-safety}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-branch-count-reg}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-caller-saves}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-caret-diagnostics}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-char8_t}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-check-array-temporaries}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-check-new}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-clangir}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-common}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-const-strings}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-constant-cfstrings}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-convergent-functions}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-coro-aligned-allocation}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-coroutines}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-cray-pointer}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-cuda-host-device-constexpr}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-cx-fortran-rules}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-cx-limited-range}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-cxx-exceptions}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-cxx-modules}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-d-lines-as-code}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-d-lines-as-comments}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-data-sections}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-debug-info-for-profiling}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-debug-pass-manager}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-debug-ranges-base-address}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-debug-types-section}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-declspec}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-default-inline}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-define-target-os-macros}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-deprecated-macro}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-devirtualize}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-devirtualize-speculatively}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-diagnostics-fixit-info}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-diagnostics-show-hotness}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-diagnostics-show-line-numbers}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-diagnostics-show-note-include-stack}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-diagnostics-show-option}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-diagnostics-use-presumed-location}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-digraphs}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-direct-access-external-data}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-directives-only}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-disable-block-signature-string}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-discard-value-names}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-dllexport-inlines}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-dollar-ok}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-dollars-in-identifiers}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-double-square-bracket-attributes}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-dump-fortran-optimized}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-dump-fortran-original}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-dump-parse-tree}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-dwarf2-cfi-asm}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-dwarf-directory-asm}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-elide-constructors}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-elide-type}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-eliminate-unused-debug-symbols}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-emit-compact-unwind-non-canonical}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-emulated-tls}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-escaping-block-tail-calls}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-exceptions}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-experimental-isel}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-experimental-late-parse-attributes}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-experimental-library}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-experimental-omit-vtable-rtti}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-experimental-relative-c\+\+-abi-vtables}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-external-blas}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-f2c}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-fast-math}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-fat-lto-objects}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-implicit-modules-use-lock}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-fine-grained-bitfield-accesses}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-finite-loops}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-finite-math-only}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-inline-limit}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-fixed-point}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-float-store}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-for-scope}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-force-dwarf-frame}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-force-enable-int128}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-friend-injection}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-frontend-optimize}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-function-attribute-list}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-function-sections}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-gcse}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-gcse-after-reload}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-gcse-las}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-gcse-sm}}
// CLOptionCHECK8: {{(unknown argument ignored in clang-cl).*-fno-global-isel}}
// RUN: not %clang_cl -fno-gnu -fno-gnu89-inline -fno-gnu-inline-asm -fno-gnu-keywords -fno-gpu-approx-transcendentals -fno-honor-infinities -fno-honor-nans -fno-implement-inlines -fno-implicit-module-maps -fno-implicit-none -fno-implicit-none-ext -fno-implicit-templates -fno-init-global-zero -fno-init-local-zero -fno-inline -fno-inline-functions -fno-inline-functions-called-once -fno-inline-small-functions -fno-integer-4-integer-8 -fno-integrated-as -fno-ipa-cp -fno-ivopts -fno-jmc -fno-jump-tables -fno-keep-persistent-storage-variables -fno-keep-static-consts -fno-keep-system-includes -fno-lax-vector-conversions -fno-logical-abbreviations -fno-loop-interchange -fno-version-loops-for-stride -fno-lto-unit -fno-math-builtin -fno-math-errno -fno-max-identifier-length -fno-max-type-align -fno-memory-profile -fno-merge-all-constants -fno-merge-constants -fno-minimize-whitespace -fno-module-file-deps -fno-module-maps -fno-module-private -fno-modulemap-allow-subdirectory-search -fno-modules-check-relocated -fno-modules-error-recovery -fno-modules-global-index -fno-modules-prune-non-affecting-module-map-files -fno-modules-share-filemanager -fno-modules-skip-diagnostic-options -fno-modules-skip-header-search-paths -fno-strict-modules-decluse -fno_modules-validate-input-files-content -fno-modules-validate-system-headers -fno-modules-validate-textual-header-includes -fno-modulo-sched -fno-modulo-sched-allow-regmoves -fno-ms-tls-guards -fno-ms-volatile -fno-new-infallible -fno-non-call-exceptions -fno-objc-arc -fno-objc-arc-exceptions -fno-objc-avoid-heapify-local-blocks -fno-objc-convert-messages-to-runtime-calls -fno-objc-encode-cxx-class-template-spec -fno-objc-exceptions -fno-objc-infer-related-result-type -fno-objc-legacy-dispatch -fno-objc-nonfragile-abi -fno-objc-weak -fno-offload-uniform-block -fno-omit-frame-pointer -fno-openmp -fno-openmp-assume-teams-oversubscription -fno-openmp-assume-threads-oversubscription -fno-openmp-cuda-mode -fno-openmp-extensions -fno-openmp-new-driver -fno-openmp-optimistic-collapse -fno-openmp-simd -fno-openmp-target-debug -fno-openmp-target-new-runtime -fno-operator-names -fno-optimize-sibling-calls -fno-pack-derived -fno-pack-struct -fno-padding-on-unsigned-fixed-point -fno-pascal-strings -fno-pch-codegen -fno-pch-debuginfo -fno-pch-timestamp -fno_pch-validate-input-files-content -fno-peel-loops -fno-permissive -fno-pic -fno-pie -fno-plt -fno-pointer-tbaa -fno-ppc-native-vector-element-order  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK9 %s

// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-gnu}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-gnu89-inline}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-gnu-inline-asm}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-gnu-keywords}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-gpu-approx-transcendentals}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-honor-infinities}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-honor-nans}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-implement-inlines}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-implicit-module-maps}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-implicit-none}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-implicit-none-ext}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-implicit-templates}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-init-global-zero}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-init-local-zero}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-inline}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-inline-functions}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-inline-functions-called-once}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-inline-small-functions}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-integer-4-integer-8}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-integrated-as}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-ipa-cp}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-ivopts}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-jmc}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-jump-tables}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-keep-persistent-storage-variables}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-keep-static-consts}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-keep-system-includes}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-lax-vector-conversions}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-logical-abbreviations}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-loop-interchange}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-version-loops-for-stride}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-lto-unit}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-math-builtin}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-math-errno}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-max-identifier-length}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-max-type-align}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-memory-profile}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-merge-all-constants}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-merge-constants}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-minimize-whitespace}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-module-file-deps}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-module-maps}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-module-private}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-modulemap-allow-subdirectory-search}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-modules-check-relocated}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-modules-error-recovery}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-modules-global-index}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-modules-prune-non-affecting-module-map-files}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-modules-share-filemanager}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-modules-skip-diagnostic-options}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-modules-skip-header-search-paths}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-strict-modules-decluse}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno_modules-validate-input-files-content}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-modules-validate-system-headers}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-modules-validate-textual-header-includes}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-modulo-sched}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-modulo-sched-allow-regmoves}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-ms-tls-guards}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-ms-volatile}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-new-infallible}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-non-call-exceptions}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-objc-arc}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-objc-arc-exceptions}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-objc-avoid-heapify-local-blocks}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-objc-convert-messages-to-runtime-calls}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-objc-encode-cxx-class-template-spec}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-objc-exceptions}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-objc-infer-related-result-type}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-objc-legacy-dispatch}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-objc-nonfragile-abi}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-objc-weak}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-offload-uniform-block}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-omit-frame-pointer}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-openmp}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-openmp-assume-teams-oversubscription}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-openmp-assume-threads-oversubscription}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-openmp-cuda-mode}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-openmp-extensions}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-openmp-new-driver}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-openmp-optimistic-collapse}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-openmp-simd}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-openmp-target-debug}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-openmp-target-new-runtime}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-operator-names}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-optimize-sibling-calls}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-pack-derived}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-pack-struct}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-padding-on-unsigned-fixed-point}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-pascal-strings}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-pch-codegen}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-pch-debuginfo}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-pch-timestamp}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno_pch-validate-input-files-content}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-peel-loops}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-permissive}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-pic}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-pie}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-plt}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-pointer-tbaa}}
// CLOptionCHECK9: {{(unknown argument ignored in clang-cl).*-fno-ppc-native-vector-element-order}}
// RUN: not %clang_cl -fno-prebuilt-implicit-modules -fno-prefetch-loop-arrays -fno-preserve-as-comments -fno-printf -fno-profile -fno-profile-arcs -fno-profile-correction -fno-profile-generate-sampling -fno-profile-reusedist -fno-profile-sample-accurate -fno-profile-use -fno-profile-values -fno-protect-parens -fno-pseudo-probe-for-profiling -fno-ptrauth-auth-traps -fno-ptrauth-calls -fno-ptrauth-elf-got -fno-ptrauth-function-pointer-type-discrimination -fno-ptrauth-indirect-gotos -fno-ptrauth-init-fini -fno-ptrauth-init-fini-address-discrimination -fno-ptrauth-intrinsics -fno-ptrauth-returns -fno-ptrauth-type-info-vtable-pointer-discrimination -fno-ptrauth-vtable-pointer-address-discrimination -fno-ptrauth-vtable-pointer-type-discrimination -fno-range-check -fno-raw-string-literals -fno-real-4-real-10 -fno-real-4-real-16 -fno-real-4-real-8 -fno-real-8-real-10 -fno-real-8-real-16 -fno-real-8-real-4 -fno-realloc-lhs -fno-reciprocal-math -fno-record-command-line -fno-recovery-ast -fno-recovery-ast-type -fno-recursive -fno-reformat -fno-register-global-dtors-with-atexit -fno-regs-graph -fno-rename-registers -fno-reorder-blocks -fno-repack-arrays -fno-retain-subst-template-type-parm-type-ast-nodes -fno-rewrite-imports -fno-rewrite-includes -fno-ripa -fno-ropi -fno-rounding-math -fno-rtlib-add-rpath -fno-rtti -fno-rtti-data -fno-rwpi -fno-safe-buffer-usage-suggestions -fno-save-main-program -fno-save-optimization-record -fno-schedule-insns -fno-schedule-insns2 -fno-second-underscore -fno-see -fno-semantic-interposition -fno-separate-named-sections -fno-short-enums -fno-short-wchar -fno-show-column -fno-show-source-location -fno-sign-zero -fno-signaling-math -fno-signaling-nans -fno-signed-char -fno-signed-wchar -fno-signed-zeros -fno-single-precision-constant -fno-sized-deallocation -fno-skip-odr-check-in-gmf -fno-slp-vectorize -fno-spec-constr-count -fno-spell-checking -fno-split-dwarf-inlining -fno-split-machine-functions -fno-split-stack -fno-stack-arrays -fno-stack-check -fno-stack-clash-protection -fno-stack-protector -fno-stack-size-section -fno-strength-reduce -fno-strict-enums -fno-strict-float-cast-overflow -fno-strict-return -fno-strict-vtable-pointers -fno-struct-path-tbaa -fno-test-coverage -fno-threadsafe-statics -fno-tls-model -fno-tracer -fno-trapping-math  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK10 %s

// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-prebuilt-implicit-modules}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-prefetch-loop-arrays}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-preserve-as-comments}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-printf}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-profile}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-profile-arcs}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-profile-correction}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-profile-generate-sampling}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-profile-reusedist}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-profile-sample-accurate}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-profile-use}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-profile-values}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-protect-parens}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-pseudo-probe-for-profiling}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ptrauth-auth-traps}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ptrauth-calls}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ptrauth-elf-got}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ptrauth-function-pointer-type-discrimination}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ptrauth-indirect-gotos}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ptrauth-init-fini}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ptrauth-init-fini-address-discrimination}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ptrauth-intrinsics}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ptrauth-returns}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ptrauth-type-info-vtable-pointer-discrimination}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ptrauth-vtable-pointer-address-discrimination}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ptrauth-vtable-pointer-type-discrimination}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-range-check}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-raw-string-literals}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-real-4-real-10}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-real-4-real-16}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-real-4-real-8}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-real-8-real-10}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-real-8-real-16}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-real-8-real-4}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-realloc-lhs}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-reciprocal-math}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-record-command-line}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-recovery-ast}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-recovery-ast-type}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-recursive}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-reformat}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-register-global-dtors-with-atexit}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-regs-graph}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-rename-registers}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-reorder-blocks}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-repack-arrays}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-retain-subst-template-type-parm-type-ast-nodes}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-rewrite-imports}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-rewrite-includes}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ripa}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-ropi}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-rounding-math}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-rtlib-add-rpath}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-rtti}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-rtti-data}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-rwpi}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-safe-buffer-usage-suggestions}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-save-main-program}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-save-optimization-record}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-schedule-insns}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-schedule-insns2}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-second-underscore}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-see}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-semantic-interposition}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-separate-named-sections}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-short-enums}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-short-wchar}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-show-column}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-show-source-location}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-sign-zero}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-signaling-math}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-signaling-nans}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-signed-char}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-signed-wchar}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-signed-zeros}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-single-precision-constant}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-sized-deallocation}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-skip-odr-check-in-gmf}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-slp-vectorize}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-spec-constr-count}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-spell-checking}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-split-dwarf-inlining}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-split-machine-functions}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-split-stack}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-stack-arrays}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-stack-check}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-stack-clash-protection}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-stack-protector}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-stack-size-section}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-strength-reduce}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-strict-enums}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-strict-float-cast-overflow}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-strict-return}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-strict-vtable-pointers}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-struct-path-tbaa}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-test-coverage}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-threadsafe-statics}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-tls-model}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-tracer}}
// CLOptionCHECK10: {{(unknown argument ignored in clang-cl).*-fno-trapping-math}}
// RUN: not %clang_cl -fno-tree-dce -fno-tree-salias -fno-tree-ter -fno-tree-vectorizer-verbose -fno-tree-vrp -fno-trigraphs -fno-underscoring -fno-unified-lto -fno-unique-basic-block-section-names -fno-unique-internal-linkage-names -fno-unique-section-names -fno-unroll-all-loops -fno-unroll-loops -fno-unsafe-loop-optimizations -fno-unsafe-math-optimizations -fno-unsigned -fno-unsigned-char -fno-unswitch-loops -fno-unwind-tables -fno-use-ctor-homing -fno-use-cxa-atexit -fno-use-init-array -fno-use-line-directives -fno-use-linker-plugin -fno-validate-pch -fno-var-tracking -fno-variable-expansion-in-unroller -fno-vect-cost-model -fno-vectorize -fno-verbose-asm -fno-visibility-from-dllstorageclass -fno-visibility-inlines-hidden -fno-visibility-inlines-hidden-static-local-var -fno-wchar -fno-web -fno-whole-file -fno-whole-program -fno-working-directory -fno-xl-pragma-pack -fno-xor-operator -fno-xray-always-emit-customevents -fno-xray-always-emit-typedevents -fno-xray-function-index -fno-xray-ignore-loops -fno-xray-instrument -fno-xray-link-deps -fno-xray-shared -fno-zero-initialized-in-bss -fno-zos-extensions -fno-zvector -fnon-call-exceptions -fnoopenmp-relocatable-target -fnoopenmp-use-tls -fobjc-abi-version= -fobjc-arc -fobjc-arc-cxxlib= -fobjc-arc-exceptions -fobjc-atdefs -fobjc-avoid-heapify-local-blocks -fobjc-call-cxx-cdtors -fobjc-convert-messages-to-runtime-calls -fobjc-disable-direct-methods-for-testing -fobjc-dispatch-method= -fobjc-encode-cxx-class-template-spec -fobjc-exceptions -fobjc-gc -fobjc-gc-only -fobjc-infer-related-result-type -fobjc-legacy-dispatch -fobjc-link-runtime -fobjc-new-property -fobjc-nonfragile-abi -fobjc-nonfragile-abi-version= -fobjc-runtime-has-weak -fobjc-sender-dependent-dispatch -fobjc-subscripting-legacy-runtime -fobjc-weak -foffload-uniform-block -fomit-frame-pointer -fopenacc -fopenmp -fopenmp= -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state -fopenmp-assume-teams-oversubscription -fopenmp-assume-threads-oversubscription -fopenmp-cuda-blocks-per-sm= -fopenmp-cuda-mode -fopenmp-cuda-number-of-sm= -fopenmp-cuda-teams-reduction-recs-num= -fopenmp-enable-irbuilder -fopenmp-extensions -fopenmp-force-usm -fopenmp-host-ir-file-path -fopenmp-is-target-device -fopenmp-new-driver -fopenmp-offload-mandatory -fopenmp-optimistic-collapse -fopenmp-relocatable-target -fopenmp-simd  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK11 %s

// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-tree-dce}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-tree-salias}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-tree-ter}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-tree-vectorizer-verbose}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-tree-vrp}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-trigraphs}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-underscoring}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-unified-lto}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-unique-basic-block-section-names}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-unique-internal-linkage-names}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-unique-section-names}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-unroll-all-loops}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-unroll-loops}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-unsafe-loop-optimizations}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-unsafe-math-optimizations}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-unsigned}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-unsigned-char}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-unswitch-loops}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-unwind-tables}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-use-ctor-homing}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-use-cxa-atexit}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-use-init-array}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-use-line-directives}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-use-linker-plugin}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-validate-pch}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-var-tracking}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-variable-expansion-in-unroller}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-vect-cost-model}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-vectorize}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-verbose-asm}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-visibility-from-dllstorageclass}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-visibility-inlines-hidden}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-visibility-inlines-hidden-static-local-var}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-wchar}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-web}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-whole-file}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-whole-program}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-working-directory}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-xl-pragma-pack}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-xor-operator}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-xray-always-emit-customevents}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-xray-always-emit-typedevents}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-xray-function-index}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-xray-ignore-loops}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-xray-instrument}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-xray-link-deps}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-xray-shared}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-zero-initialized-in-bss}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-zos-extensions}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fno-zvector}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fnon-call-exceptions}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fnoopenmp-relocatable-target}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fnoopenmp-use-tls}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-abi-version=}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-arc}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-arc-cxxlib=}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-arc-exceptions}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-atdefs}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-avoid-heapify-local-blocks}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-call-cxx-cdtors}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-convert-messages-to-runtime-calls}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-disable-direct-methods-for-testing}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-dispatch-method=}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-encode-cxx-class-template-spec}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-exceptions}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-gc}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-gc-only}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-infer-related-result-type}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-legacy-dispatch}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-link-runtime}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-new-property}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-nonfragile-abi}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-nonfragile-abi-version=}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-runtime-has-weak}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-sender-dependent-dispatch}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-subscripting-legacy-runtime}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fobjc-weak}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-foffload-uniform-block}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fomit-frame-pointer}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenacc}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp=}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-assume-no-nested-parallelism}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-assume-no-thread-state}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-assume-teams-oversubscription}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-assume-threads-oversubscription}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-cuda-blocks-per-sm=}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-cuda-mode}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-cuda-number-of-sm=}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-cuda-teams-reduction-recs-num=}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-enable-irbuilder}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-extensions}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-force-usm}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-host-ir-file-path}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-is-target-device}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-new-driver}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-offload-mandatory}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-optimistic-collapse}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-relocatable-target}}
// CLOptionCHECK11: {{(unknown argument ignored in clang-cl).*-fopenmp-simd}}
// RUN: not %clang_cl -fopenmp-target-debug -fopenmp-target-debug= -fopenmp-target-new-runtime -fopenmp-targets= -fopenmp-use-tls -fopenmp-version= -foperator-arrow-depth= -foperator-names -foptimization-record-file= -foptimization-record-passes= -foptimize-sibling-calls -force_cpusubtype_ALL -force_flat_namespace -force_load -fforce-addr -foutput-class-dir= -foverride-record-layout= -fpack-derived -fpack-struct -fpack-struct= -fpadding-on-unsigned-fixed-point -fparse-all-comments -fpascal-strings -fpass-by-value-is-noalias -fpass-plugin= -fpatchable-function-entry= -fpatchable-function-entry-offset= -fpcc-struct-return -fpch-codegen -fpch-debuginfo -fpch-preprocess -fpch-validate-input-files-content -fpeel-loops -fpermissive -fpic -fpie -fplt -fplugin= -fplugin-arg- -fpointer-tbaa -fppc-native-vector-element-order -fprebuilt-implicit-modules -fprefetch-loop-arrays -fpreprocess-include-lines -fpreserve-as-comments -fprintf -fproc-stat-report -fproc-stat-report= -fprofile -fprofile-arcs -fprofile-continuous -fprofile-correction -fprofile-dir= -fprofile-function-groups= -fprofile-generate-sampling -fprofile-instrument= -fprofile-instrument-path= -fprofile-instrument-use-path= -fprofile-reusedist -fprofile-sample-accurate -fprofile-selected-function-group= -fprofile-values -fpseudo-probe-for-profiling -fptrauth-auth-traps -fptrauth-calls -fptrauth-elf-got -fptrauth-function-pointer-type-discrimination -fptrauth-indirect-gotos -fptrauth-init-fini -fptrauth-init-fini-address-discrimination -fptrauth-intrinsics -fptrauth-returns -fptrauth-type-info-vtable-pointer-discrimination -fptrauth-vtable-pointer-address-discrimination -fptrauth-vtable-pointer-type-discrimination -framework -frandom-seed= -frandomize-layout-seed= -frandomize-layout-seed-file= -frange-check -fraw-string-literals -freal-4-real-10 -freal-4-real-16 -freal-4-real-8 -freal-8-real-10 -freal-8-real-16 -freal-8-real-4 -frealloc-lhs -freciprocal-math -frecord-command-line -frecord-marker= -frecovery-ast -frecovery-ast-type -frecursive -freg-struct-return -fregister-global-dtors-with-atexit -fregs-graph -frename-registers -freorder-blocks -frepack-arrays  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK12 %s

// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fopenmp-target-debug}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fopenmp-target-debug=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fopenmp-target-new-runtime}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fopenmp-targets=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fopenmp-use-tls}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fopenmp-version=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-foperator-arrow-depth=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-foperator-names}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-foptimization-record-file=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-foptimization-record-passes=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-foptimize-sibling-calls}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-force_cpusubtype_ALL}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-force_flat_namespace}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-force_load}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fforce-addr}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-foutput-class-dir=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-foverride-record-layout=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpack-derived}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpack-struct}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpack-struct=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpadding-on-unsigned-fixed-point}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fparse-all-comments}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpascal-strings}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpass-by-value-is-noalias}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpass-plugin=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpatchable-function-entry=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpatchable-function-entry-offset=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpcc-struct-return}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpch-codegen}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpch-debuginfo}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpch-preprocess}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpch-validate-input-files-content}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpeel-loops}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpermissive}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpic}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpie}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fplt}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fplugin=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fplugin-arg-}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpointer-tbaa}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fppc-native-vector-element-order}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprebuilt-implicit-modules}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprefetch-loop-arrays}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpreprocess-include-lines}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpreserve-as-comments}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprintf}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fproc-stat-report}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fproc-stat-report=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-arcs}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-continuous}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-correction}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-dir=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-function-groups=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-generate-sampling}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-instrument=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-instrument-path=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-instrument-use-path=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-reusedist}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-sample-accurate}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-selected-function-group=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fprofile-values}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fpseudo-probe-for-profiling}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fptrauth-auth-traps}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fptrauth-calls}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fptrauth-elf-got}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fptrauth-function-pointer-type-discrimination}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fptrauth-indirect-gotos}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fptrauth-init-fini}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fptrauth-init-fini-address-discrimination}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fptrauth-intrinsics}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fptrauth-returns}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fptrauth-type-info-vtable-pointer-discrimination}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fptrauth-vtable-pointer-address-discrimination}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fptrauth-vtable-pointer-type-discrimination}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-framework}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-frandom-seed=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-frandomize-layout-seed=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-frandomize-layout-seed-file=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-frange-check}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fraw-string-literals}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-freal-4-real-10}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-freal-4-real-16}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-freal-4-real-8}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-freal-8-real-10}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-freal-8-real-16}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-freal-8-real-4}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-frealloc-lhs}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-freciprocal-math}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-frecord-command-line}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-frecord-marker=}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-frecovery-ast}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-frecovery-ast-type}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-frecursive}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-freg-struct-return}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fregister-global-dtors-with-atexit}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-fregs-graph}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-frename-registers}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-freorder-blocks}}
// CLOptionCHECK12: {{(unknown argument ignored in clang-cl).*-frepack-arrays}}
// RUN: not %clang_cl -fretain-comments-from-system-headers -fretain-subst-template-type-parm-type-ast-nodes -frewrite-imports -frewrite-includes -fripa -fropi -frounding-math -frtlib-add-rpath -frtti -frtti-data -frwpi -fsafe-buffer-usage-suggestions -fsample-profile-use-profi -fsanitize-coverage-8bit-counters -fsanitize-coverage-control-flow -fsanitize-coverage-indirect-calls -fsanitize-coverage-inline-8bit-counters -fsanitize-coverage-inline-bool-flag -fsanitize-coverage-no-prune -fsanitize-coverage-pc-table -fsanitize-coverage-stack-depth -fsanitize-coverage-trace-bb -fsanitize-coverage-trace-cmp -fsanitize-coverage-trace-div -fsanitize-coverage-trace-gep -fsanitize-coverage-trace-loads -fsanitize-coverage-trace-pc -fsanitize-coverage-trace-pc-guard -fsanitize-coverage-trace-stores -fsanitize-coverage-type= -fsave-main-program -fsave-optimization-record -fsave-optimization-record= -fschedule-insns -fschedule-insns2 -fsecond-underscore -fsee -fseh-exceptions -fsemantic-interposition -fseparate-named-sections -fshort-enums -fshort-wchar -fshow-column -fshow-overloads= -fshow-skipped-includes -fshow-source-location -fsign-zero -fsignaling-math -fsignaling-nans -fsigned-bitfields -fsigned-char -fsigned-wchar -fsigned-zeros -fsingle-precision-constant -fsized-deallocation -fsjlj-exceptions -fskip-odr-check-in-gmf -fslp-vectorize -fspec-constr-count -fspell-checking -fspell-checking-limit= -fsplit-dwarf-inlining -fsplit-machine-functions -fsplit-stack -fspv-target-env= -fstack-arrays -fstack-check -fstack-clash-protection -fstack-protector -fstack-protector-all -fstack-protector-strong -fstack-size-section -fstack-usage -fstrength-reduce -fstrict-enums -fstrict-flex-arrays= -fstrict-float-cast-overflow -fstrict-return -fstrict-vtable-pointers -fstruct-path-tbaa -fsycl-is-device -fsycl-is-host -fsymbol-partition= -ftabstop -ftabstop= -ftemplate-backtrace-limit= -ftemplate-depth= -ftest-coverage -ftest-module-file-extension= -fthreadsafe-statics -ftime-report -ftime-report= -ftls-model -ftls-model= -ftracer -ftrap-function= -ftrapping-math -ftrapv -ftrapv-handler -ftrapv-handler=  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK13 %s

// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fretain-comments-from-system-headers}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fretain-subst-template-type-parm-type-ast-nodes}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-frewrite-imports}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-frewrite-includes}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fripa}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fropi}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-frounding-math}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-frtlib-add-rpath}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-frtti}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-frtti-data}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-frwpi}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsafe-buffer-usage-suggestions}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsample-profile-use-profi}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-8bit-counters}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-control-flow}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-indirect-calls}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-inline-8bit-counters}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-inline-bool-flag}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-no-prune}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-pc-table}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-stack-depth}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-trace-bb}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-trace-cmp}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-trace-div}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-trace-gep}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-trace-loads}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-trace-pc}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-trace-pc-guard}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-trace-stores}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsanitize-coverage-type=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsave-main-program}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsave-optimization-record}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsave-optimization-record=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fschedule-insns}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fschedule-insns2}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsecond-underscore}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsee}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fseh-exceptions}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsemantic-interposition}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fseparate-named-sections}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fshort-enums}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fshort-wchar}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fshow-column}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fshow-overloads=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fshow-skipped-includes}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fshow-source-location}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsign-zero}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsignaling-math}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsignaling-nans}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsigned-bitfields}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsigned-char}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsigned-wchar}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsigned-zeros}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsingle-precision-constant}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsized-deallocation}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsjlj-exceptions}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fskip-odr-check-in-gmf}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fslp-vectorize}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fspec-constr-count}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fspell-checking}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fspell-checking-limit=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsplit-dwarf-inlining}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsplit-machine-functions}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsplit-stack}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fspv-target-env=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstack-arrays}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstack-check}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstack-clash-protection}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstack-protector}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstack-protector-all}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstack-protector-strong}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstack-size-section}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstack-usage}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstrength-reduce}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstrict-enums}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstrict-flex-arrays=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstrict-float-cast-overflow}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstrict-return}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstrict-vtable-pointers}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fstruct-path-tbaa}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsycl-is-device}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsycl-is-host}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fsymbol-partition=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftabstop}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftabstop=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftemplate-backtrace-limit=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftemplate-depth=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftest-coverage}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftest-module-file-extension=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-fthreadsafe-statics}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftime-report}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftime-report=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftls-model}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftls-model=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftracer}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftrap-function=}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftrapping-math}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftrapv}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftrapv-handler}}
// CLOptionCHECK13: {{(unknown argument ignored in clang-cl).*-ftrapv-handler=}}
// RUN: not %clang_cl -ftree-dce -ftree-salias -ftree-ter -ftree-vectorizer-verbose -ftree-vrp -ftrigraphs -ftype-visibility= -function-alignment -funderscoring -funified-lto -funique-basic-block-section-names -funique-internal-linkage-names -funique-section-names -funknown-anytype -funroll-all-loops -funroll-loops -funsafe-loop-optimizations -funsafe-math-optimizations -funsigned -funsigned-bitfields -funsigned-char -funswitch-loops -funwind-tables -funwind-tables= -fuse-ctor-homing -fuse-cxa-atexit -fuse-init-array -fuse-line-directives -fuse-linker-plugin -fuse-lipo= -fuse-register-sized-bitfield-access -fvalidate-ast-input-files-content -fvariable-expansion-in-unroller -fveclib= -fvect-cost-model -fvectorize -fverbose-asm -fverify-debuginfo-preserve -fverify-debuginfo-preserve-export= -fvisibility= -fvisibility-dllexport= -fvisibility-externs-dllimport= -fvisibility-externs-nodllstorageclass= -fvisibility-from-dllstorageclass -fvisibility-global-new-delete= -fvisibility-global-new-delete-hidden -fvisibility-inlines-hidden -fvisibility-inlines-hidden-static-local-var -fvisibility-ms-compat -fvisibility-nodllstorageclass= -fwarn-stack-size= -fwasm-exceptions -fwchar-type= -fweb -fwhole-file -fwhole-program -fwritable-strings -fxl-pragma-pack -fxor-operator -fxray-always-emit-customevents -fxray-always-emit-typedevents -fxray-always-instrument= -fxray-attr-list= -fxray-function-groups= -fxray-function-index -fxray-ignore-loops -fxray-instruction-threshold= -fxray-instrument -fxray-instrumentation-bundle= -fxray-link-deps -fxray-modes= -fxray-never-instrument= -fxray-selected-function-group= -fxray-shared -fzero-call-used-regs= -fzero-initialized-in-bss -fzos-extensions -fzvector -g0 -g1 -g2 -g3 --gcc-install-dir= --gcc-toolchain= --gcc-triple= -gcoff -gdbx -gdwarf32 -gdwarf64 -gdwarf-2 -gdwarf-3 -gdwarf-4 -gdwarf-5 -gdwarf-aranges -gembed-source -gfull -ggdb -ggdb0 -ggdb1 -ggdb2  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK14 %s

// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-ftree-dce}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-ftree-salias}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-ftree-ter}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-ftree-vectorizer-verbose}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-ftree-vrp}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-ftrigraphs}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-ftype-visibility=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-function-alignment}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funderscoring}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funified-lto}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funique-basic-block-section-names}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funique-internal-linkage-names}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funique-section-names}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funknown-anytype}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funroll-all-loops}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funroll-loops}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funsafe-loop-optimizations}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funsafe-math-optimizations}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funsigned}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funsigned-bitfields}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funsigned-char}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funswitch-loops}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funwind-tables}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-funwind-tables=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fuse-ctor-homing}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fuse-cxa-atexit}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fuse-init-array}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fuse-line-directives}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fuse-linker-plugin}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fuse-lipo=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fuse-register-sized-bitfield-access}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvalidate-ast-input-files-content}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvariable-expansion-in-unroller}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fveclib=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvect-cost-model}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvectorize}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fverbose-asm}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fverify-debuginfo-preserve}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fverify-debuginfo-preserve-export=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvisibility=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvisibility-dllexport=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvisibility-externs-dllimport=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvisibility-externs-nodllstorageclass=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvisibility-from-dllstorageclass}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvisibility-global-new-delete=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvisibility-global-new-delete-hidden}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvisibility-inlines-hidden}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvisibility-inlines-hidden-static-local-var}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvisibility-ms-compat}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fvisibility-nodllstorageclass=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fwarn-stack-size=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fwasm-exceptions}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fwchar-type=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fweb}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fwhole-file}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fwhole-program}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fwritable-strings}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxl-pragma-pack}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxor-operator}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-always-emit-customevents}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-always-emit-typedevents}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-always-instrument=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-attr-list=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-function-groups=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-function-index}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-ignore-loops}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-instruction-threshold=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-instrument}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-instrumentation-bundle=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-link-deps}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-modes=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-never-instrument=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-selected-function-group=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fxray-shared}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fzero-call-used-regs=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fzero-initialized-in-bss}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fzos-extensions}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-fzvector}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-g0}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-g1}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-g2}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-g3}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*--gcc-install-dir=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*--gcc-toolchain=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*--gcc-triple=}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-gcoff}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-gdbx}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-gdwarf32}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-gdwarf64}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-gdwarf-2}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-gdwarf-3}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-gdwarf-4}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-gdwarf-5}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-gdwarf-aranges}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-gembed-source}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-gfull}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-ggdb}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-ggdb0}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-ggdb1}}
// CLOptionCHECK14: {{(unknown argument ignored in clang-cl).*-ggdb2}}
// RUN: not %clang_cl -ggdb3 -ggnu-pubnames -glldb -gmlt -gmodules -gno-embed-source -gno-gnu-pubnames -gno-modules -gno-pubnames -gno-record-command-line -gno-simple-template-names -gno-template-alias -gpubnames -gpulibc -grecord-command-line -gsce -gsimple-template-names -gsimple-template-names= -gsrc-hash= -gstabs -gtemplate-alias -gtoggle -gused -gvms -gxcoff -gz -gz= -header-include-file -header-include-filtering= -header-include-format= -headerpad_max_install_names -hlsl-entry -iapinotes-modules -ibuiltininc -idirafter -iframework -iframeworkwithsysroot -imacros -image_base -import-call-optimization -imultilib -include -include-pch -init -init-only -install_name -interface-stub-version= -internal-externc-isystem -internal-isystem -iprefix -iquote -isysroot -isystem -isystem-after -ivfsoverlay -iwithprefix -iwithprefixbefore -iwithsysroot -keep_private_externs -l -lazy_framework -lazy_library --ld-path= --libomptarget-amdgcn-bc-path= --libomptarget-amdgpu-bc-path= --libomptarget-nvptx-bc-path= --libomptarget-spirv-bc-path= -llvm-verify-each -load -m3dnow -m3dnowa -m80387 -main-file-name -mappletvos-version-min= -mappletvsimulator-version-min= -mapxf -marm -massembler-fatal-warnings -massembler-no-warn -mavx10.1 -mavx10.2 -mbranch-protection-pauth-lr -mbranch-target-enforce -mdebug-pass -menable-no-infs -menable-no-nans -mfloat-abi -mfpmath -mframe-pointer= -mguarded-control-stack -no-finalize-removal -no-ns-alloc-error -mlimit-float-precision -mlink-bitcode-file -mlink-builtin-bitcode -mmapsyms=implicit -mmfcrf -mmpx -mno-3dnow -mno-3dnowa  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK15 %s

// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-ggdb3}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-ggnu-pubnames}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-glldb}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gmlt}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gmodules}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gno-embed-source}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gno-gnu-pubnames}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gno-modules}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gno-pubnames}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gno-record-command-line}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gno-simple-template-names}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gno-template-alias}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gpubnames}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gpulibc}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-grecord-command-line}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gsce}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gsimple-template-names}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gsimple-template-names=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gsrc-hash=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gstabs}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gtemplate-alias}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gtoggle}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gused}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gvms}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gxcoff}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gz}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-gz=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-header-include-file}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-header-include-filtering=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-header-include-format=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-headerpad_max_install_names}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-hlsl-entry}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-iapinotes-modules}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-ibuiltininc}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-idirafter}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-iframework}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-iframeworkwithsysroot}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-imacros}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-image_base}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-import-call-optimization}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-imultilib}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-include}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-include-pch}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-init}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-init-only}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-install_name}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-interface-stub-version=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-internal-externc-isystem}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-internal-isystem}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-iprefix}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-iquote}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-isysroot}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-isystem}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-isystem-after}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-ivfsoverlay}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-iwithprefix}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-iwithprefixbefore}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-iwithsysroot}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-keep_private_externs}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-l}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-lazy_framework}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-lazy_library}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*--ld-path=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*--libomptarget-amdgcn-bc-path=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*--libomptarget-amdgpu-bc-path=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*--libomptarget-nvptx-bc-path=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*--libomptarget-spirv-bc-path=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-llvm-verify-each}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-load}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-m3dnow}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-m3dnowa}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-m80387}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-main-file-name}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mappletvos-version-min=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mappletvsimulator-version-min=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mapxf}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-marm}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-massembler-fatal-warnings}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-massembler-no-warn}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mavx10.1}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mavx10.2}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mbranch-protection-pauth-lr}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mbranch-target-enforce}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mdebug-pass}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-menable-no-infs}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-menable-no-nans}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mfloat-abi}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mfpmath}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mframe-pointer=}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mguarded-control-stack}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-no-finalize-removal}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-no-ns-alloc-error}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mlimit-float-precision}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mlink-bitcode-file}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mlink-builtin-bitcode}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mmapsyms=implicit}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mmfcrf}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mmpx}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mno-3dnow}}
// CLOptionCHECK15: {{(unknown argument ignored in clang-cl).*-mno-3dnowa}}
// RUN: not %clang_cl -mno-80387 -mno-apxf -mno-avx10.1 -mno-avx10.1-512 -mnoexecstack -mno-fmv -mno-fp-ret-in-387 -mno-mfcrf -mno-mpx -mno-outline -mno-outline-atomics -mno-pascal-strings -mno-pure-code -mno-sse4 -mno-type-check -mno-zvector -module-dependency-dir -module-dir -module-file-deps -module-file-info -module-suffix -fmodules-reduced-bmi -moutline -moutline-atomics -mpascal-strings -mpure-code -mreassociate -mregparm -mrelax-relocations=no -mrelocation-model -msave-temp-labels -msign-return-address-key= -msmall-data-limit -msse4 -mtp -mtvos-simulator-version-min= -multi_module -multi-lib-config= -multiply_defined -multiply_defined_unused -mvscale-max= -mvscale-min= -mwatchsimulator-version-min= -mxcoff-build-id= -mzos-hlq-clang= -mzos-hlq-csslib= -mzos-hlq-le= -mzos-sys-include= -mzvector -n -new-struct-path-tbaa -no_dead_strip_inits_and_terms -no-clear-ast-before-backend -no-code-completion-globals -no-code-completion-ns-level-decls -no-cpp-precomp -fno-c++-static-destructors -no-emit-llvm-uselists -no-enable-noundef-analysis -no-implicit-float -no-integrated-cpp --no-offload-add-rpath --no-offloadlib -no-pedantic -no-pie -no-pointer-tbaa -no-pthread -no-round-trip-args -no-struct-path-tbaa --no-system-header-prefix= -nocpp -nodefaultlibs -nodriverkitlib -nofixprebinding -nogpuinc -nogpulibc -nohipwrapperinc -nolibc -nomultidefs -nopie -noprebind -noprofilelib -noseglinkedit -nostartfiles -nostdinc++ -nostdlib -nostdlibinc -nostdlib++ -nostdsysteminc -fexperimental-openacc-macro-override -fexperimental-openacc-macro-override= -p -pagezero_size -pass-exit-codes -pch-through-hdrstop-create -pch-through-hdrstop-use -pch-through-header= -pedantic -pedantic-errors -pg  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK16 %s

// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-80387}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-apxf}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-avx10.1}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-avx10.1-512}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mnoexecstack}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-fmv}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-fp-ret-in-387}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-mfcrf}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-mpx}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-outline}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-outline-atomics}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-pascal-strings}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-pure-code}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-sse4}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-type-check}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mno-zvector}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-module-dependency-dir}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-module-dir}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-module-file-deps}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-module-file-info}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-module-suffix}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-fmodules-reduced-bmi}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-moutline}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-moutline-atomics}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mpascal-strings}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mpure-code}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mreassociate}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mregparm}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mrelax-relocations=no}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mrelocation-model}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-msave-temp-labels}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-msign-return-address-key=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-msmall-data-limit}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-msse4}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mtp}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mtvos-simulator-version-min=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-multi_module}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-multi-lib-config=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-multiply_defined}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-multiply_defined_unused}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mvscale-max=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mvscale-min=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mwatchsimulator-version-min=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mxcoff-build-id=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mzos-hlq-clang=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mzos-hlq-csslib=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mzos-hlq-le=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mzos-sys-include=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-mzvector}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-n}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-new-struct-path-tbaa}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no_dead_strip_inits_and_terms}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-clear-ast-before-backend}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-code-completion-globals}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-code-completion-ns-level-decls}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-cpp-precomp}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-fno-c\+\+-static-destructors}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-emit-llvm-uselists}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-enable-noundef-analysis}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-implicit-float}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-integrated-cpp}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*--no-offload-add-rpath}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*--no-offloadlib}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-pedantic}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-pie}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-pointer-tbaa}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-pthread}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-round-trip-args}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-no-struct-path-tbaa}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*--no-system-header-prefix=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nocpp}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nodefaultlibs}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nodriverkitlib}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nofixprebinding}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nogpuinc}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nogpulibc}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nohipwrapperinc}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nolibc}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nomultidefs}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nopie}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-noprebind}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-noprofilelib}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-noseglinkedit}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nostartfiles}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nostdinc\+\+}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nostdlib}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nostdlibinc}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nostdlib\+\+}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-nostdsysteminc}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-fexperimental-openacc-macro-override}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-fexperimental-openacc-macro-override=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-p}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-pagezero_size}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-pass-exit-codes}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-pch-through-hdrstop-create}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-pch-through-hdrstop-use}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-pch-through-header=}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-pedantic}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-pedantic-errors}}
// CLOptionCHECK16: {{(unknown argument ignored in clang-cl).*-pg}}
// RUN: not %clang_cl -pic-is-pie -pic-level -pie -pipe -plugin -plugin-arg- -pointer-tbaa -preamble-bytes= -prebind -prebind_all_twolevel_modules -preload -print-dependency-directives-minimized-source -print-ivar-layout -print-multi-directory -print-multi-flags-experimental -print-multi-lib -print-multi-os-directory -print-preamble -print-stats -private_bundle --product-name= -pthread -pthreads -r -rdynamic -read_only_relocs -record-command-line -reexport_framework -reexport-l -reexport_library -relaxed-aliasing -relocatable-pch -remap -remap-file -rewrite-legacy-objc -rewrite-macros -rewrite-objc -rewrite-test -round-trip-args -rpath -s -save-stats -save-stats= -save-temps -save-temps= -sectalign -sectcreate -sectobjectsymbols -sectorder -seg1addr -seg_addr_table -seg_addr_table_filename -segaddr -segcreate -seglinkedit -segprot -segs_read_ -segs_read_only_addr -segs_read_write_addr -setup-static-analyzer -shared -shared-libgcc -shared-libsan -show-encoding --show-includes -show-inst -single_module -skip-function-bodies -source-date-epoch -specs -specs= -spirv -split-dwarf-file -split-dwarf-output -stack-protector -stack-protector-buffer-size -stack-usage-file -startfiles -static -static-define -static-libclosure -static-libgcc -static-libgfortran -static-libsan -static-libstdc++ -static-openmp -static-pie -stats-file= -stats-file-append -std= -std-default= -stdlib -stdlib= -stdlib++-isystem -sub_library -sub_umbrella --symbol-graph-dir= -sys-header-deps --system-header-prefix= -t  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK17 %s

// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-pic-is-pie}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-pic-level}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-pie}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-pipe}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-plugin}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-plugin-arg-}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-pointer-tbaa}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-preamble-bytes=}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-prebind}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-prebind_all_twolevel_modules}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-preload}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-print-dependency-directives-minimized-source}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-print-ivar-layout}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-print-multi-directory}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-print-multi-flags-experimental}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-print-multi-lib}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-print-multi-os-directory}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-print-preamble}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-print-stats}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-private_bundle}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*--product-name=}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-pthread}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-pthreads}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-r}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-rdynamic}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-read_only_relocs}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-record-command-line}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-reexport_framework}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-reexport-l}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-reexport_library}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-relaxed-aliasing}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-relocatable-pch}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-remap}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-remap-file}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-rewrite-legacy-objc}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-rewrite-macros}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-rewrite-objc}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-rewrite-test}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-round-trip-args}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-rpath}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-s}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-save-stats}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-save-stats=}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-save-temps}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-save-temps=}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-sectalign}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-sectcreate}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-sectobjectsymbols}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-sectorder}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-seg1addr}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-seg_addr_table}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-seg_addr_table_filename}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-segaddr}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-segcreate}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-seglinkedit}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-segprot}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-segs_read_}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-segs_read_only_addr}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-segs_read_write_addr}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-setup-static-analyzer}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-shared}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-shared-libgcc}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-shared-libsan}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-show-encoding}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*--show-includes}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-show-inst}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-single_module}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-skip-function-bodies}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-source-date-epoch}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-specs}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-specs=}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-spirv}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-split-dwarf-file}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-split-dwarf-output}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-stack-protector}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-stack-protector-buffer-size}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-stack-usage-file}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-startfiles}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-static}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-static-define}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-static-libclosure}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-static-libgcc}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-static-libgfortran}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-static-libsan}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-static-libstdc\+\+}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-static-openmp}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-static-pie}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-stats-file=}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-stats-file-append}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-std=}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-std-default=}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-stdlib}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-stdlib=}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-stdlib\+\+-isystem}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-sub_library}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-sub_umbrella}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*--symbol-graph-dir=}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-sys-header-deps}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*--system-header-prefix=}}
// CLOptionCHECK17: {{(unknown argument ignored in clang-cl).*-t}}
// RUN: not %clang_cl -target-abi -target-cpu -target-feature -target-linker-version -T -target-sdk-version= -templight-dump -test-io -time -traditional -traditional-cpp -trigraphs -trim-egraph -triple -triple= -tune-cpu -twolevel_namespace -twolevel_namespace_hints -umbrella -undef -undefined -unexported_symbols_list -unwindlib= -vectorize-loops -vectorize-slp -verify -verify= --verify-debug-info -verify-ignore-unexpected -verify-ignore-unexpected= -verify-pch -vtordisp-mode= -y -z  -### /c /WX -Werror 2>&1 | FileCheck -check-prefix=CLOptionCHECK18 %s

// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-target-abi}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-target-cpu}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-target-feature}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-target-linker-version}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-T}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-target-sdk-version=}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-templight-dump}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-test-io}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-time}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-traditional}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-traditional-cpp}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-trigraphs}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-trim-egraph}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-triple}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-triple=}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-tune-cpu}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-twolevel_namespace}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-twolevel_namespace_hints}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-umbrella}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-undef}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-undefined}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-unexported_symbols_list}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-unwindlib=}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-vectorize-loops}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-vectorize-slp}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-verify}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-verify=}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*--verify-debug-info}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-verify-ignore-unexpected}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-verify-ignore-unexpected=}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-verify-pch}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-vtordisp-mode=}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-y}}
// CLOptionCHECK18: {{(unknown argument ignored in clang-cl).*-z}}
// RUN: not %clang_dxc -A -A- -B -C -CC -F -faapcs-bitfield-load -G -G= -H -J -K -L -M -MD -MF -MG -MJ -MM -MMD -MP -MQ -MT -MV -Mach -P -Q -Qn -Qy -R -Rpass= -Rpass-analysis= -Rpass-missed= -S -U -V -X -Xanalyzer -Xarch_ -Xarch_device -Xarch_host -Xassembler -Xcuda-fatbinary -Xcuda-ptxas -Xflang -Xlinker -Xoffload-linker -Xopenmp-target -Xopenmp-target= -Xpreprocessor -Z -Z-Xlinker-no-demangle -Z-reserved-lib-cckext -Z-reserved-lib-stdc++ -Zlinker-input --CLASSPATH --CLASSPATH= -AI -Brepro -Brepro- -Bt -Bt+ -C -F -FA -FC -FI -FR -FS -FU -Fa -Fd -Fe -Fe: -Fi -Fi: -Fm -Fp -Fp: -Fr -Fx -G1 -G2 -GA -GF -GF- -GH -GL -GL- -GR -GR- -GS -GS- -GT -GX -GX- -GZ -Gd -Ge -Gh  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK0 %s

// DXCOptionCHECK0: {{(unknown argument).*-A}}
// DXCOptionCHECK0: {{(unknown argument).*-A-}}
// DXCOptionCHECK0: {{(unknown argument).*-B}}
// DXCOptionCHECK0: {{(unknown argument).*-C}}
// DXCOptionCHECK0: {{(unknown argument).*-CC}}
// DXCOptionCHECK0: {{(unknown argument).*-F}}
// DXCOptionCHECK0: {{(unknown argument).*-faapcs-bitfield-load}}
// DXCOptionCHECK0: {{(unknown argument).*-G}}
// DXCOptionCHECK0: {{(unknown argument).*-G=}}
// DXCOptionCHECK0: {{(unknown argument).*-H}}
// DXCOptionCHECK0: {{(unknown argument).*-J}}
// DXCOptionCHECK0: {{(unknown argument).*-K}}
// DXCOptionCHECK0: {{(unknown argument).*-L}}
// DXCOptionCHECK0: {{(unknown argument).*-M}}
// DXCOptionCHECK0: {{(unknown argument).*-MD}}
// DXCOptionCHECK0: {{(unknown argument).*-MF}}
// DXCOptionCHECK0: {{(unknown argument).*-MG}}
// DXCOptionCHECK0: {{(unknown argument).*-MJ}}
// DXCOptionCHECK0: {{(unknown argument).*-MM}}
// DXCOptionCHECK0: {{(unknown argument).*-MMD}}
// DXCOptionCHECK0: {{(unknown argument).*-MP}}
// DXCOptionCHECK0: {{(unknown argument).*-MQ}}
// DXCOptionCHECK0: {{(unknown argument).*-MT}}
// DXCOptionCHECK0: {{(unknown argument).*-MV}}
// DXCOptionCHECK0: {{(unknown argument).*-Mach}}
// DXCOptionCHECK0: {{(unknown argument).*-P}}
// DXCOptionCHECK0: {{(unknown argument).*-Q}}
// DXCOptionCHECK0: {{(unknown argument).*-Qn}}
// DXCOptionCHECK0: {{(unknown argument).*-Qy}}
// DXCOptionCHECK0: {{(unknown argument).*-R}}
// DXCOptionCHECK0: {{(unknown argument).*-Rpass=}}
// DXCOptionCHECK0: {{(unknown argument).*-Rpass-analysis=}}
// DXCOptionCHECK0: {{(unknown argument).*-Rpass-missed=}}
// DXCOptionCHECK0: {{(unknown argument).*-S}}
// DXCOptionCHECK0: {{(unknown argument).*-U}}
// DXCOptionCHECK0: {{(unknown argument).*-V}}
// DXCOptionCHECK0: {{(unknown argument).*-X}}
// DXCOptionCHECK0: {{(unknown argument).*-Xanalyzer}}
// DXCOptionCHECK0: {{(unknown argument).*-Xarch_}}
// DXCOptionCHECK0: {{(unknown argument).*-Xarch_device}}
// DXCOptionCHECK0: {{(unknown argument).*-Xarch_host}}
// DXCOptionCHECK0: {{(unknown argument).*-Xassembler}}
// DXCOptionCHECK0: {{(unknown argument).*-Xcuda-fatbinary}}
// DXCOptionCHECK0: {{(unknown argument).*-Xcuda-ptxas}}
// DXCOptionCHECK0: {{(unknown argument).*-Xflang}}
// DXCOptionCHECK0: {{(unknown argument).*-Xlinker}}
// DXCOptionCHECK0: {{(unknown argument).*-Xoffload-linker}}
// DXCOptionCHECK0: {{(unknown argument).*-Xopenmp-target}}
// DXCOptionCHECK0: {{(unknown argument).*-Xopenmp-target=}}
// DXCOptionCHECK0: {{(unknown argument).*-Xpreprocessor}}
// DXCOptionCHECK0: {{(unknown argument).*-Z}}
// DXCOptionCHECK0: {{(unknown argument).*-Z-Xlinker-no-demangle}}
// DXCOptionCHECK0: {{(unknown argument).*-Z-reserved-lib-cckext}}
// DXCOptionCHECK0: {{(unknown argument).*-Z-reserved-lib-stdc\+\+}}
// DXCOptionCHECK0: {{(unknown argument).*-Zlinker-input}}
// DXCOptionCHECK0: {{(unknown argument).*--CLASSPATH}}
// DXCOptionCHECK0: {{(unknown argument).*--CLASSPATH=}}
// DXCOptionCHECK0: {{(unknown argument).*-AI}}
// DXCOptionCHECK0: {{(unknown argument).*-Brepro}}
// DXCOptionCHECK0: {{(unknown argument).*-Brepro-}}
// DXCOptionCHECK0: {{(unknown argument).*-Bt}}
// DXCOptionCHECK0: {{(unknown argument).*-Bt\+}}
// DXCOptionCHECK0: {{(unknown argument).*-C}}
// DXCOptionCHECK0: {{(unknown argument).*-F}}
// DXCOptionCHECK0: {{(unknown argument).*-FA}}
// DXCOptionCHECK0: {{(unknown argument).*-FC}}
// DXCOptionCHECK0: {{(unknown argument).*-FI}}
// DXCOptionCHECK0: {{(unknown argument).*-FR}}
// DXCOptionCHECK0: {{(unknown argument).*-FS}}
// DXCOptionCHECK0: {{(unknown argument).*-FU}}
// DXCOptionCHECK0: {{(unknown argument).*-Fa}}
// DXCOptionCHECK0: {{(unknown argument).*-Fd}}
// DXCOptionCHECK0: {{(unknown argument).*-Fe}}
// DXCOptionCHECK0: {{(unknown argument).*-Fe:}}
// DXCOptionCHECK0: {{(unknown argument).*-Fi}}
// DXCOptionCHECK0: {{(unknown argument).*-Fi:}}
// DXCOptionCHECK0: {{(unknown argument).*-Fm}}
// DXCOptionCHECK0: {{(unknown argument).*-Fp}}
// DXCOptionCHECK0: {{(unknown argument).*-Fp:}}
// DXCOptionCHECK0: {{(unknown argument).*-Fr}}
// DXCOptionCHECK0: {{(unknown argument).*-Fx}}
// DXCOptionCHECK0: {{(unknown argument).*-G1}}
// DXCOptionCHECK0: {{(unknown argument).*-G2}}
// DXCOptionCHECK0: {{(unknown argument).*-GA}}
// DXCOptionCHECK0: {{(unknown argument).*-GF}}
// DXCOptionCHECK0: {{(unknown argument).*-GF-}}
// DXCOptionCHECK0: {{(unknown argument).*-GH}}
// DXCOptionCHECK0: {{(unknown argument).*-GL}}
// DXCOptionCHECK0: {{(unknown argument).*-GL-}}
// DXCOptionCHECK0: {{(unknown argument).*-GR}}
// DXCOptionCHECK0: {{(unknown argument).*-GR-}}
// DXCOptionCHECK0: {{(unknown argument).*-GS}}
// DXCOptionCHECK0: {{(unknown argument).*-GS-}}
// DXCOptionCHECK0: {{(unknown argument).*-GT}}
// DXCOptionCHECK0: {{(unknown argument).*-GX}}
// DXCOptionCHECK0: {{(unknown argument).*-GX-}}
// DXCOptionCHECK0: {{(unknown argument).*-GZ}}
// DXCOptionCHECK0: {{(unknown argument).*-Gd}}
// DXCOptionCHECK0: {{(unknown argument).*-Ge}}
// DXCOptionCHECK0: {{(unknown argument).*-Gh}}
// RUN: not %clang_dxc -Gm -Gm- -Gr -Gregcall -Gregcall4 -Gs -Gv -Gw -Gw- -Gy -Gy- -Gz -H -HELP -J -JMC -JMC- -LD -LDd -LN -MD -MDd -MP -MT -MTd -P -QIfist -QIntel-jcc-erratum -? -Qfast_transcendentals -Qimprecise_fwaits -Qpar -Qpar-report -Qsafe_fp_loads -Qspectre -Qspectre-load -Qspectre-load-cf -Qvec -Qvec- -Qvec-report -RTC -U -V -X -Y- -Yc -Yd -Yl -Yu -Z7 -ZH:MD5 -ZH:SHA1 -ZH:SHA_256 -ZI -ZW -Za -Zc: -Zc:__STDC__ -Zc:__cplusplus -Zc:alignedNew -Zc:alignedNew- -Zc:auto -Zc:char8_t -Zc:char8_t- -Zc:dllexportInlines -Zc:dllexportInlines- -Zc:forScope -Zc:inline -Zc:rvalueCast -Zc:sizedDealloc -Zc:sizedDealloc- -Zc:strictStrings -Zc:ternary -Zc:threadSafeInit -Zc:threadSafeInit- -Zc:tlsGuards -Zc:tlsGuards- -Zc:trigraphs -Zc:trigraphs- -Zc:twoPhase -Zc:twoPhase- -Zc:wchar_t -Zc:wchar_t- -Ze -Zg -Zl -Zm -Zo -Zo- -Zp -Zp -Zs -analyze- -arch: -arm64EC -await -await: -bigobj -c -cgthreads  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK1 %s

// DXCOptionCHECK1: {{(unknown argument).*-Gm}}
// DXCOptionCHECK1: {{(unknown argument).*-Gm-}}
// DXCOptionCHECK1: {{(unknown argument).*-Gr}}
// DXCOptionCHECK1: {{(unknown argument).*-Gregcall}}
// DXCOptionCHECK1: {{(unknown argument).*-Gregcall4}}
// DXCOptionCHECK1: {{(unknown argument).*-Gs}}
// DXCOptionCHECK1: {{(unknown argument).*-Gv}}
// DXCOptionCHECK1: {{(unknown argument).*-Gw}}
// DXCOptionCHECK1: {{(unknown argument).*-Gw-}}
// DXCOptionCHECK1: {{(unknown argument).*-Gy}}
// DXCOptionCHECK1: {{(unknown argument).*-Gy-}}
// DXCOptionCHECK1: {{(unknown argument).*-Gz}}
// DXCOptionCHECK1: {{(unknown argument).*-H}}
// DXCOptionCHECK1: {{(unknown argument).*-HELP}}
// DXCOptionCHECK1: {{(unknown argument).*-J}}
// DXCOptionCHECK1: {{(unknown argument).*-JMC}}
// DXCOptionCHECK1: {{(unknown argument).*-JMC-}}
// DXCOptionCHECK1: {{(unknown argument).*-LD}}
// DXCOptionCHECK1: {{(unknown argument).*-LDd}}
// DXCOptionCHECK1: {{(unknown argument).*-LN}}
// DXCOptionCHECK1: {{(unknown argument).*-MD}}
// DXCOptionCHECK1: {{(unknown argument).*-MDd}}
// DXCOptionCHECK1: {{(unknown argument).*-MP}}
// DXCOptionCHECK1: {{(unknown argument).*-MT}}
// DXCOptionCHECK1: {{(unknown argument).*-MTd}}
// DXCOptionCHECK1: {{(unknown argument).*-P}}
// DXCOptionCHECK1: {{(unknown argument).*-QIfist}}
// DXCOptionCHECK1: {{(unknown argument).*-QIntel-jcc-erratum}}
// DXCOptionCHECK1: {{(unknown argument).*-?}}
// DXCOptionCHECK1: {{(unknown argument).*-Qfast_transcendentals}}
// DXCOptionCHECK1: {{(unknown argument).*-Qimprecise_fwaits}}
// DXCOptionCHECK1: {{(unknown argument).*-Qpar}}
// DXCOptionCHECK1: {{(unknown argument).*-Qpar-report}}
// DXCOptionCHECK1: {{(unknown argument).*-Qsafe_fp_loads}}
// DXCOptionCHECK1: {{(unknown argument).*-Qspectre}}
// DXCOptionCHECK1: {{(unknown argument).*-Qspectre-load}}
// DXCOptionCHECK1: {{(unknown argument).*-Qspectre-load-cf}}
// DXCOptionCHECK1: {{(unknown argument).*-Qvec}}
// DXCOptionCHECK1: {{(unknown argument).*-Qvec-}}
// DXCOptionCHECK1: {{(unknown argument).*-Qvec-report}}
// DXCOptionCHECK1: {{(unknown argument).*-RTC}}
// DXCOptionCHECK1: {{(unknown argument).*-U}}
// DXCOptionCHECK1: {{(unknown argument).*-V}}
// DXCOptionCHECK1: {{(unknown argument).*-X}}
// DXCOptionCHECK1: {{(unknown argument).*-Y-}}
// DXCOptionCHECK1: {{(unknown argument).*-Yc}}
// DXCOptionCHECK1: {{(unknown argument).*-Yd}}
// DXCOptionCHECK1: {{(unknown argument).*-Yl}}
// DXCOptionCHECK1: {{(unknown argument).*-Yu}}
// DXCOptionCHECK1: {{(unknown argument).*-Z7}}
// DXCOptionCHECK1: {{(unknown argument).*-ZH:MD5}}
// DXCOptionCHECK1: {{(unknown argument).*-ZH:SHA1}}
// DXCOptionCHECK1: {{(unknown argument).*-ZH:SHA_256}}
// DXCOptionCHECK1: {{(unknown argument).*-ZI}}
// DXCOptionCHECK1: {{(unknown argument).*-ZW}}
// DXCOptionCHECK1: {{(unknown argument).*-Za}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:__STDC__}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:__cplusplus}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:alignedNew}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:alignedNew-}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:auto}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:char8_t}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:char8_t-}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:dllexportInlines}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:dllexportInlines-}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:forScope}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:inline}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:rvalueCast}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:sizedDealloc}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:sizedDealloc-}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:strictStrings}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:ternary}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:threadSafeInit}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:threadSafeInit-}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:tlsGuards}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:tlsGuards-}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:trigraphs}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:trigraphs-}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:twoPhase}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:twoPhase-}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:wchar_t}}
// DXCOptionCHECK1: {{(unknown argument).*-Zc:wchar_t-}}
// DXCOptionCHECK1: {{(unknown argument).*-Ze}}
// DXCOptionCHECK1: {{(unknown argument).*-Zg}}
// DXCOptionCHECK1: {{(unknown argument).*-Zl}}
// DXCOptionCHECK1: {{(unknown argument).*-Zm}}
// DXCOptionCHECK1: {{(unknown argument).*-Zo}}
// DXCOptionCHECK1: {{(unknown argument).*-Zo-}}
// DXCOptionCHECK1: {{(unknown argument).*-Zp}}
// DXCOptionCHECK1: {{(unknown argument).*-Zp}}
// DXCOptionCHECK1: {{(unknown argument).*-Zs}}
// DXCOptionCHECK1: {{(unknown argument).*-analyze-}}
// DXCOptionCHECK1: {{(unknown argument).*-arch:}}
// DXCOptionCHECK1: {{(unknown argument).*-arm64EC}}
// DXCOptionCHECK1: {{(unknown argument).*-await}}
// DXCOptionCHECK1: {{(unknown argument).*-await:}}
// DXCOptionCHECK1: {{(unknown argument).*-bigobj}}
// DXCOptionCHECK1: {{(unknown argument).*-c}}
// DXCOptionCHECK1: {{(unknown argument).*-cgthreads}}
// RUN: not %clang_dxc -clang: -clr -constexpr: -d1 -d1PP -d1reportAllClassLayout -d2 -d2FastFail -d2Zi+ -diagnostics:caret -diagnostics:classic -diagnostics:column -diasdkdir -doc -errorReport -execution-charset: -experimental: -exportHeader -external: -external:I -external:W0 -external:W1 -external:W2 -external:W3 -external:W4 -external:env: -favor -fno-sanitize-address-vcasan-lib -fp:contract -fp:except -fp:except- -fp:fast -fp:precise -fp:strict -fsanitize=address -fsanitize-address-use-after-return -guard: -headerUnit -headerUnit:angle -headerUnit:quote -headerName: -homeparams -hotpatch -imsvc -kernel -kernel- -link -nologo -o -openmp -openmp- -openmp:experimental -permissive -permissive- -reference -sdl -sdl- -showFilenames -showFilenames- -showIncludes -showIncludes:user -sourceDependencies -sourceDependencies:directives -source-charset: -std: -translateInclude -tune: -u -utf-8 -validate-charset -validate-charset- -vctoolsdir -vctoolsversion -vd -vmb -vmg -vmm -vms -vmv -volatile:iso -volatile:ms -wd -winsdkdir -winsdkversion -winsysroot --all-warnings --analyze --analyzer-no-default-checks --analyzer-output --assemble --assert --assert= --bootclasspath --bootclasspath= --classpath --classpath= --comments --comments-in-macros --compile --constant-cfstrings  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK2 %s

// DXCOptionCHECK2: {{(unknown argument).*-clang:}}
// DXCOptionCHECK2: {{(unknown argument).*-clr}}
// DXCOptionCHECK2: {{(unknown argument).*-constexpr:}}
// DXCOptionCHECK2: {{(unknown argument).*-d1}}
// DXCOptionCHECK2: {{(unknown argument).*-d1PP}}
// DXCOptionCHECK2: {{(unknown argument).*-d1reportAllClassLayout}}
// DXCOptionCHECK2: {{(unknown argument).*-d2}}
// DXCOptionCHECK2: {{(unknown argument).*-d2FastFail}}
// DXCOptionCHECK2: {{(unknown argument).*-d2Zi\+}}
// DXCOptionCHECK2: {{(unknown argument).*-diagnostics:caret}}
// DXCOptionCHECK2: {{(unknown argument).*-diagnostics:classic}}
// DXCOptionCHECK2: {{(unknown argument).*-diagnostics:column}}
// DXCOptionCHECK2: {{(unknown argument).*-diasdkdir}}
// DXCOptionCHECK2: {{(unknown argument).*-doc}}
// DXCOptionCHECK2: {{(unknown argument).*-errorReport}}
// DXCOptionCHECK2: {{(unknown argument).*-execution-charset:}}
// DXCOptionCHECK2: {{(unknown argument).*-experimental:}}
// DXCOptionCHECK2: {{(unknown argument).*-exportHeader}}
// DXCOptionCHECK2: {{(unknown argument).*-external:}}
// DXCOptionCHECK2: {{(unknown argument).*-external:I}}
// DXCOptionCHECK2: {{(unknown argument).*-external:W0}}
// DXCOptionCHECK2: {{(unknown argument).*-external:W1}}
// DXCOptionCHECK2: {{(unknown argument).*-external:W2}}
// DXCOptionCHECK2: {{(unknown argument).*-external:W3}}
// DXCOptionCHECK2: {{(unknown argument).*-external:W4}}
// DXCOptionCHECK2: {{(unknown argument).*-external:env:}}
// DXCOptionCHECK2: {{(unknown argument).*-favor}}
// DXCOptionCHECK2: {{(unknown argument).*-fno-sanitize-address-vcasan-lib}}
// DXCOptionCHECK2: {{(unknown argument).*-fp:contract}}
// DXCOptionCHECK2: {{(unknown argument).*-fp:except}}
// DXCOptionCHECK2: {{(unknown argument).*-fp:except-}}
// DXCOptionCHECK2: {{(unknown argument).*-fp:fast}}
// DXCOptionCHECK2: {{(unknown argument).*-fp:precise}}
// DXCOptionCHECK2: {{(unknown argument).*-fp:strict}}
// DXCOptionCHECK2: {{(unknown argument).*-fsanitize=address}}
// DXCOptionCHECK2: {{(unknown argument).*-fsanitize-address-use-after-return}}
// DXCOptionCHECK2: {{(unknown argument).*-guard:}}
// DXCOptionCHECK2: {{(unknown argument).*-headerUnit}}
// DXCOptionCHECK2: {{(unknown argument).*-headerUnit:angle}}
// DXCOptionCHECK2: {{(unknown argument).*-headerUnit:quote}}
// DXCOptionCHECK2: {{(unknown argument).*-headerName:}}
// DXCOptionCHECK2: {{(unknown argument).*-homeparams}}
// DXCOptionCHECK2: {{(unknown argument).*-hotpatch}}
// DXCOptionCHECK2: {{(unknown argument).*-imsvc}}
// DXCOptionCHECK2: {{(unknown argument).*-kernel}}
// DXCOptionCHECK2: {{(unknown argument).*-kernel-}}
// DXCOptionCHECK2: {{(unknown argument).*-link}}
// DXCOptionCHECK2: {{(unknown argument).*-nologo}}
// DXCOptionCHECK2: {{(unknown argument).*-o}}
// DXCOptionCHECK2: {{(unknown argument).*-openmp}}
// DXCOptionCHECK2: {{(unknown argument).*-openmp-}}
// DXCOptionCHECK2: {{(unknown argument).*-openmp:experimental}}
// DXCOptionCHECK2: {{(unknown argument).*-permissive}}
// DXCOptionCHECK2: {{(unknown argument).*-permissive-}}
// DXCOptionCHECK2: {{(unknown argument).*-reference}}
// DXCOptionCHECK2: {{(unknown argument).*-sdl}}
// DXCOptionCHECK2: {{(unknown argument).*-sdl-}}
// DXCOptionCHECK2: {{(unknown argument).*-showFilenames}}
// DXCOptionCHECK2: {{(unknown argument).*-showFilenames-}}
// DXCOptionCHECK2: {{(unknown argument).*-showIncludes}}
// DXCOptionCHECK2: {{(unknown argument).*-showIncludes:user}}
// DXCOptionCHECK2: {{(unknown argument).*-sourceDependencies}}
// DXCOptionCHECK2: {{(unknown argument).*-sourceDependencies:directives}}
// DXCOptionCHECK2: {{(unknown argument).*-source-charset:}}
// DXCOptionCHECK2: {{(unknown argument).*-std:}}
// DXCOptionCHECK2: {{(unknown argument).*-translateInclude}}
// DXCOptionCHECK2: {{(unknown argument).*-tune:}}
// DXCOptionCHECK2: {{(unknown argument).*-u}}
// DXCOptionCHECK2: {{(unknown argument).*-utf-8}}
// DXCOptionCHECK2: {{(unknown argument).*-validate-charset}}
// DXCOptionCHECK2: {{(unknown argument).*-validate-charset-}}
// DXCOptionCHECK2: {{(unknown argument).*-vctoolsdir}}
// DXCOptionCHECK2: {{(unknown argument).*-vctoolsversion}}
// DXCOptionCHECK2: {{(unknown argument).*-vd}}
// DXCOptionCHECK2: {{(unknown argument).*-vmb}}
// DXCOptionCHECK2: {{(unknown argument).*-vmg}}
// DXCOptionCHECK2: {{(unknown argument).*-vmm}}
// DXCOptionCHECK2: {{(unknown argument).*-vms}}
// DXCOptionCHECK2: {{(unknown argument).*-vmv}}
// DXCOptionCHECK2: {{(unknown argument).*-volatile:iso}}
// DXCOptionCHECK2: {{(unknown argument).*-volatile:ms}}
// DXCOptionCHECK2: {{(unknown argument).*-wd}}
// DXCOptionCHECK2: {{(unknown argument).*-winsdkdir}}
// DXCOptionCHECK2: {{(unknown argument).*-winsdkversion}}
// DXCOptionCHECK2: {{(unknown argument).*-winsysroot}}
// DXCOptionCHECK2: {{(unknown argument).*--all-warnings}}
// DXCOptionCHECK2: {{(unknown argument).*--analyze}}
// DXCOptionCHECK2: {{(unknown argument).*--analyzer-no-default-checks}}
// DXCOptionCHECK2: {{(unknown argument).*--analyzer-output}}
// DXCOptionCHECK2: {{(unknown argument).*--assemble}}
// DXCOptionCHECK2: {{(unknown argument).*--assert}}
// DXCOptionCHECK2: {{(unknown argument).*--assert=}}
// DXCOptionCHECK2: {{(unknown argument).*--bootclasspath}}
// DXCOptionCHECK2: {{(unknown argument).*--bootclasspath=}}
// DXCOptionCHECK2: {{(unknown argument).*--classpath}}
// DXCOptionCHECK2: {{(unknown argument).*--classpath=}}
// DXCOptionCHECK2: {{(unknown argument).*--comments}}
// DXCOptionCHECK2: {{(unknown argument).*--comments-in-macros}}
// DXCOptionCHECK2: {{(unknown argument).*--compile}}
// DXCOptionCHECK2: {{(unknown argument).*--constant-cfstrings}}
// RUN: not %clang_dxc --debug --debug= --define-macro --define-macro= --dependencies --dyld-prefix --dyld-prefix= --encoding --encoding= --entry --extdirs --extdirs= --extra-warnings --for-linker --for-linker= --force-link --force-link= --help-hidden --imacros= --include= --include-barrier --include-directory --include-directory= --include-directory-after --include-directory-after= --include-prefix --include-prefix= --include-with-prefix --include-with-prefix= --include-with-prefix-after --include-with-prefix-after= --include-with-prefix-before --include-with-prefix-before= --language --language= --library-directory --library-directory= --mhwdiv --mhwdiv= --no-line-commands --no-standard-includes --no-standard-libraries --no-undefined --optimize --optimize= --output --output= --output-class-directory --output-class-directory= --param --param= --precompile --prefix --prefix= --preprocess --print-diagnostic-categories --print-file-name --print-missing-file-dependencies --print-prog-name --profile --resource --resource= --rtlib -serialize-diagnostics --signed-char --std --stdlib --sysroot --sysroot= --target-help --trace-includes --undefine-macro --undefine-macro= --unsigned-char --user-dependencies --verbose --warn- --warn-= --write-dependencies --write-user-dependencies -add-plugin -alias_list -faligned-alloc-unavailable -all_load -allowable_client -faltivec-src-compat= --amdgpu-arch-tool= -cfg-add-implicit-dtors -unoptimized-cfg -analyze -analyze-function -analyze-function= -analyzer-checker -analyzer-checker= -analyzer-checker-help -analyzer-checker-help-alpha -analyzer-checker-help-developer -analyzer-checker-option-help -analyzer-checker-option-help-alpha -analyzer-checker-option-help-developer  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK3 %s

// DXCOptionCHECK3: {{(unknown argument).*--debug}}
// DXCOptionCHECK3: {{(unknown argument).*--debug=}}
// DXCOptionCHECK3: {{(unknown argument).*--define-macro}}
// DXCOptionCHECK3: {{(unknown argument).*--define-macro=}}
// DXCOptionCHECK3: {{(unknown argument).*--dependencies}}
// DXCOptionCHECK3: {{(unknown argument).*--dyld-prefix}}
// DXCOptionCHECK3: {{(unknown argument).*--dyld-prefix=}}
// DXCOptionCHECK3: {{(unknown argument).*--encoding}}
// DXCOptionCHECK3: {{(unknown argument).*--encoding=}}
// DXCOptionCHECK3: {{(unknown argument).*--entry}}
// DXCOptionCHECK3: {{(unknown argument).*--extdirs}}
// DXCOptionCHECK3: {{(unknown argument).*--extdirs=}}
// DXCOptionCHECK3: {{(unknown argument).*--extra-warnings}}
// DXCOptionCHECK3: {{(unknown argument).*--for-linker}}
// DXCOptionCHECK3: {{(unknown argument).*--for-linker=}}
// DXCOptionCHECK3: {{(unknown argument).*--force-link}}
// DXCOptionCHECK3: {{(unknown argument).*--force-link=}}
// DXCOptionCHECK3: {{(unknown argument).*--help-hidden}}
// DXCOptionCHECK3: {{(unknown argument).*--imacros=}}
// DXCOptionCHECK3: {{(unknown argument).*--include=}}
// DXCOptionCHECK3: {{(unknown argument).*--include-barrier}}
// DXCOptionCHECK3: {{(unknown argument).*--include-directory}}
// DXCOptionCHECK3: {{(unknown argument).*--include-directory=}}
// DXCOptionCHECK3: {{(unknown argument).*--include-directory-after}}
// DXCOptionCHECK3: {{(unknown argument).*--include-directory-after=}}
// DXCOptionCHECK3: {{(unknown argument).*--include-prefix}}
// DXCOptionCHECK3: {{(unknown argument).*--include-prefix=}}
// DXCOptionCHECK3: {{(unknown argument).*--include-with-prefix}}
// DXCOptionCHECK3: {{(unknown argument).*--include-with-prefix=}}
// DXCOptionCHECK3: {{(unknown argument).*--include-with-prefix-after}}
// DXCOptionCHECK3: {{(unknown argument).*--include-with-prefix-after=}}
// DXCOptionCHECK3: {{(unknown argument).*--include-with-prefix-before}}
// DXCOptionCHECK3: {{(unknown argument).*--include-with-prefix-before=}}
// DXCOptionCHECK3: {{(unknown argument).*--language}}
// DXCOptionCHECK3: {{(unknown argument).*--language=}}
// DXCOptionCHECK3: {{(unknown argument).*--library-directory}}
// DXCOptionCHECK3: {{(unknown argument).*--library-directory=}}
// DXCOptionCHECK3: {{(unknown argument).*--mhwdiv}}
// DXCOptionCHECK3: {{(unknown argument).*--mhwdiv=}}
// DXCOptionCHECK3: {{(unknown argument).*--no-line-commands}}
// DXCOptionCHECK3: {{(unknown argument).*--no-standard-includes}}
// DXCOptionCHECK3: {{(unknown argument).*--no-standard-libraries}}
// DXCOptionCHECK3: {{(unknown argument).*--no-undefined}}
// DXCOptionCHECK3: {{(unknown argument).*--optimize}}
// DXCOptionCHECK3: {{(unknown argument).*--optimize=}}
// DXCOptionCHECK3: {{(unknown argument).*--output}}
// DXCOptionCHECK3: {{(unknown argument).*--output=}}
// DXCOptionCHECK3: {{(unknown argument).*--output-class-directory}}
// DXCOptionCHECK3: {{(unknown argument).*--output-class-directory=}}
// DXCOptionCHECK3: {{(unknown argument).*--param}}
// DXCOptionCHECK3: {{(unknown argument).*--param=}}
// DXCOptionCHECK3: {{(unknown argument).*--precompile}}
// DXCOptionCHECK3: {{(unknown argument).*--prefix}}
// DXCOptionCHECK3: {{(unknown argument).*--prefix=}}
// DXCOptionCHECK3: {{(unknown argument).*--preprocess}}
// DXCOptionCHECK3: {{(unknown argument).*--print-diagnostic-categories}}
// DXCOptionCHECK3: {{(unknown argument).*--print-file-name}}
// DXCOptionCHECK3: {{(unknown argument).*--print-missing-file-dependencies}}
// DXCOptionCHECK3: {{(unknown argument).*--print-prog-name}}
// DXCOptionCHECK3: {{(unknown argument).*--profile}}
// DXCOptionCHECK3: {{(unknown argument).*--resource}}
// DXCOptionCHECK3: {{(unknown argument).*--resource=}}
// DXCOptionCHECK3: {{(unknown argument).*--rtlib}}
// DXCOptionCHECK3: {{(unknown argument).*-serialize-diagnostics}}
// DXCOptionCHECK3: {{(unknown argument).*--signed-char}}
// DXCOptionCHECK3: {{(unknown argument).*--std}}
// DXCOptionCHECK3: {{(unknown argument).*--stdlib}}
// DXCOptionCHECK3: {{(unknown argument).*--sysroot}}
// DXCOptionCHECK3: {{(unknown argument).*--sysroot=}}
// DXCOptionCHECK3: {{(unknown argument).*--target-help}}
// DXCOptionCHECK3: {{(unknown argument).*--trace-includes}}
// DXCOptionCHECK3: {{(unknown argument).*--undefine-macro}}
// DXCOptionCHECK3: {{(unknown argument).*--undefine-macro=}}
// DXCOptionCHECK3: {{(unknown argument).*--unsigned-char}}
// DXCOptionCHECK3: {{(unknown argument).*--user-dependencies}}
// DXCOptionCHECK3: {{(unknown argument).*--verbose}}
// DXCOptionCHECK3: {{(unknown argument).*--warn-}}
// DXCOptionCHECK3: {{(unknown argument).*--warn-=}}
// DXCOptionCHECK3: {{(unknown argument).*--write-dependencies}}
// DXCOptionCHECK3: {{(unknown argument).*--write-user-dependencies}}
// DXCOptionCHECK3: {{(unknown argument).*-add-plugin}}
// DXCOptionCHECK3: {{(unknown argument).*-alias_list}}
// DXCOptionCHECK3: {{(unknown argument).*-faligned-alloc-unavailable}}
// DXCOptionCHECK3: {{(unknown argument).*-all_load}}
// DXCOptionCHECK3: {{(unknown argument).*-allowable_client}}
// DXCOptionCHECK3: {{(unknown argument).*-faltivec-src-compat=}}
// DXCOptionCHECK3: {{(unknown argument).*--amdgpu-arch-tool=}}
// DXCOptionCHECK3: {{(unknown argument).*-cfg-add-implicit-dtors}}
// DXCOptionCHECK3: {{(unknown argument).*-unoptimized-cfg}}
// DXCOptionCHECK3: {{(unknown argument).*-analyze}}
// DXCOptionCHECK3: {{(unknown argument).*-analyze-function}}
// DXCOptionCHECK3: {{(unknown argument).*-analyze-function=}}
// DXCOptionCHECK3: {{(unknown argument).*-analyzer-checker}}
// DXCOptionCHECK3: {{(unknown argument).*-analyzer-checker=}}
// DXCOptionCHECK3: {{(unknown argument).*-analyzer-checker-help}}
// DXCOptionCHECK3: {{(unknown argument).*-analyzer-checker-help-alpha}}
// DXCOptionCHECK3: {{(unknown argument).*-analyzer-checker-help-developer}}
// DXCOptionCHECK3: {{(unknown argument).*-analyzer-checker-option-help}}
// DXCOptionCHECK3: {{(unknown argument).*-analyzer-checker-option-help-alpha}}
// DXCOptionCHECK3: {{(unknown argument).*-analyzer-checker-option-help-developer}}
// RUN: not %clang_dxc -analyzer-config -analyzer-config-compatibility-mode -analyzer-config-compatibility-mode= -analyzer-config-help -analyzer-constraints -analyzer-constraints= -analyzer-disable-all-checks -analyzer-disable-checker -analyzer-disable-checker= -analyzer-disable-retry-exhausted -analyzer-display-progress -analyzer-dump-egraph -analyzer-dump-egraph= -analyzer-inline-max-stack-depth -analyzer-inline-max-stack-depth= -analyzer-inlining-mode -analyzer-inlining-mode= -analyzer-list-enabled-checkers -analyzer-max-loop -analyzer-note-analysis-entry-points -analyzer-opt-analyze-headers -analyzer-output -analyzer-output= -analyzer-purge -analyzer-purge= -analyzer-stats -analyzer-viz-egraph-graphviz -analyzer-werror -fnew-alignment -faligned-new -fno-aligned-new -fsched-interblock -ftemplate-depth- -ftree-vectorize -fno-tree-vectorize -fcuda-rdc -ftree-slp-vectorize -fno-tree-slp-vectorize -fterminated-vtables -fno-cuda-rdc --hip-device-lib-path= -grecord-gcc-switches -gno-record-gcc-switches -miphoneos-version-min= -miphonesimulator-version-min= -mmacosx-version-min= -nocudainc -nogpulib -nocudalib -print-multiarch --system-header-prefix --no-system-header-prefix -mcpu=help -mtune=help -integrated-as -no-integrated-as -coverage-data-file= -coverage-notes-file= -fopenmp-is-device -fcuda-approx-transcendentals -fno-cuda-approx-transcendentals -Gs -Qgather- -Qscatter- -Xmicrosoft-visualc-tools-root -Xmicrosoft-visualc-tools-version -Xmicrosoft-windows-sdk-root -Xmicrosoft-windows-sdk-version -Xmicrosoft-windows-sys-root -shared-libasan -static-libasan -fslp-vectorize-aggressive -frecord-gcc-switches -fno-record-gcc-switches -fno-slp-vectorize-aggressive -Xparser -Xcompiler -fexpensive-optimizations -fno-expensive-optimizations -fdefer-pop -fno-defer-pop -fextended-identifiers -fno-extended-identifiers -fsanitize-blacklist= -fno-sanitize-blacklist -fhonor-infinites -fno-honor-infinites -findirect-virtual-calls --config -ansi -arch -arch_errors_fatal -arch_only -as-secure-log-file -ast-dump -ast-dump= -ast-dump-all -ast-dump-all= -ast-dump-decl-types -ast-dump-filter  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK4 %s

// DXCOptionCHECK4: {{(unknown argument).*-analyzer-config}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-config-compatibility-mode}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-config-compatibility-mode=}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-config-help}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-constraints}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-constraints=}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-disable-all-checks}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-disable-checker}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-disable-checker=}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-disable-retry-exhausted}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-display-progress}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-dump-egraph}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-dump-egraph=}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-inline-max-stack-depth}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-inline-max-stack-depth=}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-inlining-mode}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-inlining-mode=}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-list-enabled-checkers}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-max-loop}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-note-analysis-entry-points}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-opt-analyze-headers}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-output}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-output=}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-purge}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-purge=}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-stats}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-viz-egraph-graphviz}}
// DXCOptionCHECK4: {{(unknown argument).*-analyzer-werror}}
// DXCOptionCHECK4: {{(unknown argument).*-fnew-alignment}}
// DXCOptionCHECK4: {{(unknown argument).*-faligned-new}}
// DXCOptionCHECK4: {{(unknown argument).*-fno-aligned-new}}
// DXCOptionCHECK4: {{(unknown argument).*-fsched-interblock}}
// DXCOptionCHECK4: {{(unknown argument).*-ftemplate-depth-}}
// DXCOptionCHECK4: {{(unknown argument).*-ftree-vectorize}}
// DXCOptionCHECK4: {{(unknown argument).*-fno-tree-vectorize}}
// DXCOptionCHECK4: {{(unknown argument).*-fcuda-rdc}}
// DXCOptionCHECK4: {{(unknown argument).*-ftree-slp-vectorize}}
// DXCOptionCHECK4: {{(unknown argument).*-fno-tree-slp-vectorize}}
// DXCOptionCHECK4: {{(unknown argument).*-fterminated-vtables}}
// DXCOptionCHECK4: {{(unknown argument).*-fno-cuda-rdc}}
// DXCOptionCHECK4: {{(unknown argument).*--hip-device-lib-path=}}
// DXCOptionCHECK4: {{(unknown argument).*-grecord-gcc-switches}}
// DXCOptionCHECK4: {{(unknown argument).*-gno-record-gcc-switches}}
// DXCOptionCHECK4: {{(unknown argument).*-miphoneos-version-min=}}
// DXCOptionCHECK4: {{(unknown argument).*-miphonesimulator-version-min=}}
// DXCOptionCHECK4: {{(unknown argument).*-mmacosx-version-min=}}
// DXCOptionCHECK4: {{(unknown argument).*-nocudainc}}
// DXCOptionCHECK4: {{(unknown argument).*-nogpulib}}
// DXCOptionCHECK4: {{(unknown argument).*-nocudalib}}
// DXCOptionCHECK4: {{(unknown argument).*-print-multiarch}}
// DXCOptionCHECK4: {{(unknown argument).*--system-header-prefix}}
// DXCOptionCHECK4: {{(unknown argument).*--no-system-header-prefix}}
// DXCOptionCHECK4: {{(unknown argument).*-mcpu=help}}
// DXCOptionCHECK4: {{(unknown argument).*-mtune=help}}
// DXCOptionCHECK4: {{(unknown argument).*-integrated-as}}
// DXCOptionCHECK4: {{(unknown argument).*-no-integrated-as}}
// DXCOptionCHECK4: {{(unknown argument).*-coverage-data-file=}}
// DXCOptionCHECK4: {{(unknown argument).*-coverage-notes-file=}}
// DXCOptionCHECK4: {{(unknown argument).*-fopenmp-is-device}}
// DXCOptionCHECK4: {{(unknown argument).*-fcuda-approx-transcendentals}}
// DXCOptionCHECK4: {{(unknown argument).*-fno-cuda-approx-transcendentals}}
// DXCOptionCHECK4: {{(unknown argument).*-Gs}}
// DXCOptionCHECK4: {{(unknown argument).*-Qgather-}}
// DXCOptionCHECK4: {{(unknown argument).*-Qscatter-}}
// DXCOptionCHECK4: {{(unknown argument).*-Xmicrosoft-visualc-tools-root}}
// DXCOptionCHECK4: {{(unknown argument).*-Xmicrosoft-visualc-tools-version}}
// DXCOptionCHECK4: {{(unknown argument).*-Xmicrosoft-windows-sdk-root}}
// DXCOptionCHECK4: {{(unknown argument).*-Xmicrosoft-windows-sdk-version}}
// DXCOptionCHECK4: {{(unknown argument).*-Xmicrosoft-windows-sys-root}}
// DXCOptionCHECK4: {{(unknown argument).*-shared-libasan}}
// DXCOptionCHECK4: {{(unknown argument).*-static-libasan}}
// DXCOptionCHECK4: {{(unknown argument).*-fslp-vectorize-aggressive}}
// DXCOptionCHECK4: {{(unknown argument).*-frecord-gcc-switches}}
// DXCOptionCHECK4: {{(unknown argument).*-fno-record-gcc-switches}}
// DXCOptionCHECK4: {{(unknown argument).*-fno-slp-vectorize-aggressive}}
// DXCOptionCHECK4: {{(unknown argument).*-Xparser}}
// DXCOptionCHECK4: {{(unknown argument).*-Xcompiler}}
// DXCOptionCHECK4: {{(unknown argument).*-fexpensive-optimizations}}
// DXCOptionCHECK4: {{(unknown argument).*-fno-expensive-optimizations}}
// DXCOptionCHECK4: {{(unknown argument).*-fdefer-pop}}
// DXCOptionCHECK4: {{(unknown argument).*-fno-defer-pop}}
// DXCOptionCHECK4: {{(unknown argument).*-fextended-identifiers}}
// DXCOptionCHECK4: {{(unknown argument).*-fno-extended-identifiers}}
// DXCOptionCHECK4: {{(unknown argument).*-fsanitize-blacklist=}}
// DXCOptionCHECK4: {{(unknown argument).*-fno-sanitize-blacklist}}
// DXCOptionCHECK4: {{(unknown argument).*-fhonor-infinites}}
// DXCOptionCHECK4: {{(unknown argument).*-fno-honor-infinites}}
// DXCOptionCHECK4: {{(unknown argument).*-findirect-virtual-calls}}
// DXCOptionCHECK4: {{(unknown argument).*--config}}
// DXCOptionCHECK4: {{(unknown argument).*-ansi}}
// DXCOptionCHECK4: {{(unknown argument).*-arch}}
// DXCOptionCHECK4: {{(unknown argument).*-arch_errors_fatal}}
// DXCOptionCHECK4: {{(unknown argument).*-arch_only}}
// DXCOptionCHECK4: {{(unknown argument).*-as-secure-log-file}}
// DXCOptionCHECK4: {{(unknown argument).*-ast-dump}}
// DXCOptionCHECK4: {{(unknown argument).*-ast-dump=}}
// DXCOptionCHECK4: {{(unknown argument).*-ast-dump-all}}
// DXCOptionCHECK4: {{(unknown argument).*-ast-dump-all=}}
// DXCOptionCHECK4: {{(unknown argument).*-ast-dump-decl-types}}
// DXCOptionCHECK4: {{(unknown argument).*-ast-dump-filter}}
// RUN: not %clang_dxc -ast-dump-filter= -ast-dump-lookups -ast-list -ast-merge -ast-print -ast-view --autocomplete= -aux-target-cpu -aux-target-feature -aux-triple -b -bind_at_load -building-pch-with-obj -bundle -bundle_loader -c -c-isystem -ccc- -ccc-gcc-name -cfguard -cfguard-no-checks -chain-include -cl-denorms-are-zero -cl-ext= -cl-fast-relaxed-math -cl-finite-math-only -cl-fp32-correctly-rounded-divide-sqrt -cl-kernel-arg-info -cl-mad-enable -cl-no-signed-zeros -cl-no-stdinc -cl-opt-disable -cl-single-precision-constant -cl-std= -cl-strict-aliasing -cl-uniform-work-group-size -cl-unsafe-math-optimizations -clear-ast-before-backend -client_name -code-completion-at -code-completion-at= -code-completion-brief-comments -code-completion-macros -code-completion-patterns -code-completion-with-fixits -combine -compatibility_version -compiler-options-dump -complex-range= -compress-debug-sections -compress-debug-sections= -coverage -coverage-version= -cpp -cpp-precomp --crel --cuda-compile-host-device --cuda-device-only --cuda-feature= --cuda-gpu-arch= --cuda-host-only --cuda-include-ptx= --cuda-noopt-device-debug --cuda-path= --cuda-path-ignore-env -cuid= -current_version -cxx-isystem -fc++-static-destructors -fc++-static-destructors= -dA -dD -dE -dI -dM -d -d -darwin-target-variant -darwin-target-variant-sdk-version= -darwin-target-variant-triple -dead_strip -debug-forward-template-params -debug-info-kind= -debug-info-macro -debugger-tuning= -default-function-attr --defsym -dependency-dot -dependency-file --dependent-lib= -detailed-preprocessing-record -diagnostic-log-file -serialize-diagnostic-file -disable-O0-optnone -disable-free -disable-lifetime-markers -disable-llvm-optzns -disable-llvm-passes -disable-llvm-verifier -disable-objc-default-synthesize-properties  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK5 %s

// DXCOptionCHECK5: {{(unknown argument).*-ast-dump-filter=}}
// DXCOptionCHECK5: {{(unknown argument).*-ast-dump-lookups}}
// DXCOptionCHECK5: {{(unknown argument).*-ast-list}}
// DXCOptionCHECK5: {{(unknown argument).*-ast-merge}}
// DXCOptionCHECK5: {{(unknown argument).*-ast-print}}
// DXCOptionCHECK5: {{(unknown argument).*-ast-view}}
// DXCOptionCHECK5: {{(unknown argument).*--autocomplete=}}
// DXCOptionCHECK5: {{(unknown argument).*-aux-target-cpu}}
// DXCOptionCHECK5: {{(unknown argument).*-aux-target-feature}}
// DXCOptionCHECK5: {{(unknown argument).*-aux-triple}}
// DXCOptionCHECK5: {{(unknown argument).*-b}}
// DXCOptionCHECK5: {{(unknown argument).*-bind_at_load}}
// DXCOptionCHECK5: {{(unknown argument).*-building-pch-with-obj}}
// DXCOptionCHECK5: {{(unknown argument).*-bundle}}
// DXCOptionCHECK5: {{(unknown argument).*-bundle_loader}}
// DXCOptionCHECK5: {{(unknown argument).*-c}}
// DXCOptionCHECK5: {{(unknown argument).*-c-isystem}}
// DXCOptionCHECK5: {{(unknown argument).*-ccc-}}
// DXCOptionCHECK5: {{(unknown argument).*-ccc-gcc-name}}
// DXCOptionCHECK5: {{(unknown argument).*-cfguard}}
// DXCOptionCHECK5: {{(unknown argument).*-cfguard-no-checks}}
// DXCOptionCHECK5: {{(unknown argument).*-chain-include}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-denorms-are-zero}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-ext=}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-fast-relaxed-math}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-finite-math-only}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-fp32-correctly-rounded-divide-sqrt}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-kernel-arg-info}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-mad-enable}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-no-signed-zeros}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-no-stdinc}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-opt-disable}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-single-precision-constant}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-std=}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-strict-aliasing}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-uniform-work-group-size}}
// DXCOptionCHECK5: {{(unknown argument).*-cl-unsafe-math-optimizations}}
// DXCOptionCHECK5: {{(unknown argument).*-clear-ast-before-backend}}
// DXCOptionCHECK5: {{(unknown argument).*-client_name}}
// DXCOptionCHECK5: {{(unknown argument).*-code-completion-at}}
// DXCOptionCHECK5: {{(unknown argument).*-code-completion-at=}}
// DXCOptionCHECK5: {{(unknown argument).*-code-completion-brief-comments}}
// DXCOptionCHECK5: {{(unknown argument).*-code-completion-macros}}
// DXCOptionCHECK5: {{(unknown argument).*-code-completion-patterns}}
// DXCOptionCHECK5: {{(unknown argument).*-code-completion-with-fixits}}
// DXCOptionCHECK5: {{(unknown argument).*-combine}}
// DXCOptionCHECK5: {{(unknown argument).*-compatibility_version}}
// DXCOptionCHECK5: {{(unknown argument).*-compiler-options-dump}}
// DXCOptionCHECK5: {{(unknown argument).*-complex-range=}}
// DXCOptionCHECK5: {{(unknown argument).*-compress-debug-sections}}
// DXCOptionCHECK5: {{(unknown argument).*-compress-debug-sections=}}
// DXCOptionCHECK5: {{(unknown argument).*-coverage}}
// DXCOptionCHECK5: {{(unknown argument).*-coverage-version=}}
// DXCOptionCHECK5: {{(unknown argument).*-cpp}}
// DXCOptionCHECK5: {{(unknown argument).*-cpp-precomp}}
// DXCOptionCHECK5: {{(unknown argument).*--crel}}
// DXCOptionCHECK5: {{(unknown argument).*--cuda-compile-host-device}}
// DXCOptionCHECK5: {{(unknown argument).*--cuda-device-only}}
// DXCOptionCHECK5: {{(unknown argument).*--cuda-feature=}}
// DXCOptionCHECK5: {{(unknown argument).*--cuda-gpu-arch=}}
// DXCOptionCHECK5: {{(unknown argument).*--cuda-host-only}}
// DXCOptionCHECK5: {{(unknown argument).*--cuda-include-ptx=}}
// DXCOptionCHECK5: {{(unknown argument).*--cuda-noopt-device-debug}}
// DXCOptionCHECK5: {{(unknown argument).*--cuda-path=}}
// DXCOptionCHECK5: {{(unknown argument).*--cuda-path-ignore-env}}
// DXCOptionCHECK5: {{(unknown argument).*-cuid=}}
// DXCOptionCHECK5: {{(unknown argument).*-current_version}}
// DXCOptionCHECK5: {{(unknown argument).*-cxx-isystem}}
// DXCOptionCHECK5: {{(unknown argument).*-fc\+\+-static-destructors}}
// DXCOptionCHECK5: {{(unknown argument).*-fc\+\+-static-destructors=}}
// DXCOptionCHECK5: {{(unknown argument).*-dA}}
// DXCOptionCHECK5: {{(unknown argument).*-dD}}
// DXCOptionCHECK5: {{(unknown argument).*-dE}}
// DXCOptionCHECK5: {{(unknown argument).*-dI}}
// DXCOptionCHECK5: {{(unknown argument).*-dM}}
// DXCOptionCHECK5: {{(unknown argument).*-d}}
// DXCOptionCHECK5: {{(unknown argument).*-d}}
// DXCOptionCHECK5: {{(unknown argument).*-darwin-target-variant}}
// DXCOptionCHECK5: {{(unknown argument).*-darwin-target-variant-sdk-version=}}
// DXCOptionCHECK5: {{(unknown argument).*-darwin-target-variant-triple}}
// DXCOptionCHECK5: {{(unknown argument).*-dead_strip}}
// DXCOptionCHECK5: {{(unknown argument).*-debug-forward-template-params}}
// DXCOptionCHECK5: {{(unknown argument).*-debug-info-kind=}}
// DXCOptionCHECK5: {{(unknown argument).*-debug-info-macro}}
// DXCOptionCHECK5: {{(unknown argument).*-debugger-tuning=}}
// DXCOptionCHECK5: {{(unknown argument).*-default-function-attr}}
// DXCOptionCHECK5: {{(unknown argument).*--defsym}}
// DXCOptionCHECK5: {{(unknown argument).*-dependency-dot}}
// DXCOptionCHECK5: {{(unknown argument).*-dependency-file}}
// DXCOptionCHECK5: {{(unknown argument).*--dependent-lib=}}
// DXCOptionCHECK5: {{(unknown argument).*-detailed-preprocessing-record}}
// DXCOptionCHECK5: {{(unknown argument).*-diagnostic-log-file}}
// DXCOptionCHECK5: {{(unknown argument).*-serialize-diagnostic-file}}
// DXCOptionCHECK5: {{(unknown argument).*-disable-O0-optnone}}
// DXCOptionCHECK5: {{(unknown argument).*-disable-free}}
// DXCOptionCHECK5: {{(unknown argument).*-disable-lifetime-markers}}
// DXCOptionCHECK5: {{(unknown argument).*-disable-llvm-optzns}}
// DXCOptionCHECK5: {{(unknown argument).*-disable-llvm-passes}}
// DXCOptionCHECK5: {{(unknown argument).*-disable-llvm-verifier}}
// DXCOptionCHECK5: {{(unknown argument).*-disable-objc-default-synthesize-properties}}
// RUN: not %clang_dxc -disable-pragma-debug-crash -disable-red-zone -discard-value-names -dsym-dir -dump-coverage-mapping -dump-deserialized-decls -dump-raw-tokens -dump-tokens -dumpdir -dumpmachine -dumpspecs -dumpversion -dwarf-debug-flags -dwarf-debug-producer -dwarf-explicit-import -dwarf-ext-refs -dwarf-version= -dylib_file -dylinker -dylinker_install_name -dynamic -dynamiclib -e -ehcontguard --embed-dir= -emit-cir -emit-codegen-only --emit-extension-symbol-graphs -emit-fir -emit-header-unit -emit-hlfir -emit-html -emit-interface-stubs -emit-llvm -emit-llvm-bc -emit-llvm-only -emit-llvm-uselists -emit-merged-ifs -emit-mlir -emit-module -emit-module-interface -emit-obj -emit-pch --pretty-sgf -emit-reduced-module-interface --emit-sgf-symbol-labels-for-testing --emit-static-lib -emit-symbol-graph -enable-noundef-analysis -enable-tlsdesc -error-on-deserialized-decl -error-on-deserialized-decl= -exception-model -exception-model= -fexperimental-modules-reduced-bmi -exported_symbols_list -extract-api --extract-api-ignores= -fPIC -fPIE -faapcs-bitfield-width -faarch64-jump-table-hardening -faccess-control -faddress-space-map-mangling= -faddrsig -faggressive-function-elimination -falign-commons -falign-functions -falign-functions= -falign-jumps -falign-jumps= -falign-labels -falign-labels= -falign-loops -falign-loops= -faligned-allocation -faligned-new= -fall-intrinsics -fallow-editor-placeholders -fallow-pch-with-different-modules-cache-path -fallow-pch-with-compiler-errors -fallow-pcm-with-compiler-errors -fallow-unsupported -falternative-parameter-statement -faltivec -fanalyzed-objects-for-unparse -fandroid-pad-segment -fkeep-inline-functions -funit-at-a-time -fapinotes -fapinotes-modules -fapinotes-swift-version= -fapple-kext -fapple-link-rtlib -fapple-pragma-pack -fapplication-extension -fapply-global-visibility-to-externs -fapprox-func -fasm -fasm-blocks  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK6 %s

// DXCOptionCHECK6: {{(unknown argument).*-disable-pragma-debug-crash}}
// DXCOptionCHECK6: {{(unknown argument).*-disable-red-zone}}
// DXCOptionCHECK6: {{(unknown argument).*-discard-value-names}}
// DXCOptionCHECK6: {{(unknown argument).*-dsym-dir}}
// DXCOptionCHECK6: {{(unknown argument).*-dump-coverage-mapping}}
// DXCOptionCHECK6: {{(unknown argument).*-dump-deserialized-decls}}
// DXCOptionCHECK6: {{(unknown argument).*-dump-raw-tokens}}
// DXCOptionCHECK6: {{(unknown argument).*-dump-tokens}}
// DXCOptionCHECK6: {{(unknown argument).*-dumpdir}}
// DXCOptionCHECK6: {{(unknown argument).*-dumpmachine}}
// DXCOptionCHECK6: {{(unknown argument).*-dumpspecs}}
// DXCOptionCHECK6: {{(unknown argument).*-dumpversion}}
// DXCOptionCHECK6: {{(unknown argument).*-dwarf-debug-flags}}
// DXCOptionCHECK6: {{(unknown argument).*-dwarf-debug-producer}}
// DXCOptionCHECK6: {{(unknown argument).*-dwarf-explicit-import}}
// DXCOptionCHECK6: {{(unknown argument).*-dwarf-ext-refs}}
// DXCOptionCHECK6: {{(unknown argument).*-dwarf-version=}}
// DXCOptionCHECK6: {{(unknown argument).*-dylib_file}}
// DXCOptionCHECK6: {{(unknown argument).*-dylinker}}
// DXCOptionCHECK6: {{(unknown argument).*-dylinker_install_name}}
// DXCOptionCHECK6: {{(unknown argument).*-dynamic}}
// DXCOptionCHECK6: {{(unknown argument).*-dynamiclib}}
// DXCOptionCHECK6: {{(unknown argument).*-e}}
// DXCOptionCHECK6: {{(unknown argument).*-ehcontguard}}
// DXCOptionCHECK6: {{(unknown argument).*--embed-dir=}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-cir}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-codegen-only}}
// DXCOptionCHECK6: {{(unknown argument).*--emit-extension-symbol-graphs}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-fir}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-header-unit}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-hlfir}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-html}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-interface-stubs}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-llvm}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-llvm-bc}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-llvm-only}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-llvm-uselists}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-merged-ifs}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-mlir}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-module}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-module-interface}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-obj}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-pch}}
// DXCOptionCHECK6: {{(unknown argument).*--pretty-sgf}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-reduced-module-interface}}
// DXCOptionCHECK6: {{(unknown argument).*--emit-sgf-symbol-labels-for-testing}}
// DXCOptionCHECK6: {{(unknown argument).*--emit-static-lib}}
// DXCOptionCHECK6: {{(unknown argument).*-emit-symbol-graph}}
// DXCOptionCHECK6: {{(unknown argument).*-enable-noundef-analysis}}
// DXCOptionCHECK6: {{(unknown argument).*-enable-tlsdesc}}
// DXCOptionCHECK6: {{(unknown argument).*-error-on-deserialized-decl}}
// DXCOptionCHECK6: {{(unknown argument).*-error-on-deserialized-decl=}}
// DXCOptionCHECK6: {{(unknown argument).*-exception-model}}
// DXCOptionCHECK6: {{(unknown argument).*-exception-model=}}
// DXCOptionCHECK6: {{(unknown argument).*-fexperimental-modules-reduced-bmi}}
// DXCOptionCHECK6: {{(unknown argument).*-exported_symbols_list}}
// DXCOptionCHECK6: {{(unknown argument).*-extract-api}}
// DXCOptionCHECK6: {{(unknown argument).*--extract-api-ignores=}}
// DXCOptionCHECK6: {{(unknown argument).*-fPIC}}
// DXCOptionCHECK6: {{(unknown argument).*-fPIE}}
// DXCOptionCHECK6: {{(unknown argument).*-faapcs-bitfield-width}}
// DXCOptionCHECK6: {{(unknown argument).*-faarch64-jump-table-hardening}}
// DXCOptionCHECK6: {{(unknown argument).*-faccess-control}}
// DXCOptionCHECK6: {{(unknown argument).*-faddress-space-map-mangling=}}
// DXCOptionCHECK6: {{(unknown argument).*-faddrsig}}
// DXCOptionCHECK6: {{(unknown argument).*-faggressive-function-elimination}}
// DXCOptionCHECK6: {{(unknown argument).*-falign-commons}}
// DXCOptionCHECK6: {{(unknown argument).*-falign-functions}}
// DXCOptionCHECK6: {{(unknown argument).*-falign-functions=}}
// DXCOptionCHECK6: {{(unknown argument).*-falign-jumps}}
// DXCOptionCHECK6: {{(unknown argument).*-falign-jumps=}}
// DXCOptionCHECK6: {{(unknown argument).*-falign-labels}}
// DXCOptionCHECK6: {{(unknown argument).*-falign-labels=}}
// DXCOptionCHECK6: {{(unknown argument).*-falign-loops}}
// DXCOptionCHECK6: {{(unknown argument).*-falign-loops=}}
// DXCOptionCHECK6: {{(unknown argument).*-faligned-allocation}}
// DXCOptionCHECK6: {{(unknown argument).*-faligned-new=}}
// DXCOptionCHECK6: {{(unknown argument).*-fall-intrinsics}}
// DXCOptionCHECK6: {{(unknown argument).*-fallow-editor-placeholders}}
// DXCOptionCHECK6: {{(unknown argument).*-fallow-pch-with-different-modules-cache-path}}
// DXCOptionCHECK6: {{(unknown argument).*-fallow-pch-with-compiler-errors}}
// DXCOptionCHECK6: {{(unknown argument).*-fallow-pcm-with-compiler-errors}}
// DXCOptionCHECK6: {{(unknown argument).*-fallow-unsupported}}
// DXCOptionCHECK6: {{(unknown argument).*-falternative-parameter-statement}}
// DXCOptionCHECK6: {{(unknown argument).*-faltivec}}
// DXCOptionCHECK6: {{(unknown argument).*-fanalyzed-objects-for-unparse}}
// DXCOptionCHECK6: {{(unknown argument).*-fandroid-pad-segment}}
// DXCOptionCHECK6: {{(unknown argument).*-fkeep-inline-functions}}
// DXCOptionCHECK6: {{(unknown argument).*-funit-at-a-time}}
// DXCOptionCHECK6: {{(unknown argument).*-fapinotes}}
// DXCOptionCHECK6: {{(unknown argument).*-fapinotes-modules}}
// DXCOptionCHECK6: {{(unknown argument).*-fapinotes-swift-version=}}
// DXCOptionCHECK6: {{(unknown argument).*-fapple-kext}}
// DXCOptionCHECK6: {{(unknown argument).*-fapple-link-rtlib}}
// DXCOptionCHECK6: {{(unknown argument).*-fapple-pragma-pack}}
// DXCOptionCHECK6: {{(unknown argument).*-fapplication-extension}}
// DXCOptionCHECK6: {{(unknown argument).*-fapply-global-visibility-to-externs}}
// DXCOptionCHECK6: {{(unknown argument).*-fapprox-func}}
// DXCOptionCHECK6: {{(unknown argument).*-fasm}}
// DXCOptionCHECK6: {{(unknown argument).*-fasm-blocks}}
// RUN: not %clang_dxc -fassociative-math -fassume-nothrow-exception-dtor -fassume-sane-operator-new -fassume-unique-vtables -fassumptions -fast -fastcp -fastf -fasync-exceptions -fasynchronous-unwind-tables -fauto-import -fauto-profile= -fauto-profile-accurate -fautolink -fautomatic -fbackslash -fbacktrace -fbasic-block-address-map -fbasic-block-sections= -fbfloat16-excess-precision= -fbinutils-version= -fblas-matmul-limit= -fblocks -fblocks-runtime-optional -fbootclasspath= -fborland-extensions -fbounds-check -fexperimental-bounds-safety -fbracket-depth -fbracket-depth= -fbranch-count-reg -fbuild-session-file= -fbuild-session-timestamp= -fbuiltin-headers-in-system-modules -fbuiltin-module-map -fcall-saved-x10 -fcall-saved-x11 -fcall-saved-x12 -fcall-saved-x13 -fcall-saved-x14 -fcall-saved-x15 -fcall-saved-x18 -fcall-saved-x8 -fcall-saved-x9 -fcaller-saves -fcaret-diagnostics -fcf-protection -fcf-protection= -fcf-runtime-abi= -fchar8_t -fcheck= -fcheck-array-temporaries -fcheck-new -fclang-abi-compat= -fclangir -fclasspath= -fcoarray= -fcodegen-data-generate -fcodegen-data-generate= -fcodegen-data-use -fcodegen-data-use= -fcomment-block-commands= -fcommon -fcompatibility-qualified-id-block-type-checking -fcompile-resource= -fcomplete-member-pointers -fcomplex-arithmetic= -fconst-strings -fconstant-cfstrings -fconstant-string-class -fconstant-string-class= -fconstexpr-backtrace-limit= -fconstexpr-depth= -fconstexpr-steps= -fconvergent-functions -fconvert= -fcoro-aligned-allocation -fcoroutines -fcoverage-compilation-dir= -fcoverage-mapping -fcoverage-prefix-map= -fcray-pointer -fcreate-profile -fcs-profile-generate -fcs-profile-generate= -fctor-dtor-return-this -fcuda-allow-variadic-functions -fcuda-flush-denormals-to-zero -fcuda-include-gpubinary -fcuda-is-device -fcuda-short-ptr -fcx-fortran-rules -fcx-limited-range -fc++-abi= -fcxx-exceptions -fcxx-modules -fd-lines-as-code -fd-lines-as-comments -fdata-sections -fdebug-default-version=  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK7 %s

// DXCOptionCHECK7: {{(unknown argument).*-fassociative-math}}
// DXCOptionCHECK7: {{(unknown argument).*-fassume-nothrow-exception-dtor}}
// DXCOptionCHECK7: {{(unknown argument).*-fassume-sane-operator-new}}
// DXCOptionCHECK7: {{(unknown argument).*-fassume-unique-vtables}}
// DXCOptionCHECK7: {{(unknown argument).*-fassumptions}}
// DXCOptionCHECK7: {{(unknown argument).*-fast}}
// DXCOptionCHECK7: {{(unknown argument).*-fastcp}}
// DXCOptionCHECK7: {{(unknown argument).*-fastf}}
// DXCOptionCHECK7: {{(unknown argument).*-fasync-exceptions}}
// DXCOptionCHECK7: {{(unknown argument).*-fasynchronous-unwind-tables}}
// DXCOptionCHECK7: {{(unknown argument).*-fauto-import}}
// DXCOptionCHECK7: {{(unknown argument).*-fauto-profile=}}
// DXCOptionCHECK7: {{(unknown argument).*-fauto-profile-accurate}}
// DXCOptionCHECK7: {{(unknown argument).*-fautolink}}
// DXCOptionCHECK7: {{(unknown argument).*-fautomatic}}
// DXCOptionCHECK7: {{(unknown argument).*-fbackslash}}
// DXCOptionCHECK7: {{(unknown argument).*-fbacktrace}}
// DXCOptionCHECK7: {{(unknown argument).*-fbasic-block-address-map}}
// DXCOptionCHECK7: {{(unknown argument).*-fbasic-block-sections=}}
// DXCOptionCHECK7: {{(unknown argument).*-fbfloat16-excess-precision=}}
// DXCOptionCHECK7: {{(unknown argument).*-fbinutils-version=}}
// DXCOptionCHECK7: {{(unknown argument).*-fblas-matmul-limit=}}
// DXCOptionCHECK7: {{(unknown argument).*-fblocks}}
// DXCOptionCHECK7: {{(unknown argument).*-fblocks-runtime-optional}}
// DXCOptionCHECK7: {{(unknown argument).*-fbootclasspath=}}
// DXCOptionCHECK7: {{(unknown argument).*-fborland-extensions}}
// DXCOptionCHECK7: {{(unknown argument).*-fbounds-check}}
// DXCOptionCHECK7: {{(unknown argument).*-fexperimental-bounds-safety}}
// DXCOptionCHECK7: {{(unknown argument).*-fbracket-depth}}
// DXCOptionCHECK7: {{(unknown argument).*-fbracket-depth=}}
// DXCOptionCHECK7: {{(unknown argument).*-fbranch-count-reg}}
// DXCOptionCHECK7: {{(unknown argument).*-fbuild-session-file=}}
// DXCOptionCHECK7: {{(unknown argument).*-fbuild-session-timestamp=}}
// DXCOptionCHECK7: {{(unknown argument).*-fbuiltin-headers-in-system-modules}}
// DXCOptionCHECK7: {{(unknown argument).*-fbuiltin-module-map}}
// DXCOptionCHECK7: {{(unknown argument).*-fcall-saved-x10}}
// DXCOptionCHECK7: {{(unknown argument).*-fcall-saved-x11}}
// DXCOptionCHECK7: {{(unknown argument).*-fcall-saved-x12}}
// DXCOptionCHECK7: {{(unknown argument).*-fcall-saved-x13}}
// DXCOptionCHECK7: {{(unknown argument).*-fcall-saved-x14}}
// DXCOptionCHECK7: {{(unknown argument).*-fcall-saved-x15}}
// DXCOptionCHECK7: {{(unknown argument).*-fcall-saved-x18}}
// DXCOptionCHECK7: {{(unknown argument).*-fcall-saved-x8}}
// DXCOptionCHECK7: {{(unknown argument).*-fcall-saved-x9}}
// DXCOptionCHECK7: {{(unknown argument).*-fcaller-saves}}
// DXCOptionCHECK7: {{(unknown argument).*-fcaret-diagnostics}}
// DXCOptionCHECK7: {{(unknown argument).*-fcf-protection}}
// DXCOptionCHECK7: {{(unknown argument).*-fcf-protection=}}
// DXCOptionCHECK7: {{(unknown argument).*-fcf-runtime-abi=}}
// DXCOptionCHECK7: {{(unknown argument).*-fchar8_t}}
// DXCOptionCHECK7: {{(unknown argument).*-fcheck=}}
// DXCOptionCHECK7: {{(unknown argument).*-fcheck-array-temporaries}}
// DXCOptionCHECK7: {{(unknown argument).*-fcheck-new}}
// DXCOptionCHECK7: {{(unknown argument).*-fclang-abi-compat=}}
// DXCOptionCHECK7: {{(unknown argument).*-fclangir}}
// DXCOptionCHECK7: {{(unknown argument).*-fclasspath=}}
// DXCOptionCHECK7: {{(unknown argument).*-fcoarray=}}
// DXCOptionCHECK7: {{(unknown argument).*-fcodegen-data-generate}}
// DXCOptionCHECK7: {{(unknown argument).*-fcodegen-data-generate=}}
// DXCOptionCHECK7: {{(unknown argument).*-fcodegen-data-use}}
// DXCOptionCHECK7: {{(unknown argument).*-fcodegen-data-use=}}
// DXCOptionCHECK7: {{(unknown argument).*-fcomment-block-commands=}}
// DXCOptionCHECK7: {{(unknown argument).*-fcommon}}
// DXCOptionCHECK7: {{(unknown argument).*-fcompatibility-qualified-id-block-type-checking}}
// DXCOptionCHECK7: {{(unknown argument).*-fcompile-resource=}}
// DXCOptionCHECK7: {{(unknown argument).*-fcomplete-member-pointers}}
// DXCOptionCHECK7: {{(unknown argument).*-fcomplex-arithmetic=}}
// DXCOptionCHECK7: {{(unknown argument).*-fconst-strings}}
// DXCOptionCHECK7: {{(unknown argument).*-fconstant-cfstrings}}
// DXCOptionCHECK7: {{(unknown argument).*-fconstant-string-class}}
// DXCOptionCHECK7: {{(unknown argument).*-fconstant-string-class=}}
// DXCOptionCHECK7: {{(unknown argument).*-fconstexpr-backtrace-limit=}}
// DXCOptionCHECK7: {{(unknown argument).*-fconstexpr-depth=}}
// DXCOptionCHECK7: {{(unknown argument).*-fconstexpr-steps=}}
// DXCOptionCHECK7: {{(unknown argument).*-fconvergent-functions}}
// DXCOptionCHECK7: {{(unknown argument).*-fconvert=}}
// DXCOptionCHECK7: {{(unknown argument).*-fcoro-aligned-allocation}}
// DXCOptionCHECK7: {{(unknown argument).*-fcoroutines}}
// DXCOptionCHECK7: {{(unknown argument).*-fcoverage-compilation-dir=}}
// DXCOptionCHECK7: {{(unknown argument).*-fcoverage-mapping}}
// DXCOptionCHECK7: {{(unknown argument).*-fcoverage-prefix-map=}}
// DXCOptionCHECK7: {{(unknown argument).*-fcray-pointer}}
// DXCOptionCHECK7: {{(unknown argument).*-fcreate-profile}}
// DXCOptionCHECK7: {{(unknown argument).*-fcs-profile-generate}}
// DXCOptionCHECK7: {{(unknown argument).*-fcs-profile-generate=}}
// DXCOptionCHECK7: {{(unknown argument).*-fctor-dtor-return-this}}
// DXCOptionCHECK7: {{(unknown argument).*-fcuda-allow-variadic-functions}}
// DXCOptionCHECK7: {{(unknown argument).*-fcuda-flush-denormals-to-zero}}
// DXCOptionCHECK7: {{(unknown argument).*-fcuda-include-gpubinary}}
// DXCOptionCHECK7: {{(unknown argument).*-fcuda-is-device}}
// DXCOptionCHECK7: {{(unknown argument).*-fcuda-short-ptr}}
// DXCOptionCHECK7: {{(unknown argument).*-fcx-fortran-rules}}
// DXCOptionCHECK7: {{(unknown argument).*-fcx-limited-range}}
// DXCOptionCHECK7: {{(unknown argument).*-fc\+\+-abi=}}
// DXCOptionCHECK7: {{(unknown argument).*-fcxx-exceptions}}
// DXCOptionCHECK7: {{(unknown argument).*-fcxx-modules}}
// DXCOptionCHECK7: {{(unknown argument).*-fd-lines-as-code}}
// DXCOptionCHECK7: {{(unknown argument).*-fd-lines-as-comments}}
// DXCOptionCHECK7: {{(unknown argument).*-fdata-sections}}
// DXCOptionCHECK7: {{(unknown argument).*-fdebug-default-version=}}
// RUN: not %clang_dxc -fdebug-dump-all -fdebug-dump-parse-tree -fdebug-dump-parse-tree-no-sema -fdebug-dump-parsing-log -fdebug-dump-pft -fdebug-dump-provenance -fdebug-dump-symbols -fdebug-info-for-profiling -fdebug-measure-parse-tree -fdebug-module-writer -fdebug-pass-arguments -fdebug-pass-manager -fdebug-pass-structure -fdebug-pre-fir-tree -fdebug-prefix-map= -fdebug-ranges-base-address -fdebug-types-section -fdebug-unparse -fdebug-unparse-no-sema -fdebug-unparse-with-modules -fdebug-unparse-with-symbols -fdebugger-cast-result-to-id -fdebugger-objc-literal -fdebugger-support -fdeclare-opencl-builtins -fdeclspec -fdefault-calling-conv= -fdefault-double-8 -fdefault-inline -fdefault-integer-8 -fdefault-real-8 -fdefine-target-os-macros -fdelayed-template-parsing -fdelete-null-pointer-checks -fdenormal-fp-math= -fdenormal-fp-math-f32= -fdepfile-entry= -fdeprecated-macro -fdevirtualize -fdevirtualize-speculatively -fdiagnostics-fixit-info -fdiagnostics-format -fdiagnostics-format= -fdiagnostics-hotness-threshold= -fdiagnostics-misexpect-tolerance= -fdiagnostics-print-source-range-info -fdiagnostics-show-category -fdiagnostics-show-category= -fdiagnostics-show-hotness -fdiagnostics-show-line-numbers -fdiagnostics-show-location= -fdiagnostics-show-note-include-stack -fdiagnostics-show-option -fdiagnostics-show-template-tree -fdigraphs -fdirect-access-external-data -fdirectives-only -fdisable-block-signature-string -fdisable-integer-16 -fdisable-integer-2 -fdisable-module-hash -fdisable-real-10 -fdisable-real-3 -fdollar-ok -fdollars-in-identifiers -fdouble-square-bracket-attributes -fdump-fortran-optimized -fdump-fortran-original -fdump-parse-tree -fdump-record-layouts -fdump-record-layouts-canonical -fdump-record-layouts-complete -fdump-record-layouts-simple -fdump-vtable-layouts -fdwarf2-cfi-asm -fdwarf-directory-asm -fdwarf-exceptions -felide-constructors -feliminate-unused-debug-symbols -feliminate-unused-debug-types -fembed-bitcode -fembed-bitcode= -fembed-bitcode-marker -fembed-offload-object= -femit-all-decls -femit-compact-unwind-non-canonical -femit-dwarf-unwind= -femulated-tls -fenable-matrix -fencode-extended-block-signature -fencoding= -ferror-limit -fescaping-block-tail-calls -fexceptions -fexcess-precision= -fexec-charset= -fexperimental-assignment-tracking= -fexperimental-isel -fexperimental-late-parse-attributes -fexperimental-library  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK8 %s

// DXCOptionCHECK8: {{(unknown argument).*-fdebug-dump-all}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-dump-parse-tree}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-dump-parse-tree-no-sema}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-dump-parsing-log}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-dump-pft}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-dump-provenance}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-dump-symbols}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-info-for-profiling}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-measure-parse-tree}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-module-writer}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-pass-arguments}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-pass-manager}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-pass-structure}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-pre-fir-tree}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-prefix-map=}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-ranges-base-address}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-types-section}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-unparse}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-unparse-no-sema}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-unparse-with-modules}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebug-unparse-with-symbols}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebugger-cast-result-to-id}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebugger-objc-literal}}
// DXCOptionCHECK8: {{(unknown argument).*-fdebugger-support}}
// DXCOptionCHECK8: {{(unknown argument).*-fdeclare-opencl-builtins}}
// DXCOptionCHECK8: {{(unknown argument).*-fdeclspec}}
// DXCOptionCHECK8: {{(unknown argument).*-fdefault-calling-conv=}}
// DXCOptionCHECK8: {{(unknown argument).*-fdefault-double-8}}
// DXCOptionCHECK8: {{(unknown argument).*-fdefault-inline}}
// DXCOptionCHECK8: {{(unknown argument).*-fdefault-integer-8}}
// DXCOptionCHECK8: {{(unknown argument).*-fdefault-real-8}}
// DXCOptionCHECK8: {{(unknown argument).*-fdefine-target-os-macros}}
// DXCOptionCHECK8: {{(unknown argument).*-fdelayed-template-parsing}}
// DXCOptionCHECK8: {{(unknown argument).*-fdelete-null-pointer-checks}}
// DXCOptionCHECK8: {{(unknown argument).*-fdenormal-fp-math=}}
// DXCOptionCHECK8: {{(unknown argument).*-fdenormal-fp-math-f32=}}
// DXCOptionCHECK8: {{(unknown argument).*-fdepfile-entry=}}
// DXCOptionCHECK8: {{(unknown argument).*-fdeprecated-macro}}
// DXCOptionCHECK8: {{(unknown argument).*-fdevirtualize}}
// DXCOptionCHECK8: {{(unknown argument).*-fdevirtualize-speculatively}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-fixit-info}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-format}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-format=}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-hotness-threshold=}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-misexpect-tolerance=}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-print-source-range-info}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-show-category}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-show-category=}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-show-hotness}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-show-line-numbers}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-show-location=}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-show-note-include-stack}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-show-option}}
// DXCOptionCHECK8: {{(unknown argument).*-fdiagnostics-show-template-tree}}
// DXCOptionCHECK8: {{(unknown argument).*-fdigraphs}}
// DXCOptionCHECK8: {{(unknown argument).*-fdirect-access-external-data}}
// DXCOptionCHECK8: {{(unknown argument).*-fdirectives-only}}
// DXCOptionCHECK8: {{(unknown argument).*-fdisable-block-signature-string}}
// DXCOptionCHECK8: {{(unknown argument).*-fdisable-integer-16}}
// DXCOptionCHECK8: {{(unknown argument).*-fdisable-integer-2}}
// DXCOptionCHECK8: {{(unknown argument).*-fdisable-module-hash}}
// DXCOptionCHECK8: {{(unknown argument).*-fdisable-real-10}}
// DXCOptionCHECK8: {{(unknown argument).*-fdisable-real-3}}
// DXCOptionCHECK8: {{(unknown argument).*-fdollar-ok}}
// DXCOptionCHECK8: {{(unknown argument).*-fdollars-in-identifiers}}
// DXCOptionCHECK8: {{(unknown argument).*-fdouble-square-bracket-attributes}}
// DXCOptionCHECK8: {{(unknown argument).*-fdump-fortran-optimized}}
// DXCOptionCHECK8: {{(unknown argument).*-fdump-fortran-original}}
// DXCOptionCHECK8: {{(unknown argument).*-fdump-parse-tree}}
// DXCOptionCHECK8: {{(unknown argument).*-fdump-record-layouts}}
// DXCOptionCHECK8: {{(unknown argument).*-fdump-record-layouts-canonical}}
// DXCOptionCHECK8: {{(unknown argument).*-fdump-record-layouts-complete}}
// DXCOptionCHECK8: {{(unknown argument).*-fdump-record-layouts-simple}}
// DXCOptionCHECK8: {{(unknown argument).*-fdump-vtable-layouts}}
// DXCOptionCHECK8: {{(unknown argument).*-fdwarf2-cfi-asm}}
// DXCOptionCHECK8: {{(unknown argument).*-fdwarf-directory-asm}}
// DXCOptionCHECK8: {{(unknown argument).*-fdwarf-exceptions}}
// DXCOptionCHECK8: {{(unknown argument).*-felide-constructors}}
// DXCOptionCHECK8: {{(unknown argument).*-feliminate-unused-debug-symbols}}
// DXCOptionCHECK8: {{(unknown argument).*-feliminate-unused-debug-types}}
// DXCOptionCHECK8: {{(unknown argument).*-fembed-bitcode}}
// DXCOptionCHECK8: {{(unknown argument).*-fembed-bitcode=}}
// DXCOptionCHECK8: {{(unknown argument).*-fembed-bitcode-marker}}
// DXCOptionCHECK8: {{(unknown argument).*-fembed-offload-object=}}
// DXCOptionCHECK8: {{(unknown argument).*-femit-all-decls}}
// DXCOptionCHECK8: {{(unknown argument).*-femit-compact-unwind-non-canonical}}
// DXCOptionCHECK8: {{(unknown argument).*-femit-dwarf-unwind=}}
// DXCOptionCHECK8: {{(unknown argument).*-femulated-tls}}
// DXCOptionCHECK8: {{(unknown argument).*-fenable-matrix}}
// DXCOptionCHECK8: {{(unknown argument).*-fencode-extended-block-signature}}
// DXCOptionCHECK8: {{(unknown argument).*-fencoding=}}
// DXCOptionCHECK8: {{(unknown argument).*-ferror-limit}}
// DXCOptionCHECK8: {{(unknown argument).*-fescaping-block-tail-calls}}
// DXCOptionCHECK8: {{(unknown argument).*-fexceptions}}
// DXCOptionCHECK8: {{(unknown argument).*-fexcess-precision=}}
// DXCOptionCHECK8: {{(unknown argument).*-fexec-charset=}}
// DXCOptionCHECK8: {{(unknown argument).*-fexperimental-assignment-tracking=}}
// DXCOptionCHECK8: {{(unknown argument).*-fexperimental-isel}}
// DXCOptionCHECK8: {{(unknown argument).*-fexperimental-late-parse-attributes}}
// DXCOptionCHECK8: {{(unknown argument).*-fexperimental-library}}
// RUN: not %clang_dxc -fexperimental-max-bitint-width= -fexperimental-new-constant-interpreter -fexperimental-omit-vtable-rtti -fexperimental-relative-c++-abi-vtables -fexperimental-sanitize-metadata= -fexperimental-sanitize-metadata=atomics -fexperimental-sanitize-metadata=covered -fexperimental-sanitize-metadata=uar -fexperimental-sanitize-metadata-ignorelist= -fexperimental-strict-floating-point -fextdirs= -fextend-arguments= -fextend-variable-liveness -fextend-variable-liveness= -fexternal-blas -fexternc-nounwind -ff2c -ffake-address-space-map -ffast-math -ffat-lto-objects -ffile-prefix-map= -ffile-reproducible -fimplicit-modules-use-lock -ffine-grained-bitfield-accesses -ffinite-loops -ffinite-math-only -finline-limit -ffixed-a0 -ffixed-a1 -ffixed-a2 -ffixed-a3 -ffixed-a4 -ffixed-a5 -ffixed-a6 -ffixed-d0 -ffixed-d1 -ffixed-d2 -ffixed-d3 -ffixed-d4 -ffixed-d5 -ffixed-d6 -ffixed-d7 -ffixed-form -ffixed-g1 -ffixed-g2 -ffixed-g3 -ffixed-g4 -ffixed-g5 -ffixed-g6 -ffixed-g7 -ffixed-i0 -ffixed-i1 -ffixed-i2 -ffixed-i3 -ffixed-i4 -ffixed-i5 -ffixed-l0 -ffixed-l1 -ffixed-l2 -ffixed-l3 -ffixed-l4 -ffixed-l5 -ffixed-l6 -ffixed-l7 -ffixed-line-length= -ffixed-line-length- -ffixed-o0 -ffixed-o1 -ffixed-o2 -ffixed-o3 -ffixed-o4 -ffixed-o5 -ffixed-point -ffixed-r19 -ffixed-r9 -ffixed-x1 -ffixed-x10 -ffixed-x11 -ffixed-x12 -ffixed-x13 -ffixed-x14 -ffixed-x15 -ffixed-x16 -ffixed-x17 -ffixed-x18 -ffixed-x19 -ffixed-x2 -ffixed-x20 -ffixed-x21 -ffixed-x22 -ffixed-x23 -ffixed-x24 -ffixed-x25 -ffixed-x26 -ffixed-x27 -ffixed-x28 -ffixed-x29 -ffixed-x3 -ffixed-x30 -ffixed-x31  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK9 %s

// DXCOptionCHECK9: {{(unknown argument).*-fexperimental-max-bitint-width=}}
// DXCOptionCHECK9: {{(unknown argument).*-fexperimental-new-constant-interpreter}}
// DXCOptionCHECK9: {{(unknown argument).*-fexperimental-omit-vtable-rtti}}
// DXCOptionCHECK9: {{(unknown argument).*-fexperimental-relative-c\+\+-abi-vtables}}
// DXCOptionCHECK9: {{(unknown argument).*-fexperimental-sanitize-metadata=}}
// DXCOptionCHECK9: {{(unknown argument).*-fexperimental-sanitize-metadata=atomics}}
// DXCOptionCHECK9: {{(unknown argument).*-fexperimental-sanitize-metadata=covered}}
// DXCOptionCHECK9: {{(unknown argument).*-fexperimental-sanitize-metadata=uar}}
// DXCOptionCHECK9: {{(unknown argument).*-fexperimental-sanitize-metadata-ignorelist=}}
// DXCOptionCHECK9: {{(unknown argument).*-fexperimental-strict-floating-point}}
// DXCOptionCHECK9: {{(unknown argument).*-fextdirs=}}
// DXCOptionCHECK9: {{(unknown argument).*-fextend-arguments=}}
// DXCOptionCHECK9: {{(unknown argument).*-fextend-variable-liveness}}
// DXCOptionCHECK9: {{(unknown argument).*-fextend-variable-liveness=}}
// DXCOptionCHECK9: {{(unknown argument).*-fexternal-blas}}
// DXCOptionCHECK9: {{(unknown argument).*-fexternc-nounwind}}
// DXCOptionCHECK9: {{(unknown argument).*-ff2c}}
// DXCOptionCHECK9: {{(unknown argument).*-ffake-address-space-map}}
// DXCOptionCHECK9: {{(unknown argument).*-ffast-math}}
// DXCOptionCHECK9: {{(unknown argument).*-ffat-lto-objects}}
// DXCOptionCHECK9: {{(unknown argument).*-ffile-prefix-map=}}
// DXCOptionCHECK9: {{(unknown argument).*-ffile-reproducible}}
// DXCOptionCHECK9: {{(unknown argument).*-fimplicit-modules-use-lock}}
// DXCOptionCHECK9: {{(unknown argument).*-ffine-grained-bitfield-accesses}}
// DXCOptionCHECK9: {{(unknown argument).*-ffinite-loops}}
// DXCOptionCHECK9: {{(unknown argument).*-ffinite-math-only}}
// DXCOptionCHECK9: {{(unknown argument).*-finline-limit}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-a0}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-a1}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-a2}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-a3}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-a4}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-a5}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-a6}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-d0}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-d1}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-d2}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-d3}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-d4}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-d5}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-d6}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-d7}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-form}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-g1}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-g2}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-g3}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-g4}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-g5}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-g6}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-g7}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-i0}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-i1}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-i2}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-i3}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-i4}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-i5}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-l0}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-l1}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-l2}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-l3}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-l4}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-l5}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-l6}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-l7}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-line-length=}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-line-length-}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-o0}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-o1}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-o2}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-o3}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-o4}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-o5}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-point}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-r19}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-r9}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x1}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x10}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x11}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x12}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x13}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x14}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x15}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x16}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x17}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x18}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x19}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x2}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x20}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x21}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x22}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x23}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x24}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x25}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x26}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x27}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x28}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x29}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x3}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x30}}
// DXCOptionCHECK9: {{(unknown argument).*-ffixed-x31}}
// RUN: not %clang_dxc -ffixed-x4 -ffixed-x5 -ffixed-x6 -ffixed-x7 -ffixed-x8 -ffixed-x9 -ffloat16-excess-precision= -ffloat-store -ffor-scope -fforbid-guard-variables -fforce-check-cxx20-modules-input-files -fforce-dwarf-frame -fforce-emit-vtables -fforce-enable-int128 -ffp-contract= -ffp-eval-method= -ffp-exception-behavior= -ffp-model= -ffpe-trap= -ffree-form -ffree-line-length- -ffreestanding -ffriend-injection -ffrontend-optimize -ffuchsia-api-level= -ffunction-attribute-list -ffunction-sections -fgcse -fgcse-after-reload -fgcse-las -fgcse-sm -fget-definition -fget-symbols-sources -fglobal-isel -fgnu -fgnu89-inline -fgnu-inline-asm -fgnu-keywords -fgnu-runtime -fgnuc-version= -fgpu-allow-device-init -fgpu-approx-transcendentals -fgpu-default-stream= -fgpu-defer-diag -fgpu-exclude-wrong-side-overloads -fgpu-flush-denormals-to-zero -fgpu-inline-threshold= -fgpu-rdc -fgpu-sanitize -fhalf-no-semantic-interposition -fheinous-gnu-extensions -fhermetic-module-files -fhip-dump-offload-linker-script -fhip-emit-relocatable -fhip-fp32-correctly-rounded-divide-sqrt -fhip-kernel-arg-name -fhip-new-launch-api -fhlsl-strict-availability -fhonor-infinities -fhonor-nans -fhosted -fignore-exceptions -filelist -filetype -fimplement-inlines -fimplicit-module-maps -fimplicit-modules -fimplicit-none -fimplicit-none-ext -fimplicit-templates -finclude-default-header -fincremental-extensions -finit-character= -finit-global-zero -finit-integer= -finit-local-zero -finit-logical= -finit-real= -finline -finline-functions -finline-functions-called-once -finline-hint-functions -finline-limit= -finline-max-stacksize= -finline-small-functions -finput-charset= -finstrument-function-entry-bare -finstrument-functions -finstrument-functions-after-inlining -finteger-4-integer-8 -fintegrated-as -fintegrated-objemitter -fintrinsic-modules-path -fipa-cp -fivopts -fix-only-warnings -fix-what-you-can -fixit -fixit= -fixit-recompile  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK10 %s

// DXCOptionCHECK10: {{(unknown argument).*-ffixed-x4}}
// DXCOptionCHECK10: {{(unknown argument).*-ffixed-x5}}
// DXCOptionCHECK10: {{(unknown argument).*-ffixed-x6}}
// DXCOptionCHECK10: {{(unknown argument).*-ffixed-x7}}
// DXCOptionCHECK10: {{(unknown argument).*-ffixed-x8}}
// DXCOptionCHECK10: {{(unknown argument).*-ffixed-x9}}
// DXCOptionCHECK10: {{(unknown argument).*-ffloat16-excess-precision=}}
// DXCOptionCHECK10: {{(unknown argument).*-ffloat-store}}
// DXCOptionCHECK10: {{(unknown argument).*-ffor-scope}}
// DXCOptionCHECK10: {{(unknown argument).*-fforbid-guard-variables}}
// DXCOptionCHECK10: {{(unknown argument).*-fforce-check-cxx20-modules-input-files}}
// DXCOptionCHECK10: {{(unknown argument).*-fforce-dwarf-frame}}
// DXCOptionCHECK10: {{(unknown argument).*-fforce-emit-vtables}}
// DXCOptionCHECK10: {{(unknown argument).*-fforce-enable-int128}}
// DXCOptionCHECK10: {{(unknown argument).*-ffp-contract=}}
// DXCOptionCHECK10: {{(unknown argument).*-ffp-eval-method=}}
// DXCOptionCHECK10: {{(unknown argument).*-ffp-exception-behavior=}}
// DXCOptionCHECK10: {{(unknown argument).*-ffp-model=}}
// DXCOptionCHECK10: {{(unknown argument).*-ffpe-trap=}}
// DXCOptionCHECK10: {{(unknown argument).*-ffree-form}}
// DXCOptionCHECK10: {{(unknown argument).*-ffree-line-length-}}
// DXCOptionCHECK10: {{(unknown argument).*-ffreestanding}}
// DXCOptionCHECK10: {{(unknown argument).*-ffriend-injection}}
// DXCOptionCHECK10: {{(unknown argument).*-ffrontend-optimize}}
// DXCOptionCHECK10: {{(unknown argument).*-ffuchsia-api-level=}}
// DXCOptionCHECK10: {{(unknown argument).*-ffunction-attribute-list}}
// DXCOptionCHECK10: {{(unknown argument).*-ffunction-sections}}
// DXCOptionCHECK10: {{(unknown argument).*-fgcse}}
// DXCOptionCHECK10: {{(unknown argument).*-fgcse-after-reload}}
// DXCOptionCHECK10: {{(unknown argument).*-fgcse-las}}
// DXCOptionCHECK10: {{(unknown argument).*-fgcse-sm}}
// DXCOptionCHECK10: {{(unknown argument).*-fget-definition}}
// DXCOptionCHECK10: {{(unknown argument).*-fget-symbols-sources}}
// DXCOptionCHECK10: {{(unknown argument).*-fglobal-isel}}
// DXCOptionCHECK10: {{(unknown argument).*-fgnu}}
// DXCOptionCHECK10: {{(unknown argument).*-fgnu89-inline}}
// DXCOptionCHECK10: {{(unknown argument).*-fgnu-inline-asm}}
// DXCOptionCHECK10: {{(unknown argument).*-fgnu-keywords}}
// DXCOptionCHECK10: {{(unknown argument).*-fgnu-runtime}}
// DXCOptionCHECK10: {{(unknown argument).*-fgnuc-version=}}
// DXCOptionCHECK10: {{(unknown argument).*-fgpu-allow-device-init}}
// DXCOptionCHECK10: {{(unknown argument).*-fgpu-approx-transcendentals}}
// DXCOptionCHECK10: {{(unknown argument).*-fgpu-default-stream=}}
// DXCOptionCHECK10: {{(unknown argument).*-fgpu-defer-diag}}
// DXCOptionCHECK10: {{(unknown argument).*-fgpu-exclude-wrong-side-overloads}}
// DXCOptionCHECK10: {{(unknown argument).*-fgpu-flush-denormals-to-zero}}
// DXCOptionCHECK10: {{(unknown argument).*-fgpu-inline-threshold=}}
// DXCOptionCHECK10: {{(unknown argument).*-fgpu-rdc}}
// DXCOptionCHECK10: {{(unknown argument).*-fgpu-sanitize}}
// DXCOptionCHECK10: {{(unknown argument).*-fhalf-no-semantic-interposition}}
// DXCOptionCHECK10: {{(unknown argument).*-fheinous-gnu-extensions}}
// DXCOptionCHECK10: {{(unknown argument).*-fhermetic-module-files}}
// DXCOptionCHECK10: {{(unknown argument).*-fhip-dump-offload-linker-script}}
// DXCOptionCHECK10: {{(unknown argument).*-fhip-emit-relocatable}}
// DXCOptionCHECK10: {{(unknown argument).*-fhip-fp32-correctly-rounded-divide-sqrt}}
// DXCOptionCHECK10: {{(unknown argument).*-fhip-kernel-arg-name}}
// DXCOptionCHECK10: {{(unknown argument).*-fhip-new-launch-api}}
// DXCOptionCHECK10: {{(unknown argument).*-fhlsl-strict-availability}}
// DXCOptionCHECK10: {{(unknown argument).*-fhonor-infinities}}
// DXCOptionCHECK10: {{(unknown argument).*-fhonor-nans}}
// DXCOptionCHECK10: {{(unknown argument).*-fhosted}}
// DXCOptionCHECK10: {{(unknown argument).*-fignore-exceptions}}
// DXCOptionCHECK10: {{(unknown argument).*-filelist}}
// DXCOptionCHECK10: {{(unknown argument).*-filetype}}
// DXCOptionCHECK10: {{(unknown argument).*-fimplement-inlines}}
// DXCOptionCHECK10: {{(unknown argument).*-fimplicit-module-maps}}
// DXCOptionCHECK10: {{(unknown argument).*-fimplicit-modules}}
// DXCOptionCHECK10: {{(unknown argument).*-fimplicit-none}}
// DXCOptionCHECK10: {{(unknown argument).*-fimplicit-none-ext}}
// DXCOptionCHECK10: {{(unknown argument).*-fimplicit-templates}}
// DXCOptionCHECK10: {{(unknown argument).*-finclude-default-header}}
// DXCOptionCHECK10: {{(unknown argument).*-fincremental-extensions}}
// DXCOptionCHECK10: {{(unknown argument).*-finit-character=}}
// DXCOptionCHECK10: {{(unknown argument).*-finit-global-zero}}
// DXCOptionCHECK10: {{(unknown argument).*-finit-integer=}}
// DXCOptionCHECK10: {{(unknown argument).*-finit-local-zero}}
// DXCOptionCHECK10: {{(unknown argument).*-finit-logical=}}
// DXCOptionCHECK10: {{(unknown argument).*-finit-real=}}
// DXCOptionCHECK10: {{(unknown argument).*-finline}}
// DXCOptionCHECK10: {{(unknown argument).*-finline-functions}}
// DXCOptionCHECK10: {{(unknown argument).*-finline-functions-called-once}}
// DXCOptionCHECK10: {{(unknown argument).*-finline-hint-functions}}
// DXCOptionCHECK10: {{(unknown argument).*-finline-limit=}}
// DXCOptionCHECK10: {{(unknown argument).*-finline-max-stacksize=}}
// DXCOptionCHECK10: {{(unknown argument).*-finline-small-functions}}
// DXCOptionCHECK10: {{(unknown argument).*-finput-charset=}}
// DXCOptionCHECK10: {{(unknown argument).*-finstrument-function-entry-bare}}
// DXCOptionCHECK10: {{(unknown argument).*-finstrument-functions}}
// DXCOptionCHECK10: {{(unknown argument).*-finstrument-functions-after-inlining}}
// DXCOptionCHECK10: {{(unknown argument).*-finteger-4-integer-8}}
// DXCOptionCHECK10: {{(unknown argument).*-fintegrated-as}}
// DXCOptionCHECK10: {{(unknown argument).*-fintegrated-objemitter}}
// DXCOptionCHECK10: {{(unknown argument).*-fintrinsic-modules-path}}
// DXCOptionCHECK10: {{(unknown argument).*-fipa-cp}}
// DXCOptionCHECK10: {{(unknown argument).*-fivopts}}
// DXCOptionCHECK10: {{(unknown argument).*-fix-only-warnings}}
// DXCOptionCHECK10: {{(unknown argument).*-fix-what-you-can}}
// DXCOptionCHECK10: {{(unknown argument).*-fixit}}
// DXCOptionCHECK10: {{(unknown argument).*-fixit=}}
// DXCOptionCHECK10: {{(unknown argument).*-fixit-recompile}}
// RUN: not %clang_dxc -fixit-to-temporary -fjmc -fjump-tables -fkeep-persistent-storage-variables -fkeep-static-consts -fkeep-system-includes -flang-deprecated-no-hlfir -flang-experimental-hlfir -flarge-sizes -flat_namespace -flax-vector-conversions -flax-vector-conversions= -flimited-precision= -flogical-abbreviations -floop-interchange -fversion-loops-for-stride -flto -flto= -flto=auto -flto=jobserver -flto-jobs= -flto-unit -flto-visibility-public-std -fmacro-backtrace-limit= -fmacro-prefix-map= -fmath-errno -fmax-array-constructor= -fmax-errors= -fmax-identifier-length -fmax-stack-var-size= -fmax-subrecord-length= -fmax-tokens= -fmax-type-align= -fcoverage-mcdc -fmcdc-max-conditions= -fmcdc-max-test-vectors= -fmemory-profile -fmemory-profile= -fmemory-profile-use= -fmerge-all-constants -fmerge-constants -fmerge-functions -fmessage-length= -fminimize-whitespace -fmodule-feature -fmodule-file= -fmodule-file-deps -fmodule-file-home-is-cwd -fmodule-format= -fmodule-header -fmodule-header= -fmodule-implementation-of -fmodule-map-file= -fmodule-map-file-home-is-cwd -fmodule-maps -fmodule-name= -fmodule-output -fmodule-output= -fmodule-private -fmodulemap-allow-subdirectory-search -fmodules -fmodules-cache-path= -fmodules-codegen -fmodules-debuginfo -fmodules-decluse -fmodules-disable-diagnostic-validation -fmodules-embed-all-files -fmodules-embed-file= -fmodules-hash-content -fmodules-ignore-macro= -fmodules-local-submodule-visibility -fmodules-prune-after= -fmodules-prune-interval= -fmodules-search-all -fmodules-skip-diagnostic-options -fmodules-skip-header-search-paths -fmodules-strict-context-hash -fmodules-strict-decluse -fmodules-user-build-path -fmodules-validate-input-files-content -fmodules-validate-once-per-build-session -fmodules-validate-system-headers -fmodulo-sched -fmodulo-sched-allow-regmoves -fms-compatibility -fms-compatibility-version= -fms-define-stdc -fms-extensions -fms-hotpatch -fms-kernel -fms-memptr-rep= -fms-omit-default-lib -fms-runtime-lib= -fms-tls-guards -fms-volatile -fmsc-version= -fmudflap -fmudflapth -fmultilib-flag= -fnative-half-arguments-and-returns  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK11 %s

// DXCOptionCHECK11: {{(unknown argument).*-fixit-to-temporary}}
// DXCOptionCHECK11: {{(unknown argument).*-fjmc}}
// DXCOptionCHECK11: {{(unknown argument).*-fjump-tables}}
// DXCOptionCHECK11: {{(unknown argument).*-fkeep-persistent-storage-variables}}
// DXCOptionCHECK11: {{(unknown argument).*-fkeep-static-consts}}
// DXCOptionCHECK11: {{(unknown argument).*-fkeep-system-includes}}
// DXCOptionCHECK11: {{(unknown argument).*-flang-deprecated-no-hlfir}}
// DXCOptionCHECK11: {{(unknown argument).*-flang-experimental-hlfir}}
// DXCOptionCHECK11: {{(unknown argument).*-flarge-sizes}}
// DXCOptionCHECK11: {{(unknown argument).*-flat_namespace}}
// DXCOptionCHECK11: {{(unknown argument).*-flax-vector-conversions}}
// DXCOptionCHECK11: {{(unknown argument).*-flax-vector-conversions=}}
// DXCOptionCHECK11: {{(unknown argument).*-flimited-precision=}}
// DXCOptionCHECK11: {{(unknown argument).*-flogical-abbreviations}}
// DXCOptionCHECK11: {{(unknown argument).*-floop-interchange}}
// DXCOptionCHECK11: {{(unknown argument).*-fversion-loops-for-stride}}
// DXCOptionCHECK11: {{(unknown argument).*-flto}}
// DXCOptionCHECK11: {{(unknown argument).*-flto=}}
// DXCOptionCHECK11: {{(unknown argument).*-flto=auto}}
// DXCOptionCHECK11: {{(unknown argument).*-flto=jobserver}}
// DXCOptionCHECK11: {{(unknown argument).*-flto-jobs=}}
// DXCOptionCHECK11: {{(unknown argument).*-flto-unit}}
// DXCOptionCHECK11: {{(unknown argument).*-flto-visibility-public-std}}
// DXCOptionCHECK11: {{(unknown argument).*-fmacro-backtrace-limit=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmacro-prefix-map=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmath-errno}}
// DXCOptionCHECK11: {{(unknown argument).*-fmax-array-constructor=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmax-errors=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmax-identifier-length}}
// DXCOptionCHECK11: {{(unknown argument).*-fmax-stack-var-size=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmax-subrecord-length=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmax-tokens=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmax-type-align=}}
// DXCOptionCHECK11: {{(unknown argument).*-fcoverage-mcdc}}
// DXCOptionCHECK11: {{(unknown argument).*-fmcdc-max-conditions=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmcdc-max-test-vectors=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmemory-profile}}
// DXCOptionCHECK11: {{(unknown argument).*-fmemory-profile=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmemory-profile-use=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmerge-all-constants}}
// DXCOptionCHECK11: {{(unknown argument).*-fmerge-constants}}
// DXCOptionCHECK11: {{(unknown argument).*-fmerge-functions}}
// DXCOptionCHECK11: {{(unknown argument).*-fmessage-length=}}
// DXCOptionCHECK11: {{(unknown argument).*-fminimize-whitespace}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-feature}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-file=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-file-deps}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-file-home-is-cwd}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-format=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-header}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-header=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-implementation-of}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-map-file=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-map-file-home-is-cwd}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-maps}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-name=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-output}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-output=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodule-private}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodulemap-allow-subdirectory-search}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-cache-path=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-codegen}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-debuginfo}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-decluse}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-disable-diagnostic-validation}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-embed-all-files}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-embed-file=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-hash-content}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-ignore-macro=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-local-submodule-visibility}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-prune-after=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-prune-interval=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-search-all}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-skip-diagnostic-options}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-skip-header-search-paths}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-strict-context-hash}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-strict-decluse}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-user-build-path}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-validate-input-files-content}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-validate-once-per-build-session}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodules-validate-system-headers}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodulo-sched}}
// DXCOptionCHECK11: {{(unknown argument).*-fmodulo-sched-allow-regmoves}}
// DXCOptionCHECK11: {{(unknown argument).*-fms-compatibility}}
// DXCOptionCHECK11: {{(unknown argument).*-fms-compatibility-version=}}
// DXCOptionCHECK11: {{(unknown argument).*-fms-define-stdc}}
// DXCOptionCHECK11: {{(unknown argument).*-fms-extensions}}
// DXCOptionCHECK11: {{(unknown argument).*-fms-hotpatch}}
// DXCOptionCHECK11: {{(unknown argument).*-fms-kernel}}
// DXCOptionCHECK11: {{(unknown argument).*-fms-memptr-rep=}}
// DXCOptionCHECK11: {{(unknown argument).*-fms-omit-default-lib}}
// DXCOptionCHECK11: {{(unknown argument).*-fms-runtime-lib=}}
// DXCOptionCHECK11: {{(unknown argument).*-fms-tls-guards}}
// DXCOptionCHECK11: {{(unknown argument).*-fms-volatile}}
// DXCOptionCHECK11: {{(unknown argument).*-fmsc-version=}}
// DXCOptionCHECK11: {{(unknown argument).*-fmudflap}}
// DXCOptionCHECK11: {{(unknown argument).*-fmudflapth}}
// DXCOptionCHECK11: {{(unknown argument).*-fmultilib-flag=}}
// DXCOptionCHECK11: {{(unknown argument).*-fnative-half-arguments-and-returns}}
// RUN: not %clang_dxc -fnative-half-type -fnested-functions -fnew-alignment= -fnew-infallible -fnext-runtime -fno-PIC -fno-PIE -fno-aapcs-bitfield-width -fno-aarch64-jump-table-hardening -fno-access-control -fno-addrsig -fno-aggressive-function-elimination -fno-align-commons -fno-align-functions -fno-align-jumps -fno-align-labels -fno-align-loops -fno-aligned-allocation -fno-all-intrinsics -fno-allow-editor-placeholders -fno-altivec -fno-analyzed-objects-for-unparse -fno-android-pad-segment -fno-keep-inline-functions -fno-unit-at-a-time -fno-apinotes -fno-apinotes-modules -fno-apple-pragma-pack -fno-application-extension -fno-approx-func -fno-asm -fno-asm-blocks -fno-associative-math -fno-assume-nothrow-exception-dtor -fno-assume-sane-operator-new -fno-assume-unique-vtables -fno-assumptions -fno-async-exceptions -fno-asynchronous-unwind-tables -fno-auto-import -fno-auto-profile -fno-auto-profile-accurate -fno-autolink -fno-automatic -fno-backslash -fno-backtrace -fno-basic-block-address-map -fno-bitfield-type-align -fno-blocks -fno-borland-extensions -fno-bounds-check -fno-experimental-bounds-safety -fno-branch-count-reg -fno-caller-saves -fno-caret-diagnostics -fno-char8_t -fno-check-array-temporaries -fno-check-new -fno-clangir -fno-common -fno-complete-member-pointers -fno-const-strings -fno-constant-cfstrings -fno-convergent-functions -fno-coro-aligned-allocation -fno-coroutines -fno-coverage-mapping -fno-cray-pointer -fno-cuda-flush-denormals-to-zero -fno-cuda-host-device-constexpr -fno-cuda-short-ptr -fno-cx-fortran-rules -fno-cx-limited-range -fno-cxx-exceptions -fno-cxx-modules -fno-d-lines-as-code -fno-d-lines-as-comments -fno-data-sections -fno-debug-info-for-profiling -fno-debug-pass-manager -fno-debug-ranges-base-address -fno-debug-types-section -fno-declspec -fno-default-inline -fno-define-target-os-macros -fno-delayed-template-parsing -fno-delete-null-pointer-checks -fno-deprecated-macro -fno-devirtualize -fno-devirtualize-speculatively -fno-diagnostics-fixit-info -fno-diagnostics-show-hotness -fno-diagnostics-show-line-numbers -fno-diagnostics-show-note-include-stack -fno-diagnostics-show-option -fno-diagnostics-use-presumed-location -fno-digraphs -fno-direct-access-external-data -fno-directives-only -fno-disable-block-signature-string  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK12 %s

// DXCOptionCHECK12: {{(unknown argument).*-fnative-half-type}}
// DXCOptionCHECK12: {{(unknown argument).*-fnested-functions}}
// DXCOptionCHECK12: {{(unknown argument).*-fnew-alignment=}}
// DXCOptionCHECK12: {{(unknown argument).*-fnew-infallible}}
// DXCOptionCHECK12: {{(unknown argument).*-fnext-runtime}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-PIC}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-PIE}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-aapcs-bitfield-width}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-aarch64-jump-table-hardening}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-access-control}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-addrsig}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-aggressive-function-elimination}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-align-commons}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-align-functions}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-align-jumps}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-align-labels}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-align-loops}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-aligned-allocation}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-all-intrinsics}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-allow-editor-placeholders}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-altivec}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-analyzed-objects-for-unparse}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-android-pad-segment}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-keep-inline-functions}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-unit-at-a-time}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-apinotes}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-apinotes-modules}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-apple-pragma-pack}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-application-extension}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-approx-func}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-asm}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-asm-blocks}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-associative-math}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-assume-nothrow-exception-dtor}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-assume-sane-operator-new}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-assume-unique-vtables}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-assumptions}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-async-exceptions}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-asynchronous-unwind-tables}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-auto-import}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-auto-profile}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-auto-profile-accurate}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-autolink}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-automatic}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-backslash}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-backtrace}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-basic-block-address-map}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-bitfield-type-align}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-blocks}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-borland-extensions}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-bounds-check}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-experimental-bounds-safety}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-branch-count-reg}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-caller-saves}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-caret-diagnostics}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-char8_t}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-check-array-temporaries}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-check-new}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-clangir}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-common}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-complete-member-pointers}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-const-strings}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-constant-cfstrings}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-convergent-functions}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-coro-aligned-allocation}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-coroutines}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-coverage-mapping}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-cray-pointer}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-cuda-flush-denormals-to-zero}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-cuda-host-device-constexpr}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-cuda-short-ptr}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-cx-fortran-rules}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-cx-limited-range}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-cxx-exceptions}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-cxx-modules}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-d-lines-as-code}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-d-lines-as-comments}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-data-sections}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-debug-info-for-profiling}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-debug-pass-manager}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-debug-ranges-base-address}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-debug-types-section}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-declspec}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-default-inline}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-define-target-os-macros}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-delayed-template-parsing}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-delete-null-pointer-checks}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-deprecated-macro}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-devirtualize}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-devirtualize-speculatively}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-diagnostics-fixit-info}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-diagnostics-show-hotness}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-diagnostics-show-line-numbers}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-diagnostics-show-note-include-stack}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-diagnostics-show-option}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-diagnostics-use-presumed-location}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-digraphs}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-direct-access-external-data}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-directives-only}}
// DXCOptionCHECK12: {{(unknown argument).*-fno-disable-block-signature-string}}
// RUN: not %clang_dxc -fno-dllexport-inlines -fno-dollar-ok -fno-dollars-in-identifiers -fno-double-square-bracket-attributes -fno-dump-fortran-optimized -fno-dump-fortran-original -fno-dump-parse-tree -fno-dwarf2-cfi-asm -fno-dwarf-directory-asm -fno-elide-constructors -fno-elide-type -fno-eliminate-unused-debug-symbols -fno-eliminate-unused-debug-types -fno-emit-compact-unwind-non-canonical -fno-emulated-tls -fno-escaping-block-tail-calls -fno-exceptions -fno-experimental-isel -fno-experimental-late-parse-attributes -fno-experimental-library -fno-experimental-omit-vtable-rtti -fno-experimental-relative-c++-abi-vtables -fno-experimental-sanitize-metadata= -fno-external-blas -fno-f2c -fno-fast-math -fno-fat-lto-objects -fno-file-reproducible -fno-implicit-modules-use-lock -fno-fine-grained-bitfield-accesses -fno-finite-loops -fno-finite-math-only -fno-inline-limit -fno-fixed-point -fno-float-store -fno-for-scope -fno-force-dwarf-frame -fno-force-emit-vtables -fno-force-enable-int128 -fno-friend-injection -fno-frontend-optimize -fno-function-attribute-list -fno-function-sections -fno-gcse -fno-gcse-after-reload -fno-gcse-las -fno-gcse-sm -fno-global-isel -fno-gnu -fno-gnu89-inline -fno-gnu-inline-asm -fno-gnu-keywords -fno-gpu-allow-device-init -fno-gpu-approx-transcendentals -fno-gpu-defer-diag -fno-gpu-exclude-wrong-side-overloads -fno-gpu-flush-denormals-to-zero -fno-gpu-rdc -fno-gpu-sanitize -fno-hip-emit-relocatable -fno-hip-fp32-correctly-rounded-divide-sqrt -fno-hip-kernel-arg-name -fno-hip-new-launch-api -fno-honor-infinities -fno-honor-nans -fno-implement-inlines -fno-implicit-module-maps -fno-implicit-modules -fno-implicit-none -fno-implicit-none-ext -fno-implicit-templates -fno-init-global-zero -fno-init-local-zero -fno-inline -fno-inline-functions -fno-inline-functions-called-once -fno-inline-small-functions -fno-integer-4-integer-8 -fno-integrated-as -fno-integrated-objemitter -fno-ipa-cp -fno-ivopts -fno-jmc -fno-jump-tables -fno-keep-persistent-storage-variables -fno-keep-static-consts -fno-keep-system-includes -fno-knr-functions -fno-lax-vector-conversions -fno-logical-abbreviations -fno-loop-interchange -fno-version-loops-for-stride -fno-lto-unit -fno-math-builtin -fno-math-errno -fno-max-identifier-length -fno-max-type-align -fno-coverage-mcdc -fno-memory-profile -fno-merge-all-constants  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK13 %s

// DXCOptionCHECK13: {{(unknown argument).*-fno-dllexport-inlines}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-dollar-ok}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-dollars-in-identifiers}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-double-square-bracket-attributes}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-dump-fortran-optimized}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-dump-fortran-original}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-dump-parse-tree}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-dwarf2-cfi-asm}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-dwarf-directory-asm}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-elide-constructors}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-elide-type}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-eliminate-unused-debug-symbols}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-eliminate-unused-debug-types}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-emit-compact-unwind-non-canonical}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-emulated-tls}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-escaping-block-tail-calls}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-exceptions}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-experimental-isel}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-experimental-late-parse-attributes}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-experimental-library}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-experimental-omit-vtable-rtti}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-experimental-relative-c\+\+-abi-vtables}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-experimental-sanitize-metadata=}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-external-blas}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-f2c}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-fast-math}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-fat-lto-objects}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-file-reproducible}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-implicit-modules-use-lock}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-fine-grained-bitfield-accesses}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-finite-loops}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-finite-math-only}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-inline-limit}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-fixed-point}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-float-store}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-for-scope}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-force-dwarf-frame}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-force-emit-vtables}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-force-enable-int128}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-friend-injection}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-frontend-optimize}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-function-attribute-list}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-function-sections}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gcse}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gcse-after-reload}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gcse-las}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gcse-sm}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-global-isel}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gnu}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gnu89-inline}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gnu-inline-asm}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gnu-keywords}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gpu-allow-device-init}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gpu-approx-transcendentals}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gpu-defer-diag}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gpu-exclude-wrong-side-overloads}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gpu-flush-denormals-to-zero}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gpu-rdc}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-gpu-sanitize}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-hip-emit-relocatable}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-hip-fp32-correctly-rounded-divide-sqrt}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-hip-kernel-arg-name}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-hip-new-launch-api}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-honor-infinities}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-honor-nans}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-implement-inlines}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-implicit-module-maps}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-implicit-modules}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-implicit-none}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-implicit-none-ext}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-implicit-templates}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-init-global-zero}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-init-local-zero}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-inline}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-inline-functions}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-inline-functions-called-once}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-inline-small-functions}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-integer-4-integer-8}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-integrated-as}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-integrated-objemitter}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-ipa-cp}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-ivopts}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-jmc}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-jump-tables}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-keep-persistent-storage-variables}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-keep-static-consts}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-keep-system-includes}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-knr-functions}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-lax-vector-conversions}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-logical-abbreviations}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-loop-interchange}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-version-loops-for-stride}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-lto-unit}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-math-builtin}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-math-errno}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-max-identifier-length}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-max-type-align}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-coverage-mcdc}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-memory-profile}}
// DXCOptionCHECK13: {{(unknown argument).*-fno-merge-all-constants}}
// RUN: not %clang_dxc -fno-merge-constants -fno-minimize-whitespace -fno-module-file-deps -fno-module-maps -fno-module-private -fno-modulemap-allow-subdirectory-search -fno-modules -fno-modules-check-relocated -fno-modules-decluse -fno-modules-error-recovery -fno-modules-global-index -fno-modules-prune-non-affecting-module-map-files -fno-modules-search-all -fno-modules-share-filemanager -fno-modules-skip-diagnostic-options -fno-modules-skip-header-search-paths -fno-strict-modules-decluse -fno_modules-validate-input-files-content -fno-modules-validate-system-headers -fno-modules-validate-textual-header-includes -fno-modulo-sched -fno-modulo-sched-allow-regmoves -fno-ms-compatibility -fno-ms-extensions -fno-ms-tls-guards -fno-ms-volatile -fno-new-infallible -fno-non-call-exceptions -fno-objc-arc -fno-objc-arc-exceptions -fno-objc-avoid-heapify-local-blocks -fno-objc-convert-messages-to-runtime-calls -fno-objc-encode-cxx-class-template-spec -fno-objc-exceptions -fno-objc-infer-related-result-type -fno-objc-legacy-dispatch -fno-objc-nonfragile-abi -fno-objc-weak -fno-offload-implicit-host-device-templates -fno-offload-lto -fno-offload-uniform-block -fno-offload-via-llvm -fno-omit-frame-pointer -fno-openmp -fno-openmp-assume-teams-oversubscription -fno-openmp-assume-threads-oversubscription -fno-openmp-cuda-mode -fno-openmp-extensions -fno-openmp-new-driver -fno-openmp-optimistic-collapse -fno-openmp-simd -fno-openmp-target-debug -fno-openmp-target-jit -fno-openmp-target-new-runtime -fno-operator-names -fno-optimize-sibling-calls -fno-pack-derived -fno-pack-struct -fno-padding-on-unsigned-fixed-point -fno-pascal-strings -fno-pch-codegen -fno-pch-debuginfo -fno-pch-instantiate-templates -fno-pch-timestamp -fno_pch-validate-input-files-content -fno-peel-loops -fno-permissive -fno-pic -fno-pie -fno-plt -fno-pointer-tbaa -fno-ppc-native-vector-element-order -fno-prebuilt-implicit-modules -fno-prefetch-loop-arrays -fno-preserve-as-comments -fno-printf -fno-profile -fno-profile-arcs -fno-profile-correction -fno-profile-generate -fno-profile-generate-sampling -fno-profile-instr-generate -fno-profile-instr-use -fno-profile-reusedist -fno-profile-sample-accurate -fno-profile-sample-use -fno-profile-use -fno-profile-values -fno-protect-parens -fno-pseudo-probe-for-profiling -fno-ptrauth-auth-traps -fno-ptrauth-calls -fno-ptrauth-elf-got -fno-ptrauth-function-pointer-type-discrimination -fno-ptrauth-indirect-gotos -fno-ptrauth-init-fini -fno-ptrauth-init-fini-address-discrimination -fno-ptrauth-intrinsics -fno-ptrauth-returns -fno-ptrauth-type-info-vtable-pointer-discrimination  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK14 %s

// DXCOptionCHECK14: {{(unknown argument).*-fno-merge-constants}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-minimize-whitespace}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-module-file-deps}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-module-maps}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-module-private}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modulemap-allow-subdirectory-search}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modules}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modules-check-relocated}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modules-decluse}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modules-error-recovery}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modules-global-index}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modules-prune-non-affecting-module-map-files}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modules-search-all}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modules-share-filemanager}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modules-skip-diagnostic-options}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modules-skip-header-search-paths}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-strict-modules-decluse}}
// DXCOptionCHECK14: {{(unknown argument).*-fno_modules-validate-input-files-content}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modules-validate-system-headers}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modules-validate-textual-header-includes}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modulo-sched}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-modulo-sched-allow-regmoves}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ms-compatibility}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ms-extensions}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ms-tls-guards}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ms-volatile}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-new-infallible}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-non-call-exceptions}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-objc-arc}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-objc-arc-exceptions}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-objc-avoid-heapify-local-blocks}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-objc-convert-messages-to-runtime-calls}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-objc-encode-cxx-class-template-spec}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-objc-exceptions}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-objc-infer-related-result-type}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-objc-legacy-dispatch}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-objc-nonfragile-abi}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-objc-weak}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-offload-implicit-host-device-templates}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-offload-lto}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-offload-uniform-block}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-offload-via-llvm}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-omit-frame-pointer}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-openmp}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-openmp-assume-teams-oversubscription}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-openmp-assume-threads-oversubscription}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-openmp-cuda-mode}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-openmp-extensions}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-openmp-new-driver}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-openmp-optimistic-collapse}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-openmp-simd}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-openmp-target-debug}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-openmp-target-jit}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-openmp-target-new-runtime}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-operator-names}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-optimize-sibling-calls}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-pack-derived}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-pack-struct}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-padding-on-unsigned-fixed-point}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-pascal-strings}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-pch-codegen}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-pch-debuginfo}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-pch-instantiate-templates}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-pch-timestamp}}
// DXCOptionCHECK14: {{(unknown argument).*-fno_pch-validate-input-files-content}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-peel-loops}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-permissive}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-pic}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-pie}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-plt}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-pointer-tbaa}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ppc-native-vector-element-order}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-prebuilt-implicit-modules}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-prefetch-loop-arrays}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-preserve-as-comments}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-printf}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-profile}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-profile-arcs}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-profile-correction}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-profile-generate}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-profile-generate-sampling}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-profile-instr-generate}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-profile-instr-use}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-profile-reusedist}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-profile-sample-accurate}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-profile-sample-use}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-profile-use}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-profile-values}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-protect-parens}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-pseudo-probe-for-profiling}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ptrauth-auth-traps}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ptrauth-calls}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ptrauth-elf-got}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ptrauth-function-pointer-type-discrimination}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ptrauth-indirect-gotos}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ptrauth-init-fini}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ptrauth-init-fini-address-discrimination}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ptrauth-intrinsics}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ptrauth-returns}}
// DXCOptionCHECK14: {{(unknown argument).*-fno-ptrauth-type-info-vtable-pointer-discrimination}}
// RUN: not %clang_dxc -fno-ptrauth-vtable-pointer-address-discrimination -fno-ptrauth-vtable-pointer-type-discrimination -fno-range-check -fno-raw-string-literals -fno-real-4-real-10 -fno-real-4-real-16 -fno-real-4-real-8 -fno-real-8-real-10 -fno-real-8-real-16 -fno-real-8-real-4 -fno-realloc-lhs -fno-reciprocal-math -fno-record-command-line -fno-recovery-ast -fno-recovery-ast-type -fno-recursive -fno-reformat -fno-register-global-dtors-with-atexit -fno-regs-graph -fno-rename-registers -fno-reorder-blocks -fno-repack-arrays -fno-retain-subst-template-type-parm-type-ast-nodes -fno-rewrite-imports -fno-rewrite-includes -fno-ripa -fno-ropi -fno-rounding-math -fno-rtlib-add-rpath -fno-rtlib-defaultlib -fno-rtti -fno-rtti-data -fno-rwpi -fno-safe-buffer-usage-suggestions -fno-sanitize= -fno-sanitize-address-globals-dead-stripping -fno-sanitize-address-outline-instrumentation -fno-sanitize-address-poison-custom-array-cookie -fno-sanitize-address-use-after-scope -fno-sanitize-address-use-odr-indicator -fno-sanitize-cfi-canonical-jump-tables -fno-sanitize-cfi-cross-dso -fno-sanitize-coverage= -fno-sanitize-hwaddress-experimental-aliasing -fno-sanitize-ignorelist -fno-sanitize-link-c++-runtime -fno-sanitize-link-runtime -fno-sanitize-memory-param-retval -fno-sanitize-memory-track-origins -fno-sanitize-memory-use-after-dtor -fno-sanitize-merge -fno-sanitize-merge= -fno-sanitize-minimal-runtime -fno-sanitize-recover -fno-sanitize-recover= -fno-sanitize-stable-abi -fno-sanitize-stats -fno-sanitize-thread-atomics -fno-sanitize-thread-func-entry-exit -fno-sanitize-thread-memory-access -fno-sanitize-trap -fno-sanitize-trap= -fno-sanitize-undefined-trap-on-error -fno-save-main-program -fno-save-optimization-record -fno-schedule-insns -fno-schedule-insns2 -fno-second-underscore -fno-see -fno-semantic-interposition -fno-separate-named-sections -fno-short-enums -fno-short-wchar -fno-show-column -fno-show-source-location -fno-sign-zero -fno-signaling-math -fno-signaling-nans -fno-signed-char -fno-signed-wchar -fno-signed-zeros -fno-single-precision-constant -fno-sized-deallocation -fno-skip-odr-check-in-gmf -fno-slp-vectorize -fno-spec-constr-count -fno-spell-checking -fno-split-dwarf-inlining -fno-split-lto-unit -fno-split-machine-functions -fno-split-stack -fno-stack-arrays -fno-stack-check -fno-stack-clash-protection -fno-stack-protector -fno-stack-size-section -fno-strength-reduce -fno-strict-enums -fno-strict-float-cast-overflow -fno-strict-overflow  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK15 %s

// DXCOptionCHECK15: {{(unknown argument).*-fno-ptrauth-vtable-pointer-address-discrimination}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-ptrauth-vtable-pointer-type-discrimination}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-range-check}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-raw-string-literals}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-real-4-real-10}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-real-4-real-16}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-real-4-real-8}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-real-8-real-10}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-real-8-real-16}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-real-8-real-4}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-realloc-lhs}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-reciprocal-math}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-record-command-line}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-recovery-ast}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-recovery-ast-type}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-recursive}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-reformat}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-register-global-dtors-with-atexit}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-regs-graph}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-rename-registers}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-reorder-blocks}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-repack-arrays}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-retain-subst-template-type-parm-type-ast-nodes}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-rewrite-imports}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-rewrite-includes}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-ripa}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-ropi}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-rounding-math}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-rtlib-add-rpath}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-rtlib-defaultlib}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-rtti}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-rtti-data}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-rwpi}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-safe-buffer-usage-suggestions}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize=}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-address-globals-dead-stripping}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-address-outline-instrumentation}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-address-poison-custom-array-cookie}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-address-use-after-scope}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-address-use-odr-indicator}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-cfi-canonical-jump-tables}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-cfi-cross-dso}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-coverage=}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-hwaddress-experimental-aliasing}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-ignorelist}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-link-c\+\+-runtime}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-link-runtime}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-memory-param-retval}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-memory-track-origins}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-memory-use-after-dtor}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-merge}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-merge=}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-minimal-runtime}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-recover}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-recover=}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-stable-abi}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-stats}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-thread-atomics}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-thread-func-entry-exit}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-thread-memory-access}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-trap}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-trap=}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sanitize-undefined-trap-on-error}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-save-main-program}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-save-optimization-record}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-schedule-insns}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-schedule-insns2}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-second-underscore}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-see}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-semantic-interposition}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-separate-named-sections}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-short-enums}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-short-wchar}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-show-column}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-show-source-location}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sign-zero}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-signaling-math}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-signaling-nans}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-signed-char}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-signed-wchar}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-signed-zeros}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-single-precision-constant}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-sized-deallocation}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-skip-odr-check-in-gmf}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-slp-vectorize}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-spec-constr-count}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-spell-checking}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-split-dwarf-inlining}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-split-lto-unit}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-split-machine-functions}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-split-stack}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-stack-arrays}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-stack-check}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-stack-clash-protection}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-stack-protector}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-stack-size-section}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-strength-reduce}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-strict-enums}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-strict-float-cast-overflow}}
// DXCOptionCHECK15: {{(unknown argument).*-fno-strict-overflow}}
// RUN: not %clang_dxc -fno-strict-return -fno-strict-vtable-pointers -fno-struct-path-tbaa -fno-sycl -fno-test-coverage -fno-threadsafe-statics -fno-tls-model -fno-tracer -fno-trapping-math -fno-tree-dce -fno-tree-salias -fno-tree-ter -fno-tree-vectorizer-verbose -fno-tree-vrp -fno-trigraphs -fno-underscoring -fno-unified-lto -fno-unique-basic-block-section-names -fno-unique-internal-linkage-names -fno-unique-section-names -fno-unroll-all-loops -fno-unroll-loops -fno-unsafe-loop-optimizations -fno-unsafe-math-optimizations -fno-unsigned -fno-unsigned-char -fno-unswitch-loops -fno-unwind-tables -fno-use-ctor-homing -fno-use-cxa-atexit -fno-use-init-array -fno-use-line-directives -fno-use-linker-plugin -fno-validate-pch -fno-var-tracking -fno-variable-expansion-in-unroller -fno-vect-cost-model -fno-vectorize -fno-verbose-asm -fno-virtual-function-elimination -fno-visibility-from-dllstorageclass -fno-visibility-inlines-hidden -fno-visibility-inlines-hidden-static-local-var -fno-wchar -fno-web -fno-whole-file -fno-whole-program -fno-whole-program-vtables -fno-working-directory -fno-wrapv -fno-wrapv-pointer -fno-xl-pragma-pack -fno-xor-operator -fno-xray-always-emit-customevents -fno-xray-always-emit-typedevents -fno-xray-function-index -fno-xray-ignore-loops -fno-xray-instrument -fno-xray-link-deps -fno-xray-shared -fno-zero-initialized-in-bss -fno-zos-extensions -fno-zvector -fnon-call-exceptions -fnoopenmp-relocatable-target -fnoopenmp-use-tls -fobjc-abi-version= -fobjc-arc -fobjc-arc-cxxlib= -fobjc-arc-exceptions -fobjc-atdefs -fobjc-avoid-heapify-local-blocks -fobjc-call-cxx-cdtors -fobjc-convert-messages-to-runtime-calls -fobjc-disable-direct-methods-for-testing -fobjc-dispatch-method= -fobjc-encode-cxx-class-template-spec -fobjc-exceptions -fobjc-gc -fobjc-gc-only -fobjc-infer-related-result-type -fobjc-legacy-dispatch -fobjc-link-runtime -fobjc-new-property -fobjc-nonfragile-abi -fobjc-nonfragile-abi-version= -fobjc-runtime= -fobjc-runtime-has-weak -fobjc-sender-dependent-dispatch -fobjc-subscripting-legacy-runtime -fobjc-weak -foffload-implicit-host-device-templates -foffload-lto -foffload-lto= -foffload-uniform-block -foffload-via-llvm -fomit-frame-pointer -fopenacc -fopenmp -fopenmp=  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK16 %s

// DXCOptionCHECK16: {{(unknown argument).*-fno-strict-return}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-strict-vtable-pointers}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-struct-path-tbaa}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-sycl}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-test-coverage}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-threadsafe-statics}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-tls-model}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-tracer}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-trapping-math}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-tree-dce}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-tree-salias}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-tree-ter}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-tree-vectorizer-verbose}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-tree-vrp}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-trigraphs}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-underscoring}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-unified-lto}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-unique-basic-block-section-names}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-unique-internal-linkage-names}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-unique-section-names}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-unroll-all-loops}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-unroll-loops}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-unsafe-loop-optimizations}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-unsafe-math-optimizations}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-unsigned}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-unsigned-char}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-unswitch-loops}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-unwind-tables}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-use-ctor-homing}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-use-cxa-atexit}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-use-init-array}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-use-line-directives}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-use-linker-plugin}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-validate-pch}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-var-tracking}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-variable-expansion-in-unroller}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-vect-cost-model}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-vectorize}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-verbose-asm}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-virtual-function-elimination}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-visibility-from-dllstorageclass}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-visibility-inlines-hidden}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-visibility-inlines-hidden-static-local-var}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-wchar}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-web}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-whole-file}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-whole-program}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-whole-program-vtables}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-working-directory}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-wrapv}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-wrapv-pointer}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-xl-pragma-pack}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-xor-operator}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-xray-always-emit-customevents}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-xray-always-emit-typedevents}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-xray-function-index}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-xray-ignore-loops}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-xray-instrument}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-xray-link-deps}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-xray-shared}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-zero-initialized-in-bss}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-zos-extensions}}
// DXCOptionCHECK16: {{(unknown argument).*-fno-zvector}}
// DXCOptionCHECK16: {{(unknown argument).*-fnon-call-exceptions}}
// DXCOptionCHECK16: {{(unknown argument).*-fnoopenmp-relocatable-target}}
// DXCOptionCHECK16: {{(unknown argument).*-fnoopenmp-use-tls}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-abi-version=}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-arc}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-arc-cxxlib=}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-arc-exceptions}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-atdefs}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-avoid-heapify-local-blocks}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-call-cxx-cdtors}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-convert-messages-to-runtime-calls}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-disable-direct-methods-for-testing}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-dispatch-method=}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-encode-cxx-class-template-spec}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-exceptions}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-gc}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-gc-only}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-infer-related-result-type}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-legacy-dispatch}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-link-runtime}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-new-property}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-nonfragile-abi}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-nonfragile-abi-version=}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-runtime=}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-runtime-has-weak}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-sender-dependent-dispatch}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-subscripting-legacy-runtime}}
// DXCOptionCHECK16: {{(unknown argument).*-fobjc-weak}}
// DXCOptionCHECK16: {{(unknown argument).*-foffload-implicit-host-device-templates}}
// DXCOptionCHECK16: {{(unknown argument).*-foffload-lto}}
// DXCOptionCHECK16: {{(unknown argument).*-foffload-lto=}}
// DXCOptionCHECK16: {{(unknown argument).*-foffload-uniform-block}}
// DXCOptionCHECK16: {{(unknown argument).*-foffload-via-llvm}}
// DXCOptionCHECK16: {{(unknown argument).*-fomit-frame-pointer}}
// DXCOptionCHECK16: {{(unknown argument).*-fopenacc}}
// DXCOptionCHECK16: {{(unknown argument).*-fopenmp}}
// DXCOptionCHECK16: {{(unknown argument).*-fopenmp=}}
// RUN: not %clang_dxc -fopenmp-assume-no-nested-parallelism -fopenmp-assume-no-thread-state -fopenmp-assume-teams-oversubscription -fopenmp-assume-threads-oversubscription -fopenmp-cuda-blocks-per-sm= -fopenmp-cuda-mode -fopenmp-cuda-number-of-sm= -fopenmp-cuda-teams-reduction-recs-num= -fopenmp-enable-irbuilder -fopenmp-extensions -fopenmp-force-usm -fopenmp-host-ir-file-path -fopenmp-is-target-device -fopenmp-new-driver -fopenmp-offload-mandatory -fopenmp-optimistic-collapse -fopenmp-relocatable-target -fopenmp-simd -fopenmp-target-debug -fopenmp-target-debug= -fopenmp-target-jit -fopenmp-target-new-runtime -fopenmp-targets= -fopenmp-use-tls -fopenmp-version= -foperator-arrow-depth= -foperator-names -foptimization-record-file= -foptimization-record-passes= -foptimize-sibling-calls -force_cpusubtype_ALL -force_flat_namespace -force_load -fforce-addr -forder-file-instrumentation -foutput-class-dir= -foverride-record-layout= -fpack-derived -fpack-struct -fpack-struct= -fpadding-on-unsigned-fixed-point -fparse-all-comments -fpascal-strings -fpass-by-value-is-noalias -fpass-plugin= -fpatchable-function-entry= -fpatchable-function-entry-offset= -fpcc-struct-return -fpch-codegen -fpch-debuginfo -fpch-instantiate-templates -fpch-preprocess -fpch-validate-input-files-content -fpeel-loops -fpermissive -fpic -fpie -fplt -fplugin= -fplugin-arg- -fpointer-tbaa -fppc-native-vector-element-order -fprebuilt-implicit-modules -fprebuilt-module-path= -fprefetch-loop-arrays -fpreprocess-include-lines -fpreserve-as-comments -fprintf -fproc-stat-report -fproc-stat-report= -fprofile -fprofile-arcs -fprofile-continuous -fprofile-correction -fprofile-dir= -fprofile-exclude-files= -fprofile-filter-files= -fprofile-function-groups= -fprofile-generate -fprofile-generate= -fprofile-generate-cold-function-coverage -fprofile-generate-cold-function-coverage= -fprofile-generate-sampling -fprofile-instr-generate -fprofile-instr-generate= -fprofile-instr-use -fprofile-instr-use= -fprofile-instrument= -fprofile-instrument-path= -fprofile-instrument-use-path= -fprofile-list= -fprofile-remapping-file= -fprofile-reusedist -fprofile-sample-accurate -fprofile-sample-use= -fprofile-selected-function-group= -fprofile-update= -fprofile-use -fprofile-use= -fprofile-values  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK17 %s

// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-assume-no-nested-parallelism}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-assume-no-thread-state}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-assume-teams-oversubscription}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-assume-threads-oversubscription}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-cuda-blocks-per-sm=}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-cuda-mode}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-cuda-number-of-sm=}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-cuda-teams-reduction-recs-num=}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-enable-irbuilder}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-extensions}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-force-usm}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-host-ir-file-path}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-is-target-device}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-new-driver}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-offload-mandatory}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-optimistic-collapse}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-relocatable-target}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-simd}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-target-debug}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-target-debug=}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-target-jit}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-target-new-runtime}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-targets=}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-use-tls}}
// DXCOptionCHECK17: {{(unknown argument).*-fopenmp-version=}}
// DXCOptionCHECK17: {{(unknown argument).*-foperator-arrow-depth=}}
// DXCOptionCHECK17: {{(unknown argument).*-foperator-names}}
// DXCOptionCHECK17: {{(unknown argument).*-foptimization-record-file=}}
// DXCOptionCHECK17: {{(unknown argument).*-foptimization-record-passes=}}
// DXCOptionCHECK17: {{(unknown argument).*-foptimize-sibling-calls}}
// DXCOptionCHECK17: {{(unknown argument).*-force_cpusubtype_ALL}}
// DXCOptionCHECK17: {{(unknown argument).*-force_flat_namespace}}
// DXCOptionCHECK17: {{(unknown argument).*-force_load}}
// DXCOptionCHECK17: {{(unknown argument).*-fforce-addr}}
// DXCOptionCHECK17: {{(unknown argument).*-forder-file-instrumentation}}
// DXCOptionCHECK17: {{(unknown argument).*-foutput-class-dir=}}
// DXCOptionCHECK17: {{(unknown argument).*-foverride-record-layout=}}
// DXCOptionCHECK17: {{(unknown argument).*-fpack-derived}}
// DXCOptionCHECK17: {{(unknown argument).*-fpack-struct}}
// DXCOptionCHECK17: {{(unknown argument).*-fpack-struct=}}
// DXCOptionCHECK17: {{(unknown argument).*-fpadding-on-unsigned-fixed-point}}
// DXCOptionCHECK17: {{(unknown argument).*-fparse-all-comments}}
// DXCOptionCHECK17: {{(unknown argument).*-fpascal-strings}}
// DXCOptionCHECK17: {{(unknown argument).*-fpass-by-value-is-noalias}}
// DXCOptionCHECK17: {{(unknown argument).*-fpass-plugin=}}
// DXCOptionCHECK17: {{(unknown argument).*-fpatchable-function-entry=}}
// DXCOptionCHECK17: {{(unknown argument).*-fpatchable-function-entry-offset=}}
// DXCOptionCHECK17: {{(unknown argument).*-fpcc-struct-return}}
// DXCOptionCHECK17: {{(unknown argument).*-fpch-codegen}}
// DXCOptionCHECK17: {{(unknown argument).*-fpch-debuginfo}}
// DXCOptionCHECK17: {{(unknown argument).*-fpch-instantiate-templates}}
// DXCOptionCHECK17: {{(unknown argument).*-fpch-preprocess}}
// DXCOptionCHECK17: {{(unknown argument).*-fpch-validate-input-files-content}}
// DXCOptionCHECK17: {{(unknown argument).*-fpeel-loops}}
// DXCOptionCHECK17: {{(unknown argument).*-fpermissive}}
// DXCOptionCHECK17: {{(unknown argument).*-fpic}}
// DXCOptionCHECK17: {{(unknown argument).*-fpie}}
// DXCOptionCHECK17: {{(unknown argument).*-fplt}}
// DXCOptionCHECK17: {{(unknown argument).*-fplugin=}}
// DXCOptionCHECK17: {{(unknown argument).*-fplugin-arg-}}
// DXCOptionCHECK17: {{(unknown argument).*-fpointer-tbaa}}
// DXCOptionCHECK17: {{(unknown argument).*-fppc-native-vector-element-order}}
// DXCOptionCHECK17: {{(unknown argument).*-fprebuilt-implicit-modules}}
// DXCOptionCHECK17: {{(unknown argument).*-fprebuilt-module-path=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprefetch-loop-arrays}}
// DXCOptionCHECK17: {{(unknown argument).*-fpreprocess-include-lines}}
// DXCOptionCHECK17: {{(unknown argument).*-fpreserve-as-comments}}
// DXCOptionCHECK17: {{(unknown argument).*-fprintf}}
// DXCOptionCHECK17: {{(unknown argument).*-fproc-stat-report}}
// DXCOptionCHECK17: {{(unknown argument).*-fproc-stat-report=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-arcs}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-continuous}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-correction}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-dir=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-exclude-files=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-filter-files=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-function-groups=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-generate}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-generate=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-generate-cold-function-coverage}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-generate-cold-function-coverage=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-generate-sampling}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-instr-generate}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-instr-generate=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-instr-use}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-instr-use=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-instrument=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-instrument-path=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-instrument-use-path=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-list=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-remapping-file=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-reusedist}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-sample-accurate}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-sample-use=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-selected-function-group=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-update=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-use}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-use=}}
// DXCOptionCHECK17: {{(unknown argument).*-fprofile-values}}
// RUN: not %clang_dxc -fprotect-parens -fpseudo-probe-for-profiling -fptrauth-auth-traps -fptrauth-calls -fptrauth-elf-got -fptrauth-function-pointer-type-discrimination -fptrauth-indirect-gotos -fptrauth-init-fini -fptrauth-init-fini-address-discrimination -fptrauth-intrinsics -fptrauth-returns -fptrauth-type-info-vtable-pointer-discrimination -fptrauth-vtable-pointer-address-discrimination -fptrauth-vtable-pointer-type-discrimination -framework -frandom-seed= -frandomize-layout-seed= -frandomize-layout-seed-file= -frange-check -fraw-string-literals -freal-4-real-10 -freal-4-real-16 -freal-4-real-8 -freal-8-real-10 -freal-8-real-16 -freal-8-real-4 -frealloc-lhs -freciprocal-math -frecord-command-line -frecord-marker= -frecovery-ast -frecovery-ast-type -frecursive -freg-struct-return -fregister-global-dtors-with-atexit -fregs-graph -frename-registers -freorder-blocks -frepack-arrays -fretain-comments-from-system-headers -fretain-subst-template-type-parm-type-ast-nodes -frewrite-imports -frewrite-includes -fripa -fropi -frounding-math -frtlib-add-rpath -frtlib-defaultlib -frtti -frtti-data -frwpi -fsafe-buffer-usage-suggestions -fsample-profile-use-profi -fsanitize= -fsanitize-address-field-padding= -fsanitize-address-globals-dead-stripping -fsanitize-address-outline-instrumentation -fsanitize-address-poison-custom-array-cookie -fsanitize-address-use-after-scope -fsanitize-address-use-odr-indicator -fsanitize-cfi-canonical-jump-tables -fsanitize-cfi-cross-dso -fsanitize-cfi-icall-generalize-pointers -fsanitize-cfi-icall-experimental-normalize-integers -fsanitize-coverage= -fsanitize-coverage-8bit-counters -fsanitize-coverage-allowlist= -fsanitize-coverage-control-flow -fsanitize-coverage-ignorelist= -fsanitize-coverage-indirect-calls -fsanitize-coverage-inline-8bit-counters -fsanitize-coverage-inline-bool-flag -fsanitize-coverage-no-prune -fsanitize-coverage-pc-table -fsanitize-coverage-stack-depth -fsanitize-coverage-trace-bb -fsanitize-coverage-trace-cmp -fsanitize-coverage-trace-div -fsanitize-coverage-trace-gep -fsanitize-coverage-trace-loads -fsanitize-coverage-trace-pc -fsanitize-coverage-trace-pc-guard -fsanitize-coverage-trace-stores -fsanitize-coverage-type= -fsanitize-hwaddress-abi= -fsanitize-hwaddress-experimental-aliasing -fsanitize-ignorelist= -fsanitize-kcfi-arity -fsanitize-link-c++-runtime -fsanitize-link-runtime -fsanitize-memory-param-retval -fsanitize-memory-track-origins -fsanitize-memory-track-origins= -fsanitize-memory-use-after-dtor -fsanitize-memtag-mode= -fsanitize-merge -fsanitize-merge= -fsanitize-minimal-runtime -fsanitize-recover -fsanitize-recover=  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK18 %s

// DXCOptionCHECK18: {{(unknown argument).*-fprotect-parens}}
// DXCOptionCHECK18: {{(unknown argument).*-fpseudo-probe-for-profiling}}
// DXCOptionCHECK18: {{(unknown argument).*-fptrauth-auth-traps}}
// DXCOptionCHECK18: {{(unknown argument).*-fptrauth-calls}}
// DXCOptionCHECK18: {{(unknown argument).*-fptrauth-elf-got}}
// DXCOptionCHECK18: {{(unknown argument).*-fptrauth-function-pointer-type-discrimination}}
// DXCOptionCHECK18: {{(unknown argument).*-fptrauth-indirect-gotos}}
// DXCOptionCHECK18: {{(unknown argument).*-fptrauth-init-fini}}
// DXCOptionCHECK18: {{(unknown argument).*-fptrauth-init-fini-address-discrimination}}
// DXCOptionCHECK18: {{(unknown argument).*-fptrauth-intrinsics}}
// DXCOptionCHECK18: {{(unknown argument).*-fptrauth-returns}}
// DXCOptionCHECK18: {{(unknown argument).*-fptrauth-type-info-vtable-pointer-discrimination}}
// DXCOptionCHECK18: {{(unknown argument).*-fptrauth-vtable-pointer-address-discrimination}}
// DXCOptionCHECK18: {{(unknown argument).*-fptrauth-vtable-pointer-type-discrimination}}
// DXCOptionCHECK18: {{(unknown argument).*-framework}}
// DXCOptionCHECK18: {{(unknown argument).*-frandom-seed=}}
// DXCOptionCHECK18: {{(unknown argument).*-frandomize-layout-seed=}}
// DXCOptionCHECK18: {{(unknown argument).*-frandomize-layout-seed-file=}}
// DXCOptionCHECK18: {{(unknown argument).*-frange-check}}
// DXCOptionCHECK18: {{(unknown argument).*-fraw-string-literals}}
// DXCOptionCHECK18: {{(unknown argument).*-freal-4-real-10}}
// DXCOptionCHECK18: {{(unknown argument).*-freal-4-real-16}}
// DXCOptionCHECK18: {{(unknown argument).*-freal-4-real-8}}
// DXCOptionCHECK18: {{(unknown argument).*-freal-8-real-10}}
// DXCOptionCHECK18: {{(unknown argument).*-freal-8-real-16}}
// DXCOptionCHECK18: {{(unknown argument).*-freal-8-real-4}}
// DXCOptionCHECK18: {{(unknown argument).*-frealloc-lhs}}
// DXCOptionCHECK18: {{(unknown argument).*-freciprocal-math}}
// DXCOptionCHECK18: {{(unknown argument).*-frecord-command-line}}
// DXCOptionCHECK18: {{(unknown argument).*-frecord-marker=}}
// DXCOptionCHECK18: {{(unknown argument).*-frecovery-ast}}
// DXCOptionCHECK18: {{(unknown argument).*-frecovery-ast-type}}
// DXCOptionCHECK18: {{(unknown argument).*-frecursive}}
// DXCOptionCHECK18: {{(unknown argument).*-freg-struct-return}}
// DXCOptionCHECK18: {{(unknown argument).*-fregister-global-dtors-with-atexit}}
// DXCOptionCHECK18: {{(unknown argument).*-fregs-graph}}
// DXCOptionCHECK18: {{(unknown argument).*-frename-registers}}
// DXCOptionCHECK18: {{(unknown argument).*-freorder-blocks}}
// DXCOptionCHECK18: {{(unknown argument).*-frepack-arrays}}
// DXCOptionCHECK18: {{(unknown argument).*-fretain-comments-from-system-headers}}
// DXCOptionCHECK18: {{(unknown argument).*-fretain-subst-template-type-parm-type-ast-nodes}}
// DXCOptionCHECK18: {{(unknown argument).*-frewrite-imports}}
// DXCOptionCHECK18: {{(unknown argument).*-frewrite-includes}}
// DXCOptionCHECK18: {{(unknown argument).*-fripa}}
// DXCOptionCHECK18: {{(unknown argument).*-fropi}}
// DXCOptionCHECK18: {{(unknown argument).*-frounding-math}}
// DXCOptionCHECK18: {{(unknown argument).*-frtlib-add-rpath}}
// DXCOptionCHECK18: {{(unknown argument).*-frtlib-defaultlib}}
// DXCOptionCHECK18: {{(unknown argument).*-frtti}}
// DXCOptionCHECK18: {{(unknown argument).*-frtti-data}}
// DXCOptionCHECK18: {{(unknown argument).*-frwpi}}
// DXCOptionCHECK18: {{(unknown argument).*-fsafe-buffer-usage-suggestions}}
// DXCOptionCHECK18: {{(unknown argument).*-fsample-profile-use-profi}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize=}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-address-field-padding=}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-address-globals-dead-stripping}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-address-outline-instrumentation}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-address-poison-custom-array-cookie}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-address-use-after-scope}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-address-use-odr-indicator}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-cfi-canonical-jump-tables}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-cfi-cross-dso}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-cfi-icall-generalize-pointers}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-cfi-icall-experimental-normalize-integers}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage=}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-8bit-counters}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-allowlist=}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-control-flow}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-ignorelist=}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-indirect-calls}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-inline-8bit-counters}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-inline-bool-flag}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-no-prune}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-pc-table}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-stack-depth}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-trace-bb}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-trace-cmp}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-trace-div}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-trace-gep}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-trace-loads}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-trace-pc}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-trace-pc-guard}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-trace-stores}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-coverage-type=}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-hwaddress-abi=}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-hwaddress-experimental-aliasing}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-ignorelist=}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-kcfi-arity}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-link-c\+\+-runtime}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-link-runtime}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-memory-param-retval}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-memory-track-origins}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-memory-track-origins=}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-memory-use-after-dtor}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-memtag-mode=}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-merge}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-merge=}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-minimal-runtime}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-recover}}
// DXCOptionCHECK18: {{(unknown argument).*-fsanitize-recover=}}
// RUN: not %clang_dxc -fsanitize-skip-hot-cutoff= -fsanitize-stable-abi -fsanitize-stats -fsanitize-system-ignorelist= -fsanitize-thread-atomics -fsanitize-thread-func-entry-exit -fsanitize-thread-memory-access -fsanitize-trap -fsanitize-trap= -fsanitize-undefined-ignore-overflow-pattern= -fsanitize-undefined-strip-path-components= -fsanitize-undefined-trap-on-error -fsave-main-program -fsave-optimization-record -fsave-optimization-record= -fschedule-insns -fschedule-insns2 -fsecond-underscore -fsee -fseh-exceptions -fsemantic-interposition -fseparate-named-sections -fshort-enums -fshort-wchar -fshow-column -fshow-overloads= -fshow-skipped-includes -fshow-source-location -fsign-zero -fsignaling-math -fsignaling-nans -fsigned-bitfields -fsigned-char -fsigned-wchar -fsigned-zeros -fsingle-precision-constant -fsized-deallocation -fsjlj-exceptions -fskip-odr-check-in-gmf -fslp-vectorize -fspec-constr-count -fspell-checking -fspell-checking-limit= -fsplit-dwarf-inlining -fsplit-lto-unit -fsplit-machine-functions -fsplit-stack -fstack-arrays -fstack-check -fstack-clash-protection -fstack-protector -fstack-protector-all -fstack-protector-strong -fstack-size-section -fstack-usage -fstrength-reduce -fstrict-enums -fstrict-flex-arrays= -fstrict-float-cast-overflow -fstrict-overflow -fstrict-return -fstrict-vtable-pointers -fstruct-path-tbaa -fswift-async-fp= -fsycl -fsycl-device-only -fsycl-host-only -fsycl-is-device -fsycl-is-host -fsymbol-partition= -fsystem-module -ftabstop -ftabstop= -ftemplate-backtrace-limit= -ftemplate-depth= -ftemporal-profile -ftest-coverage -ftest-module-file-extension= -fthin-link-bitcode= -fthinlto-index= -fthreadsafe-statics -ftime-report -ftime-report= -ftls-model -ftls-model= -ftracer -ftrap-function= -ftrapping-math -ftrapv -ftrapv-handler -ftrapv-handler= -ftree-dce -ftree-salias -ftree-ter -ftree-vectorizer-verbose -ftree-vrp -ftrigraphs -ftype-visibility= -function-alignment -funderscoring  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK19 %s

// DXCOptionCHECK19: {{(unknown argument).*-fsanitize-skip-hot-cutoff=}}
// DXCOptionCHECK19: {{(unknown argument).*-fsanitize-stable-abi}}
// DXCOptionCHECK19: {{(unknown argument).*-fsanitize-stats}}
// DXCOptionCHECK19: {{(unknown argument).*-fsanitize-system-ignorelist=}}
// DXCOptionCHECK19: {{(unknown argument).*-fsanitize-thread-atomics}}
// DXCOptionCHECK19: {{(unknown argument).*-fsanitize-thread-func-entry-exit}}
// DXCOptionCHECK19: {{(unknown argument).*-fsanitize-thread-memory-access}}
// DXCOptionCHECK19: {{(unknown argument).*-fsanitize-trap}}
// DXCOptionCHECK19: {{(unknown argument).*-fsanitize-trap=}}
// DXCOptionCHECK19: {{(unknown argument).*-fsanitize-undefined-ignore-overflow-pattern=}}
// DXCOptionCHECK19: {{(unknown argument).*-fsanitize-undefined-strip-path-components=}}
// DXCOptionCHECK19: {{(unknown argument).*-fsanitize-undefined-trap-on-error}}
// DXCOptionCHECK19: {{(unknown argument).*-fsave-main-program}}
// DXCOptionCHECK19: {{(unknown argument).*-fsave-optimization-record}}
// DXCOptionCHECK19: {{(unknown argument).*-fsave-optimization-record=}}
// DXCOptionCHECK19: {{(unknown argument).*-fschedule-insns}}
// DXCOptionCHECK19: {{(unknown argument).*-fschedule-insns2}}
// DXCOptionCHECK19: {{(unknown argument).*-fsecond-underscore}}
// DXCOptionCHECK19: {{(unknown argument).*-fsee}}
// DXCOptionCHECK19: {{(unknown argument).*-fseh-exceptions}}
// DXCOptionCHECK19: {{(unknown argument).*-fsemantic-interposition}}
// DXCOptionCHECK19: {{(unknown argument).*-fseparate-named-sections}}
// DXCOptionCHECK19: {{(unknown argument).*-fshort-enums}}
// DXCOptionCHECK19: {{(unknown argument).*-fshort-wchar}}
// DXCOptionCHECK19: {{(unknown argument).*-fshow-column}}
// DXCOptionCHECK19: {{(unknown argument).*-fshow-overloads=}}
// DXCOptionCHECK19: {{(unknown argument).*-fshow-skipped-includes}}
// DXCOptionCHECK19: {{(unknown argument).*-fshow-source-location}}
// DXCOptionCHECK19: {{(unknown argument).*-fsign-zero}}
// DXCOptionCHECK19: {{(unknown argument).*-fsignaling-math}}
// DXCOptionCHECK19: {{(unknown argument).*-fsignaling-nans}}
// DXCOptionCHECK19: {{(unknown argument).*-fsigned-bitfields}}
// DXCOptionCHECK19: {{(unknown argument).*-fsigned-char}}
// DXCOptionCHECK19: {{(unknown argument).*-fsigned-wchar}}
// DXCOptionCHECK19: {{(unknown argument).*-fsigned-zeros}}
// DXCOptionCHECK19: {{(unknown argument).*-fsingle-precision-constant}}
// DXCOptionCHECK19: {{(unknown argument).*-fsized-deallocation}}
// DXCOptionCHECK19: {{(unknown argument).*-fsjlj-exceptions}}
// DXCOptionCHECK19: {{(unknown argument).*-fskip-odr-check-in-gmf}}
// DXCOptionCHECK19: {{(unknown argument).*-fslp-vectorize}}
// DXCOptionCHECK19: {{(unknown argument).*-fspec-constr-count}}
// DXCOptionCHECK19: {{(unknown argument).*-fspell-checking}}
// DXCOptionCHECK19: {{(unknown argument).*-fspell-checking-limit=}}
// DXCOptionCHECK19: {{(unknown argument).*-fsplit-dwarf-inlining}}
// DXCOptionCHECK19: {{(unknown argument).*-fsplit-lto-unit}}
// DXCOptionCHECK19: {{(unknown argument).*-fsplit-machine-functions}}
// DXCOptionCHECK19: {{(unknown argument).*-fsplit-stack}}
// DXCOptionCHECK19: {{(unknown argument).*-fstack-arrays}}
// DXCOptionCHECK19: {{(unknown argument).*-fstack-check}}
// DXCOptionCHECK19: {{(unknown argument).*-fstack-clash-protection}}
// DXCOptionCHECK19: {{(unknown argument).*-fstack-protector}}
// DXCOptionCHECK19: {{(unknown argument).*-fstack-protector-all}}
// DXCOptionCHECK19: {{(unknown argument).*-fstack-protector-strong}}
// DXCOptionCHECK19: {{(unknown argument).*-fstack-size-section}}
// DXCOptionCHECK19: {{(unknown argument).*-fstack-usage}}
// DXCOptionCHECK19: {{(unknown argument).*-fstrength-reduce}}
// DXCOptionCHECK19: {{(unknown argument).*-fstrict-enums}}
// DXCOptionCHECK19: {{(unknown argument).*-fstrict-flex-arrays=}}
// DXCOptionCHECK19: {{(unknown argument).*-fstrict-float-cast-overflow}}
// DXCOptionCHECK19: {{(unknown argument).*-fstrict-overflow}}
// DXCOptionCHECK19: {{(unknown argument).*-fstrict-return}}
// DXCOptionCHECK19: {{(unknown argument).*-fstrict-vtable-pointers}}
// DXCOptionCHECK19: {{(unknown argument).*-fstruct-path-tbaa}}
// DXCOptionCHECK19: {{(unknown argument).*-fswift-async-fp=}}
// DXCOptionCHECK19: {{(unknown argument).*-fsycl}}
// DXCOptionCHECK19: {{(unknown argument).*-fsycl-device-only}}
// DXCOptionCHECK19: {{(unknown argument).*-fsycl-host-only}}
// DXCOptionCHECK19: {{(unknown argument).*-fsycl-is-device}}
// DXCOptionCHECK19: {{(unknown argument).*-fsycl-is-host}}
// DXCOptionCHECK19: {{(unknown argument).*-fsymbol-partition=}}
// DXCOptionCHECK19: {{(unknown argument).*-fsystem-module}}
// DXCOptionCHECK19: {{(unknown argument).*-ftabstop}}
// DXCOptionCHECK19: {{(unknown argument).*-ftabstop=}}
// DXCOptionCHECK19: {{(unknown argument).*-ftemplate-backtrace-limit=}}
// DXCOptionCHECK19: {{(unknown argument).*-ftemplate-depth=}}
// DXCOptionCHECK19: {{(unknown argument).*-ftemporal-profile}}
// DXCOptionCHECK19: {{(unknown argument).*-ftest-coverage}}
// DXCOptionCHECK19: {{(unknown argument).*-ftest-module-file-extension=}}
// DXCOptionCHECK19: {{(unknown argument).*-fthin-link-bitcode=}}
// DXCOptionCHECK19: {{(unknown argument).*-fthinlto-index=}}
// DXCOptionCHECK19: {{(unknown argument).*-fthreadsafe-statics}}
// DXCOptionCHECK19: {{(unknown argument).*-ftime-report}}
// DXCOptionCHECK19: {{(unknown argument).*-ftime-report=}}
// DXCOptionCHECK19: {{(unknown argument).*-ftls-model}}
// DXCOptionCHECK19: {{(unknown argument).*-ftls-model=}}
// DXCOptionCHECK19: {{(unknown argument).*-ftracer}}
// DXCOptionCHECK19: {{(unknown argument).*-ftrap-function=}}
// DXCOptionCHECK19: {{(unknown argument).*-ftrapping-math}}
// DXCOptionCHECK19: {{(unknown argument).*-ftrapv}}
// DXCOptionCHECK19: {{(unknown argument).*-ftrapv-handler}}
// DXCOptionCHECK19: {{(unknown argument).*-ftrapv-handler=}}
// DXCOptionCHECK19: {{(unknown argument).*-ftree-dce}}
// DXCOptionCHECK19: {{(unknown argument).*-ftree-salias}}
// DXCOptionCHECK19: {{(unknown argument).*-ftree-ter}}
// DXCOptionCHECK19: {{(unknown argument).*-ftree-vectorizer-verbose}}
// DXCOptionCHECK19: {{(unknown argument).*-ftree-vrp}}
// DXCOptionCHECK19: {{(unknown argument).*-ftrigraphs}}
// DXCOptionCHECK19: {{(unknown argument).*-ftype-visibility=}}
// DXCOptionCHECK19: {{(unknown argument).*-function-alignment}}
// DXCOptionCHECK19: {{(unknown argument).*-funderscoring}}
// RUN: not %clang_dxc -funified-lto -funique-basic-block-section-names -funique-internal-linkage-names -funique-section-names -funknown-anytype -funroll-all-loops -funroll-loops -funsafe-loop-optimizations -funsafe-math-optimizations -funsigned -funsigned-bitfields -funsigned-char -funswitch-loops -funwind-tables -funwind-tables= -fuse-ctor-homing -fuse-cuid= -fuse-cxa-atexit -fuse-init-array -fuse-ld= -fuse-line-directives -fuse-linker-plugin -fuse-lipo= -fuse-register-sized-bitfield-access -fvalidate-ast-input-files-content -fvariable-expansion-in-unroller -fveclib= -fvect-cost-model -fvectorize -fverbose-asm -fverify-debuginfo-preserve -fverify-debuginfo-preserve-export= -fvirtual-function-elimination -fvisibility= -fvisibility-dllexport= -fvisibility-externs-dllimport= -fvisibility-externs-nodllstorageclass= -fvisibility-from-dllstorageclass -fvisibility-global-new-delete= -fvisibility-global-new-delete-hidden -fvisibility-inlines-hidden -fvisibility-inlines-hidden-static-local-var -fvisibility-ms-compat -fvisibility-nodllstorageclass= -fwarn-stack-size= -fwasm-exceptions -fwchar-type= -fweb -fwhole-file -fwhole-program -fwhole-program-vtables -fwrapv -fwrapv-pointer -fwritable-strings -fxl-pragma-pack -fxor-operator -fxray-always-emit-customevents -fxray-always-emit-typedevents -fxray-always-instrument= -fxray-attr-list= -fxray-function-groups= -fxray-function-index -fxray-ignore-loops -fxray-instruction-threshold= -fxray-instrument -fxray-instrumentation-bundle= -fxray-link-deps -fxray-modes= -fxray-never-instrument= -fxray-selected-function-group= -fxray-shared -fzero-call-used-regs= -fzero-initialized-in-bss -fzos-extensions -fzvector -g0 -g1 -g2 -g3 --gcc-install-dir= --gcc-toolchain= --gcc-triple= -gcoff -gdbx -gdwarf32 -gdwarf64 -gdwarf-2 -gdwarf-3 -gdwarf-4 -gdwarf-5 -gdwarf-aranges -gembed-source -gfull -ggdb -ggdb0 -ggdb1 -ggdb2 -ggdb3 -ggnu-pubnames -glldb  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK20 %s

// DXCOptionCHECK20: {{(unknown argument).*-funified-lto}}
// DXCOptionCHECK20: {{(unknown argument).*-funique-basic-block-section-names}}
// DXCOptionCHECK20: {{(unknown argument).*-funique-internal-linkage-names}}
// DXCOptionCHECK20: {{(unknown argument).*-funique-section-names}}
// DXCOptionCHECK20: {{(unknown argument).*-funknown-anytype}}
// DXCOptionCHECK20: {{(unknown argument).*-funroll-all-loops}}
// DXCOptionCHECK20: {{(unknown argument).*-funroll-loops}}
// DXCOptionCHECK20: {{(unknown argument).*-funsafe-loop-optimizations}}
// DXCOptionCHECK20: {{(unknown argument).*-funsafe-math-optimizations}}
// DXCOptionCHECK20: {{(unknown argument).*-funsigned}}
// DXCOptionCHECK20: {{(unknown argument).*-funsigned-bitfields}}
// DXCOptionCHECK20: {{(unknown argument).*-funsigned-char}}
// DXCOptionCHECK20: {{(unknown argument).*-funswitch-loops}}
// DXCOptionCHECK20: {{(unknown argument).*-funwind-tables}}
// DXCOptionCHECK20: {{(unknown argument).*-funwind-tables=}}
// DXCOptionCHECK20: {{(unknown argument).*-fuse-ctor-homing}}
// DXCOptionCHECK20: {{(unknown argument).*-fuse-cuid=}}
// DXCOptionCHECK20: {{(unknown argument).*-fuse-cxa-atexit}}
// DXCOptionCHECK20: {{(unknown argument).*-fuse-init-array}}
// DXCOptionCHECK20: {{(unknown argument).*-fuse-ld=}}
// DXCOptionCHECK20: {{(unknown argument).*-fuse-line-directives}}
// DXCOptionCHECK20: {{(unknown argument).*-fuse-linker-plugin}}
// DXCOptionCHECK20: {{(unknown argument).*-fuse-lipo=}}
// DXCOptionCHECK20: {{(unknown argument).*-fuse-register-sized-bitfield-access}}
// DXCOptionCHECK20: {{(unknown argument).*-fvalidate-ast-input-files-content}}
// DXCOptionCHECK20: {{(unknown argument).*-fvariable-expansion-in-unroller}}
// DXCOptionCHECK20: {{(unknown argument).*-fveclib=}}
// DXCOptionCHECK20: {{(unknown argument).*-fvect-cost-model}}
// DXCOptionCHECK20: {{(unknown argument).*-fvectorize}}
// DXCOptionCHECK20: {{(unknown argument).*-fverbose-asm}}
// DXCOptionCHECK20: {{(unknown argument).*-fverify-debuginfo-preserve}}
// DXCOptionCHECK20: {{(unknown argument).*-fverify-debuginfo-preserve-export=}}
// DXCOptionCHECK20: {{(unknown argument).*-fvirtual-function-elimination}}
// DXCOptionCHECK20: {{(unknown argument).*-fvisibility=}}
// DXCOptionCHECK20: {{(unknown argument).*-fvisibility-dllexport=}}
// DXCOptionCHECK20: {{(unknown argument).*-fvisibility-externs-dllimport=}}
// DXCOptionCHECK20: {{(unknown argument).*-fvisibility-externs-nodllstorageclass=}}
// DXCOptionCHECK20: {{(unknown argument).*-fvisibility-from-dllstorageclass}}
// DXCOptionCHECK20: {{(unknown argument).*-fvisibility-global-new-delete=}}
// DXCOptionCHECK20: {{(unknown argument).*-fvisibility-global-new-delete-hidden}}
// DXCOptionCHECK20: {{(unknown argument).*-fvisibility-inlines-hidden}}
// DXCOptionCHECK20: {{(unknown argument).*-fvisibility-inlines-hidden-static-local-var}}
// DXCOptionCHECK20: {{(unknown argument).*-fvisibility-ms-compat}}
// DXCOptionCHECK20: {{(unknown argument).*-fvisibility-nodllstorageclass=}}
// DXCOptionCHECK20: {{(unknown argument).*-fwarn-stack-size=}}
// DXCOptionCHECK20: {{(unknown argument).*-fwasm-exceptions}}
// DXCOptionCHECK20: {{(unknown argument).*-fwchar-type=}}
// DXCOptionCHECK20: {{(unknown argument).*-fweb}}
// DXCOptionCHECK20: {{(unknown argument).*-fwhole-file}}
// DXCOptionCHECK20: {{(unknown argument).*-fwhole-program}}
// DXCOptionCHECK20: {{(unknown argument).*-fwhole-program-vtables}}
// DXCOptionCHECK20: {{(unknown argument).*-fwrapv}}
// DXCOptionCHECK20: {{(unknown argument).*-fwrapv-pointer}}
// DXCOptionCHECK20: {{(unknown argument).*-fwritable-strings}}
// DXCOptionCHECK20: {{(unknown argument).*-fxl-pragma-pack}}
// DXCOptionCHECK20: {{(unknown argument).*-fxor-operator}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-always-emit-customevents}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-always-emit-typedevents}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-always-instrument=}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-attr-list=}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-function-groups=}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-function-index}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-ignore-loops}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-instruction-threshold=}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-instrument}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-instrumentation-bundle=}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-link-deps}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-modes=}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-never-instrument=}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-selected-function-group=}}
// DXCOptionCHECK20: {{(unknown argument).*-fxray-shared}}
// DXCOptionCHECK20: {{(unknown argument).*-fzero-call-used-regs=}}
// DXCOptionCHECK20: {{(unknown argument).*-fzero-initialized-in-bss}}
// DXCOptionCHECK20: {{(unknown argument).*-fzos-extensions}}
// DXCOptionCHECK20: {{(unknown argument).*-fzvector}}
// DXCOptionCHECK20: {{(unknown argument).*-g0}}
// DXCOptionCHECK20: {{(unknown argument).*-g1}}
// DXCOptionCHECK20: {{(unknown argument).*-g2}}
// DXCOptionCHECK20: {{(unknown argument).*-g3}}
// DXCOptionCHECK20: {{(unknown argument).*--gcc-install-dir=}}
// DXCOptionCHECK20: {{(unknown argument).*--gcc-toolchain=}}
// DXCOptionCHECK20: {{(unknown argument).*--gcc-triple=}}
// DXCOptionCHECK20: {{(unknown argument).*-gcoff}}
// DXCOptionCHECK20: {{(unknown argument).*-gdbx}}
// DXCOptionCHECK20: {{(unknown argument).*-gdwarf32}}
// DXCOptionCHECK20: {{(unknown argument).*-gdwarf64}}
// DXCOptionCHECK20: {{(unknown argument).*-gdwarf-2}}
// DXCOptionCHECK20: {{(unknown argument).*-gdwarf-3}}
// DXCOptionCHECK20: {{(unknown argument).*-gdwarf-4}}
// DXCOptionCHECK20: {{(unknown argument).*-gdwarf-5}}
// DXCOptionCHECK20: {{(unknown argument).*-gdwarf-aranges}}
// DXCOptionCHECK20: {{(unknown argument).*-gembed-source}}
// DXCOptionCHECK20: {{(unknown argument).*-gfull}}
// DXCOptionCHECK20: {{(unknown argument).*-ggdb}}
// DXCOptionCHECK20: {{(unknown argument).*-ggdb0}}
// DXCOptionCHECK20: {{(unknown argument).*-ggdb1}}
// DXCOptionCHECK20: {{(unknown argument).*-ggdb2}}
// DXCOptionCHECK20: {{(unknown argument).*-ggdb3}}
// DXCOptionCHECK20: {{(unknown argument).*-ggnu-pubnames}}
// DXCOptionCHECK20: {{(unknown argument).*-glldb}}
// RUN: not %clang_dxc -gmlt -gmodules -gno-embed-source -gno-gnu-pubnames -gno-modules -gno-pubnames -gno-record-command-line -gno-simple-template-names -gno-template-alias --gpu-bundle-output --gpu-instrument-lib= --gpu-max-threads-per-block= --gpu-use-aux-triple-only -gpubnames -gpulibc -grecord-command-line -gsce -gsimple-template-names -gsimple-template-names= -gsrc-hash= -gstabs -gtemplate-alias -gtoggle -gused -gvms -gxcoff -gz -gz= -header-include-file -header-include-filtering= -header-include-format= -headerpad_max_install_names --hip-device-lib= --hip-link --hip-path= --hip-version= --hipspv-pass-plugin= --hipstdpar --hipstdpar-interpose-alloc --hipstdpar-path= --hipstdpar-prim-path= --hipstdpar-thrust-path= -iapinotes-modules -ibuiltininc -idirafter -iframework -iframeworkwithsysroot -imacros -image_base -import-call-optimization -imultilib -include -include-pch -init -init-only -inline-asm= -install_name -interface-stub-version= -internal-externc-isystem -internal-isystem -iprefix -iquote -isysroot -isystem -isystem-after -ivfsoverlay -iwithprefix -iwithprefixbefore -iwithsysroot -keep_private_externs -l -lazy_framework -lazy_library --ld-path= --libomptarget-amdgcn-bc-path= --libomptarget-amdgpu-bc-path= --libomptarget-nvptx-bc-path= --libomptarget-spirv-bc-path= --linker-option= -llvm-verify-each -load -m3dnow -m3dnowa -m68000 -m68010 -m68020 -m68030 -m68040 -m68060 -m68881 -m80387 -mseses -mabi= -mabi=ieeelongdouble -mabi=quadword-atomics -mabi=vec-extabi -mabicalls -mabs= -madx -maes  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK21 %s

// DXCOptionCHECK21: {{(unknown argument).*-gmlt}}
// DXCOptionCHECK21: {{(unknown argument).*-gmodules}}
// DXCOptionCHECK21: {{(unknown argument).*-gno-embed-source}}
// DXCOptionCHECK21: {{(unknown argument).*-gno-gnu-pubnames}}
// DXCOptionCHECK21: {{(unknown argument).*-gno-modules}}
// DXCOptionCHECK21: {{(unknown argument).*-gno-pubnames}}
// DXCOptionCHECK21: {{(unknown argument).*-gno-record-command-line}}
// DXCOptionCHECK21: {{(unknown argument).*-gno-simple-template-names}}
// DXCOptionCHECK21: {{(unknown argument).*-gno-template-alias}}
// DXCOptionCHECK21: {{(unknown argument).*--gpu-bundle-output}}
// DXCOptionCHECK21: {{(unknown argument).*--gpu-instrument-lib=}}
// DXCOptionCHECK21: {{(unknown argument).*--gpu-max-threads-per-block=}}
// DXCOptionCHECK21: {{(unknown argument).*--gpu-use-aux-triple-only}}
// DXCOptionCHECK21: {{(unknown argument).*-gpubnames}}
// DXCOptionCHECK21: {{(unknown argument).*-gpulibc}}
// DXCOptionCHECK21: {{(unknown argument).*-grecord-command-line}}
// DXCOptionCHECK21: {{(unknown argument).*-gsce}}
// DXCOptionCHECK21: {{(unknown argument).*-gsimple-template-names}}
// DXCOptionCHECK21: {{(unknown argument).*-gsimple-template-names=}}
// DXCOptionCHECK21: {{(unknown argument).*-gsrc-hash=}}
// DXCOptionCHECK21: {{(unknown argument).*-gstabs}}
// DXCOptionCHECK21: {{(unknown argument).*-gtemplate-alias}}
// DXCOptionCHECK21: {{(unknown argument).*-gtoggle}}
// DXCOptionCHECK21: {{(unknown argument).*-gused}}
// DXCOptionCHECK21: {{(unknown argument).*-gvms}}
// DXCOptionCHECK21: {{(unknown argument).*-gxcoff}}
// DXCOptionCHECK21: {{(unknown argument).*-gz}}
// DXCOptionCHECK21: {{(unknown argument).*-gz=}}
// DXCOptionCHECK21: {{(unknown argument).*-header-include-file}}
// DXCOptionCHECK21: {{(unknown argument).*-header-include-filtering=}}
// DXCOptionCHECK21: {{(unknown argument).*-header-include-format=}}
// DXCOptionCHECK21: {{(unknown argument).*-headerpad_max_install_names}}
// DXCOptionCHECK21: {{(unknown argument).*--hip-device-lib=}}
// DXCOptionCHECK21: {{(unknown argument).*--hip-link}}
// DXCOptionCHECK21: {{(unknown argument).*--hip-path=}}
// DXCOptionCHECK21: {{(unknown argument).*--hip-version=}}
// DXCOptionCHECK21: {{(unknown argument).*--hipspv-pass-plugin=}}
// DXCOptionCHECK21: {{(unknown argument).*--hipstdpar}}
// DXCOptionCHECK21: {{(unknown argument).*--hipstdpar-interpose-alloc}}
// DXCOptionCHECK21: {{(unknown argument).*--hipstdpar-path=}}
// DXCOptionCHECK21: {{(unknown argument).*--hipstdpar-prim-path=}}
// DXCOptionCHECK21: {{(unknown argument).*--hipstdpar-thrust-path=}}
// DXCOptionCHECK21: {{(unknown argument).*-iapinotes-modules}}
// DXCOptionCHECK21: {{(unknown argument).*-ibuiltininc}}
// DXCOptionCHECK21: {{(unknown argument).*-idirafter}}
// DXCOptionCHECK21: {{(unknown argument).*-iframework}}
// DXCOptionCHECK21: {{(unknown argument).*-iframeworkwithsysroot}}
// DXCOptionCHECK21: {{(unknown argument).*-imacros}}
// DXCOptionCHECK21: {{(unknown argument).*-image_base}}
// DXCOptionCHECK21: {{(unknown argument).*-import-call-optimization}}
// DXCOptionCHECK21: {{(unknown argument).*-imultilib}}
// DXCOptionCHECK21: {{(unknown argument).*-include}}
// DXCOptionCHECK21: {{(unknown argument).*-include-pch}}
// DXCOptionCHECK21: {{(unknown argument).*-init}}
// DXCOptionCHECK21: {{(unknown argument).*-init-only}}
// DXCOptionCHECK21: {{(unknown argument).*-inline-asm=}}
// DXCOptionCHECK21: {{(unknown argument).*-install_name}}
// DXCOptionCHECK21: {{(unknown argument).*-interface-stub-version=}}
// DXCOptionCHECK21: {{(unknown argument).*-internal-externc-isystem}}
// DXCOptionCHECK21: {{(unknown argument).*-internal-isystem}}
// DXCOptionCHECK21: {{(unknown argument).*-iprefix}}
// DXCOptionCHECK21: {{(unknown argument).*-iquote}}
// DXCOptionCHECK21: {{(unknown argument).*-isysroot}}
// DXCOptionCHECK21: {{(unknown argument).*-isystem}}
// DXCOptionCHECK21: {{(unknown argument).*-isystem-after}}
// DXCOptionCHECK21: {{(unknown argument).*-ivfsoverlay}}
// DXCOptionCHECK21: {{(unknown argument).*-iwithprefix}}
// DXCOptionCHECK21: {{(unknown argument).*-iwithprefixbefore}}
// DXCOptionCHECK21: {{(unknown argument).*-iwithsysroot}}
// DXCOptionCHECK21: {{(unknown argument).*-keep_private_externs}}
// DXCOptionCHECK21: {{(unknown argument).*-l}}
// DXCOptionCHECK21: {{(unknown argument).*-lazy_framework}}
// DXCOptionCHECK21: {{(unknown argument).*-lazy_library}}
// DXCOptionCHECK21: {{(unknown argument).*--ld-path=}}
// DXCOptionCHECK21: {{(unknown argument).*--libomptarget-amdgcn-bc-path=}}
// DXCOptionCHECK21: {{(unknown argument).*--libomptarget-amdgpu-bc-path=}}
// DXCOptionCHECK21: {{(unknown argument).*--libomptarget-nvptx-bc-path=}}
// DXCOptionCHECK21: {{(unknown argument).*--libomptarget-spirv-bc-path=}}
// DXCOptionCHECK21: {{(unknown argument).*--linker-option=}}
// DXCOptionCHECK21: {{(unknown argument).*-llvm-verify-each}}
// DXCOptionCHECK21: {{(unknown argument).*-load}}
// DXCOptionCHECK21: {{(unknown argument).*-m3dnow}}
// DXCOptionCHECK21: {{(unknown argument).*-m3dnowa}}
// DXCOptionCHECK21: {{(unknown argument).*-m68000}}
// DXCOptionCHECK21: {{(unknown argument).*-m68010}}
// DXCOptionCHECK21: {{(unknown argument).*-m68020}}
// DXCOptionCHECK21: {{(unknown argument).*-m68030}}
// DXCOptionCHECK21: {{(unknown argument).*-m68040}}
// DXCOptionCHECK21: {{(unknown argument).*-m68060}}
// DXCOptionCHECK21: {{(unknown argument).*-m68881}}
// DXCOptionCHECK21: {{(unknown argument).*-m80387}}
// DXCOptionCHECK21: {{(unknown argument).*-mseses}}
// DXCOptionCHECK21: {{(unknown argument).*-mabi=}}
// DXCOptionCHECK21: {{(unknown argument).*-mabi=ieeelongdouble}}
// DXCOptionCHECK21: {{(unknown argument).*-mabi=quadword-atomics}}
// DXCOptionCHECK21: {{(unknown argument).*-mabi=vec-extabi}}
// DXCOptionCHECK21: {{(unknown argument).*-mabicalls}}
// DXCOptionCHECK21: {{(unknown argument).*-mabs=}}
// DXCOptionCHECK21: {{(unknown argument).*-madx}}
// DXCOptionCHECK21: {{(unknown argument).*-maes}}
// RUN: not %clang_dxc -main-file-name -maix32 -maix64 -maix-shared-lib-tls-model-opt -maix-small-local-dynamic-tls -maix-small-local-exec-tls -maix-struct-return -malign-branch= -malign-branch-boundary= -malign-double -malign-functions= -malign-jumps= -malign-loops= -maltivec -mamdgpu-ieee -mamdgpu-precise-memory-op -mamx-avx512 -mamx-bf16 -mamx-complex -mamx-fp16 -mamx-fp8 -mamx-int8 -mamx-movrs -mamx-tf32 -mamx-tile -mamx-transpose -mannotate-tablejump -mappletvos-version-min= -mappletvsimulator-version-min= -mapx-features= -mapx-inline-asm-use-gpr32 -mapxf -marm -marm64x -masm= -massembler-fatal-warnings -massembler-no-warn -matomics -mavx -mavx10.1 -mavx10.1-256 -mavx10.1-512 -mavx10.2 -mavx10.2-256 -mavx10.2-512 -mavx2 -mavx512bf16 -mavx512bitalg -mavx512bw -mavx512cd -mavx512dq -mavx512f -mavx512fp16 -mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vl -mavx512vnni -mavx512vp2intersect -mavx512vpopcntdq -mavxifma -mavxneconvert -mavxvnni -mavxvnniint16 -mavxvnniint8 -mbackchain -mbig-endian -mbmi -mbmi2 -mbranch-likely -mbranch-protection= -mbranch-protection-pauth-lr -mbranch-target-enforce -mbranches-within-32B-boundaries -mbulk-memory -mbulk-memory-opt -mcabac -mcall-indirect-overlong -mcf-branch-label-scheme= -mcheck-zero-division -mcldemote -mclflushopt -mclwb -mclzero -mcmodel= -mcmpb -mcmpccxadd -mcmse -mcode-object-version= -mcompact-branches= -mconsole -mconstant-cfstrings -mconstructor-aliases -mcpu= -mcrbits -mcrc -mcrc32 -mcumode -mcx16 -mdaz-ftz  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK22 %s

// DXCOptionCHECK22: {{(unknown argument).*-main-file-name}}
// DXCOptionCHECK22: {{(unknown argument).*-maix32}}
// DXCOptionCHECK22: {{(unknown argument).*-maix64}}
// DXCOptionCHECK22: {{(unknown argument).*-maix-shared-lib-tls-model-opt}}
// DXCOptionCHECK22: {{(unknown argument).*-maix-small-local-dynamic-tls}}
// DXCOptionCHECK22: {{(unknown argument).*-maix-small-local-exec-tls}}
// DXCOptionCHECK22: {{(unknown argument).*-maix-struct-return}}
// DXCOptionCHECK22: {{(unknown argument).*-malign-branch=}}
// DXCOptionCHECK22: {{(unknown argument).*-malign-branch-boundary=}}
// DXCOptionCHECK22: {{(unknown argument).*-malign-double}}
// DXCOptionCHECK22: {{(unknown argument).*-malign-functions=}}
// DXCOptionCHECK22: {{(unknown argument).*-malign-jumps=}}
// DXCOptionCHECK22: {{(unknown argument).*-malign-loops=}}
// DXCOptionCHECK22: {{(unknown argument).*-maltivec}}
// DXCOptionCHECK22: {{(unknown argument).*-mamdgpu-ieee}}
// DXCOptionCHECK22: {{(unknown argument).*-mamdgpu-precise-memory-op}}
// DXCOptionCHECK22: {{(unknown argument).*-mamx-avx512}}
// DXCOptionCHECK22: {{(unknown argument).*-mamx-bf16}}
// DXCOptionCHECK22: {{(unknown argument).*-mamx-complex}}
// DXCOptionCHECK22: {{(unknown argument).*-mamx-fp16}}
// DXCOptionCHECK22: {{(unknown argument).*-mamx-fp8}}
// DXCOptionCHECK22: {{(unknown argument).*-mamx-int8}}
// DXCOptionCHECK22: {{(unknown argument).*-mamx-movrs}}
// DXCOptionCHECK22: {{(unknown argument).*-mamx-tf32}}
// DXCOptionCHECK22: {{(unknown argument).*-mamx-tile}}
// DXCOptionCHECK22: {{(unknown argument).*-mamx-transpose}}
// DXCOptionCHECK22: {{(unknown argument).*-mannotate-tablejump}}
// DXCOptionCHECK22: {{(unknown argument).*-mappletvos-version-min=}}
// DXCOptionCHECK22: {{(unknown argument).*-mappletvsimulator-version-min=}}
// DXCOptionCHECK22: {{(unknown argument).*-mapx-features=}}
// DXCOptionCHECK22: {{(unknown argument).*-mapx-inline-asm-use-gpr32}}
// DXCOptionCHECK22: {{(unknown argument).*-mapxf}}
// DXCOptionCHECK22: {{(unknown argument).*-marm}}
// DXCOptionCHECK22: {{(unknown argument).*-marm64x}}
// DXCOptionCHECK22: {{(unknown argument).*-masm=}}
// DXCOptionCHECK22: {{(unknown argument).*-massembler-fatal-warnings}}
// DXCOptionCHECK22: {{(unknown argument).*-massembler-no-warn}}
// DXCOptionCHECK22: {{(unknown argument).*-matomics}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx10.1}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx10.1-256}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx10.1-512}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx10.2}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx10.2-256}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx10.2-512}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx2}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512bf16}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512bitalg}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512bw}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512cd}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512dq}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512f}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512fp16}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512ifma}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512vbmi}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512vbmi2}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512vl}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512vnni}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512vp2intersect}}
// DXCOptionCHECK22: {{(unknown argument).*-mavx512vpopcntdq}}
// DXCOptionCHECK22: {{(unknown argument).*-mavxifma}}
// DXCOptionCHECK22: {{(unknown argument).*-mavxneconvert}}
// DXCOptionCHECK22: {{(unknown argument).*-mavxvnni}}
// DXCOptionCHECK22: {{(unknown argument).*-mavxvnniint16}}
// DXCOptionCHECK22: {{(unknown argument).*-mavxvnniint8}}
// DXCOptionCHECK22: {{(unknown argument).*-mbackchain}}
// DXCOptionCHECK22: {{(unknown argument).*-mbig-endian}}
// DXCOptionCHECK22: {{(unknown argument).*-mbmi}}
// DXCOptionCHECK22: {{(unknown argument).*-mbmi2}}
// DXCOptionCHECK22: {{(unknown argument).*-mbranch-likely}}
// DXCOptionCHECK22: {{(unknown argument).*-mbranch-protection=}}
// DXCOptionCHECK22: {{(unknown argument).*-mbranch-protection-pauth-lr}}
// DXCOptionCHECK22: {{(unknown argument).*-mbranch-target-enforce}}
// DXCOptionCHECK22: {{(unknown argument).*-mbranches-within-32B-boundaries}}
// DXCOptionCHECK22: {{(unknown argument).*-mbulk-memory}}
// DXCOptionCHECK22: {{(unknown argument).*-mbulk-memory-opt}}
// DXCOptionCHECK22: {{(unknown argument).*-mcabac}}
// DXCOptionCHECK22: {{(unknown argument).*-mcall-indirect-overlong}}
// DXCOptionCHECK22: {{(unknown argument).*-mcf-branch-label-scheme=}}
// DXCOptionCHECK22: {{(unknown argument).*-mcheck-zero-division}}
// DXCOptionCHECK22: {{(unknown argument).*-mcldemote}}
// DXCOptionCHECK22: {{(unknown argument).*-mclflushopt}}
// DXCOptionCHECK22: {{(unknown argument).*-mclwb}}
// DXCOptionCHECK22: {{(unknown argument).*-mclzero}}
// DXCOptionCHECK22: {{(unknown argument).*-mcmodel=}}
// DXCOptionCHECK22: {{(unknown argument).*-mcmpb}}
// DXCOptionCHECK22: {{(unknown argument).*-mcmpccxadd}}
// DXCOptionCHECK22: {{(unknown argument).*-mcmse}}
// DXCOptionCHECK22: {{(unknown argument).*-mcode-object-version=}}
// DXCOptionCHECK22: {{(unknown argument).*-mcompact-branches=}}
// DXCOptionCHECK22: {{(unknown argument).*-mconsole}}
// DXCOptionCHECK22: {{(unknown argument).*-mconstant-cfstrings}}
// DXCOptionCHECK22: {{(unknown argument).*-mconstructor-aliases}}
// DXCOptionCHECK22: {{(unknown argument).*-mcpu=}}
// DXCOptionCHECK22: {{(unknown argument).*-mcrbits}}
// DXCOptionCHECK22: {{(unknown argument).*-mcrc}}
// DXCOptionCHECK22: {{(unknown argument).*-mcrc32}}
// DXCOptionCHECK22: {{(unknown argument).*-mcumode}}
// DXCOptionCHECK22: {{(unknown argument).*-mcx16}}
// DXCOptionCHECK22: {{(unknown argument).*-mdaz-ftz}}
// RUN: not %clang_dxc -mdebug-pass -mdefault-build-attributes -mdefault-visibility-export-mapping= -mdirect-move -mdiv32 -mdll -mdouble= -mdouble-float -mdsp -mdspr2 -mdynamic-no-pic -meabi -mefpu2 -membedded-data -menable-experimental-extensions -menable-no-infs -menable-no-nans -menqcmd -mevex512 -mexception-handling -mexec-model= -mexecute-only -mextended-const -mextern-sdata -mf16c -mfancy-math-387 -mfentry -mfix4300 -mfix-and-continue -mfix-cmse-cve-2021-35465 -mfix-cortex-a53-835769 -mfix-cortex-a57-aes-1742098 -mfix-cortex-a72-aes-1655431 -mfix-gr712rc -mfix-ut700 -mfloat128 -mfloat-abi -mfloat-abi= -mfma -mfma4 -mfp16 -mfp32 -mfp64 -mfpmath -mfpmath= -mfprnd -mfpu -mfpu= -mfpxx -mframe-chain= -mframe-pointer= -mfrecipe -mfsgsbase -mfsmuld -mfunction-return= -mfxsr -mgeneral-regs-only -mgfni -mginv -mglibc -mglobal-merge -mgpopt -mguard= -mguarded-control-stack -mhard-float -mhard-quad-float -mharden-sls= -mhvx -mhvx= -mhvx-ieee-fp -mhvx-length= -mhvx-qfloat -mhreset -mhtm -mhwdiv= -mhwmult= -miamcu -mieee-fp -mieee-rnd-near -mignore-xcoff-visibility -no-finalize-removal -no-ns-alloc-error -mimplicit-float -mimplicit-it= -mincremental-linker-compatible -mindirect-branch-cs-prefix -mindirect-jump= -minline-all-stringops -minvariant-function-descriptors -minvpcid -mios-simulator-version-min= -mios-version-min= -mips1 -mips16 -mips2 -mips3 -mips32 -mips32r2 -mips32r3 -mips32r5  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK23 %s

// DXCOptionCHECK23: {{(unknown argument).*-mdebug-pass}}
// DXCOptionCHECK23: {{(unknown argument).*-mdefault-build-attributes}}
// DXCOptionCHECK23: {{(unknown argument).*-mdefault-visibility-export-mapping=}}
// DXCOptionCHECK23: {{(unknown argument).*-mdirect-move}}
// DXCOptionCHECK23: {{(unknown argument).*-mdiv32}}
// DXCOptionCHECK23: {{(unknown argument).*-mdll}}
// DXCOptionCHECK23: {{(unknown argument).*-mdouble=}}
// DXCOptionCHECK23: {{(unknown argument).*-mdouble-float}}
// DXCOptionCHECK23: {{(unknown argument).*-mdsp}}
// DXCOptionCHECK23: {{(unknown argument).*-mdspr2}}
// DXCOptionCHECK23: {{(unknown argument).*-mdynamic-no-pic}}
// DXCOptionCHECK23: {{(unknown argument).*-meabi}}
// DXCOptionCHECK23: {{(unknown argument).*-mefpu2}}
// DXCOptionCHECK23: {{(unknown argument).*-membedded-data}}
// DXCOptionCHECK23: {{(unknown argument).*-menable-experimental-extensions}}
// DXCOptionCHECK23: {{(unknown argument).*-menable-no-infs}}
// DXCOptionCHECK23: {{(unknown argument).*-menable-no-nans}}
// DXCOptionCHECK23: {{(unknown argument).*-menqcmd}}
// DXCOptionCHECK23: {{(unknown argument).*-mevex512}}
// DXCOptionCHECK23: {{(unknown argument).*-mexception-handling}}
// DXCOptionCHECK23: {{(unknown argument).*-mexec-model=}}
// DXCOptionCHECK23: {{(unknown argument).*-mexecute-only}}
// DXCOptionCHECK23: {{(unknown argument).*-mextended-const}}
// DXCOptionCHECK23: {{(unknown argument).*-mextern-sdata}}
// DXCOptionCHECK23: {{(unknown argument).*-mf16c}}
// DXCOptionCHECK23: {{(unknown argument).*-mfancy-math-387}}
// DXCOptionCHECK23: {{(unknown argument).*-mfentry}}
// DXCOptionCHECK23: {{(unknown argument).*-mfix4300}}
// DXCOptionCHECK23: {{(unknown argument).*-mfix-and-continue}}
// DXCOptionCHECK23: {{(unknown argument).*-mfix-cmse-cve-2021-35465}}
// DXCOptionCHECK23: {{(unknown argument).*-mfix-cortex-a53-835769}}
// DXCOptionCHECK23: {{(unknown argument).*-mfix-cortex-a57-aes-1742098}}
// DXCOptionCHECK23: {{(unknown argument).*-mfix-cortex-a72-aes-1655431}}
// DXCOptionCHECK23: {{(unknown argument).*-mfix-gr712rc}}
// DXCOptionCHECK23: {{(unknown argument).*-mfix-ut700}}
// DXCOptionCHECK23: {{(unknown argument).*-mfloat128}}
// DXCOptionCHECK23: {{(unknown argument).*-mfloat-abi}}
// DXCOptionCHECK23: {{(unknown argument).*-mfloat-abi=}}
// DXCOptionCHECK23: {{(unknown argument).*-mfma}}
// DXCOptionCHECK23: {{(unknown argument).*-mfma4}}
// DXCOptionCHECK23: {{(unknown argument).*-mfp16}}
// DXCOptionCHECK23: {{(unknown argument).*-mfp32}}
// DXCOptionCHECK23: {{(unknown argument).*-mfp64}}
// DXCOptionCHECK23: {{(unknown argument).*-mfpmath}}
// DXCOptionCHECK23: {{(unknown argument).*-mfpmath=}}
// DXCOptionCHECK23: {{(unknown argument).*-mfprnd}}
// DXCOptionCHECK23: {{(unknown argument).*-mfpu}}
// DXCOptionCHECK23: {{(unknown argument).*-mfpu=}}
// DXCOptionCHECK23: {{(unknown argument).*-mfpxx}}
// DXCOptionCHECK23: {{(unknown argument).*-mframe-chain=}}
// DXCOptionCHECK23: {{(unknown argument).*-mframe-pointer=}}
// DXCOptionCHECK23: {{(unknown argument).*-mfrecipe}}
// DXCOptionCHECK23: {{(unknown argument).*-mfsgsbase}}
// DXCOptionCHECK23: {{(unknown argument).*-mfsmuld}}
// DXCOptionCHECK23: {{(unknown argument).*-mfunction-return=}}
// DXCOptionCHECK23: {{(unknown argument).*-mfxsr}}
// DXCOptionCHECK23: {{(unknown argument).*-mgeneral-regs-only}}
// DXCOptionCHECK23: {{(unknown argument).*-mgfni}}
// DXCOptionCHECK23: {{(unknown argument).*-mginv}}
// DXCOptionCHECK23: {{(unknown argument).*-mglibc}}
// DXCOptionCHECK23: {{(unknown argument).*-mglobal-merge}}
// DXCOptionCHECK23: {{(unknown argument).*-mgpopt}}
// DXCOptionCHECK23: {{(unknown argument).*-mguard=}}
// DXCOptionCHECK23: {{(unknown argument).*-mguarded-control-stack}}
// DXCOptionCHECK23: {{(unknown argument).*-mhard-float}}
// DXCOptionCHECK23: {{(unknown argument).*-mhard-quad-float}}
// DXCOptionCHECK23: {{(unknown argument).*-mharden-sls=}}
// DXCOptionCHECK23: {{(unknown argument).*-mhvx}}
// DXCOptionCHECK23: {{(unknown argument).*-mhvx=}}
// DXCOptionCHECK23: {{(unknown argument).*-mhvx-ieee-fp}}
// DXCOptionCHECK23: {{(unknown argument).*-mhvx-length=}}
// DXCOptionCHECK23: {{(unknown argument).*-mhvx-qfloat}}
// DXCOptionCHECK23: {{(unknown argument).*-mhreset}}
// DXCOptionCHECK23: {{(unknown argument).*-mhtm}}
// DXCOptionCHECK23: {{(unknown argument).*-mhwdiv=}}
// DXCOptionCHECK23: {{(unknown argument).*-mhwmult=}}
// DXCOptionCHECK23: {{(unknown argument).*-miamcu}}
// DXCOptionCHECK23: {{(unknown argument).*-mieee-fp}}
// DXCOptionCHECK23: {{(unknown argument).*-mieee-rnd-near}}
// DXCOptionCHECK23: {{(unknown argument).*-mignore-xcoff-visibility}}
// DXCOptionCHECK23: {{(unknown argument).*-no-finalize-removal}}
// DXCOptionCHECK23: {{(unknown argument).*-no-ns-alloc-error}}
// DXCOptionCHECK23: {{(unknown argument).*-mimplicit-float}}
// DXCOptionCHECK23: {{(unknown argument).*-mimplicit-it=}}
// DXCOptionCHECK23: {{(unknown argument).*-mincremental-linker-compatible}}
// DXCOptionCHECK23: {{(unknown argument).*-mindirect-branch-cs-prefix}}
// DXCOptionCHECK23: {{(unknown argument).*-mindirect-jump=}}
// DXCOptionCHECK23: {{(unknown argument).*-minline-all-stringops}}
// DXCOptionCHECK23: {{(unknown argument).*-minvariant-function-descriptors}}
// DXCOptionCHECK23: {{(unknown argument).*-minvpcid}}
// DXCOptionCHECK23: {{(unknown argument).*-mios-simulator-version-min=}}
// DXCOptionCHECK23: {{(unknown argument).*-mios-version-min=}}
// DXCOptionCHECK23: {{(unknown argument).*-mips1}}
// DXCOptionCHECK23: {{(unknown argument).*-mips16}}
// DXCOptionCHECK23: {{(unknown argument).*-mips2}}
// DXCOptionCHECK23: {{(unknown argument).*-mips3}}
// DXCOptionCHECK23: {{(unknown argument).*-mips32}}
// DXCOptionCHECK23: {{(unknown argument).*-mips32r2}}
// DXCOptionCHECK23: {{(unknown argument).*-mips32r3}}
// DXCOptionCHECK23: {{(unknown argument).*-mips32r5}}
// RUN: not %clang_dxc -mips32r6 -mips4 -mips5 -mips64 -mips64r2 -mips64r3 -mips64r5 -mips64r6 -misel -mkernel -mkl -mlam-bh -mlamcas -mlarge-data-threshold= -mlasx -mld-seq-sa -mldc1-sdc1 -mlimit-float-precision -mlink-bitcode-file -mlink-builtin-bitcode -mlink-builtin-bitcode-postopt -mlinker-version= -mlittle-endian -mlocal-sdata -mlong-calls -mlong-double-128 -mlong-double-64 -mlong-double-80 -mlongcall -mlr-for-calls-only -mlsx -mlvi-cfi -mlvi-hardening -mlwp -mlzcnt -mmacos-version-min= -mmadd4 -mmapsyms=implicit -mmark-bti-property -mmcu= -mmemops -mmfcrf -mmfocrf -mmicromips -mmlir -mmma -mmmx -mmovbe -mmovdir64b -mmovdiri -mmovrs -mmpx -mms-bitfields -mmsa -mmt -mmultimemory -mmultivalue -mmutable-globals -mmwaitx -mnan= -mno-3dnow -mno-3dnowa -mno-80387 -mno-abicalls -mno-adx -mno-aes -mno-altivec -mno-amdgpu-ieee -mno-amdgpu-precise-memory-op -mno-amx-avx512 -mno-amx-bf16 -mno-amx-complex -mno-amx-fp16 -mno-amx-fp8 -mno-amx-int8 -mno-amx-movrs -mno-amx-tf32 -mno-amx-tile -mno-amx-transpose -mno-annotate-tablejump -mno-apx-features= -mno-apxf -mno-atomics -mno-avx -mno-avx10.1 -mno-avx10.1-256 -mno-avx10.1-512 -mno-avx10.2 -mno-avx2 -mno-avx512bf16 -mno-avx512bitalg -mno-avx512bw -mno-avx512cd -mno-avx512dq -mno-avx512f -mno-avx512fp16 -mno-avx512ifma -mno-avx512vbmi -mno-avx512vbmi2 -mno-avx512vl  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK24 %s

// DXCOptionCHECK24: {{(unknown argument).*-mips32r6}}
// DXCOptionCHECK24: {{(unknown argument).*-mips4}}
// DXCOptionCHECK24: {{(unknown argument).*-mips5}}
// DXCOptionCHECK24: {{(unknown argument).*-mips64}}
// DXCOptionCHECK24: {{(unknown argument).*-mips64r2}}
// DXCOptionCHECK24: {{(unknown argument).*-mips64r3}}
// DXCOptionCHECK24: {{(unknown argument).*-mips64r5}}
// DXCOptionCHECK24: {{(unknown argument).*-mips64r6}}
// DXCOptionCHECK24: {{(unknown argument).*-misel}}
// DXCOptionCHECK24: {{(unknown argument).*-mkernel}}
// DXCOptionCHECK24: {{(unknown argument).*-mkl}}
// DXCOptionCHECK24: {{(unknown argument).*-mlam-bh}}
// DXCOptionCHECK24: {{(unknown argument).*-mlamcas}}
// DXCOptionCHECK24: {{(unknown argument).*-mlarge-data-threshold=}}
// DXCOptionCHECK24: {{(unknown argument).*-mlasx}}
// DXCOptionCHECK24: {{(unknown argument).*-mld-seq-sa}}
// DXCOptionCHECK24: {{(unknown argument).*-mldc1-sdc1}}
// DXCOptionCHECK24: {{(unknown argument).*-mlimit-float-precision}}
// DXCOptionCHECK24: {{(unknown argument).*-mlink-bitcode-file}}
// DXCOptionCHECK24: {{(unknown argument).*-mlink-builtin-bitcode}}
// DXCOptionCHECK24: {{(unknown argument).*-mlink-builtin-bitcode-postopt}}
// DXCOptionCHECK24: {{(unknown argument).*-mlinker-version=}}
// DXCOptionCHECK24: {{(unknown argument).*-mlittle-endian}}
// DXCOptionCHECK24: {{(unknown argument).*-mlocal-sdata}}
// DXCOptionCHECK24: {{(unknown argument).*-mlong-calls}}
// DXCOptionCHECK24: {{(unknown argument).*-mlong-double-128}}
// DXCOptionCHECK24: {{(unknown argument).*-mlong-double-64}}
// DXCOptionCHECK24: {{(unknown argument).*-mlong-double-80}}
// DXCOptionCHECK24: {{(unknown argument).*-mlongcall}}
// DXCOptionCHECK24: {{(unknown argument).*-mlr-for-calls-only}}
// DXCOptionCHECK24: {{(unknown argument).*-mlsx}}
// DXCOptionCHECK24: {{(unknown argument).*-mlvi-cfi}}
// DXCOptionCHECK24: {{(unknown argument).*-mlvi-hardening}}
// DXCOptionCHECK24: {{(unknown argument).*-mlwp}}
// DXCOptionCHECK24: {{(unknown argument).*-mlzcnt}}
// DXCOptionCHECK24: {{(unknown argument).*-mmacos-version-min=}}
// DXCOptionCHECK24: {{(unknown argument).*-mmadd4}}
// DXCOptionCHECK24: {{(unknown argument).*-mmapsyms=implicit}}
// DXCOptionCHECK24: {{(unknown argument).*-mmark-bti-property}}
// DXCOptionCHECK24: {{(unknown argument).*-mmcu=}}
// DXCOptionCHECK24: {{(unknown argument).*-mmemops}}
// DXCOptionCHECK24: {{(unknown argument).*-mmfcrf}}
// DXCOptionCHECK24: {{(unknown argument).*-mmfocrf}}
// DXCOptionCHECK24: {{(unknown argument).*-mmicromips}}
// DXCOptionCHECK24: {{(unknown argument).*-mmlir}}
// DXCOptionCHECK24: {{(unknown argument).*-mmma}}
// DXCOptionCHECK24: {{(unknown argument).*-mmmx}}
// DXCOptionCHECK24: {{(unknown argument).*-mmovbe}}
// DXCOptionCHECK24: {{(unknown argument).*-mmovdir64b}}
// DXCOptionCHECK24: {{(unknown argument).*-mmovdiri}}
// DXCOptionCHECK24: {{(unknown argument).*-mmovrs}}
// DXCOptionCHECK24: {{(unknown argument).*-mmpx}}
// DXCOptionCHECK24: {{(unknown argument).*-mms-bitfields}}
// DXCOptionCHECK24: {{(unknown argument).*-mmsa}}
// DXCOptionCHECK24: {{(unknown argument).*-mmt}}
// DXCOptionCHECK24: {{(unknown argument).*-mmultimemory}}
// DXCOptionCHECK24: {{(unknown argument).*-mmultivalue}}
// DXCOptionCHECK24: {{(unknown argument).*-mmutable-globals}}
// DXCOptionCHECK24: {{(unknown argument).*-mmwaitx}}
// DXCOptionCHECK24: {{(unknown argument).*-mnan=}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-3dnow}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-3dnowa}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-80387}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-abicalls}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-adx}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-aes}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-altivec}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-amdgpu-ieee}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-amdgpu-precise-memory-op}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-amx-avx512}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-amx-bf16}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-amx-complex}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-amx-fp16}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-amx-fp8}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-amx-int8}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-amx-movrs}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-amx-tf32}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-amx-tile}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-amx-transpose}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-annotate-tablejump}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-apx-features=}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-apxf}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-atomics}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx10.1}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx10.1-256}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx10.1-512}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx10.2}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx2}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx512bf16}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx512bitalg}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx512bw}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx512cd}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx512dq}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx512f}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx512fp16}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx512ifma}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx512vbmi}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx512vbmi2}}
// DXCOptionCHECK24: {{(unknown argument).*-mno-avx512vl}}
// RUN: not %clang_dxc -mno-avx512vnni -mno-avx512vp2intersect -mno-avx512vpopcntdq -mno-avxifma -mno-avxneconvert -mno-avxvnni -mno-avxvnniint16 -mno-avxvnniint8 -mno-backchain -mno-bmi -mno-bmi2 -mno-branch-likely -mno-bti-at-return-twice -mno-bulk-memory -mno-bulk-memory-opt -mno-call-indirect-overlong -mno-check-zero-division -mno-cldemote -mno-clflushopt -mno-clwb -mno-clzero -mno-cmpb -mno-cmpccxadd -mno-constant-cfstrings -mno-constructor-aliases -mno-crbits -mno-crc -mno-crc32 -mno-cumode -mno-cx16 -mno-daz-ftz -mno-default-build-attributes -mno-div32 -mno-dsp -mno-dspr2 -mno-embedded-data -mno-enqcmd -mno-evex512 -mno-exception-handling -mnoexecstack -mno-execute-only -mno-extended-const -mno-extern-sdata -mno-f16c -mno-fix-cmse-cve-2021-35465 -mno-fix-cortex-a53-835769 -mno-fix-cortex-a57-aes-1742098 -mno-fix-cortex-a72-aes-1655431 -mno-float128 -mno-fma -mno-fma4 -mno-fmv -mno-fp16 -mno-fp-ret-in-387 -mno-fprnd -mno-fpu -mno-frecipe -mno-fsgsbase -mno-fsmuld -mno-fxsr -mno-gather -mno-gfni -mno-ginv -mno-global-merge -mno-gpopt -mno-hvx -mno-hvx-ieee-fp -mno-hvx-qfloat -mno-hreset -mno-htm -mno-iamcu -mno-implicit-float -mno-incremental-linker-compatible -mno-inline-all-stringops -mno-invariant-function-descriptors -mno-invpcid -mno-isel -mno-kl -mno-lam-bh -mno-lamcas -mno-lasx -mno-ld-seq-sa -mno-ldc1-sdc1 -mno-link-builtin-bitcode-postopt -mno-local-sdata -mno-long-calls -mno-longcall -mno-lsx -mno-lvi-cfi -mno-lvi-hardening -mno-lwp -mno-lzcnt -mno-madd4 -mno-memops -mno-mfcrf -mno-mfocrf -mno-micromips -mno-mips16 -mno-mma -mno-mmx  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK25 %s

// DXCOptionCHECK25: {{(unknown argument).*-mno-avx512vnni}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-avx512vp2intersect}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-avx512vpopcntdq}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-avxifma}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-avxneconvert}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-avxvnni}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-avxvnniint16}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-avxvnniint8}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-backchain}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-bmi}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-bmi2}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-branch-likely}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-bti-at-return-twice}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-bulk-memory}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-bulk-memory-opt}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-call-indirect-overlong}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-check-zero-division}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-cldemote}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-clflushopt}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-clwb}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-clzero}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-cmpb}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-cmpccxadd}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-constant-cfstrings}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-constructor-aliases}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-crbits}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-crc}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-crc32}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-cumode}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-cx16}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-daz-ftz}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-default-build-attributes}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-div32}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-dsp}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-dspr2}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-embedded-data}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-enqcmd}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-evex512}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-exception-handling}}
// DXCOptionCHECK25: {{(unknown argument).*-mnoexecstack}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-execute-only}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-extended-const}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-extern-sdata}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-f16c}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fix-cmse-cve-2021-35465}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fix-cortex-a53-835769}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fix-cortex-a57-aes-1742098}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fix-cortex-a72-aes-1655431}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-float128}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fma}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fma4}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fmv}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fp16}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fp-ret-in-387}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fprnd}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fpu}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-frecipe}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fsgsbase}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fsmuld}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-fxsr}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-gather}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-gfni}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-ginv}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-global-merge}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-gpopt}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-hvx}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-hvx-ieee-fp}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-hvx-qfloat}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-hreset}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-htm}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-iamcu}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-implicit-float}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-incremental-linker-compatible}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-inline-all-stringops}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-invariant-function-descriptors}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-invpcid}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-isel}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-kl}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-lam-bh}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-lamcas}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-lasx}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-ld-seq-sa}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-ldc1-sdc1}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-link-builtin-bitcode-postopt}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-local-sdata}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-long-calls}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-longcall}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-lsx}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-lvi-cfi}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-lvi-hardening}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-lwp}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-lzcnt}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-madd4}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-memops}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-mfcrf}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-mfocrf}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-micromips}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-mips16}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-mma}}
// DXCOptionCHECK25: {{(unknown argument).*-mno-mmx}}
// RUN: not %clang_dxc -mno-movbe -mno-movdir64b -mno-movdiri -mno-movrs -mno-movt -mno-mpx -mno-ms-bitfields -mno-msa -mno-mt -mno-multimemory -mno-multivalue -mno-mutable-globals -mno-mwaitx -mno-neg-immediates -mno-nontrapping-fptoint -mno-nvj -mno-nvs -mno-odd-spreg -mno-omit-leaf-frame-pointer -mno-outline -mno-outline-atomics -mno-packed-stack -mno-packets -mno-pascal-strings -mno-pclmul -mno-pconfig -mno-pcrel -mno-pic-data-is-text-relative -mno-pku -mno-popc -mno-popcnt -mno-popcntd -mno-power10-vector -mno-power8-vector -mno-power9-vector -mno-prefetchi -mno-prefixed -mno-prfchw -mno-ptwrite -mno-pure-code -mno-raoint -mno-rdpid -mno-rdpru -mno-rdrnd -mno-rdseed -mno-red-zone -mno-reference-types -mno-regnames -mno-relax -mno-relax-all -mno-relax-pic-calls -mno-relaxed-simd -mno-restrict-it -mno-retpoline -mno-retpoline-external-thunk -mno-rtd -mno-rtm -mno-sahf -mno-save-restore -mno-scalar-strict-align -mno-scatter -mno-scq -mno-serialize -mno-seses -mno-sgx -mno-sha -mno-sha512 -mno-shstk -mno-sign-ext -mno-simd128 -mno-skip-rax-setup -mno-sm3 -mno-sm4 -mno-soft-float -mno-spe -mno-speculative-load-hardening -mno-sse -mno-sse2 -mno-sse3 -mno-sse4 -mno-sse4.1 -mno-sse4.2 -mno-sse4a -mno-ssse3 -mno-stack-arg-probe -mno-stackrealign -mno-strict-align -mno-tail-call -mno-tbm -mno-tgsplit -mno-thumb -mno-tls-direct-seg-refs -mno-tocdata -mno-tocdata= -mno-tsxldtrk -mno-type-check -mno-uintr -mno-unaligned-access -mno-unaligned-symbols -mno-unsafe-fp-atomics  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK26 %s

// DXCOptionCHECK26: {{(unknown argument).*-mno-movbe}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-movdir64b}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-movdiri}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-movrs}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-movt}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-mpx}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-ms-bitfields}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-msa}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-mt}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-multimemory}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-multivalue}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-mutable-globals}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-mwaitx}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-neg-immediates}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-nontrapping-fptoint}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-nvj}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-nvs}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-odd-spreg}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-omit-leaf-frame-pointer}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-outline}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-outline-atomics}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-packed-stack}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-packets}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-pascal-strings}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-pclmul}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-pconfig}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-pcrel}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-pic-data-is-text-relative}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-pku}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-popc}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-popcnt}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-popcntd}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-power10-vector}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-power8-vector}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-power9-vector}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-prefetchi}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-prefixed}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-prfchw}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-ptwrite}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-pure-code}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-raoint}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-rdpid}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-rdpru}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-rdrnd}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-rdseed}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-red-zone}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-reference-types}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-regnames}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-relax}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-relax-all}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-relax-pic-calls}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-relaxed-simd}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-restrict-it}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-retpoline}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-retpoline-external-thunk}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-rtd}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-rtm}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sahf}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-save-restore}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-scalar-strict-align}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-scatter}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-scq}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-serialize}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-seses}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sgx}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sha}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sha512}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-shstk}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sign-ext}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-simd128}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-skip-rax-setup}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sm3}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sm4}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-soft-float}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-spe}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-speculative-load-hardening}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sse}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sse2}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sse3}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sse4}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sse4.1}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sse4.2}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-sse4a}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-ssse3}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-stack-arg-probe}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-stackrealign}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-strict-align}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-tail-call}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-tbm}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-tgsplit}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-thumb}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-tls-direct-seg-refs}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-tocdata}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-tocdata=}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-tsxldtrk}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-type-check}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-uintr}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-unaligned-access}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-unaligned-symbols}}
// DXCOptionCHECK26: {{(unknown argument).*-mno-unsafe-fp-atomics}}
// RUN: not %clang_dxc -mno-usermsr -mno-v8plus -mno-vaes -mno-vector-strict-align -mno-vevpu -mno-virt -mno-vis -mno-vis2 -mno-vis3 -mno-vpclmulqdq -mno-vsx -mno-vx -mno-vzeroupper -mno-waitpkg -mno-warn-nonportable-cfstrings -mno-wavefrontsize64 -mno-wbnoinvd -mno-wide-arithmetic -mno-widekl -mno-x87 -mno-xcoff-roptr -mno-xgot -mno-xop -mno-xsave -mno-xsavec -mno-xsaveopt -mno-xsaves -mno-zvector -mnocrc -mno-direct-move -mnontrapping-fptoint -mnop-mcount -mno-paired-vector-memops -mno-crypto -mnvj -mnvs -modd-spreg -module-dependency-dir -module-dir -module-file-deps -module-file-info -module-suffix -fmodules-reduced-bmi -momit-leaf-frame-pointer -moslib= -moutline -moutline-atomics -mpacked-stack -mpackets -mpad-max-prefix-size= -mpaired-vector-memops -mpascal-strings -mpclmul -mpconfig -mpcrel -mpic-data-is-text-relative -mpku -mpopc -mpopcnt -mpopcntd -mpower10-vector -mcrypto -mpower8-vector -mpower9-vector -mprefer-vector-width= -mprefetchi -mprefixed -mprfchw -mprintf-kind= -mprivileged -mptwrite -mpure-code -mqdsp6-compat -mraoint -mrdpid -mrdpru -mrdrnd -mrdseed -mreassociate -mrecip -mrecip= -mrecord-mcount -mred-zone -mreference-types -mregnames -mregparm -mregparm= -mrelax -mrelax-all -mrelax-pic-calls -mrelax-relocations=no -mrelaxed-simd -mrelocation-model -mrestrict-it -mretpoline -mretpoline-external-thunk -mrop-protect -mrtd -mrtm -mrvv-vector-bits=  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK27 %s

// DXCOptionCHECK27: {{(unknown argument).*-mno-usermsr}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-v8plus}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-vaes}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-vector-strict-align}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-vevpu}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-virt}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-vis}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-vis2}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-vis3}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-vpclmulqdq}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-vsx}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-vx}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-vzeroupper}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-waitpkg}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-warn-nonportable-cfstrings}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-wavefrontsize64}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-wbnoinvd}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-wide-arithmetic}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-widekl}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-x87}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-xcoff-roptr}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-xgot}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-xop}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-xsave}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-xsavec}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-xsaveopt}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-xsaves}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-zvector}}
// DXCOptionCHECK27: {{(unknown argument).*-mnocrc}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-direct-move}}
// DXCOptionCHECK27: {{(unknown argument).*-mnontrapping-fptoint}}
// DXCOptionCHECK27: {{(unknown argument).*-mnop-mcount}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-paired-vector-memops}}
// DXCOptionCHECK27: {{(unknown argument).*-mno-crypto}}
// DXCOptionCHECK27: {{(unknown argument).*-mnvj}}
// DXCOptionCHECK27: {{(unknown argument).*-mnvs}}
// DXCOptionCHECK27: {{(unknown argument).*-modd-spreg}}
// DXCOptionCHECK27: {{(unknown argument).*-module-dependency-dir}}
// DXCOptionCHECK27: {{(unknown argument).*-module-dir}}
// DXCOptionCHECK27: {{(unknown argument).*-module-file-deps}}
// DXCOptionCHECK27: {{(unknown argument).*-module-file-info}}
// DXCOptionCHECK27: {{(unknown argument).*-module-suffix}}
// DXCOptionCHECK27: {{(unknown argument).*-fmodules-reduced-bmi}}
// DXCOptionCHECK27: {{(unknown argument).*-momit-leaf-frame-pointer}}
// DXCOptionCHECK27: {{(unknown argument).*-moslib=}}
// DXCOptionCHECK27: {{(unknown argument).*-moutline}}
// DXCOptionCHECK27: {{(unknown argument).*-moutline-atomics}}
// DXCOptionCHECK27: {{(unknown argument).*-mpacked-stack}}
// DXCOptionCHECK27: {{(unknown argument).*-mpackets}}
// DXCOptionCHECK27: {{(unknown argument).*-mpad-max-prefix-size=}}
// DXCOptionCHECK27: {{(unknown argument).*-mpaired-vector-memops}}
// DXCOptionCHECK27: {{(unknown argument).*-mpascal-strings}}
// DXCOptionCHECK27: {{(unknown argument).*-mpclmul}}
// DXCOptionCHECK27: {{(unknown argument).*-mpconfig}}
// DXCOptionCHECK27: {{(unknown argument).*-mpcrel}}
// DXCOptionCHECK27: {{(unknown argument).*-mpic-data-is-text-relative}}
// DXCOptionCHECK27: {{(unknown argument).*-mpku}}
// DXCOptionCHECK27: {{(unknown argument).*-mpopc}}
// DXCOptionCHECK27: {{(unknown argument).*-mpopcnt}}
// DXCOptionCHECK27: {{(unknown argument).*-mpopcntd}}
// DXCOptionCHECK27: {{(unknown argument).*-mpower10-vector}}
// DXCOptionCHECK27: {{(unknown argument).*-mcrypto}}
// DXCOptionCHECK27: {{(unknown argument).*-mpower8-vector}}
// DXCOptionCHECK27: {{(unknown argument).*-mpower9-vector}}
// DXCOptionCHECK27: {{(unknown argument).*-mprefer-vector-width=}}
// DXCOptionCHECK27: {{(unknown argument).*-mprefetchi}}
// DXCOptionCHECK27: {{(unknown argument).*-mprefixed}}
// DXCOptionCHECK27: {{(unknown argument).*-mprfchw}}
// DXCOptionCHECK27: {{(unknown argument).*-mprintf-kind=}}
// DXCOptionCHECK27: {{(unknown argument).*-mprivileged}}
// DXCOptionCHECK27: {{(unknown argument).*-mptwrite}}
// DXCOptionCHECK27: {{(unknown argument).*-mpure-code}}
// DXCOptionCHECK27: {{(unknown argument).*-mqdsp6-compat}}
// DXCOptionCHECK27: {{(unknown argument).*-mraoint}}
// DXCOptionCHECK27: {{(unknown argument).*-mrdpid}}
// DXCOptionCHECK27: {{(unknown argument).*-mrdpru}}
// DXCOptionCHECK27: {{(unknown argument).*-mrdrnd}}
// DXCOptionCHECK27: {{(unknown argument).*-mrdseed}}
// DXCOptionCHECK27: {{(unknown argument).*-mreassociate}}
// DXCOptionCHECK27: {{(unknown argument).*-mrecip}}
// DXCOptionCHECK27: {{(unknown argument).*-mrecip=}}
// DXCOptionCHECK27: {{(unknown argument).*-mrecord-mcount}}
// DXCOptionCHECK27: {{(unknown argument).*-mred-zone}}
// DXCOptionCHECK27: {{(unknown argument).*-mreference-types}}
// DXCOptionCHECK27: {{(unknown argument).*-mregnames}}
// DXCOptionCHECK27: {{(unknown argument).*-mregparm}}
// DXCOptionCHECK27: {{(unknown argument).*-mregparm=}}
// DXCOptionCHECK27: {{(unknown argument).*-mrelax}}
// DXCOptionCHECK27: {{(unknown argument).*-mrelax-all}}
// DXCOptionCHECK27: {{(unknown argument).*-mrelax-pic-calls}}
// DXCOptionCHECK27: {{(unknown argument).*-mrelax-relocations=no}}
// DXCOptionCHECK27: {{(unknown argument).*-mrelaxed-simd}}
// DXCOptionCHECK27: {{(unknown argument).*-mrelocation-model}}
// DXCOptionCHECK27: {{(unknown argument).*-mrestrict-it}}
// DXCOptionCHECK27: {{(unknown argument).*-mretpoline}}
// DXCOptionCHECK27: {{(unknown argument).*-mretpoline-external-thunk}}
// DXCOptionCHECK27: {{(unknown argument).*-mrop-protect}}
// DXCOptionCHECK27: {{(unknown argument).*-mrtd}}
// DXCOptionCHECK27: {{(unknown argument).*-mrtm}}
// DXCOptionCHECK27: {{(unknown argument).*-mrvv-vector-bits=}}
// RUN: not %clang_dxc -msahf -msave-reg-params -msave-restore -msave-temp-labels -mscalar-strict-align -mscq -msecure-plt -mserialize -msgx -msha -msha512 -mshstk -msign-ext -msign-return-address= -msign-return-address-key= -msim -msimd128 -msimd= -msingle-float -mskip-rax-setup -msm3 -msm4 -msmall-data-limit -msmall-data-limit= -msmall-data-threshold= -msoft-float -msoft-quad-float -mspe -mspeculative-load-hardening -msse -msse2 -msse2avx -msse3 -msse4 -msse4.1 -msse4.2 -msse4a -mssse3 -mstack-alignment= -mstack-arg-probe -mstack-probe-size= -mstack-protector-guard= -mstack-protector-guard-offset= -mstack-protector-guard-reg= -mstack-protector-guard-symbol= -mstackrealign -mstrict-align -msve-vector-bits= -msvr4-struct-return -mtail-call -mtargetos= -mtbm -mtgsplit -mthread-model -mthreads -mthumb -mtls-dialect= -mtls-direct-seg-refs -mtls-size= -mtocdata -mtocdata= -mtp -mtp= -mtsxldtrk -mtune= -mtvos-simulator-version-min= -mtvos-version-min= -muclibc -muintr -multi_module -multi-lib-config= -multiply_defined -multiply_defined_unused -munaligned-access -munaligned-symbols -municode -munsafe-fp-atomics -musermsr -mv5 -mv55 -mv60 -mv62 -mv65 -mv66 -mv67 -mv67t -mv68 -mv69 -mv71 -mv71t -mv73 -mv75 -mv79 -mv8plus -mvaes -mvector-strict-align -mvevpu -mvirt -mvis -mvis2  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK28 %s

// DXCOptionCHECK28: {{(unknown argument).*-msahf}}
// DXCOptionCHECK28: {{(unknown argument).*-msave-reg-params}}
// DXCOptionCHECK28: {{(unknown argument).*-msave-restore}}
// DXCOptionCHECK28: {{(unknown argument).*-msave-temp-labels}}
// DXCOptionCHECK28: {{(unknown argument).*-mscalar-strict-align}}
// DXCOptionCHECK28: {{(unknown argument).*-mscq}}
// DXCOptionCHECK28: {{(unknown argument).*-msecure-plt}}
// DXCOptionCHECK28: {{(unknown argument).*-mserialize}}
// DXCOptionCHECK28: {{(unknown argument).*-msgx}}
// DXCOptionCHECK28: {{(unknown argument).*-msha}}
// DXCOptionCHECK28: {{(unknown argument).*-msha512}}
// DXCOptionCHECK28: {{(unknown argument).*-mshstk}}
// DXCOptionCHECK28: {{(unknown argument).*-msign-ext}}
// DXCOptionCHECK28: {{(unknown argument).*-msign-return-address=}}
// DXCOptionCHECK28: {{(unknown argument).*-msign-return-address-key=}}
// DXCOptionCHECK28: {{(unknown argument).*-msim}}
// DXCOptionCHECK28: {{(unknown argument).*-msimd128}}
// DXCOptionCHECK28: {{(unknown argument).*-msimd=}}
// DXCOptionCHECK28: {{(unknown argument).*-msingle-float}}
// DXCOptionCHECK28: {{(unknown argument).*-mskip-rax-setup}}
// DXCOptionCHECK28: {{(unknown argument).*-msm3}}
// DXCOptionCHECK28: {{(unknown argument).*-msm4}}
// DXCOptionCHECK28: {{(unknown argument).*-msmall-data-limit}}
// DXCOptionCHECK28: {{(unknown argument).*-msmall-data-limit=}}
// DXCOptionCHECK28: {{(unknown argument).*-msmall-data-threshold=}}
// DXCOptionCHECK28: {{(unknown argument).*-msoft-float}}
// DXCOptionCHECK28: {{(unknown argument).*-msoft-quad-float}}
// DXCOptionCHECK28: {{(unknown argument).*-mspe}}
// DXCOptionCHECK28: {{(unknown argument).*-mspeculative-load-hardening}}
// DXCOptionCHECK28: {{(unknown argument).*-msse}}
// DXCOptionCHECK28: {{(unknown argument).*-msse2}}
// DXCOptionCHECK28: {{(unknown argument).*-msse2avx}}
// DXCOptionCHECK28: {{(unknown argument).*-msse3}}
// DXCOptionCHECK28: {{(unknown argument).*-msse4}}
// DXCOptionCHECK28: {{(unknown argument).*-msse4.1}}
// DXCOptionCHECK28: {{(unknown argument).*-msse4.2}}
// DXCOptionCHECK28: {{(unknown argument).*-msse4a}}
// DXCOptionCHECK28: {{(unknown argument).*-mssse3}}
// DXCOptionCHECK28: {{(unknown argument).*-mstack-alignment=}}
// DXCOptionCHECK28: {{(unknown argument).*-mstack-arg-probe}}
// DXCOptionCHECK28: {{(unknown argument).*-mstack-probe-size=}}
// DXCOptionCHECK28: {{(unknown argument).*-mstack-protector-guard=}}
// DXCOptionCHECK28: {{(unknown argument).*-mstack-protector-guard-offset=}}
// DXCOptionCHECK28: {{(unknown argument).*-mstack-protector-guard-reg=}}
// DXCOptionCHECK28: {{(unknown argument).*-mstack-protector-guard-symbol=}}
// DXCOptionCHECK28: {{(unknown argument).*-mstackrealign}}
// DXCOptionCHECK28: {{(unknown argument).*-mstrict-align}}
// DXCOptionCHECK28: {{(unknown argument).*-msve-vector-bits=}}
// DXCOptionCHECK28: {{(unknown argument).*-msvr4-struct-return}}
// DXCOptionCHECK28: {{(unknown argument).*-mtail-call}}
// DXCOptionCHECK28: {{(unknown argument).*-mtargetos=}}
// DXCOptionCHECK28: {{(unknown argument).*-mtbm}}
// DXCOptionCHECK28: {{(unknown argument).*-mtgsplit}}
// DXCOptionCHECK28: {{(unknown argument).*-mthread-model}}
// DXCOptionCHECK28: {{(unknown argument).*-mthreads}}
// DXCOptionCHECK28: {{(unknown argument).*-mthumb}}
// DXCOptionCHECK28: {{(unknown argument).*-mtls-dialect=}}
// DXCOptionCHECK28: {{(unknown argument).*-mtls-direct-seg-refs}}
// DXCOptionCHECK28: {{(unknown argument).*-mtls-size=}}
// DXCOptionCHECK28: {{(unknown argument).*-mtocdata}}
// DXCOptionCHECK28: {{(unknown argument).*-mtocdata=}}
// DXCOptionCHECK28: {{(unknown argument).*-mtp}}
// DXCOptionCHECK28: {{(unknown argument).*-mtp=}}
// DXCOptionCHECK28: {{(unknown argument).*-mtsxldtrk}}
// DXCOptionCHECK28: {{(unknown argument).*-mtune=}}
// DXCOptionCHECK28: {{(unknown argument).*-mtvos-simulator-version-min=}}
// DXCOptionCHECK28: {{(unknown argument).*-mtvos-version-min=}}
// DXCOptionCHECK28: {{(unknown argument).*-muclibc}}
// DXCOptionCHECK28: {{(unknown argument).*-muintr}}
// DXCOptionCHECK28: {{(unknown argument).*-multi_module}}
// DXCOptionCHECK28: {{(unknown argument).*-multi-lib-config=}}
// DXCOptionCHECK28: {{(unknown argument).*-multiply_defined}}
// DXCOptionCHECK28: {{(unknown argument).*-multiply_defined_unused}}
// DXCOptionCHECK28: {{(unknown argument).*-munaligned-access}}
// DXCOptionCHECK28: {{(unknown argument).*-munaligned-symbols}}
// DXCOptionCHECK28: {{(unknown argument).*-municode}}
// DXCOptionCHECK28: {{(unknown argument).*-munsafe-fp-atomics}}
// DXCOptionCHECK28: {{(unknown argument).*-musermsr}}
// DXCOptionCHECK28: {{(unknown argument).*-mv5}}
// DXCOptionCHECK28: {{(unknown argument).*-mv55}}
// DXCOptionCHECK28: {{(unknown argument).*-mv60}}
// DXCOptionCHECK28: {{(unknown argument).*-mv62}}
// DXCOptionCHECK28: {{(unknown argument).*-mv65}}
// DXCOptionCHECK28: {{(unknown argument).*-mv66}}
// DXCOptionCHECK28: {{(unknown argument).*-mv67}}
// DXCOptionCHECK28: {{(unknown argument).*-mv67t}}
// DXCOptionCHECK28: {{(unknown argument).*-mv68}}
// DXCOptionCHECK28: {{(unknown argument).*-mv69}}
// DXCOptionCHECK28: {{(unknown argument).*-mv71}}
// DXCOptionCHECK28: {{(unknown argument).*-mv71t}}
// DXCOptionCHECK28: {{(unknown argument).*-mv73}}
// DXCOptionCHECK28: {{(unknown argument).*-mv75}}
// DXCOptionCHECK28: {{(unknown argument).*-mv79}}
// DXCOptionCHECK28: {{(unknown argument).*-mv8plus}}
// DXCOptionCHECK28: {{(unknown argument).*-mvaes}}
// DXCOptionCHECK28: {{(unknown argument).*-mvector-strict-align}}
// DXCOptionCHECK28: {{(unknown argument).*-mvevpu}}
// DXCOptionCHECK28: {{(unknown argument).*-mvirt}}
// DXCOptionCHECK28: {{(unknown argument).*-mvis}}
// DXCOptionCHECK28: {{(unknown argument).*-mvis2}}
// RUN: not %clang_dxc -mvis3 -mvpclmulqdq -mvscale-max= -mvscale-min= -mvsx -mvx -mvzeroupper -mwaitpkg -mwarn-nonportable-cfstrings -mwatchos-simulator-version-min= -mwatchos-version-min= -mwatchsimulator-version-min= -mwavefrontsize64 -mwbnoinvd -mwide-arithmetic -mwidekl -mwindows -mx87 -mxcoff-build-id= -mxcoff-roptr -mxgot -mxop -mxsave -mxsavec -mxsaveopt -mxsaves -mzos-hlq-clang= -mzos-hlq-csslib= -mzos-hlq-le= -mzos-sys-include= -mzos-target= -mzvector -n -new-struct-path-tbaa -no_dead_strip_inits_and_terms -no-clear-ast-before-backend -no-code-completion-globals -no-code-completion-ns-level-decls -no-cpp-precomp --no-cuda-gpu-arch= --no-cuda-include-ptx= --no-cuda-noopt-device-debug --no-cuda-version-check -fno-c++-static-destructors -no-emit-llvm-uselists -no-enable-noundef-analysis --no-gpu-bundle-output -no-hip-rt -no-implicit-float -no-integrated-cpp --no-offload-add-rpath --no-offload-arch= --no-offload-compress --no-offload-new-driver --no-offloadlib -no-pedantic -no-pie -no-pointer-tbaa -no-pthread -no-round-trip-args -no-struct-path-tbaa --no-system-header-prefix= --no-wasm-opt -nocpp -nodefaultlibs -nodriverkitlib -nofixprebinding -nogpuinc -nogpulibc -nohipwrapperinc -nolibc -nomultidefs -nopie -noprebind -noprofilelib -noseglinkedit -nostartfiles -nostdinc++ -nostdlib -nostdlibinc -nostdlib++ -nostdsysteminc --nvptx-arch-tool= -o -objc-isystem -objcxx-isystem -object --offload= --offload-add-rpath --offload-arch= --offload-compress --offload-compression-level= --offload-device-only --offload-host-device --offload-host-only --offload-link --offload-new-driver --offloadlib -fexperimental-openacc-macro-override -fexperimental-openacc-macro-override=  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK29 %s

// DXCOptionCHECK29: {{(unknown argument).*-mvis3}}
// DXCOptionCHECK29: {{(unknown argument).*-mvpclmulqdq}}
// DXCOptionCHECK29: {{(unknown argument).*-mvscale-max=}}
// DXCOptionCHECK29: {{(unknown argument).*-mvscale-min=}}
// DXCOptionCHECK29: {{(unknown argument).*-mvsx}}
// DXCOptionCHECK29: {{(unknown argument).*-mvx}}
// DXCOptionCHECK29: {{(unknown argument).*-mvzeroupper}}
// DXCOptionCHECK29: {{(unknown argument).*-mwaitpkg}}
// DXCOptionCHECK29: {{(unknown argument).*-mwarn-nonportable-cfstrings}}
// DXCOptionCHECK29: {{(unknown argument).*-mwatchos-simulator-version-min=}}
// DXCOptionCHECK29: {{(unknown argument).*-mwatchos-version-min=}}
// DXCOptionCHECK29: {{(unknown argument).*-mwatchsimulator-version-min=}}
// DXCOptionCHECK29: {{(unknown argument).*-mwavefrontsize64}}
// DXCOptionCHECK29: {{(unknown argument).*-mwbnoinvd}}
// DXCOptionCHECK29: {{(unknown argument).*-mwide-arithmetic}}
// DXCOptionCHECK29: {{(unknown argument).*-mwidekl}}
// DXCOptionCHECK29: {{(unknown argument).*-mwindows}}
// DXCOptionCHECK29: {{(unknown argument).*-mx87}}
// DXCOptionCHECK29: {{(unknown argument).*-mxcoff-build-id=}}
// DXCOptionCHECK29: {{(unknown argument).*-mxcoff-roptr}}
// DXCOptionCHECK29: {{(unknown argument).*-mxgot}}
// DXCOptionCHECK29: {{(unknown argument).*-mxop}}
// DXCOptionCHECK29: {{(unknown argument).*-mxsave}}
// DXCOptionCHECK29: {{(unknown argument).*-mxsavec}}
// DXCOptionCHECK29: {{(unknown argument).*-mxsaveopt}}
// DXCOptionCHECK29: {{(unknown argument).*-mxsaves}}
// DXCOptionCHECK29: {{(unknown argument).*-mzos-hlq-clang=}}
// DXCOptionCHECK29: {{(unknown argument).*-mzos-hlq-csslib=}}
// DXCOptionCHECK29: {{(unknown argument).*-mzos-hlq-le=}}
// DXCOptionCHECK29: {{(unknown argument).*-mzos-sys-include=}}
// DXCOptionCHECK29: {{(unknown argument).*-mzos-target=}}
// DXCOptionCHECK29: {{(unknown argument).*-mzvector}}
// DXCOptionCHECK29: {{(unknown argument).*-n}}
// DXCOptionCHECK29: {{(unknown argument).*-new-struct-path-tbaa}}
// DXCOptionCHECK29: {{(unknown argument).*-no_dead_strip_inits_and_terms}}
// DXCOptionCHECK29: {{(unknown argument).*-no-clear-ast-before-backend}}
// DXCOptionCHECK29: {{(unknown argument).*-no-code-completion-globals}}
// DXCOptionCHECK29: {{(unknown argument).*-no-code-completion-ns-level-decls}}
// DXCOptionCHECK29: {{(unknown argument).*-no-cpp-precomp}}
// DXCOptionCHECK29: {{(unknown argument).*--no-cuda-gpu-arch=}}
// DXCOptionCHECK29: {{(unknown argument).*--no-cuda-include-ptx=}}
// DXCOptionCHECK29: {{(unknown argument).*--no-cuda-noopt-device-debug}}
// DXCOptionCHECK29: {{(unknown argument).*--no-cuda-version-check}}
// DXCOptionCHECK29: {{(unknown argument).*-fno-c\+\+-static-destructors}}
// DXCOptionCHECK29: {{(unknown argument).*-no-emit-llvm-uselists}}
// DXCOptionCHECK29: {{(unknown argument).*-no-enable-noundef-analysis}}
// DXCOptionCHECK29: {{(unknown argument).*--no-gpu-bundle-output}}
// DXCOptionCHECK29: {{(unknown argument).*-no-hip-rt}}
// DXCOptionCHECK29: {{(unknown argument).*-no-implicit-float}}
// DXCOptionCHECK29: {{(unknown argument).*-no-integrated-cpp}}
// DXCOptionCHECK29: {{(unknown argument).*--no-offload-add-rpath}}
// DXCOptionCHECK29: {{(unknown argument).*--no-offload-arch=}}
// DXCOptionCHECK29: {{(unknown argument).*--no-offload-compress}}
// DXCOptionCHECK29: {{(unknown argument).*--no-offload-new-driver}}
// DXCOptionCHECK29: {{(unknown argument).*--no-offloadlib}}
// DXCOptionCHECK29: {{(unknown argument).*-no-pedantic}}
// DXCOptionCHECK29: {{(unknown argument).*-no-pie}}
// DXCOptionCHECK29: {{(unknown argument).*-no-pointer-tbaa}}
// DXCOptionCHECK29: {{(unknown argument).*-no-pthread}}
// DXCOptionCHECK29: {{(unknown argument).*-no-round-trip-args}}
// DXCOptionCHECK29: {{(unknown argument).*-no-struct-path-tbaa}}
// DXCOptionCHECK29: {{(unknown argument).*--no-system-header-prefix=}}
// DXCOptionCHECK29: {{(unknown argument).*--no-wasm-opt}}
// DXCOptionCHECK29: {{(unknown argument).*-nocpp}}
// DXCOptionCHECK29: {{(unknown argument).*-nodefaultlibs}}
// DXCOptionCHECK29: {{(unknown argument).*-nodriverkitlib}}
// DXCOptionCHECK29: {{(unknown argument).*-nofixprebinding}}
// DXCOptionCHECK29: {{(unknown argument).*-nogpuinc}}
// DXCOptionCHECK29: {{(unknown argument).*-nogpulibc}}
// DXCOptionCHECK29: {{(unknown argument).*-nohipwrapperinc}}
// DXCOptionCHECK29: {{(unknown argument).*-nolibc}}
// DXCOptionCHECK29: {{(unknown argument).*-nomultidefs}}
// DXCOptionCHECK29: {{(unknown argument).*-nopie}}
// DXCOptionCHECK29: {{(unknown argument).*-noprebind}}
// DXCOptionCHECK29: {{(unknown argument).*-noprofilelib}}
// DXCOptionCHECK29: {{(unknown argument).*-noseglinkedit}}
// DXCOptionCHECK29: {{(unknown argument).*-nostartfiles}}
// DXCOptionCHECK29: {{(unknown argument).*-nostdinc\+\+}}
// DXCOptionCHECK29: {{(unknown argument).*-nostdlib}}
// DXCOptionCHECK29: {{(unknown argument).*-nostdlibinc}}
// DXCOptionCHECK29: {{(unknown argument).*-nostdlib\+\+}}
// DXCOptionCHECK29: {{(unknown argument).*-nostdsysteminc}}
// DXCOptionCHECK29: {{(unknown argument).*--nvptx-arch-tool=}}
// DXCOptionCHECK29: {{(unknown argument).*-o}}
// DXCOptionCHECK29: {{(unknown argument).*-objc-isystem}}
// DXCOptionCHECK29: {{(unknown argument).*-objcxx-isystem}}
// DXCOptionCHECK29: {{(unknown argument).*-object}}
// DXCOptionCHECK29: {{(unknown argument).*--offload=}}
// DXCOptionCHECK29: {{(unknown argument).*--offload-add-rpath}}
// DXCOptionCHECK29: {{(unknown argument).*--offload-arch=}}
// DXCOptionCHECK29: {{(unknown argument).*--offload-compress}}
// DXCOptionCHECK29: {{(unknown argument).*--offload-compression-level=}}
// DXCOptionCHECK29: {{(unknown argument).*--offload-device-only}}
// DXCOptionCHECK29: {{(unknown argument).*--offload-host-device}}
// DXCOptionCHECK29: {{(unknown argument).*--offload-host-only}}
// DXCOptionCHECK29: {{(unknown argument).*--offload-link}}
// DXCOptionCHECK29: {{(unknown argument).*--offload-new-driver}}
// DXCOptionCHECK29: {{(unknown argument).*--offloadlib}}
// DXCOptionCHECK29: {{(unknown argument).*-fexperimental-openacc-macro-override}}
// DXCOptionCHECK29: {{(unknown argument).*-fexperimental-openacc-macro-override=}}
// RUN: not %clang_dxc -opt-record-file -opt-record-format -opt-record-passes --output-asm-variant= -p -pagezero_size -pass-exit-codes -pch-through-hdrstop-create -pch-through-hdrstop-use -pch-through-header= -pedantic -pedantic-errors -pg -pic-is-pie -pic-level -pie -pipe -plugin -plugin-arg- -pointer-tbaa -preamble-bytes= -prebind -prebind_all_twolevel_modules -preload -print-dependency-directives-minimized-source -print-diagnostic-options -print-effective-triple -print-enabled-extensions -print-file-name= -print-ivar-layout -print-libgcc-file-name -print-multi-directory -print-multi-flags-experimental -print-multi-lib -print-multi-os-directory -print-preamble -print-prog-name= -print-resource-dir -print-rocm-search-dirs -print-runtime-dir -print-search-dirs -print-stats -print-library-module-manifest-path -print-supported-cpus -print-supported-extensions -print-target-triple -print-targets -private_bundle --product-name= -pthread -pthreads --ptxas-path= -r -rdynamic -read_only_relocs -record-command-line -reexport_framework -reexport-l -reexport_library -regcall4 -relaxed-aliasing -relocatable-pch -remap -remap-file -rewrite-legacy-objc -rewrite-macros -rewrite-objc -rewrite-test --rocm-device-lib-path= --rocm-path= -round-trip-args -rpath -rtlib= -s -fsanitize-address-destructor= -fsanitize-address-use-after-return= -save-stats -save-stats= -save-temps -save-temps= -sectalign -sectcreate -sectobjectsymbols -sectorder -seg1addr -seg_addr_table -seg_addr_table_filename -segaddr -segcreate -seglinkedit -segprot -segs_read_ -segs_read_only_addr -segs_read_write_addr -setup-static-analyzer -shared -shared-libgcc -shared-libsan -show-encoding --show-includes  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK30 %s

// DXCOptionCHECK30: {{(unknown argument).*-opt-record-file}}
// DXCOptionCHECK30: {{(unknown argument).*-opt-record-format}}
// DXCOptionCHECK30: {{(unknown argument).*-opt-record-passes}}
// DXCOptionCHECK30: {{(unknown argument).*--output-asm-variant=}}
// DXCOptionCHECK30: {{(unknown argument).*-p}}
// DXCOptionCHECK30: {{(unknown argument).*-pagezero_size}}
// DXCOptionCHECK30: {{(unknown argument).*-pass-exit-codes}}
// DXCOptionCHECK30: {{(unknown argument).*-pch-through-hdrstop-create}}
// DXCOptionCHECK30: {{(unknown argument).*-pch-through-hdrstop-use}}
// DXCOptionCHECK30: {{(unknown argument).*-pch-through-header=}}
// DXCOptionCHECK30: {{(unknown argument).*-pedantic}}
// DXCOptionCHECK30: {{(unknown argument).*-pedantic-errors}}
// DXCOptionCHECK30: {{(unknown argument).*-pg}}
// DXCOptionCHECK30: {{(unknown argument).*-pic-is-pie}}
// DXCOptionCHECK30: {{(unknown argument).*-pic-level}}
// DXCOptionCHECK30: {{(unknown argument).*-pie}}
// DXCOptionCHECK30: {{(unknown argument).*-pipe}}
// DXCOptionCHECK30: {{(unknown argument).*-plugin}}
// DXCOptionCHECK30: {{(unknown argument).*-plugin-arg-}}
// DXCOptionCHECK30: {{(unknown argument).*-pointer-tbaa}}
// DXCOptionCHECK30: {{(unknown argument).*-preamble-bytes=}}
// DXCOptionCHECK30: {{(unknown argument).*-prebind}}
// DXCOptionCHECK30: {{(unknown argument).*-prebind_all_twolevel_modules}}
// DXCOptionCHECK30: {{(unknown argument).*-preload}}
// DXCOptionCHECK30: {{(unknown argument).*-print-dependency-directives-minimized-source}}
// DXCOptionCHECK30: {{(unknown argument).*-print-diagnostic-options}}
// DXCOptionCHECK30: {{(unknown argument).*-print-effective-triple}}
// DXCOptionCHECK30: {{(unknown argument).*-print-enabled-extensions}}
// DXCOptionCHECK30: {{(unknown argument).*-print-file-name=}}
// DXCOptionCHECK30: {{(unknown argument).*-print-ivar-layout}}
// DXCOptionCHECK30: {{(unknown argument).*-print-libgcc-file-name}}
// DXCOptionCHECK30: {{(unknown argument).*-print-multi-directory}}
// DXCOptionCHECK30: {{(unknown argument).*-print-multi-flags-experimental}}
// DXCOptionCHECK30: {{(unknown argument).*-print-multi-lib}}
// DXCOptionCHECK30: {{(unknown argument).*-print-multi-os-directory}}
// DXCOptionCHECK30: {{(unknown argument).*-print-preamble}}
// DXCOptionCHECK30: {{(unknown argument).*-print-prog-name=}}
// DXCOptionCHECK30: {{(unknown argument).*-print-resource-dir}}
// DXCOptionCHECK30: {{(unknown argument).*-print-rocm-search-dirs}}
// DXCOptionCHECK30: {{(unknown argument).*-print-runtime-dir}}
// DXCOptionCHECK30: {{(unknown argument).*-print-search-dirs}}
// DXCOptionCHECK30: {{(unknown argument).*-print-stats}}
// DXCOptionCHECK30: {{(unknown argument).*-print-library-module-manifest-path}}
// DXCOptionCHECK30: {{(unknown argument).*-print-supported-cpus}}
// DXCOptionCHECK30: {{(unknown argument).*-print-supported-extensions}}
// DXCOptionCHECK30: {{(unknown argument).*-print-target-triple}}
// DXCOptionCHECK30: {{(unknown argument).*-print-targets}}
// DXCOptionCHECK30: {{(unknown argument).*-private_bundle}}
// DXCOptionCHECK30: {{(unknown argument).*--product-name=}}
// DXCOptionCHECK30: {{(unknown argument).*-pthread}}
// DXCOptionCHECK30: {{(unknown argument).*-pthreads}}
// DXCOptionCHECK30: {{(unknown argument).*--ptxas-path=}}
// DXCOptionCHECK30: {{(unknown argument).*-r}}
// DXCOptionCHECK30: {{(unknown argument).*-rdynamic}}
// DXCOptionCHECK30: {{(unknown argument).*-read_only_relocs}}
// DXCOptionCHECK30: {{(unknown argument).*-record-command-line}}
// DXCOptionCHECK30: {{(unknown argument).*-reexport_framework}}
// DXCOptionCHECK30: {{(unknown argument).*-reexport-l}}
// DXCOptionCHECK30: {{(unknown argument).*-reexport_library}}
// DXCOptionCHECK30: {{(unknown argument).*-regcall4}}
// DXCOptionCHECK30: {{(unknown argument).*-relaxed-aliasing}}
// DXCOptionCHECK30: {{(unknown argument).*-relocatable-pch}}
// DXCOptionCHECK30: {{(unknown argument).*-remap}}
// DXCOptionCHECK30: {{(unknown argument).*-remap-file}}
// DXCOptionCHECK30: {{(unknown argument).*-rewrite-legacy-objc}}
// DXCOptionCHECK30: {{(unknown argument).*-rewrite-macros}}
// DXCOptionCHECK30: {{(unknown argument).*-rewrite-objc}}
// DXCOptionCHECK30: {{(unknown argument).*-rewrite-test}}
// DXCOptionCHECK30: {{(unknown argument).*--rocm-device-lib-path=}}
// DXCOptionCHECK30: {{(unknown argument).*--rocm-path=}}
// DXCOptionCHECK30: {{(unknown argument).*-round-trip-args}}
// DXCOptionCHECK30: {{(unknown argument).*-rpath}}
// DXCOptionCHECK30: {{(unknown argument).*-rtlib=}}
// DXCOptionCHECK30: {{(unknown argument).*-s}}
// DXCOptionCHECK30: {{(unknown argument).*-fsanitize-address-destructor=}}
// DXCOptionCHECK30: {{(unknown argument).*-fsanitize-address-use-after-return=}}
// DXCOptionCHECK30: {{(unknown argument).*-save-stats}}
// DXCOptionCHECK30: {{(unknown argument).*-save-stats=}}
// DXCOptionCHECK30: {{(unknown argument).*-save-temps}}
// DXCOptionCHECK30: {{(unknown argument).*-save-temps=}}
// DXCOptionCHECK30: {{(unknown argument).*-sectalign}}
// DXCOptionCHECK30: {{(unknown argument).*-sectcreate}}
// DXCOptionCHECK30: {{(unknown argument).*-sectobjectsymbols}}
// DXCOptionCHECK30: {{(unknown argument).*-sectorder}}
// DXCOptionCHECK30: {{(unknown argument).*-seg1addr}}
// DXCOptionCHECK30: {{(unknown argument).*-seg_addr_table}}
// DXCOptionCHECK30: {{(unknown argument).*-seg_addr_table_filename}}
// DXCOptionCHECK30: {{(unknown argument).*-segaddr}}
// DXCOptionCHECK30: {{(unknown argument).*-segcreate}}
// DXCOptionCHECK30: {{(unknown argument).*-seglinkedit}}
// DXCOptionCHECK30: {{(unknown argument).*-segprot}}
// DXCOptionCHECK30: {{(unknown argument).*-segs_read_}}
// DXCOptionCHECK30: {{(unknown argument).*-segs_read_only_addr}}
// DXCOptionCHECK30: {{(unknown argument).*-segs_read_write_addr}}
// DXCOptionCHECK30: {{(unknown argument).*-setup-static-analyzer}}
// DXCOptionCHECK30: {{(unknown argument).*-shared}}
// DXCOptionCHECK30: {{(unknown argument).*-shared-libgcc}}
// DXCOptionCHECK30: {{(unknown argument).*-shared-libsan}}
// DXCOptionCHECK30: {{(unknown argument).*-show-encoding}}
// DXCOptionCHECK30: {{(unknown argument).*--show-includes}}
// RUN: not %clang_dxc -show-inst -single_module -skip-function-bodies -source-date-epoch -specs -specs= -split-dwarf-file -split-dwarf-output -stack-protector -stack-protector-buffer-size -stack-usage-file -startfiles -static -static-define -static-libclosure -static-libgcc -static-libgfortran -static-libsan -static-libstdc++ -static-openmp -static-pie -stats-file= -stats-file-append -std= -std-default= -stdlib -stdlib= -stdlib++-isystem -sub_library -sub_umbrella --sycl-link -sycl-std= --symbol-graph-dir= -sys-header-deps --system-header-prefix= -t -target-abi -target-cpu -target-feature -target-linker-version -target-sdk-version= -templight-dump -test-io -time -traditional -traditional-cpp -trigraphs -trim-egraph -triple -triple= -tune-cpu -twolevel_namespace -twolevel_namespace_hints -u -umbrella -undef -undefined -unexported_symbols_list -unwindlib= -vectorize-loops -vectorize-slp -verify -verify= --verify-debug-info -verify-ignore-unexpected -verify-ignore-unexpected= -verify-pch -vtordisp-mode= --warning-suppression-mappings= --wasm-opt -weak_framework -weak_library -weak_reference_mismatches -weak-l -whatsloaded -why_load -whyload -working-directory -working-directory= -x -y -z  -### /T lib_6_7 2>&1 | FileCheck -check-prefix=DXCOptionCHECK31 %s

// DXCOptionCHECK31: {{(unknown argument).*-show-inst}}
// DXCOptionCHECK31: {{(unknown argument).*-single_module}}
// DXCOptionCHECK31: {{(unknown argument).*-skip-function-bodies}}
// DXCOptionCHECK31: {{(unknown argument).*-source-date-epoch}}
// DXCOptionCHECK31: {{(unknown argument).*-specs}}
// DXCOptionCHECK31: {{(unknown argument).*-specs=}}
// DXCOptionCHECK31: {{(unknown argument).*-split-dwarf-file}}
// DXCOptionCHECK31: {{(unknown argument).*-split-dwarf-output}}
// DXCOptionCHECK31: {{(unknown argument).*-stack-protector}}
// DXCOptionCHECK31: {{(unknown argument).*-stack-protector-buffer-size}}
// DXCOptionCHECK31: {{(unknown argument).*-stack-usage-file}}
// DXCOptionCHECK31: {{(unknown argument).*-startfiles}}
// DXCOptionCHECK31: {{(unknown argument).*-static}}
// DXCOptionCHECK31: {{(unknown argument).*-static-define}}
// DXCOptionCHECK31: {{(unknown argument).*-static-libclosure}}
// DXCOptionCHECK31: {{(unknown argument).*-static-libgcc}}
// DXCOptionCHECK31: {{(unknown argument).*-static-libgfortran}}
// DXCOptionCHECK31: {{(unknown argument).*-static-libsan}}
// DXCOptionCHECK31: {{(unknown argument).*-static-libstdc\+\+}}
// DXCOptionCHECK31: {{(unknown argument).*-static-openmp}}
// DXCOptionCHECK31: {{(unknown argument).*-static-pie}}
// DXCOptionCHECK31: {{(unknown argument).*-stats-file=}}
// DXCOptionCHECK31: {{(unknown argument).*-stats-file-append}}
// DXCOptionCHECK31: {{(unknown argument).*-std=}}
// DXCOptionCHECK31: {{(unknown argument).*-std-default=}}
// DXCOptionCHECK31: {{(unknown argument).*-stdlib}}
// DXCOptionCHECK31: {{(unknown argument).*-stdlib=}}
// DXCOptionCHECK31: {{(unknown argument).*-stdlib\+\+-isystem}}
// DXCOptionCHECK31: {{(unknown argument).*-sub_library}}
// DXCOptionCHECK31: {{(unknown argument).*-sub_umbrella}}
// DXCOptionCHECK31: {{(unknown argument).*--sycl-link}}
// DXCOptionCHECK31: {{(unknown argument).*-sycl-std=}}
// DXCOptionCHECK31: {{(unknown argument).*--symbol-graph-dir=}}
// DXCOptionCHECK31: {{(unknown argument).*-sys-header-deps}}
// DXCOptionCHECK31: {{(unknown argument).*--system-header-prefix=}}
// DXCOptionCHECK31: {{(unknown argument).*-t}}
// DXCOptionCHECK31: {{(unknown argument).*-target-abi}}
// DXCOptionCHECK31: {{(unknown argument).*-target-cpu}}
// DXCOptionCHECK31: {{(unknown argument).*-target-feature}}
// DXCOptionCHECK31: {{(unknown argument).*-target-linker-version}}
// DXCOptionCHECK31: {{(unknown argument).*-target-sdk-version=}}
// DXCOptionCHECK31: {{(unknown argument).*-templight-dump}}
// DXCOptionCHECK31: {{(unknown argument).*-test-io}}
// DXCOptionCHECK31: {{(unknown argument).*-time}}
// DXCOptionCHECK31: {{(unknown argument).*-traditional}}
// DXCOptionCHECK31: {{(unknown argument).*-traditional-cpp}}
// DXCOptionCHECK31: {{(unknown argument).*-trigraphs}}
// DXCOptionCHECK31: {{(unknown argument).*-trim-egraph}}
// DXCOptionCHECK31: {{(unknown argument).*-triple}}
// DXCOptionCHECK31: {{(unknown argument).*-triple=}}
// DXCOptionCHECK31: {{(unknown argument).*-tune-cpu}}
// DXCOptionCHECK31: {{(unknown argument).*-twolevel_namespace}}
// DXCOptionCHECK31: {{(unknown argument).*-twolevel_namespace_hints}}
// DXCOptionCHECK31: {{(unknown argument).*-u}}
// DXCOptionCHECK31: {{(unknown argument).*-umbrella}}
// DXCOptionCHECK31: {{(unknown argument).*-undef}}
// DXCOptionCHECK31: {{(unknown argument).*-undefined}}
// DXCOptionCHECK31: {{(unknown argument).*-unexported_symbols_list}}
// DXCOptionCHECK31: {{(unknown argument).*-unwindlib=}}
// DXCOptionCHECK31: {{(unknown argument).*-vectorize-loops}}
// DXCOptionCHECK31: {{(unknown argument).*-vectorize-slp}}
// DXCOptionCHECK31: {{(unknown argument).*-verify}}
// DXCOptionCHECK31: {{(unknown argument).*-verify=}}
// DXCOptionCHECK31: {{(unknown argument).*--verify-debug-info}}
// DXCOptionCHECK31: {{(unknown argument).*-verify-ignore-unexpected}}
// DXCOptionCHECK31: {{(unknown argument).*-verify-ignore-unexpected=}}
// DXCOptionCHECK31: {{(unknown argument).*-verify-pch}}
// DXCOptionCHECK31: {{(unknown argument).*-vtordisp-mode=}}
// DXCOptionCHECK31: {{(unknown argument).*--warning-suppression-mappings=}}
// DXCOptionCHECK31: {{(unknown argument).*--wasm-opt}}
// DXCOptionCHECK31: {{(unknown argument).*-weak_framework}}
// DXCOptionCHECK31: {{(unknown argument).*-weak_library}}
// DXCOptionCHECK31: {{(unknown argument).*-weak_reference_mismatches}}
// DXCOptionCHECK31: {{(unknown argument).*-weak-l}}
// DXCOptionCHECK31: {{(unknown argument).*-whatsloaded}}
// DXCOptionCHECK31: {{(unknown argument).*-why_load}}
// DXCOptionCHECK31: {{(unknown argument).*-whyload}}
// DXCOptionCHECK31: {{(unknown argument).*-working-directory}}
// DXCOptionCHECK31: {{(unknown argument).*-working-directory=}}
// DXCOptionCHECK31: {{(unknown argument).*-x}}
// DXCOptionCHECK31: {{(unknown argument).*-y}}
// DXCOptionCHECK31: {{(unknown argument).*-z}}
// RUN: not %clang -Eonly -J -Xflang -EH -EP -HELP -J -JMC -JMC- -MDd -QIfist -QIntel-jcc-erratum -? -Qfast_transcendentals -Qimprecise_fwaits -Qpar -Qpar-report -Qsafe_fp_loads -Qspectre -Qspectre-load -Qspectre-load-cf -Qvec -Qvec- -Qvec-report -Y- -Yc -Yd -Yl -Yu -Z7 -ZH:MD5 -ZH:SHA1 -ZH:SHA_256 -ZI -ZW -Za -Zc: -Zc:__STDC__ -Zc:__cplusplus -Zc:alignedNew -Zc:alignedNew- -Zc:auto -Zc:char8_t -Zc:char8_t- -Zc:dllexportInlines -Zc:dllexportInlines- -Zc:forScope -Zc:inline -Zc:rvalueCast -Zc:sizedDealloc -Zc:sizedDealloc- -Zc:strictStrings -Zc:ternary -Zc:threadSafeInit -Zc:threadSafeInit- -Zc:tlsGuards -Zc:tlsGuards- -Zc:trigraphs -Zc:trigraphs- -Zc:twoPhase -Zc:twoPhase- -Zc:wchar_t -Zc:wchar_t- -Ze -Zg -Zi -Zl -Zm -Zo -Zo- -Zp -Zp -Zs -analyze- -arch: -arm64EC -await -await: -cgthreads -clang: -clr -constexpr: -errorReport -execution-charset: -experimental: -exportHeader -external: -external:I -external:W0 -external:W1 -external:W2 -external:W3 -external:W4 -external:env: -favor -fno-sanitize-address-vcasan-lib -fp:contract -fp:except -fp:except- -fp:fast  -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVisCHECK0 %s

// DefaultVisCHECK0: {{(unknown argument).*-Eonly}}
// DefaultVisCHECK0: {{(unknown argument).*-J}}
// DefaultVisCHECK0: {{(unknown argument).*-Xflang}}
// DefaultVisCHECK0: {{(unknown argument).*-EH}}
// DefaultVisCHECK0: {{(unknown argument).*-EP}}
// DefaultVisCHECK0: {{(unknown argument).*-HELP}}
// DefaultVisCHECK0: {{(unknown argument).*-J}}
// DefaultVisCHECK0: {{(unknown argument).*-JMC}}
// DefaultVisCHECK0: {{(unknown argument).*-JMC-}}
// DefaultVisCHECK0: {{(unknown argument).*-MDd}}
// DefaultVisCHECK0: {{(unknown argument).*-QIfist}}
// DefaultVisCHECK0: {{(unknown argument).*-QIntel-jcc-erratum}}
// DefaultVisCHECK0: {{(unknown argument).*-?}}
// DefaultVisCHECK0: {{(unknown argument).*-Qfast_transcendentals}}
// DefaultVisCHECK0: {{(unknown argument).*-Qimprecise_fwaits}}
// DefaultVisCHECK0: {{(unknown argument).*-Qpar}}
// DefaultVisCHECK0: {{(unknown argument).*-Qpar-report}}
// DefaultVisCHECK0: {{(unknown argument).*-Qsafe_fp_loads}}
// DefaultVisCHECK0: {{(unknown argument).*-Qspectre}}
// DefaultVisCHECK0: {{(unknown argument).*-Qspectre-load}}
// DefaultVisCHECK0: {{(unknown argument).*-Qspectre-load-cf}}
// DefaultVisCHECK0: {{(unknown argument).*-Qvec}}
// DefaultVisCHECK0: {{(unknown argument).*-Qvec-}}
// DefaultVisCHECK0: {{(unknown argument).*-Qvec-report}}
// DefaultVisCHECK0: {{(unknown argument).*-Y-}}
// DefaultVisCHECK0: {{(unknown argument).*-Yc}}
// DefaultVisCHECK0: {{(unknown argument).*-Yd}}
// DefaultVisCHECK0: {{(unknown argument).*-Yl}}
// DefaultVisCHECK0: {{(unknown argument).*-Yu}}
// DefaultVisCHECK0: {{(unknown argument).*-Z7}}
// DefaultVisCHECK0: {{(unknown argument).*-ZH:MD5}}
// DefaultVisCHECK0: {{(unknown argument).*-ZH:SHA1}}
// DefaultVisCHECK0: {{(unknown argument).*-ZH:SHA_256}}
// DefaultVisCHECK0: {{(unknown argument).*-ZI}}
// DefaultVisCHECK0: {{(unknown argument).*-ZW}}
// DefaultVisCHECK0: {{(unknown argument).*-Za}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:__STDC__}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:__cplusplus}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:alignedNew}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:alignedNew-}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:auto}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:char8_t}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:char8_t-}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:dllexportInlines}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:dllexportInlines-}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:forScope}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:inline}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:rvalueCast}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:sizedDealloc}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:sizedDealloc-}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:strictStrings}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:ternary}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:threadSafeInit}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:threadSafeInit-}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:tlsGuards}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:tlsGuards-}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:trigraphs}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:trigraphs-}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:twoPhase}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:twoPhase-}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:wchar_t}}
// DefaultVisCHECK0: {{(unknown argument).*-Zc:wchar_t-}}
// DefaultVisCHECK0: {{(unknown argument).*-Ze}}
// DefaultVisCHECK0: {{(unknown argument).*-Zg}}
// DefaultVisCHECK0: {{(unknown argument).*-Zi}}
// DefaultVisCHECK0: {{(unknown argument).*-Zl}}
// DefaultVisCHECK0: {{(unknown argument).*-Zm}}
// DefaultVisCHECK0: {{(unknown argument).*-Zo}}
// DefaultVisCHECK0: {{(unknown argument).*-Zo-}}
// DefaultVisCHECK0: {{(unknown argument).*-Zp}}
// DefaultVisCHECK0: {{(unknown argument).*-Zp}}
// DefaultVisCHECK0: {{(unknown argument).*-Zs}}
// DefaultVisCHECK0: {{(unknown argument).*-analyze-}}
// DefaultVisCHECK0: {{(unknown argument).*-arch:}}
// DefaultVisCHECK0: {{(unknown argument).*-arm64EC}}
// DefaultVisCHECK0: {{(unknown argument).*-await}}
// DefaultVisCHECK0: {{(unknown argument).*-await:}}
// DefaultVisCHECK0: {{(unknown argument).*-cgthreads}}
// DefaultVisCHECK0: {{(unknown argument).*-clang:}}
// DefaultVisCHECK0: {{(unknown argument).*-clr}}
// DefaultVisCHECK0: {{(unknown argument).*-constexpr:}}
// DefaultVisCHECK0: {{(unknown argument).*-errorReport}}
// DefaultVisCHECK0: {{(unknown argument).*-execution-charset:}}
// DefaultVisCHECK0: {{(unknown argument).*-experimental:}}
// DefaultVisCHECK0: {{(unknown argument).*-exportHeader}}
// DefaultVisCHECK0: {{(unknown argument).*-external:}}
// DefaultVisCHECK0: {{(unknown argument).*-external:I}}
// DefaultVisCHECK0: {{(unknown argument).*-external:W0}}
// DefaultVisCHECK0: {{(unknown argument).*-external:W1}}
// DefaultVisCHECK0: {{(unknown argument).*-external:W2}}
// DefaultVisCHECK0: {{(unknown argument).*-external:W3}}
// DefaultVisCHECK0: {{(unknown argument).*-external:W4}}
// DefaultVisCHECK0: {{(unknown argument).*-external:env:}}
// DefaultVisCHECK0: {{(unknown argument).*-favor}}
// DefaultVisCHECK0: {{(unknown argument).*-fno-sanitize-address-vcasan-lib}}
// DefaultVisCHECK0: {{(unknown argument).*-fp:contract}}
// DefaultVisCHECK0: {{(unknown argument).*-fp:except}}
// DefaultVisCHECK0: {{(unknown argument).*-fp:except-}}
// DefaultVisCHECK0: {{(unknown argument).*-fp:fast}}
// RUN: not %clang -fp:precise -fp:strict -fsanitize-address-use-after-return -guard: -headerUnit -headerUnit:angle -headerUnit:quote -headerName: -homeparams -hotpatch -imsvc -kernel -kernel- -nologo -permissive -permissive- -reference -sdl -sdl- -showFilenames -showFilenames- -showIncludes -showIncludes:user -sourceDependencies -sourceDependencies:directives -source-charset: -std: -translateInclude -tune: -validate-charset -validate-charset- -vctoolsdir -vctoolsversion -vd -vmb -vmg -vmm -vms -vmv -volatile:iso -volatile:ms -wd -winsdkdir -winsdkversion -winsysroot -add-plugin -faligned-alloc-unavailable -cfg-add-implicit-dtors -analyze-function -analyze-function= -analyzer-checker -analyzer-checker= -analyzer-checker-help -analyzer-checker-help-alpha -analyzer-checker-help-developer -analyzer-checker-option-help -analyzer-checker-option-help-alpha -analyzer-checker-option-help-developer -analyzer-config -analyzer-config-compatibility-mode -analyzer-config-compatibility-mode= -analyzer-config-help -analyzer-constraints -analyzer-constraints= -analyzer-disable-all-checks -analyzer-disable-checker -analyzer-disable-checker= -analyzer-disable-retry-exhausted -analyzer-display-progress -analyzer-dump-egraph -analyzer-dump-egraph= -analyzer-inline-max-stack-depth -analyzer-inline-max-stack-depth= -analyzer-inlining-mode -analyzer-inlining-mode= -analyzer-list-enabled-checkers -analyzer-max-loop -analyzer-note-analysis-entry-points -analyzer-opt-analyze-headers -analyzer-purge -analyzer-purge= -analyzer-stats -analyzer-viz-egraph-graphviz -analyzer-werror -coverage-data-file= -coverage-notes-file= -fopenmp-is-device -Qgather- -Qscatter- -Qembed_debug -as-secure-log-file -ast-dump -ast-dump= -ast-dump-all -ast-dump-all= -ast-dump-decl-types -ast-dump-filter -ast-dump-filter= -ast-dump-lookups -ast-list  -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVisCHECK1 %s

// DefaultVisCHECK1: {{(unknown argument).*-fp:precise}}
// DefaultVisCHECK1: {{(unknown argument).*-fp:strict}}
// DefaultVisCHECK1: {{(unknown argument).*-fsanitize-address-use-after-return}}
// DefaultVisCHECK1: {{(unknown argument).*-guard:}}
// DefaultVisCHECK1: {{(unknown argument).*-headerUnit}}
// DefaultVisCHECK1: {{(unknown argument).*-headerUnit:angle}}
// DefaultVisCHECK1: {{(unknown argument).*-headerUnit:quote}}
// DefaultVisCHECK1: {{(unknown argument).*-headerName:}}
// DefaultVisCHECK1: {{(unknown argument).*-homeparams}}
// DefaultVisCHECK1: {{(unknown argument).*-hotpatch}}
// DefaultVisCHECK1: {{(unknown argument).*-imsvc}}
// DefaultVisCHECK1: {{(unknown argument).*-kernel}}
// DefaultVisCHECK1: {{(unknown argument).*-kernel-}}
// DefaultVisCHECK1: {{(unknown argument).*-nologo}}
// DefaultVisCHECK1: {{(unknown argument).*-permissive}}
// DefaultVisCHECK1: {{(unknown argument).*-permissive-}}
// DefaultVisCHECK1: {{(unknown argument).*-reference}}
// DefaultVisCHECK1: {{(unknown argument).*-sdl}}
// DefaultVisCHECK1: {{(unknown argument).*-sdl-}}
// DefaultVisCHECK1: {{(unknown argument).*-showFilenames}}
// DefaultVisCHECK1: {{(unknown argument).*-showFilenames-}}
// DefaultVisCHECK1: {{(unknown argument).*-showIncludes}}
// DefaultVisCHECK1: {{(unknown argument).*-showIncludes:user}}
// DefaultVisCHECK1: {{(unknown argument).*-sourceDependencies}}
// DefaultVisCHECK1: {{(unknown argument).*-sourceDependencies:directives}}
// DefaultVisCHECK1: {{(unknown argument).*-source-charset:}}
// DefaultVisCHECK1: {{(unknown argument).*-std:}}
// DefaultVisCHECK1: {{(unknown argument).*-translateInclude}}
// DefaultVisCHECK1: {{(unknown argument).*-tune:}}
// DefaultVisCHECK1: {{(unknown argument).*-validate-charset}}
// DefaultVisCHECK1: {{(unknown argument).*-validate-charset-}}
// DefaultVisCHECK1: {{(unknown argument).*-vctoolsdir}}
// DefaultVisCHECK1: {{(unknown argument).*-vctoolsversion}}
// DefaultVisCHECK1: {{(unknown argument).*-vd}}
// DefaultVisCHECK1: {{(unknown argument).*-vmb}}
// DefaultVisCHECK1: {{(unknown argument).*-vmg}}
// DefaultVisCHECK1: {{(unknown argument).*-vmm}}
// DefaultVisCHECK1: {{(unknown argument).*-vms}}
// DefaultVisCHECK1: {{(unknown argument).*-vmv}}
// DefaultVisCHECK1: {{(unknown argument).*-volatile:iso}}
// DefaultVisCHECK1: {{(unknown argument).*-volatile:ms}}
// DefaultVisCHECK1: {{(unknown argument).*-wd}}
// DefaultVisCHECK1: {{(unknown argument).*-winsdkdir}}
// DefaultVisCHECK1: {{(unknown argument).*-winsdkversion}}
// DefaultVisCHECK1: {{(unknown argument).*-winsysroot}}
// DefaultVisCHECK1: {{(unknown argument).*-add-plugin}}
// DefaultVisCHECK1: {{(unknown argument).*-faligned-alloc-unavailable}}
// DefaultVisCHECK1: {{(unknown argument).*-cfg-add-implicit-dtors}}
// DefaultVisCHECK1: {{(unknown argument).*-analyze-function}}
// DefaultVisCHECK1: {{(unknown argument).*-analyze-function=}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-checker}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-checker=}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-checker-help}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-checker-help-alpha}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-checker-help-developer}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-checker-option-help}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-checker-option-help-alpha}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-checker-option-help-developer}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-config}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-config-compatibility-mode}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-config-compatibility-mode=}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-config-help}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-constraints}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-constraints=}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-disable-all-checks}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-disable-checker}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-disable-checker=}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-disable-retry-exhausted}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-display-progress}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-dump-egraph}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-dump-egraph=}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-inline-max-stack-depth}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-inline-max-stack-depth=}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-inlining-mode}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-inlining-mode=}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-list-enabled-checkers}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-max-loop}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-note-analysis-entry-points}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-opt-analyze-headers}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-purge}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-purge=}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-stats}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-viz-egraph-graphviz}}
// DefaultVisCHECK1: {{(unknown argument).*-analyzer-werror}}
// DefaultVisCHECK1: {{(unknown argument).*-coverage-data-file=}}
// DefaultVisCHECK1: {{(unknown argument).*-coverage-notes-file=}}
// DefaultVisCHECK1: {{(unknown argument).*-fopenmp-is-device}}
// DefaultVisCHECK1: {{(unknown argument).*-Qgather-}}
// DefaultVisCHECK1: {{(unknown argument).*-Qscatter-}}
// DefaultVisCHECK1: {{(unknown argument).*-Qembed_debug}}
// DefaultVisCHECK1: {{(unknown argument).*-as-secure-log-file}}
// DefaultVisCHECK1: {{(unknown argument).*-ast-dump}}
// DefaultVisCHECK1: {{(unknown argument).*-ast-dump=}}
// DefaultVisCHECK1: {{(unknown argument).*-ast-dump-all}}
// DefaultVisCHECK1: {{(unknown argument).*-ast-dump-all=}}
// DefaultVisCHECK1: {{(unknown argument).*-ast-dump-decl-types}}
// DefaultVisCHECK1: {{(unknown argument).*-ast-dump-filter}}
// DefaultVisCHECK1: {{(unknown argument).*-ast-dump-filter=}}
// DefaultVisCHECK1: {{(unknown argument).*-ast-dump-lookups}}
// DefaultVisCHECK1: {{(unknown argument).*-ast-list}}
// RUN: not %clang -ast-merge -ast-print -ast-view -aux-target-cpu -aux-target-feature -aux-triple -c-isystem -cfguard -cfguard-no-checks -chain-include -clear-ast-before-backend -code-completion-at -code-completion-at= -code-completion-brief-comments -code-completion-macros -code-completion-patterns -code-completion-with-fixits -compiler-options-dump -complex-range= -compress-debug-sections -compress-debug-sections= -coverage-version= -cpp --crel -serialize-diagnostic-file -HV -hlsl-no-stdinc -ehcontguard -emit-codegen-only -emit-fir -emit-header-unit -emit-hlfir -emit-html -emit-llvm-bc -emit-llvm-only -emit-llvm-uselists -emit-mlir -emit-module -emit-module-interface -emit-obj -emit-pch -emit-pristine-llvm -emit-reduced-module-interface --emit-sgf-symbol-labels-for-testing -enable-16bit-types -enable-noundef-analysis -enable-tlsdesc -error-on-deserialized-decl -error-on-deserialized-decl= -exception-model -exception-model= -faddress-space-map-mangling= -fallow-pch-with-different-modules-cache-path -fallow-pch-with-compiler-errors -fallow-pcm-with-compiler-errors -falternative-parameter-statement -fanalyzed-objects-for-unparse -fapply-global-visibility-to-externs -fbackslash -fbfloat16-excess-precision= -fblocks-runtime-optional -fexperimental-bounds-safety -fbracket-depth -fbuiltin-headers-in-system-modules -fcgl -fcompatibility-qualified-id-block-type-checking -fconst-strings -fconstant-string-class -fconvert= -fctor-dtor-return-this -fcuda-allow-variadic-functions -fcuda-include-gpubinary -fcuda-is-device -fdebug-dump-all -fdebug-dump-parse-tree -fdebug-dump-parse-tree-no-sema -fdebug-dump-parsing-log -fdebug-dump-pft -fdebug-dump-provenance -fdebug-dump-symbols -fdebug-measure-parse-tree -fdebug-module-writer -fdebug-pass-manager -fdebug-pre-fir-tree -fdebug-unparse -fdebug-unparse-no-sema -fdebug-unparse-with-modules -fdebug-unparse-with-symbols -fdebugger-cast-result-to-id -fdebugger-objc-literal -fdebugger-support -fdeclare-opencl-builtins -fdefault-calling-conv= -fdefault-double-8 -fdefault-integer-8 -fdefault-real-8 -fdenormal-fp-math-f32= -fdeprecated-macro -fdiagnostics-format -fdiagnostics-show-category  -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVisCHECK2 %s

// DefaultVisCHECK2: {{(unknown argument).*-ast-merge}}
// DefaultVisCHECK2: {{(unknown argument).*-ast-print}}
// DefaultVisCHECK2: {{(unknown argument).*-ast-view}}
// DefaultVisCHECK2: {{(unknown argument).*-aux-target-cpu}}
// DefaultVisCHECK2: {{(unknown argument).*-aux-target-feature}}
// DefaultVisCHECK2: {{(unknown argument).*-aux-triple}}
// DefaultVisCHECK2: {{(unknown argument).*-c-isystem}}
// DefaultVisCHECK2: {{(unknown argument).*-cfguard}}
// DefaultVisCHECK2: {{(unknown argument).*-cfguard-no-checks}}
// DefaultVisCHECK2: {{(unknown argument).*-chain-include}}
// DefaultVisCHECK2: {{(unknown argument).*-clear-ast-before-backend}}
// DefaultVisCHECK2: {{(unknown argument).*-code-completion-at}}
// DefaultVisCHECK2: {{(unknown argument).*-code-completion-at=}}
// DefaultVisCHECK2: {{(unknown argument).*-code-completion-brief-comments}}
// DefaultVisCHECK2: {{(unknown argument).*-code-completion-macros}}
// DefaultVisCHECK2: {{(unknown argument).*-code-completion-patterns}}
// DefaultVisCHECK2: {{(unknown argument).*-code-completion-with-fixits}}
// DefaultVisCHECK2: {{(unknown argument).*-compiler-options-dump}}
// DefaultVisCHECK2: {{(unknown argument).*-complex-range=}}
// DefaultVisCHECK2: {{(unknown argument).*-compress-debug-sections}}
// DefaultVisCHECK2: {{(unknown argument).*-compress-debug-sections=}}
// DefaultVisCHECK2: {{(unknown argument).*-coverage-version=}}
// DefaultVisCHECK2: {{(unknown argument).*-cpp}}
// DefaultVisCHECK2: {{(unknown argument).*--crel}}
// DefaultVisCHECK2: {{(unknown argument).*-serialize-diagnostic-file}}
// DefaultVisCHECK2: {{(unknown argument).*-HV}}
// DefaultVisCHECK2: {{(unknown argument).*-hlsl-no-stdinc}}
// DefaultVisCHECK2: {{(unknown argument).*-ehcontguard}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-codegen-only}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-fir}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-header-unit}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-hlfir}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-html}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-llvm-bc}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-llvm-only}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-llvm-uselists}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-mlir}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-module}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-module-interface}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-obj}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-pch}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-pristine-llvm}}
// DefaultVisCHECK2: {{(unknown argument).*-emit-reduced-module-interface}}
// DefaultVisCHECK2: {{(unknown argument).*--emit-sgf-symbol-labels-for-testing}}
// DefaultVisCHECK2: {{(unknown argument).*-enable-16bit-types}}
// DefaultVisCHECK2: {{(unknown argument).*-enable-noundef-analysis}}
// DefaultVisCHECK2: {{(unknown argument).*-enable-tlsdesc}}
// DefaultVisCHECK2: {{(unknown argument).*-error-on-deserialized-decl}}
// DefaultVisCHECK2: {{(unknown argument).*-error-on-deserialized-decl=}}
// DefaultVisCHECK2: {{(unknown argument).*-exception-model}}
// DefaultVisCHECK2: {{(unknown argument).*-exception-model=}}
// DefaultVisCHECK2: {{(unknown argument).*-faddress-space-map-mangling=}}
// DefaultVisCHECK2: {{(unknown argument).*-fallow-pch-with-different-modules-cache-path}}
// DefaultVisCHECK2: {{(unknown argument).*-fallow-pch-with-compiler-errors}}
// DefaultVisCHECK2: {{(unknown argument).*-fallow-pcm-with-compiler-errors}}
// DefaultVisCHECK2: {{(unknown argument).*-falternative-parameter-statement}}
// DefaultVisCHECK2: {{(unknown argument).*-fanalyzed-objects-for-unparse}}
// DefaultVisCHECK2: {{(unknown argument).*-fapply-global-visibility-to-externs}}
// DefaultVisCHECK2: {{(unknown argument).*-fbackslash}}
// DefaultVisCHECK2: {{(unknown argument).*-fbfloat16-excess-precision=}}
// DefaultVisCHECK2: {{(unknown argument).*-fblocks-runtime-optional}}
// DefaultVisCHECK2: {{(unknown argument).*-fexperimental-bounds-safety}}
// DefaultVisCHECK2: {{(unknown argument).*-fbracket-depth}}
// DefaultVisCHECK2: {{(unknown argument).*-fbuiltin-headers-in-system-modules}}
// DefaultVisCHECK2: {{(unknown argument).*-fcgl}}
// DefaultVisCHECK2: {{(unknown argument).*-fcompatibility-qualified-id-block-type-checking}}
// DefaultVisCHECK2: {{(unknown argument).*-fconst-strings}}
// DefaultVisCHECK2: {{(unknown argument).*-fconstant-string-class}}
// DefaultVisCHECK2: {{(unknown argument).*-fconvert=}}
// DefaultVisCHECK2: {{(unknown argument).*-fctor-dtor-return-this}}
// DefaultVisCHECK2: {{(unknown argument).*-fcuda-allow-variadic-functions}}
// DefaultVisCHECK2: {{(unknown argument).*-fcuda-include-gpubinary}}
// DefaultVisCHECK2: {{(unknown argument).*-fcuda-is-device}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-dump-all}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-dump-parse-tree}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-dump-parse-tree-no-sema}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-dump-parsing-log}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-dump-pft}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-dump-provenance}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-dump-symbols}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-measure-parse-tree}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-module-writer}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-pass-manager}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-pre-fir-tree}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-unparse}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-unparse-no-sema}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-unparse-with-modules}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebug-unparse-with-symbols}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebugger-cast-result-to-id}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebugger-objc-literal}}
// DefaultVisCHECK2: {{(unknown argument).*-fdebugger-support}}
// DefaultVisCHECK2: {{(unknown argument).*-fdeclare-opencl-builtins}}
// DefaultVisCHECK2: {{(unknown argument).*-fdefault-calling-conv=}}
// DefaultVisCHECK2: {{(unknown argument).*-fdefault-double-8}}
// DefaultVisCHECK2: {{(unknown argument).*-fdefault-integer-8}}
// DefaultVisCHECK2: {{(unknown argument).*-fdefault-real-8}}
// DefaultVisCHECK2: {{(unknown argument).*-fdenormal-fp-math-f32=}}
// DefaultVisCHECK2: {{(unknown argument).*-fdeprecated-macro}}
// DefaultVisCHECK2: {{(unknown argument).*-fdiagnostics-format}}
// DefaultVisCHECK2: {{(unknown argument).*-fdiagnostics-show-category}}
// RUN: not %clang -fdisable-integer-16 -fdisable-integer-2 -fdisable-module-hash -fdisable-real-10 -fdisable-real-3 -fdump-record-layouts -fdump-record-layouts-canonical -fdump-record-layouts-complete -fdump-record-layouts-simple -fdump-vtable-layouts -fencode-extended-block-signature -ferror-limit -fexperimental-assignment-tracking= -fexperimental-max-bitint-width= -fexperimental-omit-vtable-rtti -fexternc-nounwind -ffake-address-space-map -fimplicit-modules-use-lock -ffixed-form -ffixed-line-length= -ffixed-line-length- -ffloat16-excess-precision= -fforbid-guard-variables -ffree-form -fget-definition -fget-symbols-sources -fhalf-no-semantic-interposition -fhermetic-module-files -filetype -fimplicit-none -fimplicit-none-ext -finclude-default-header -fintrinsic-modules-path -fix-only-warnings -fix-what-you-can -fixit -fixit= -fixit-recompile -fixit-to-temporary -flang-deprecated-no-hlfir -flang-experimental-hlfir -flarge-sizes -flogical-abbreviations -fversion-loops-for-stride -flto-unit -flto-visibility-public-std -fmcdc-max-conditions= -fmcdc-max-test-vectors= -fmerge-functions -fmodule-feature -fmodule-file-home-is-cwd -fmodule-format= -fmodule-map-file-home-is-cwd -fmodules-codegen -fmodules-debuginfo -fmodules-embed-file= -fmodules-hash-content -fmodules-local-submodule-visibility -fmodules-skip-diagnostic-options -fmodules-skip-header-search-paths -fmodules-strict-context-hash -fms-kernel -fnative-half-arguments-and-returns -fnative-half-type -fno-analyzed-objects-for-unparse -fno-automatic -fno-backslash -fno-bitfield-type-align -fno-experimental-bounds-safety -fno-const-strings -fno-cuda-host-device-constexpr -fno-debug-pass-manager -fno-deprecated-macro -fno-diagnostics-use-presumed-location -fno-dllexport-inlines -fno-experimental-omit-vtable-rtti -fno-implicit-modules-use-lock -fno-implicit-none -fno-implicit-none-ext -fno-logical-abbreviations -fno-version-loops-for-stride -fno-lto-unit -fno-math-builtin -fno-modules-error-recovery -fno-modules-global-index -fno-modules-prune-non-affecting-module-map-files -fno-modules-share-filemanager -fno-modules-skip-diagnostic-options -fno-modules-skip-header-search-paths -fno-openmp-optimistic-collapse -fno-padding-on-unsigned-fixed-point -fno-pch-timestamp -fno-ppc-native-vector-element-order -fno-realloc-lhs -fno-recovery-ast -fno-recovery-ast-type -fno-reformat -fno-retain-subst-template-type-parm-type-ast-nodes -fno-save-main-program -fno-signed-wchar  -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVisCHECK3 %s

// DefaultVisCHECK3: {{(unknown argument).*-fdisable-integer-16}}
// DefaultVisCHECK3: {{(unknown argument).*-fdisable-integer-2}}
// DefaultVisCHECK3: {{(unknown argument).*-fdisable-module-hash}}
// DefaultVisCHECK3: {{(unknown argument).*-fdisable-real-10}}
// DefaultVisCHECK3: {{(unknown argument).*-fdisable-real-3}}
// DefaultVisCHECK3: {{(unknown argument).*-fdump-record-layouts}}
// DefaultVisCHECK3: {{(unknown argument).*-fdump-record-layouts-canonical}}
// DefaultVisCHECK3: {{(unknown argument).*-fdump-record-layouts-complete}}
// DefaultVisCHECK3: {{(unknown argument).*-fdump-record-layouts-simple}}
// DefaultVisCHECK3: {{(unknown argument).*-fdump-vtable-layouts}}
// DefaultVisCHECK3: {{(unknown argument).*-fencode-extended-block-signature}}
// DefaultVisCHECK3: {{(unknown argument).*-ferror-limit}}
// DefaultVisCHECK3: {{(unknown argument).*-fexperimental-assignment-tracking=}}
// DefaultVisCHECK3: {{(unknown argument).*-fexperimental-max-bitint-width=}}
// DefaultVisCHECK3: {{(unknown argument).*-fexperimental-omit-vtable-rtti}}
// DefaultVisCHECK3: {{(unknown argument).*-fexternc-nounwind}}
// DefaultVisCHECK3: {{(unknown argument).*-ffake-address-space-map}}
// DefaultVisCHECK3: {{(unknown argument).*-fimplicit-modules-use-lock}}
// DefaultVisCHECK3: {{(unknown argument).*-ffixed-form}}
// DefaultVisCHECK3: {{(unknown argument).*-ffixed-line-length=}}
// DefaultVisCHECK3: {{(unknown argument).*-ffixed-line-length-}}
// DefaultVisCHECK3: {{(unknown argument).*-ffloat16-excess-precision=}}
// DefaultVisCHECK3: {{(unknown argument).*-fforbid-guard-variables}}
// DefaultVisCHECK3: {{(unknown argument).*-ffree-form}}
// DefaultVisCHECK3: {{(unknown argument).*-fget-definition}}
// DefaultVisCHECK3: {{(unknown argument).*-fget-symbols-sources}}
// DefaultVisCHECK3: {{(unknown argument).*-fhalf-no-semantic-interposition}}
// DefaultVisCHECK3: {{(unknown argument).*-fhermetic-module-files}}
// DefaultVisCHECK3: {{(unknown argument).*-filetype}}
// DefaultVisCHECK3: {{(unknown argument).*-fimplicit-none}}
// DefaultVisCHECK3: {{(unknown argument).*-fimplicit-none-ext}}
// DefaultVisCHECK3: {{(unknown argument).*-finclude-default-header}}
// DefaultVisCHECK3: {{(unknown argument).*-fintrinsic-modules-path}}
// DefaultVisCHECK3: {{(unknown argument).*-fix-only-warnings}}
// DefaultVisCHECK3: {{(unknown argument).*-fix-what-you-can}}
// DefaultVisCHECK3: {{(unknown argument).*-fixit}}
// DefaultVisCHECK3: {{(unknown argument).*-fixit=}}
// DefaultVisCHECK3: {{(unknown argument).*-fixit-recompile}}
// DefaultVisCHECK3: {{(unknown argument).*-fixit-to-temporary}}
// DefaultVisCHECK3: {{(unknown argument).*-flang-deprecated-no-hlfir}}
// DefaultVisCHECK3: {{(unknown argument).*-flang-experimental-hlfir}}
// DefaultVisCHECK3: {{(unknown argument).*-flarge-sizes}}
// DefaultVisCHECK3: {{(unknown argument).*-flogical-abbreviations}}
// DefaultVisCHECK3: {{(unknown argument).*-fversion-loops-for-stride}}
// DefaultVisCHECK3: {{(unknown argument).*-flto-unit}}
// DefaultVisCHECK3: {{(unknown argument).*-flto-visibility-public-std}}
// DefaultVisCHECK3: {{(unknown argument).*-fmcdc-max-conditions=}}
// DefaultVisCHECK3: {{(unknown argument).*-fmcdc-max-test-vectors=}}
// DefaultVisCHECK3: {{(unknown argument).*-fmerge-functions}}
// DefaultVisCHECK3: {{(unknown argument).*-fmodule-feature}}
// DefaultVisCHECK3: {{(unknown argument).*-fmodule-file-home-is-cwd}}
// DefaultVisCHECK3: {{(unknown argument).*-fmodule-format=}}
// DefaultVisCHECK3: {{(unknown argument).*-fmodule-map-file-home-is-cwd}}
// DefaultVisCHECK3: {{(unknown argument).*-fmodules-codegen}}
// DefaultVisCHECK3: {{(unknown argument).*-fmodules-debuginfo}}
// DefaultVisCHECK3: {{(unknown argument).*-fmodules-embed-file=}}
// DefaultVisCHECK3: {{(unknown argument).*-fmodules-hash-content}}
// DefaultVisCHECK3: {{(unknown argument).*-fmodules-local-submodule-visibility}}
// DefaultVisCHECK3: {{(unknown argument).*-fmodules-skip-diagnostic-options}}
// DefaultVisCHECK3: {{(unknown argument).*-fmodules-skip-header-search-paths}}
// DefaultVisCHECK3: {{(unknown argument).*-fmodules-strict-context-hash}}
// DefaultVisCHECK3: {{(unknown argument).*-fms-kernel}}
// DefaultVisCHECK3: {{(unknown argument).*-fnative-half-arguments-and-returns}}
// DefaultVisCHECK3: {{(unknown argument).*-fnative-half-type}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-analyzed-objects-for-unparse}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-automatic}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-backslash}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-bitfield-type-align}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-experimental-bounds-safety}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-const-strings}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-cuda-host-device-constexpr}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-debug-pass-manager}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-deprecated-macro}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-diagnostics-use-presumed-location}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-dllexport-inlines}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-experimental-omit-vtable-rtti}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-implicit-modules-use-lock}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-implicit-none}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-implicit-none-ext}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-logical-abbreviations}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-version-loops-for-stride}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-lto-unit}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-math-builtin}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-modules-error-recovery}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-modules-global-index}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-modules-prune-non-affecting-module-map-files}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-modules-share-filemanager}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-modules-skip-diagnostic-options}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-modules-skip-header-search-paths}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-openmp-optimistic-collapse}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-padding-on-unsigned-fixed-point}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-pch-timestamp}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-ppc-native-vector-element-order}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-realloc-lhs}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-recovery-ast}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-recovery-ast-type}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-reformat}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-retain-subst-template-type-parm-type-ast-nodes}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-save-main-program}}
// DefaultVisCHECK3: {{(unknown argument).*-fno-signed-wchar}}
// RUN: not %clang -fno-stack-arrays -fno-underscoring -fno-unsigned -fno-use-ctor-homing -fno-validate-pch -fno-wchar -fno-xor-operator -fobjc-arc-cxxlib= -fobjc-dispatch-method= -fobjc-gc -fobjc-gc-only -fobjc-runtime-has-weak -fobjc-subscripting-legacy-runtime -fopenmp-host-ir-file-path -fopenmp-is-target-device -foverride-record-layout= -fpadding-on-unsigned-fixed-point -fpass-by-value-is-noalias -fpatchable-function-entry-offset= -fppc-native-vector-element-order -fpreprocess-include-lines -fprofile-instrument= -fprofile-instrument-path= -fprofile-instrument-use-path= -frealloc-lhs -frecovery-ast -frecovery-ast-type -fretain-subst-template-type-parm-type-ast-nodes -fsanitize-coverage-8bit-counters -fsanitize-coverage-control-flow -fsanitize-coverage-indirect-calls -fsanitize-coverage-inline-8bit-counters -fsanitize-coverage-inline-bool-flag -fsanitize-coverage-no-prune -fsanitize-coverage-pc-table -fsanitize-coverage-stack-depth -fsanitize-coverage-trace-bb -fsanitize-coverage-trace-cmp -fsanitize-coverage-trace-div -fsanitize-coverage-trace-gep -fsanitize-coverage-trace-loads -fsanitize-coverage-trace-pc -fsanitize-coverage-trace-pc-guard -fsanitize-coverage-trace-stores -fsanitize-coverage-type= -fsave-main-program -fsigned-wchar -fstack-arrays -fsycl-is-device -fsycl-is-host -ftabstop -ftest-module-file-extension= -ftype-visibility= -function-alignment -funderscoring -funknown-anytype -funsigned -funwind-tables= -fuse-ctor-homing -fuse-register-sized-bitfield-access -fverify-debuginfo-preserve -fverify-debuginfo-preserve-export= -fwarn-stack-size= -fwchar-type= -fxor-operator -gsimple-template-names= -gsrc-hash= -header-include-file -header-include-filtering= -header-include-format= -import-call-optimization -init-only -internal-externc-isystem -internal-isystem -main-file-name -massembler-fatal-warnings -massembler-no-warn -mbranch-protection-pauth-lr -mbranch-target-enforce -mdebug-pass -menable-no-infs -menable-no-nans -mfloat-abi -mfpmath -mframe-pointer= -mguarded-control-stack -no-finalize-removal -no-ns-alloc-error -mlimit-float-precision -mlink-bitcode-file -mlink-builtin-bitcode -mmapsyms=implicit -mnoexecstack -mno-type-check -module-dir -module-file-deps -module-suffix -mreassociate -mregparm -mrelax-relocations=no  -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVisCHECK4 %s

// DefaultVisCHECK4: {{(unknown argument).*-fno-stack-arrays}}
// DefaultVisCHECK4: {{(unknown argument).*-fno-underscoring}}
// DefaultVisCHECK4: {{(unknown argument).*-fno-unsigned}}
// DefaultVisCHECK4: {{(unknown argument).*-fno-use-ctor-homing}}
// DefaultVisCHECK4: {{(unknown argument).*-fno-validate-pch}}
// DefaultVisCHECK4: {{(unknown argument).*-fno-wchar}}
// DefaultVisCHECK4: {{(unknown argument).*-fno-xor-operator}}
// DefaultVisCHECK4: {{(unknown argument).*-fobjc-arc-cxxlib=}}
// DefaultVisCHECK4: {{(unknown argument).*-fobjc-dispatch-method=}}
// DefaultVisCHECK4: {{(unknown argument).*-fobjc-gc}}
// DefaultVisCHECK4: {{(unknown argument).*-fobjc-gc-only}}
// DefaultVisCHECK4: {{(unknown argument).*-fobjc-runtime-has-weak}}
// DefaultVisCHECK4: {{(unknown argument).*-fobjc-subscripting-legacy-runtime}}
// DefaultVisCHECK4: {{(unknown argument).*-fopenmp-host-ir-file-path}}
// DefaultVisCHECK4: {{(unknown argument).*-fopenmp-is-target-device}}
// DefaultVisCHECK4: {{(unknown argument).*-foverride-record-layout=}}
// DefaultVisCHECK4: {{(unknown argument).*-fpadding-on-unsigned-fixed-point}}
// DefaultVisCHECK4: {{(unknown argument).*-fpass-by-value-is-noalias}}
// DefaultVisCHECK4: {{(unknown argument).*-fpatchable-function-entry-offset=}}
// DefaultVisCHECK4: {{(unknown argument).*-fppc-native-vector-element-order}}
// DefaultVisCHECK4: {{(unknown argument).*-fpreprocess-include-lines}}
// DefaultVisCHECK4: {{(unknown argument).*-fprofile-instrument=}}
// DefaultVisCHECK4: {{(unknown argument).*-fprofile-instrument-path=}}
// DefaultVisCHECK4: {{(unknown argument).*-fprofile-instrument-use-path=}}
// DefaultVisCHECK4: {{(unknown argument).*-frealloc-lhs}}
// DefaultVisCHECK4: {{(unknown argument).*-frecovery-ast}}
// DefaultVisCHECK4: {{(unknown argument).*-frecovery-ast-type}}
// DefaultVisCHECK4: {{(unknown argument).*-fretain-subst-template-type-parm-type-ast-nodes}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-8bit-counters}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-control-flow}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-indirect-calls}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-inline-8bit-counters}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-inline-bool-flag}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-no-prune}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-pc-table}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-stack-depth}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-trace-bb}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-trace-cmp}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-trace-div}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-trace-gep}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-trace-loads}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-trace-pc}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-trace-pc-guard}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-trace-stores}}
// DefaultVisCHECK4: {{(unknown argument).*-fsanitize-coverage-type=}}
// DefaultVisCHECK4: {{(unknown argument).*-fsave-main-program}}
// DefaultVisCHECK4: {{(unknown argument).*-fsigned-wchar}}
// DefaultVisCHECK4: {{(unknown argument).*-fstack-arrays}}
// DefaultVisCHECK4: {{(unknown argument).*-fsycl-is-device}}
// DefaultVisCHECK4: {{(unknown argument).*-fsycl-is-host}}
// DefaultVisCHECK4: {{(unknown argument).*-ftabstop}}
// DefaultVisCHECK4: {{(unknown argument).*-ftest-module-file-extension=}}
// DefaultVisCHECK4: {{(unknown argument).*-ftype-visibility=}}
// DefaultVisCHECK4: {{(unknown argument).*-function-alignment}}
// DefaultVisCHECK4: {{(unknown argument).*-funderscoring}}
// DefaultVisCHECK4: {{(unknown argument).*-funknown-anytype}}
// DefaultVisCHECK4: {{(unknown argument).*-funsigned}}
// DefaultVisCHECK4: {{(unknown argument).*-funwind-tables=}}
// DefaultVisCHECK4: {{(unknown argument).*-fuse-ctor-homing}}
// DefaultVisCHECK4: {{(unknown argument).*-fuse-register-sized-bitfield-access}}
// DefaultVisCHECK4: {{(unknown argument).*-fverify-debuginfo-preserve}}
// DefaultVisCHECK4: {{(unknown argument).*-fverify-debuginfo-preserve-export=}}
// DefaultVisCHECK4: {{(unknown argument).*-fwarn-stack-size=}}
// DefaultVisCHECK4: {{(unknown argument).*-fwchar-type=}}
// DefaultVisCHECK4: {{(unknown argument).*-fxor-operator}}
// DefaultVisCHECK4: {{(unknown argument).*-gsimple-template-names=}}
// DefaultVisCHECK4: {{(unknown argument).*-gsrc-hash=}}
// DefaultVisCHECK4: {{(unknown argument).*-header-include-file}}
// DefaultVisCHECK4: {{(unknown argument).*-header-include-filtering=}}
// DefaultVisCHECK4: {{(unknown argument).*-header-include-format=}}
// DefaultVisCHECK4: {{(unknown argument).*-import-call-optimization}}
// DefaultVisCHECK4: {{(unknown argument).*-init-only}}
// DefaultVisCHECK4: {{(unknown argument).*-internal-externc-isystem}}
// DefaultVisCHECK4: {{(unknown argument).*-internal-isystem}}
// DefaultVisCHECK4: {{(unknown argument).*-main-file-name}}
// DefaultVisCHECK4: {{(unknown argument).*-massembler-fatal-warnings}}
// DefaultVisCHECK4: {{(unknown argument).*-massembler-no-warn}}
// DefaultVisCHECK4: {{(unknown argument).*-mbranch-protection-pauth-lr}}
// DefaultVisCHECK4: {{(unknown argument).*-mbranch-target-enforce}}
// DefaultVisCHECK4: {{(unknown argument).*-mdebug-pass}}
// DefaultVisCHECK4: {{(unknown argument).*-menable-no-infs}}
// DefaultVisCHECK4: {{(unknown argument).*-menable-no-nans}}
// DefaultVisCHECK4: {{(unknown argument).*-mfloat-abi}}
// DefaultVisCHECK4: {{(unknown argument).*-mfpmath}}
// DefaultVisCHECK4: {{(unknown argument).*-mframe-pointer=}}
// DefaultVisCHECK4: {{(unknown argument).*-mguarded-control-stack}}
// DefaultVisCHECK4: {{(unknown argument).*-no-finalize-removal}}
// DefaultVisCHECK4: {{(unknown argument).*-no-ns-alloc-error}}
// DefaultVisCHECK4: {{(unknown argument).*-mlimit-float-precision}}
// DefaultVisCHECK4: {{(unknown argument).*-mlink-bitcode-file}}
// DefaultVisCHECK4: {{(unknown argument).*-mlink-builtin-bitcode}}
// DefaultVisCHECK4: {{(unknown argument).*-mmapsyms=implicit}}
// DefaultVisCHECK4: {{(unknown argument).*-mnoexecstack}}
// DefaultVisCHECK4: {{(unknown argument).*-mno-type-check}}
// DefaultVisCHECK4: {{(unknown argument).*-module-dir}}
// DefaultVisCHECK4: {{(unknown argument).*-module-file-deps}}
// DefaultVisCHECK4: {{(unknown argument).*-module-suffix}}
// DefaultVisCHECK4: {{(unknown argument).*-mreassociate}}
// DefaultVisCHECK4: {{(unknown argument).*-mregparm}}
// DefaultVisCHECK4: {{(unknown argument).*-mrelax-relocations=no}}
// RUN: not %clang -mrelocation-model -msave-temp-labels -msign-return-address-key= -msmall-data-limit -mtp -mvscale-max= -mvscale-min= -n -new-struct-path-tbaa -no-clear-ast-before-backend -no-code-completion-globals -no-code-completion-ns-level-decls -no-emit-llvm-uselists -no-enable-noundef-analysis -no-implicit-float -no-pointer-tbaa -no-round-trip-args -no-struct-path-tbaa -nocpp -nostdsysteminc -pch-through-hdrstop-create -pch-through-hdrstop-use -pch-through-header= -pic-is-pie -pic-level -plugin -plugin-arg- -pointer-tbaa -preamble-bytes= -print-dependency-directives-minimized-source -print-preamble -print-stats -record-command-line -relaxed-aliasing -remap-file -rewrite-macros -rewrite-test -round-trip-args -setup-static-analyzer -show-encoding --show-includes -show-inst -skip-function-bodies -source-date-epoch -spirv -split-dwarf-file -split-dwarf-output -stack-protector -stack-protector-buffer-size -stack-usage-file -static-define -stats-file= -stats-file-append -sys-header-deps -target-abi -target-cpu -target-feature -target-linker-version -target-sdk-version= -templight-dump -test-io -trim-egraph -triple -triple= -tune-cpu -vectorize-loops -vectorize-slp -verify -verify= -verify-ignore-unexpected -verify-ignore-unexpected= -vtordisp-mode=  -### -x c++ -c - < /dev/null 2>&1 | FileCheck -check-prefix=DefaultVisCHECK5 %s

// DefaultVisCHECK5: {{(unknown argument).*-mrelocation-model}}
// DefaultVisCHECK5: {{(unknown argument).*-msave-temp-labels}}
// DefaultVisCHECK5: {{(unknown argument).*-msign-return-address-key=}}
// DefaultVisCHECK5: {{(unknown argument).*-msmall-data-limit}}
// DefaultVisCHECK5: {{(unknown argument).*-mtp}}
// DefaultVisCHECK5: {{(unknown argument).*-mvscale-max=}}
// DefaultVisCHECK5: {{(unknown argument).*-mvscale-min=}}
// DefaultVisCHECK5: {{(unknown argument).*-n}}
// DefaultVisCHECK5: {{(unknown argument).*-new-struct-path-tbaa}}
// DefaultVisCHECK5: {{(unknown argument).*-no-clear-ast-before-backend}}
// DefaultVisCHECK5: {{(unknown argument).*-no-code-completion-globals}}
// DefaultVisCHECK5: {{(unknown argument).*-no-code-completion-ns-level-decls}}
// DefaultVisCHECK5: {{(unknown argument).*-no-emit-llvm-uselists}}
// DefaultVisCHECK5: {{(unknown argument).*-no-enable-noundef-analysis}}
// DefaultVisCHECK5: {{(unknown argument).*-no-implicit-float}}
// DefaultVisCHECK5: {{(unknown argument).*-no-pointer-tbaa}}
// DefaultVisCHECK5: {{(unknown argument).*-no-round-trip-args}}
// DefaultVisCHECK5: {{(unknown argument).*-no-struct-path-tbaa}}
// DefaultVisCHECK5: {{(unknown argument).*-nocpp}}
// DefaultVisCHECK5: {{(unknown argument).*-nostdsysteminc}}
// DefaultVisCHECK5: {{(unknown argument).*-pch-through-hdrstop-create}}
// DefaultVisCHECK5: {{(unknown argument).*-pch-through-hdrstop-use}}
// DefaultVisCHECK5: {{(unknown argument).*-pch-through-header=}}
// DefaultVisCHECK5: {{(unknown argument).*-pic-is-pie}}
// DefaultVisCHECK5: {{(unknown argument).*-pic-level}}
// DefaultVisCHECK5: {{(unknown argument).*-plugin}}
// DefaultVisCHECK5: {{(unknown argument).*-plugin-arg-}}
// DefaultVisCHECK5: {{(unknown argument).*-pointer-tbaa}}
// DefaultVisCHECK5: {{(unknown argument).*-preamble-bytes=}}
// DefaultVisCHECK5: {{(unknown argument).*-print-dependency-directives-minimized-source}}
// DefaultVisCHECK5: {{(unknown argument).*-print-preamble}}
// DefaultVisCHECK5: {{(unknown argument).*-print-stats}}
// DefaultVisCHECK5: {{(unknown argument).*-record-command-line}}
// DefaultVisCHECK5: {{(unknown argument).*-relaxed-aliasing}}
// DefaultVisCHECK5: {{(unknown argument).*-remap-file}}
// DefaultVisCHECK5: {{(unknown argument).*-rewrite-macros}}
// DefaultVisCHECK5: {{(unknown argument).*-rewrite-test}}
// DefaultVisCHECK5: {{(unknown argument).*-round-trip-args}}
// DefaultVisCHECK5: {{(unknown argument).*-setup-static-analyzer}}
// DefaultVisCHECK5: {{(unknown argument).*-show-encoding}}
// DefaultVisCHECK5: {{(unknown argument).*--show-includes}}
// DefaultVisCHECK5: {{(unknown argument).*-show-inst}}
// DefaultVisCHECK5: {{(unknown argument).*-skip-function-bodies}}
// DefaultVisCHECK5: {{(unknown argument).*-source-date-epoch}}
// DefaultVisCHECK5: {{(unknown argument).*-spirv}}
// DefaultVisCHECK5: {{(unknown argument).*-split-dwarf-file}}
// DefaultVisCHECK5: {{(unknown argument).*-split-dwarf-output}}
// DefaultVisCHECK5: {{(unknown argument).*-stack-protector}}
// DefaultVisCHECK5: {{(unknown argument).*-stack-protector-buffer-size}}
// DefaultVisCHECK5: {{(unknown argument).*-stack-usage-file}}
// DefaultVisCHECK5: {{(unknown argument).*-static-define}}
// DefaultVisCHECK5: {{(unknown argument).*-stats-file=}}
// DefaultVisCHECK5: {{(unknown argument).*-stats-file-append}}
// DefaultVisCHECK5: {{(unknown argument).*-sys-header-deps}}
// DefaultVisCHECK5: {{(unknown argument).*-target-abi}}
// DefaultVisCHECK5: {{(unknown argument).*-target-cpu}}
// DefaultVisCHECK5: {{(unknown argument).*-target-feature}}
// DefaultVisCHECK5: {{(unknown argument).*-target-linker-version}}
// DefaultVisCHECK5: {{(unknown argument).*-target-sdk-version=}}
// DefaultVisCHECK5: {{(unknown argument).*-templight-dump}}
// DefaultVisCHECK5: {{(unknown argument).*-test-io}}
// DefaultVisCHECK5: {{(unknown argument).*-trim-egraph}}
// DefaultVisCHECK5: {{(unknown argument).*-triple}}
// DefaultVisCHECK5: {{(unknown argument).*-triple=}}
// DefaultVisCHECK5: {{(unknown argument).*-tune-cpu}}
// DefaultVisCHECK5: {{(unknown argument).*-vectorize-loops}}
// DefaultVisCHECK5: {{(unknown argument).*-vectorize-slp}}
// DefaultVisCHECK5: {{(unknown argument).*-verify}}
// DefaultVisCHECK5: {{(unknown argument).*-verify=}}
// DefaultVisCHECK5: {{(unknown argument).*-verify-ignore-unexpected}}
// DefaultVisCHECK5: {{(unknown argument).*-verify-ignore-unexpected=}}
// DefaultVisCHECK5: {{(unknown argument).*-vtordisp-mode=}}
