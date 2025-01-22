! NOTE: This lit test was automatically generated to validate unintentionally exposed arguments to various driver flavours.
! NOTE: To make changes, see /Users/georgeasante/llvm-project/clang/utils/generate_unsupported_in_drivermode.py from which it was generated.

! RUN: not not --crash %clang --driver-mode=flang -fc1 -A - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -A- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -B - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -C - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -CC - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -EB - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -EL - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Eonly - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -F - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faapcs-bitfield-load - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -G - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -G= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -H - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -K - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -L - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -M - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -MD - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -MF - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -MG - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -MJ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -MM - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -MMD - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -MP - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -MQ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -MT - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -MV - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Mach - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Q - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Qn - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Qunused-arguments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Qy - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -T - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -V - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -X - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xanalyzer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xarch_ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xarch_device - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xarch_host - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xassembler - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xclang - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xcuda-fatbinary - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xcuda-ptxas - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xflang - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xlinker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xoffload-linker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xopenmp-target - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xopenmp-target= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xpreprocessor - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Z - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Z-Xlinker-no-demangle - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Z-reserved-lib-cckext - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Z-reserved-lib-stdc++ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Zlinker-input - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --CLASSPATH - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --CLASSPATH= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -### - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /AI - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Brepro - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Brepro- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Bt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Bt+ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /C - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /EH - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /EP - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /F - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /FA - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /FC - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /FI - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /FR - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /FS - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /FU - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fe: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fi: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fo - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fo: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fp: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /G1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /G2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GA - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GF - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GF- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GH - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GL - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GL- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GR - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GR- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GS - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GS- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GT - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GX - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GX- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /GZ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Ge - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gh - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gm- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gregcall - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gregcall4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gw - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gw- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gy - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gy- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gz - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /H - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /LD - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /LDd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /LN - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /MD - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /MDd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /MP - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /MT - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /MTd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /P - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /QIfist - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /QIntel-jcc-erratum - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qfast_transcendentals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qimprecise_fwaits - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qpar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qpar-report - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qsafe_fp_loads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qspectre - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qspectre-load - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qspectre-load-cf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qvec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qvec- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qvec-report - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /TC - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /TP - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Tc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Tp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /V - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /X - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Y- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Yc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Yd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Yl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Yu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Z7 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /ZH:MD5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /ZH:SHA1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /ZH:SHA_256 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /ZI - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /ZW - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Za - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:__STDC__ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:__cplusplus - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:alignedNew - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:alignedNew- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:auto - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:char8_t - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:char8_t- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:dllexportInlines - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:dllexportInlines- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:forScope - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:inline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:rvalueCast - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:sizedDealloc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:sizedDealloc- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:ternary - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:threadSafeInit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:threadSafeInit- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:tlsGuards - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:tlsGuards- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:trigraphs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:trigraphs- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:twoPhase - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:twoPhase- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:wchar_t - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zc:wchar_t- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Ze - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zo - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zo- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Zp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /analyze- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /arch: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /arm64EC - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /await - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /await: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /bigobj - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /c - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /cgthreads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /clang: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /clr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /constexpr: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /d1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /d1PP - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /d1reportAllClassLayout - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /d2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /d2FastFail - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /d2Zi+ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /diagnostics:caret - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /diagnostics:classic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /diagnostics:column - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /diasdkdir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /doc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /errorReport - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /execution-charset: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /experimental: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /exportHeader - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /external: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /external:I - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /external:W0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /external:W1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /external:W2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /external:W3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /external:W4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /external:env: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /favor - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /fno-sanitize-address-vcasan-lib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /fp:except - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /fp:except- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /fp:precise - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /fp:strict - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /fsanitize=address - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /fsanitize-address-use-after-return - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /guard: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /headerUnit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /headerUnit:angle - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /headerUnit:quote - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /headerName: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /homeparams - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /hotpatch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /imsvc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /kernel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /kernel- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /link - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /nologo - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /permissive - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /permissive- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /reference - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /sdl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /sdl- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /showFilenames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /showFilenames- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /showIncludes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /showIncludes:user - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /sourceDependencies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /sourceDependencies:directives - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /std: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /translateInclude - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /tune: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /u - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /utf-8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /vctoolsdir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /vctoolsversion - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /vd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /vmb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /vmg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /vmm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /vms - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /vmv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /volatile:iso - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /volatile:ms - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /w - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /wd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /winsdkdir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /winsdkversion - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /winsysroot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --all-warnings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --analyze - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --analyzer-no-default-checks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --analyzer-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --assert - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --assert= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --bootclasspath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --bootclasspath= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --classpath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --classpath= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --comments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --comments-in-macros - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --compile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --constant-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --debug - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --debug= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --dyld-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --dyld-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --encoding - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --encoding= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --entry - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --extdirs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --extdirs= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --for-linker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --for-linker= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --force-link - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --force-link= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --help-hidden - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --imacros= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --include= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --include-barrier - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --include-directory-after - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --include-directory-after= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --include-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --include-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --include-with-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --include-with-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --include-with-prefix-after - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --include-with-prefix-after= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --include-with-prefix-before - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --include-with-prefix-before= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --library-directory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --library-directory= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --mhwdiv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --mhwdiv= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --migrate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-standard-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-standard-libraries - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-undefined - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --param - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --param= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --precompile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --print-diagnostic-categories - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --print-file-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --print-missing-file-dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --print-prog-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --profile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --resource - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --resource= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --rtlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -serialize-diagnostics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --signed-char - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --stdlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --sysroot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --sysroot= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --target-help - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --trace-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --unsigned-char - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --user-dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --verbose - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --write-dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --write-user-dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -add-plugin - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -alias_list - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faligned-alloc-unavailable - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -all_load - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -allowable_client - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faltivec-src-compat= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --amdgpu-arch-tool= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cfg-add-implicit-dtors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -unoptimized-cfg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyze - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyze-function - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyze-function= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-checker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-checker= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-checker-help - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-checker-help-alpha - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-checker-help-developer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-checker-option-help - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-checker-option-help-alpha - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-checker-option-help-developer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-config - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-config-compatibility-mode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-config-compatibility-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-config-help - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-constraints - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-constraints= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-disable-all-checks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-disable-checker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-disable-checker= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-disable-retry-exhausted - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-display-progress - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-dump-egraph - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-dump-egraph= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-inline-max-stack-depth - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-inline-max-stack-depth= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-inlining-mode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-inlining-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-list-enabled-checkers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-max-loop - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-note-analysis-entry-points - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-opt-analyze-headers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-output= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-purge - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-purge= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-stats - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-viz-egraph-graphviz - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -analyzer-werror - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fnew-alignment - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faligned-new - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-aligned-new - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsched-interblock - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftemplate-depth- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftree-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-tree-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftree-slp-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-tree-slp-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fterminated-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcuda-rdc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-cuda-rdc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --hip-device-lib-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -grecord-gcc-switches - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-record-gcc-switches - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -miphoneos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -miphonesimulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmacosx-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nocudainc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-multiarch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --system-header-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-system-header-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -integrated-as - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-integrated-as - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -coverage-data-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -coverage-notes-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcuda-approx-transcendentals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-cuda-approx-transcendentals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Gs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qgather- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qscatter- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xmicrosoft-visualc-tools-root - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xmicrosoft-visualc-tools-version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xmicrosoft-windows-sdk-root - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xmicrosoft-windows-sdk-version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xmicrosoft-windows-sys-root - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Qembed_debug - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -shared-libasan - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -static-libasan - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fslp-vectorize-aggressive - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fident - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ident - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-diagnostics-color - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-slp-vectorize-aggressive - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frecord-gcc-switches - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-record-gcc-switches - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xclang= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexpensive-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-expensive-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdefer-pop - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-defer-pop - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fextended-identifiers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-extended-identifiers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xparser - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -Xcompiler - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-blacklist= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-blacklist - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fhonor-infinites - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-honor-infinites - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -findirect-virtual-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --config - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ansi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -arch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -arch_errors_fatal - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -arch_only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -arcmt-action= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -arcmt-migrate-emit-errors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -arcmt-migrate-report-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -as-secure-log-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ast-dump - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ast-dump= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ast-dump-all - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ast-dump-all= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ast-dump-decl-types - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ast-dump-filter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ast-dump-filter= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ast-dump-lookups - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ast-list - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ast-merge - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ast-print - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ast-view - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --autocomplete= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -aux-target-cpu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -aux-target-feature - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -aux-triple - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -b - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -bind_at_load - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -building-pch-with-obj - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -bundle - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -bundle_loader - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -c - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -c-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -canonical-prefixes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ccc- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ccc-arcmt-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ccc-arcmt-migrate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ccc-arcmt-modify - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ccc-gcc-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ccc-install-dir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ccc-objcmt-migrate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ccc-print-bindings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ccc-print-phases - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cfguard - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cfguard-no-checks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -chain-include - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-denorms-are-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-ext= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-fast-relaxed-math - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-finite-math-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-fp32-correctly-rounded-divide-sqrt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-kernel-arg-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-mad-enable - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-no-signed-zeros - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-no-stdinc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-opt-disable - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-single-precision-constant - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-std= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-strict-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-uniform-work-group-size - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cl-unsafe-math-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -clear-ast-before-backend - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -client_name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -code-completion-at - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -code-completion-at= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -code-completion-brief-comments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -code-completion-macros - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -code-completion-patterns - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -code-completion-with-fixits - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -combine - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -compatibility_version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -compiler-options-dump - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -complex-range= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -compress-debug-sections - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -compress-debug-sections= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --config= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --config-system-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --config-user-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -coverage - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -coverage-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cpp-precomp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --crel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --cuda-compile-host-device - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --cuda-device-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --cuda-feature= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --cuda-gpu-arch= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --cuda-host-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --cuda-include-ptx= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --cuda-noopt-device-debug - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --cuda-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --cuda-path-ignore-env - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cuid= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -current_version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -cxx-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fc++-static-destructors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fc++-static-destructors= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dA - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dD - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dE - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dI - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -d - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -d - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -darwin-target-variant - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -darwin-target-variant-sdk-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -darwin-target-variant-triple - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dead_strip - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -debug-forward-template-params - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -debug-info-macro - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -debugger-tuning= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -default-function-attr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --defsym - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dependency-dot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dependency-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -detailed-preprocessing-record - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -diagnostic-log-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -serialize-diagnostic-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -disable-O0-optnone - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -disable-free - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -disable-lifetime-markers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -disable-llvm-optzns - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -disable-llvm-passes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -disable-llvm-verifier - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -disable-objc-default-synthesize-properties - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -disable-pragma-debug-crash - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -disable-red-zone - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -discard-value-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --driver-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dsym-dir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dump-coverage-mapping - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dump-deserialized-decls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dump-raw-tokens - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dump-tokens - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dumpdir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dumpmachine - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dumpspecs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dumpversion - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dwarf-debug-flags - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dwarf-debug-producer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dwarf-explicit-import - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dwarf-ext-refs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dwarf-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Fo - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /Vd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --E - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /HV - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /hlsl-no-stdinc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --dxv-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /validator-version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dylib_file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dylinker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dylinker_install_name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dynamic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -dynamiclib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -e - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ehcontguard - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --embed-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-ast - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-cir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-codegen-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --emit-extension-symbol-graphs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-header-unit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-html - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-interface-stubs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-llvm-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-llvm-uselists - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-merged-ifs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-module - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-module-interface - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-pch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --pretty-sgf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /emit-pristine-llvm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-reduced-module-interface - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --emit-sgf-symbol-labels-for-testing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --emit-static-lib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -emit-symbol-graph - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /enable-16bit-types - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -enable-noundef-analysis - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -enable-tlsdesc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --end-no-unused-arguments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -error-on-deserialized-decl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -error-on-deserialized-decl= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -exception-model - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -exception-model= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -exported_symbols_list - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -extract-api - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --extract-api-ignores= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fPIC - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fPIE - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faapcs-bitfield-width - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faarch64-jump-table-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faccess-control - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faddress-space-map-mangling= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faddrsig - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faggressive-function-elimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -falign-commons - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -falign-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -falign-functions= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -falign-jumps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -falign-jumps= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -falign-labels - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -falign-labels= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -falign-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -falign-loops= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faligned-allocation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faligned-new= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fall-intrinsics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fallow-editor-placeholders - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fallow-pch-with-different-modules-cache-path - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fallow-pch-with-compiler-errors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fallow-pcm-with-compiler-errors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fallow-unsupported - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -faltivec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fandroid-pad-segment - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fkeep-inline-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funit-at-a-time - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fansi-escape-codes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fapinotes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fapinotes-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fapinotes-swift-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fapple-kext - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fapple-link-rtlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fapple-pragma-pack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fapplication-extension - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fapply-global-visibility-to-externs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fasm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fasm-blocks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fassociative-math - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fassume-nothrow-exception-dtor - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fassume-sane-operator-new - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fassume-unique-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fassumptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fast - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fastcp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fastf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fasync-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fasynchronous-unwind-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fauto-import - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fauto-profile= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fauto-profile-accurate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fautolink - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fautomatic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbacktrace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbasic-block-address-map - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbasic-block-sections= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbfloat16-excess-precision= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbinutils-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fblas-matmul-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fblocks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fblocks-runtime-optional - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbootclasspath= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fborland-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbounds-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-bounds-safety - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbracket-depth - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbracket-depth= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbranch-count-reg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbuild-session-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbuild-session-timestamp= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbuiltin - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbuiltin-headers-in-system-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fbuiltin-module-map - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcall-saved-x10 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcall-saved-x11 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcall-saved-x12 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcall-saved-x13 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcall-saved-x14 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcall-saved-x15 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcall-saved-x18 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcall-saved-x8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcall-saved-x9 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcaller-saves - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcaret-diagnostics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcaret-diagnostics-max-lines= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcf-protection - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcf-protection= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcf-runtime-abi= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /fcgl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fchar8_t - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcheck= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcheck-array-temporaries - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcheck-new - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fclang-abi-compat= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fclangir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fclasspath= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcoarray= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcodegen-data-generate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcodegen-data-generate= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcodegen-data-use - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcodegen-data-use= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcomment-block-commands= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcommon - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcompatibility-qualified-id-block-type-checking - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcompile-resource= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcomplete-member-pointers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcomplex-arithmetic= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fconst-strings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fconstant-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fconstant-string-class - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fconstant-string-class= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fconstexpr-backtrace-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fconstexpr-depth= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fconstexpr-steps= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fconvergent-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcoro-aligned-allocation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcoroutines - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcoverage-compilation-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcoverage-mapping - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcoverage-prefix-map= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcrash-diagnostics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcrash-diagnostics= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcrash-diagnostics-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcray-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcreate-profile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcs-profile-generate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcs-profile-generate= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fctor-dtor-return-this - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcuda-allow-variadic-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcuda-flush-denormals-to-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcuda-include-gpubinary - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcuda-is-device - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcuda-short-ptr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcx-fortran-rules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcx-limited-range - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fc++-abi= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcxx-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcxx-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fd-lines-as-code - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fd-lines-as-comments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdata-sections - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebug-compilation-dir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebug-compilation-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebug-default-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebug-info-for-profiling - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebug-macro - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebug-pass-arguments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebug-pass-structure - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebug-prefix-map= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebug-ranges-base-address - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebug-types-section - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebugger-cast-result-to-id - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebugger-objc-literal - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdebugger-support - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdeclare-opencl-builtins - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdeclspec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdefault-calling-conv= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdefault-inline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdefine-target-os-macros - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdelayed-template-parsing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdelete-null-pointer-checks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdenormal-fp-math= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdenormal-fp-math-f32= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdepfile-entry= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdeprecated-macro - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdevirtualize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdevirtualize-speculatively - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-absolute-paths - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-color= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-fixit-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-format - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-format= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-hotness-threshold= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-misexpect-tolerance= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-parseable-fixits - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-print-source-range-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-show-category - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-show-category= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-show-hotness - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-show-line-numbers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-show-location= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-show-note-include-stack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-show-option - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiagnostics-show-template-tree - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdigraphs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdirect-access-external-data - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdirectives-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdisable-block-signature-string - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdisable-module-hash - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdiscard-value-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdollar-ok - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdollars-in-identifiers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdouble-square-bracket-attributes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdriver-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdump-fortran-optimized - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdump-fortran-original - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdump-parse-tree - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdump-record-layouts - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdump-record-layouts-canonical - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdump-record-layouts-complete - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdump-record-layouts-simple - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdump-vtable-layouts - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdwarf2-cfi-asm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdwarf-directory-asm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fdwarf-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -felide-constructors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -feliminate-unused-debug-symbols - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -feliminate-unused-debug-types - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fembed-bitcode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fembed-bitcode= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fembed-bitcode-marker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -femit-all-decls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -femit-compact-unwind-non-canonical - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -femit-dwarf-unwind= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -femulated-tls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fenable-matrix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fencode-extended-block-signature - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fencoding= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ferror-limit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ferror-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fescaping-block-tail-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexcess-precision= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexec-charset= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-assignment-tracking= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-isel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-late-parse-attributes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-library - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-max-bitint-width= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-new-constant-interpreter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-omit-vtable-rtti - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-relative-c++-abi-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-sanitize-metadata= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-sanitize-metadata=atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-sanitize-metadata=covered - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-sanitize-metadata=uar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-sanitize-metadata-ignorelist= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-strict-floating-point - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fextdirs= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fextend-arguments= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexternal-blas - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexternc-nounwind - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ff2c - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffake-address-space-map - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffat-lto-objects - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffile-compilation-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffile-prefix-map= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffile-reproducible - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fimplicit-modules-use-lock - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffine-grained-bitfield-accesses - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffinite-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffinite-math-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finline-limit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-a0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-a1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-a2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-a3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-a4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-a5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-a6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-d0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-d1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-d2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-d3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-d4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-d5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-d6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-d7 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-g1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-g2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-g3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-g4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-g5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-g6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-g7 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-i0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-i1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-i2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-i3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-i4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-i5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-l0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-l1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-l2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-l3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-l4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-l5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-l6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-l7 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-o0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-o1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-o2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-o3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-o4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-o5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-point - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-r19 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-r9 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x10 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x11 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x12 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x13 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x14 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x15 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x17 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x18 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x19 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x20 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x21 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x22 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x23 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x24 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x25 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x26 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x27 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x28 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x29 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x30 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x31 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x7 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffixed-x9 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffloat16-excess-precision= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffloat-store - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffor-scope - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fforbid-guard-variables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fforce-check-cxx20-modules-input-files - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fforce-dwarf-frame - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fforce-emit-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fforce-enable-int128 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffp-eval-method= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffp-exception-behavior= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffp-model= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffpe-trap= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffree-line-length- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffreestanding - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffriend-injection - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffrontend-optimize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffuchsia-api-level= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffunction-attribute-list - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ffunction-sections - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgcse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgcse-after-reload - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgcse-las - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgcse-sm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fglobal-isel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgnu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgnu89-inline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgnu-inline-asm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgnu-keywords - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgnu-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgnuc-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgpu-allow-device-init - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgpu-approx-transcendentals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgpu-default-stream= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgpu-defer-diag - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgpu-exclude-wrong-side-overloads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgpu-flush-denormals-to-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgpu-inline-threshold= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgpu-rdc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fgpu-sanitize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fhalf-no-semantic-interposition - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fhip-dump-offload-linker-script - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fhip-emit-relocatable - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fhip-fp32-correctly-rounded-divide-sqrt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fhip-kernel-arg-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fhip-new-launch-api - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fhlsl-strict-availability - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fhonor-infinities - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fhonor-nans - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fhosted - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fignore-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -filelist - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -filetype - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fimplement-inlines - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fimplicit-module-maps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fimplicit-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fimplicit-templates - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finclude-default-header - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fincremental-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finit-character= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finit-integer= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finit-local-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finit-logical= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finit-real= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finline-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finline-functions-called-once - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finline-hint-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finline-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finline-max-stacksize= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finline-small-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finstrument-function-entry-bare - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finstrument-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finstrument-functions-after-inlining - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -finteger-4-integer-8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fintegrated-as - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fintegrated-cc1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fintegrated-objemitter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fipa-cp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fivopts - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fix-only-warnings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fix-what-you-can - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fixit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fixit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fixit-recompile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fixit-to-temporary - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fjmc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fjump-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fkeep-persistent-storage-variables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fkeep-static-consts - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fkeep-system-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -flat_namespace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -flax-vector-conversions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -flax-vector-conversions= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -flimit-debug-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -flimited-precision= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -flto-jobs= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -flto-unit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -flto-visibility-public-std - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmacro-backtrace-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmacro-prefix-map= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmath-errno - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmax-array-constructor= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmax-errors= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmax-identifier-length - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmax-stack-var-size= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmax-subrecord-length= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmax-tokens= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmax-type-align= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fcoverage-mcdc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmcdc-max-conditions= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmcdc-max-test-vectors= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmemory-profile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmemory-profile= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmemory-profile-use= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmerge-all-constants - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmerge-constants - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmerge-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmessage-length= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fminimize-whitespace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-feature - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-file-deps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-file-home-is-cwd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-format= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-header - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-header= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-implementation-of - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-map-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-map-file-home-is-cwd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-maps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-name= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-output= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodule-private - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodulemap-allow-subdirectory-search - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-cache-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-codegen - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-debuginfo - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-decluse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-disable-diagnostic-validation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-embed-all-files - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-embed-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-hash-content - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-ignore-macro= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-local-submodule-visibility - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-prune-after= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-prune-interval= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-search-all - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-skip-diagnostic-options - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-skip-header-search-paths - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-strict-context-hash - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-strict-decluse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-user-build-path - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-validate-input-files-content - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-validate-once-per-build-session - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodules-validate-system-headers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodulo-sched - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmodulo-sched-allow-regmoves - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fms-compatibility - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fms-compatibility-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fms-define-stdc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fms-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fms-hotpatch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fms-kernel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fms-memptr-rep= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fms-omit-default-lib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fms-runtime-lib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fms-tls-guards - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fms-volatile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmsc-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmudflap - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fmudflapth - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fnative-half-arguments-and-returns - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fnative-half-type - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fnested-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fnew-alignment= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fnew-infallible - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fnext-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-PIC - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-PIE - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-aapcs-bitfield-width - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-aarch64-jump-table-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-access-control - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-addrsig - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-aggressive-function-elimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-align-commons - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-align-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-align-jumps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-align-labels - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-align-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-aligned-allocation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-all-intrinsics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-allow-editor-placeholders - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-altivec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-android-pad-segment - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-keep-inline-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-unit-at-a-time - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-apinotes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-apinotes-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-apple-pragma-pack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-application-extension - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-asm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-asm-blocks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-associative-math - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-assume-nothrow-exception-dtor - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-assume-sane-operator-new - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-assume-unique-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-assumptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-async-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-asynchronous-unwind-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-auto-import - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-auto-profile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-auto-profile-accurate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-autolink - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-backtrace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-basic-block-address-map - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-bitfield-type-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-blocks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-borland-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-bounds-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-experimental-bounds-safety - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-branch-count-reg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-builtin - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-builtin- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-caller-saves - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-caret-diagnostics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-char8_t - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-check-array-temporaries - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-check-new - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-clangir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-color-diagnostics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-common - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-complete-member-pointers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-const-strings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-constant-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-convergent-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-coro-aligned-allocation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-coroutines - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-coverage-mapping - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-crash-diagnostics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-cray-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-cuda-flush-denormals-to-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-cuda-host-device-constexpr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-cuda-short-ptr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-cx-fortran-rules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-cx-limited-range - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-cxx-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-cxx-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-d-lines-as-code - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-d-lines-as-comments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-data-sections - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-debug-info-for-profiling - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-debug-macro - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-debug-ranges-base-address - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-debug-types-section - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-declspec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-default-inline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-define-target-os-macros - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-delayed-template-parsing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-delete-null-pointer-checks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-deprecated-macro - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-devirtualize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-devirtualize-speculatively - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-diagnostics-fixit-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-diagnostics-show-hotness - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-diagnostics-show-line-numbers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-diagnostics-show-note-include-stack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-diagnostics-show-option - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-diagnostics-use-presumed-location - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-digraphs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-direct-access-external-data - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-directives-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-disable-block-signature-string - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-discard-value-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-dllexport-inlines - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-dollar-ok - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-dollars-in-identifiers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-double-square-bracket-attributes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-dump-fortran-optimized - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-dump-fortran-original - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-dump-parse-tree - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-dwarf2-cfi-asm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-dwarf-directory-asm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-elide-constructors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-elide-type - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-eliminate-unused-debug-symbols - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-eliminate-unused-debug-types - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-emit-compact-unwind-non-canonical - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-emulated-tls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-escaping-block-tail-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-experimental-isel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-experimental-late-parse-attributes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-experimental-library - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-experimental-omit-vtable-rtti - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-experimental-relative-c++-abi-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-experimental-sanitize-metadata= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-external-blas - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-f2c - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-fat-lto-objects - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-file-reproducible - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-implicit-modules-use-lock - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-fine-grained-bitfield-accesses - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-finite-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-finite-math-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-inline-limit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-fixed-point - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-float-store - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-for-scope - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-force-dwarf-frame - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-force-emit-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-force-enable-int128 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-friend-injection - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-frontend-optimize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-function-attribute-list - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-function-sections - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gcse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gcse-after-reload - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gcse-las - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gcse-sm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-global-isel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gnu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gnu89-inline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gnu-inline-asm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gnu-keywords - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gpu-allow-device-init - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gpu-approx-transcendentals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gpu-defer-diag - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gpu-exclude-wrong-side-overloads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gpu-flush-denormals-to-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gpu-rdc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-gpu-sanitize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-hip-emit-relocatable - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-hip-fp32-correctly-rounded-divide-sqrt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-hip-kernel-arg-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-hip-new-launch-api - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-honor-infinities - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-honor-nans - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-implement-inlines - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-implicit-module-maps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-implicit-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-implicit-templates - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-init-local-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-inline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-inline-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-inline-functions-called-once - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-inline-small-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-integer-4-integer-8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-integrated-as - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-integrated-cc1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-integrated-objemitter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ipa-cp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ivopts - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-jmc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-jump-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-keep-persistent-storage-variables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-keep-static-consts - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-keep-system-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-knr-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-lax-vector-conversions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-limit-debug-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-lto - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-lto-unit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-math-builtin - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-math-errno - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-max-identifier-length - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-max-type-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-coverage-mcdc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-memory-profile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-merge-all-constants - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-merge-constants - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-minimize-whitespace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-module-file-deps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-module-maps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-module-private - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modulemap-allow-subdirectory-search - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modules-check-relocated - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modules-decluse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modules-error-recovery - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modules-global-index - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modules-prune-non-affecting-module-map-files - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modules-search-all - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modules-share-filemanager - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modules-skip-diagnostic-options - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modules-skip-header-search-paths - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-strict-modules-decluse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno_modules-validate-input-files-content - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modules-validate-system-headers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modules-validate-textual-header-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modulo-sched - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-modulo-sched-allow-regmoves - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ms-compatibility - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ms-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ms-tls-guards - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ms-volatile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-new-infallible - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-non-call-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-objc-arc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-objc-arc-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-objc-avoid-heapify-local-blocks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-objc-convert-messages-to-runtime-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-objc-encode-cxx-class-template-spec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-objc-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-objc-infer-related-result-type - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-objc-legacy-dispatch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-objc-nonfragile-abi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-objc-weak - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-offload-implicit-host-device-templates - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-offload-lto - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-offload-uniform-block - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-offload-via-llvm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-omit-frame-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-openmp-cuda-mode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-openmp-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-openmp-new-driver - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-openmp-optimistic-collapse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-openmp-simd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-openmp-target-jit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-openmp-target-new-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-operator-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-optimize-sibling-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-pack-derived - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-pack-struct - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-padding-on-unsigned-fixed-point - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-pascal-strings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-pch-codegen - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-pch-debuginfo - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-pch-instantiate-templates - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-pch-timestamp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno_pch-validate-input-files-content - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-peel-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-permissive - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-pic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-pie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-plt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-pointer-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-prebuilt-implicit-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-prefetch-loop-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-preserve-as-comments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-printf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-profile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-profile-arcs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-profile-correction - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-profile-generate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-profile-generate-sampling - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-profile-instr-generate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-profile-instr-use - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-profile-reusedist - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-profile-sample-accurate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-profile-sample-use - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-profile-use - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-profile-values - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-protect-parens - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-pseudo-probe-for-profiling - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ptrauth-auth-traps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ptrauth-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ptrauth-elf-got - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ptrauth-function-pointer-type-discrimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ptrauth-indirect-gotos - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ptrauth-init-fini - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ptrauth-init-fini-address-discrimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ptrauth-intrinsics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ptrauth-returns - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ptrauth-type-info-vtable-pointer-discrimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ptrauth-vtable-pointer-address-discrimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ptrauth-vtable-pointer-type-discrimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-range-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-raw-string-literals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-real-4-real-10 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-real-4-real-16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-real-4-real-8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-real-8-real-10 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-real-8-real-16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-real-8-real-4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-realloc-lhs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-record-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-recovery-ast - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-recovery-ast-type - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-recursive - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-register-global-dtors-with-atexit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-regs-graph - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-relaxed-template-template-args - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-rename-registers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-reorder-blocks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-repack-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-retain-subst-template-type-parm-type-ast-nodes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-rewrite-imports - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-rewrite-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ripa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-ropi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-rounding-math - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-rtlib-add-rpath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-rtlib-defaultlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-rtti - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-rtti-data - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-rwpi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-safe-buffer-usage-suggestions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-address-globals-dead-stripping - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-address-outline-instrumentation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-address-poison-custom-array-cookie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-address-use-after-scope - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-address-use-odr-indicator - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-cfi-canonical-jump-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-cfi-cross-dso - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-coverage= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-hwaddress-experimental-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-ignorelist - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-link-c++-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-link-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-memory-param-retval - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-memory-track-origins - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-memory-use-after-dtor - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-minimal-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-recover - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-recover= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-stable-abi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-stats - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-thread-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-thread-func-entry-exit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-thread-memory-access - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-trap - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-trap= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sanitize-undefined-trap-on-error - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-save-optimization-record - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-schedule-insns - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-schedule-insns2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-second-underscore - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-see - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-semantic-interposition - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-separate-named-sections - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-short-enums - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-short-wchar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-show-column - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-show-source-location - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sign-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-signaling-math - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-signaling-nans - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-signed-char - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-signed-wchar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-single-precision-constant - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sized-deallocation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-skip-odr-check-in-gmf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-slp-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-spec-constr-count - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-spell-checking - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-split-dwarf-inlining - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-split-lto-unit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-split-machine-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-split-stack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-stack-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-stack-clash-protection - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-stack-protector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-stack-size-section - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-standalone-debug - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-strength-reduce - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-strict-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-strict-enums - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-strict-float-cast-overflow - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-strict-overflow - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-strict-return - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-strict-vtable-pointers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-struct-path-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-sycl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-temp-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-test-coverage - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-threadsafe-statics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-tls-model - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-tracer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-trapping-math - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-tree-dce - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-tree-salias - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-tree-ter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-tree-vectorizer-verbose - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-tree-vrp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-trigraphs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-unified-lto - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-unique-basic-block-section-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-unique-internal-linkage-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-unique-section-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-unroll-all-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-unroll-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-unsafe-loop-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-unsafe-math-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-unsigned-char - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-unswitch-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-unwind-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-use-ctor-homing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-use-cxa-atexit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-use-init-array - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-use-line-directives - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-use-linker-plugin - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-validate-pch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-var-tracking - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-variable-expansion-in-unroller - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-vect-cost-model - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-verbose-asm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-verify-intermediate-code - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-virtual-function-elimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-visibility-from-dllstorageclass - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-visibility-inlines-hidden - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-visibility-inlines-hidden-static-local-var - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-wchar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-web - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-whole-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-whole-program - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-whole-program-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-working-directory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-wrapv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-xl-pragma-pack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-xray-always-emit-customevents - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-xray-always-emit-typedevents - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-xray-function-index - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-xray-ignore-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-xray-instrument - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-xray-link-deps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-xray-shared - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-zero-initialized-in-bss - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-zos-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-zvector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fnon-call-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fnoopenmp-relocatable-target - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fnoopenmp-use-tls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-abi-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-arc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-arc-cxxlib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-arc-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-atdefs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-avoid-heapify-local-blocks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-call-cxx-cdtors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-convert-messages-to-runtime-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-disable-direct-methods-for-testing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-dispatch-method= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-encode-cxx-class-template-spec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-gc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-gc-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-infer-related-result-type - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-legacy-dispatch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-link-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-new-property - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-nonfragile-abi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-nonfragile-abi-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-runtime= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-runtime-has-weak - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-sender-dependent-dispatch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-subscripting-legacy-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fobjc-weak - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -foffload-implicit-host-device-templates - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -foffload-lto - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -foffload-lto= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -foffload-uniform-block - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -foffload-via-llvm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fomit-frame-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-cuda-blocks-per-sm= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-cuda-mode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-cuda-number-of-sm= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-cuda-teams-reduction-recs-num= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-enable-irbuilder - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-new-driver - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-offload-mandatory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-optimistic-collapse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-relocatable-target - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-simd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-target-jit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-target-new-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fopenmp-use-tls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -foperator-arrow-depth= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -foperator-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -foptimization-record-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -foptimization-record-passes= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -foptimize-sibling-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -force_cpusubtype_ALL - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -force_flat_namespace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -force_load - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fforce-addr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -forder-file-instrumentation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -foutput-class-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -foverride-record-layout= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpack-derived - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpack-struct - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpack-struct= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpadding-on-unsigned-fixed-point - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fparse-all-comments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpascal-strings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpass-by-value-is-noalias - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpatchable-function-entry= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpatchable-function-entry-offset= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpcc-struct-return - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpch-codegen - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpch-debuginfo - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpch-instantiate-templates - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpch-preprocess - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpch-validate-input-files-content - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpeel-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpermissive - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fplt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fplugin= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fplugin-arg- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpointer-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprebuilt-implicit-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprebuilt-module-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprefetch-loop-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpreserve-as-comments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpreserve-vec3-type - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprintf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fproc-stat-report - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fproc-stat-report= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-arcs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-correction - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-exclude-files= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-filter-files= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-function-groups= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-generate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-generate= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-generate-cold-function-coverage - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-generate-cold-function-coverage= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-generate-sampling - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-instr-generate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-instr-generate= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-instr-use - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-instr-use= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-instrument= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-instrument-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-instrument-use-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-list= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-remapping-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-reusedist - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-sample-accurate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-sample-use= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-selected-function-group= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-update= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-use - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-use= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprofile-values - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fprotect-parens - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fpseudo-probe-for-profiling - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fptrauth-auth-traps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fptrauth-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fptrauth-elf-got - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fptrauth-function-pointer-type-discrimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fptrauth-indirect-gotos - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fptrauth-init-fini - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fptrauth-init-fini-address-discrimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fptrauth-intrinsics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fptrauth-returns - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fptrauth-type-info-vtable-pointer-discrimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fptrauth-vtable-pointer-address-discrimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fptrauth-vtable-pointer-type-discrimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -framework - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frandom-seed= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frandomize-layout-seed= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frandomize-layout-seed-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frange-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fraw-string-literals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -freal-4-real-10 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -freal-4-real-16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -freal-4-real-8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -freal-8-real-10 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -freal-8-real-16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -freal-8-real-4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frealloc-lhs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frecord-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frecord-marker= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frecovery-ast - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frecovery-ast-type - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frecursive - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -freg-struct-return - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fregister-global-dtors-with-atexit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fregs-graph - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frelaxed-template-template-args - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frename-registers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -freorder-blocks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frepack-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fretain-comments-from-system-headers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fretain-subst-template-type-parm-type-ast-nodes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frewrite-imports - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frewrite-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fripa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fropi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frounding-math - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frtlib-add-rpath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frtlib-defaultlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frtti - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frtti-data - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -frwpi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsafe-buffer-usage-suggestions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsample-profile-use-profi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-address-field-padding= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-address-globals-dead-stripping - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-address-outline-instrumentation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-address-poison-custom-array-cookie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-address-use-after-scope - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-address-use-odr-indicator - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-cfi-canonical-jump-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-cfi-cross-dso - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-cfi-icall-generalize-pointers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-cfi-icall-experimental-normalize-integers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-8bit-counters - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-allowlist= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-control-flow - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-ignorelist= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-indirect-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-inline-8bit-counters - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-inline-bool-flag - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-no-prune - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-pc-table - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-stack-depth - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-bb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-cmp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-div - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-gep - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-loads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-pc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-pc-guard - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-stores - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-coverage-type= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-hwaddress-abi= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-hwaddress-experimental-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-ignorelist= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-link-c++-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-link-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-memory-param-retval - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-memory-track-origins - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-memory-track-origins= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-memory-use-after-dtor - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-memtag-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-minimal-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-recover - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-recover= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-stable-abi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-stats - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-system-ignorelist= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-thread-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-thread-func-entry-exit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-thread-memory-access - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-trap - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-trap= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-undefined-ignore-overflow-pattern= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-undefined-strip-path-components= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-undefined-trap-on-error - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsave-optimization-record - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsave-optimization-record= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fschedule-insns - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fschedule-insns2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsecond-underscore - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsee - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fseh-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsemantic-interposition - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fseparate-named-sections - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fshort-enums - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fshort-wchar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fshow-column - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fshow-overloads= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fshow-skipped-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fshow-source-location - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsign-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsignaling-math - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsignaling-nans - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsigned-bitfields - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsigned-char - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsigned-wchar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsingle-precision-constant - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsized-deallocation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsjlj-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fskip-odr-check-in-gmf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fslp-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fspec-constr-count - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fspell-checking - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fspell-checking-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsplit-dwarf-inlining - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsplit-lto-unit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsplit-machine-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsplit-stack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fspv-target-env= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstack-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstack-clash-protection - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstack-protector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstack-protector-all - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstack-protector-strong - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstack-size-section - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstack-usage - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstandalone-debug - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstrength-reduce - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstrict-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstrict-enums - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstrict-flex-arrays= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstrict-float-cast-overflow - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstrict-overflow - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstrict-return - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstrict-vtable-pointers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fstruct-path-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fswift-async-fp= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsycl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsycl-is-device - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsycl-is-host - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsymbol-partition= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsystem-module - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftabstop - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftabstop= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftemplate-backtrace-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftemplate-depth= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftest-coverage - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftest-module-file-extension= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fthin-link-bitcode= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fthinlto-index= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fthreadsafe-statics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftime-report - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftime-report= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftime-trace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftime-trace= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftime-trace-granularity= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftime-trace-verbose - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftls-model - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftls-model= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftracer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftrap-function= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftrapping-math - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftrapv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftrapv-handler - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftrapv-handler= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftree-dce - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftree-salias - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftree-ter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftree-vectorizer-verbose - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftree-vrp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftrigraphs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftrivial-auto-var-init= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftrivial-auto-var-init-max-size= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftrivial-auto-var-init-stop-after= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ftype-visibility= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -function-alignment - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funified-lto - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funique-basic-block-section-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funique-internal-linkage-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funique-section-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funknown-anytype - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funroll-all-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funroll-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funsafe-loop-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funsafe-math-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funsigned-bitfields - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funsigned-char - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funswitch-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funwind-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -funwind-tables= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fuse-ctor-homing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fuse-cuid= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fuse-cxa-atexit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fuse-init-array - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fuse-ld= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fuse-line-directives - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fuse-linker-plugin - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fuse-register-sized-bitfield-access - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvalidate-ast-input-files-content - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvariable-expansion-in-unroller - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvect-cost-model - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvectorize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fverbose-asm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fverify-debuginfo-preserve - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fverify-debuginfo-preserve-export= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fverify-intermediate-code - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvirtual-function-elimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvisibility= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvisibility-dllexport= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvisibility-externs-dllimport= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvisibility-externs-nodllstorageclass= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvisibility-from-dllstorageclass - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvisibility-global-new-delete= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvisibility-global-new-delete-hidden - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvisibility-inlines-hidden - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvisibility-inlines-hidden-static-local-var - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvisibility-ms-compat - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fvisibility-nodllstorageclass= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fwarn-stack-size= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fwasm-exceptions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fwchar-type= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fweb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fwhole-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fwhole-program - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fwhole-program-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fwritable-strings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxl-pragma-pack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-always-emit-customevents - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-always-emit-typedevents - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-always-instrument= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-attr-list= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-function-groups= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-function-index - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-ignore-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-instruction-threshold= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-instrument - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-instrumentation-bundle= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-link-deps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-modes= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-never-instrument= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-selected-function-group= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fxray-shared - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fzero-call-used-regs= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fzero-initialized-in-bss - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fzos-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fzvector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -g0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -g1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -g2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -g3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -g - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --gcc-install-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --gcc-toolchain= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --gcc-triple= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gcodeview - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gcodeview-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gcodeview-ghash - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gcoff - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gcolumn-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gdbx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gdwarf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gdwarf32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gdwarf64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gdwarf-2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gdwarf-3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gdwarf-4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gdwarf-5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gdwarf-aranges - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gembed-source - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gen-cdb-fragment-path - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gen-reproducer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gen-reproducer= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gfull - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ggdb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ggdb0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ggdb1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ggdb2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ggdb3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ggnu-pubnames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ginline-line-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gline-directives-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gline-tables-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -glldb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gmlt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gmodules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-codeview-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-codeview-ghash - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-column-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-embed-source - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-gnu-pubnames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-inline-line-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-omit-unreferenced-methods - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-pubnames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-record-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-simple-template-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-split-dwarf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-strict-dwarf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gno-template-alias - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gomit-unreferenced-methods - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --gpu-bundle-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --gpu-instrument-lib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --gpu-max-threads-per-block= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --gpu-use-aux-triple-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gpubnames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -grecord-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gsce - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gsimple-template-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gsimple-template-names= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gsplit-dwarf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gsplit-dwarf= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gsrc-hash= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gstabs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gstrict-dwarf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gtemplate-alias - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gtoggle - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gused - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gvms - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gxcoff - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gz - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -gz= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -header-include-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -header-include-filtering= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -header-include-format= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -headerpad_max_install_names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --hip-device-lib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --hip-link - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --hip-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --hip-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --hipspv-pass-plugin= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --hipstdpar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --hipstdpar-interpose-alloc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --hipstdpar-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --hipstdpar-prim-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --hipstdpar-thrust-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -hlsl-entry - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -iapinotes-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ibuiltininc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -idirafter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -iframework - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -iframeworkwithsysroot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -imacros - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -image_base - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -imultilib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -include - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -include-pch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -init - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -inline-asm= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -install_name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -interface-stub-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -internal-externc-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -internal-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -iprefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -iquote - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -isysroot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -isystem - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -isystem-after - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -ivfsoverlay - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -iwithprefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -iwithprefixbefore - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -iwithsysroot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -keep_private_externs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -l - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -lazy_framework - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -lazy_library - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --ld-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --libomptarget-amdgcn-bc-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --libomptarget-amdgpu-bc-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --libomptarget-nvptx-bc-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --linker-option= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -llvm-verify-each - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m3dnow - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m3dnowa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m68000 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m68010 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m68020 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m68030 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m68040 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m68060 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m68881 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -m80387 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mseses - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mabicalls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mabs= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -madx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -maes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -main-file-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -maix32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -maix64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -maix-shared-lib-tls-model-opt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -maix-small-local-dynamic-tls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -maix-small-local-exec-tls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -maix-struct-return - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -malign-branch= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -malign-branch-boundary= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -malign-double - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -malign-functions= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -malign-jumps= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -malign-loops= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -maltivec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mamdgpu-ieee - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mamdgpu-precise-memory-op - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mamx-avx512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mamx-bf16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mamx-complex - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mamx-fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mamx-fp8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mamx-int8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mamx-movrs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mamx-tf32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mamx-tile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mamx-transpose - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mannotate-tablejump - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mappletvos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mappletvsimulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mapx-features= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mapx-inline-asm-use-gpr32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mapxf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -march= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -marm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -marm64x - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -masm= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -massembler-fatal-warnings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -massembler-no-warn - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -matomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx10.1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx10.1-256 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx10.1-512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx10.2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx10.2-256 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx10.2-512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512bf16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512bitalg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512bw - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512cd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512dq - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512f - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512ifma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512vbmi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512vbmi2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512vl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512vnni - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512vp2intersect - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavx512vpopcntdq - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavxifma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavxneconvert - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavxvnni - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavxvnniint16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mavxvnniint8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mbackchain - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mbig-endian - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mbmi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mbmi2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mbranch-likely - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mbranch-protection= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mbranch-protection-pauth-lr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mbranch-target-enforce - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mbranches-within-32B-boundaries - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mbulk-memory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mbulk-memory-opt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcabac - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcall-indirect-overlong - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcf-branch-label-scheme= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcheck-zero-division - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcldemote - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mclflushopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mclwb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mclzero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcmpb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcmpccxadd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcmse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcompact-branches= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mconsole - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mconstant-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mconstructor-aliases - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcpu= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcrbits - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcrc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcrc32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcumode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcx16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mdaz-ftz - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mdebug-pass - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mdefault-build-attributes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mdefault-visibility-export-mapping= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mdirect-move - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mdiv32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mdll - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mdouble= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mdouble-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mdsp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mdspr2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mdynamic-no-pic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -meabi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mefpu2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -membedded-data - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -menable-experimental-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -menqcmd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mevex512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mexception-handling - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mexec-model= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mexecute-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mextended-const - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mextern-sdata - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mf16c - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfancy-math-387 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfentry - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfix4300 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfix-and-continue - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfix-cmse-cve-2021-35465 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfix-cortex-a53-835769 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfix-cortex-a57-aes-1742098 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfix-cortex-a72-aes-1655431 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfix-gr712rc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfix-ut700 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfloat128 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfloat-abi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfloat-abi= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfma4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfp16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfp32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfp64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfpmath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfpmath= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfprnd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfpu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfpu= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfpxx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mframe-chain= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfrecipe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfsgsbase - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfsmuld - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfunction-return= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mfxsr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mgeneral-regs-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mgfni - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mginv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mglibc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mglobal-merge - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mgpopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mguard= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mguarded-control-stack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mhard-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mhard-quad-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mharden-sls= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mhvx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mhvx= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mhvx-ieee-fp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mhvx-length= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mhvx-qfloat - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mhreset - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mhtm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mhwdiv= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mhwmult= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -miamcu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mieee-fp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mieee-rnd-near - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mignore-xcoff-visibility - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -migrate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-finalize-removal - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-ns-alloc-error - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mimplicit-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mimplicit-it= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mincremental-linker-compatible - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mindirect-branch-cs-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mindirect-jump= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -minline-all-stringops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -minvariant-function-descriptors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -minvpcid - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mios-simulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mios-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips32r2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips32r3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips32r5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips32r6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips64r2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips64r3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips64r5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mips64r6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -misel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mkernel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mkl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlam-bh - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlamcas - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlasx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mld-seq-sa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mldc1-sdc1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlimit-float-precision - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlink-bitcode-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlink-builtin-bitcode-postopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlinker-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlittle-endian - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlocal-sdata - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlong-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlong-double-128 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlong-double-64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlong-double-80 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlongcall - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlr-for-calls-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlsx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlvi-cfi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlvi-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlwp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mlzcnt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmacos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmadd4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmapsyms=implicit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmark-bti-property - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmcu= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmemops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmfcrf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmfocrf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmicromips - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmmx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmovbe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmovdir64b - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmovdiri - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmovrs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmpx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mms-bitfields - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmsa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmultimemory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmultivalue - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmutable-globals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mmwaitx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mnan= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-3dnow - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-3dnowa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-80387 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-abicalls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-adx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-aes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-altivec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-amdgpu-ieee - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-amdgpu-precise-memory-op - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-amx-avx512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-amx-bf16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-amx-complex - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-amx-fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-amx-fp8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-amx-int8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-amx-movrs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-amx-tf32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-amx-tile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-amx-transpose - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-annotate-tablejump - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-apx-features= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-apxf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx10.1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx10.1-256 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx10.1-512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx10.2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx10.2-256 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx10.2-512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512bf16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512bitalg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512bw - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512cd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512dq - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512f - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512ifma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512vbmi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512vbmi2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512vl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512vnni - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512vp2intersect - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avx512vpopcntdq - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avxifma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avxneconvert - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avxvnni - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avxvnniint16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-avxvnniint8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-backchain - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-bmi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-bmi2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-branch-likely - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-bti-at-return-twice - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-bulk-memory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-bulk-memory-opt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-call-indirect-overlong - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-check-zero-division - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-cldemote - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-clflushopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-clwb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-clzero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-cmpb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-cmpccxadd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-constant-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-constructor-aliases - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-crbits - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-crc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-crc32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-cumode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-cx16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-daz-ftz - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-default-build-attributes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-div32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-dsp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-dspr2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-embedded-data - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-enqcmd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-evex512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-exception-handling - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mnoexecstack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-execute-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-extended-const - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-extern-sdata - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-f16c - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fix-cmse-cve-2021-35465 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fix-cortex-a53-835769 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fix-cortex-a57-aes-1742098 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fix-cortex-a72-aes-1655431 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-float128 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fma4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fmv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fp-ret-in-387 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fprnd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fpu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-frecipe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fsgsbase - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fsmuld - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-fxsr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-gather - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-gfni - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-ginv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-global-merge - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-gpopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-hvx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-hvx-ieee-fp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-hvx-qfloat - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-hreset - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-htm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-iamcu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-implicit-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-incremental-linker-compatible - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-inline-all-stringops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-invariant-function-descriptors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-invpcid - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-isel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-kl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-lam-bh - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-lamcas - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-lasx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-ld-seq-sa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-ldc1-sdc1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-link-builtin-bitcode-postopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-local-sdata - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-long-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-longcall - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-lsx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-lvi-cfi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-lvi-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-lwp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-lzcnt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-madd4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-memops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-mfcrf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-mfocrf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-micromips - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-mips16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-mma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-mmx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-movbe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-movdir64b - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-movdiri - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-movrs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-movt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-mpx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-ms-bitfields - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-msa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-mt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-multimemory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-multivalue - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-mutable-globals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-mwaitx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-neg-immediates - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-nontrapping-fptoint - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-nvj - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-nvs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-odd-spreg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-omit-leaf-frame-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-outline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-outline-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-packed-stack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-packets - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-pascal-strings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-pclmul - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-pconfig - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-pcrel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-pic-data-is-text-relative - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-pku - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-popc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-popcnt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-popcntd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-power10-vector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-power8-vector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-power9-vector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-prefetchi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-prefixed - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-prfchw - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-ptwrite - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-pure-code - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-raoint - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-rdpid - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-rdpru - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-rdrnd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-rdseed - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-red-zone - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-reference-types - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-regnames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-relax - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-relax-all - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-relax-pic-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-relaxed-simd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-restrict-it - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-retpoline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-retpoline-external-thunk - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-rtd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-rtm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sahf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-save-restore - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-scalar-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-scatter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-serialize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-seses - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sgx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sha - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sha512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-shstk - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sign-ext - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-simd128 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-skip-rax-setup - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sm3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sm4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-soft-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-spe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-speculative-load-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sse2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sse3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sse4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sse4.1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sse4.2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-sse4a - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-ssse3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-stack-arg-probe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-stackrealign - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-tail-call - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-tbm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-tgsplit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-thumb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-tls-direct-seg-refs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-tocdata - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-tocdata= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-tsxldtrk - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-type-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-uintr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-unaligned-access - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-unaligned-symbols - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-unsafe-fp-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-usermsr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-v8plus - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-vaes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-vector-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-vevpu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-virt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-vis - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-vis2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-vis3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-vpclmulqdq - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-vsx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-vx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-vzeroupper - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-waitpkg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-warn-nonportable-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-wavefrontsize64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-wbnoinvd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-wide-arithmetic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-widekl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-x87 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-xcoff-roptr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-xgot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-xop - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-xsave - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-xsavec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-xsaveopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-xsaves - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-zvector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mnocrc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-direct-move - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mnontrapping-fptoint - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mnop-mcount - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-paired-vector-memops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mno-crypto - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mnvj - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mnvs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -modd-spreg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -module-dependency-dir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -module-file-deps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -module-file-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-modules-reduced-bmi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -momit-leaf-frame-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -moslib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -moutline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -moutline-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpacked-stack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpackets - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpad-max-prefix-size= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpaired-vector-memops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpascal-strings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpclmul - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpconfig - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpcrel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpic-data-is-text-relative - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpku - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpopc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpopcnt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpopcntd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpower10-vector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mcrypto - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpower8-vector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpower9-vector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mprefer-vector-width= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mprefetchi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mprefixed - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mprfchw - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mprintf-kind= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mprivileged - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mptwrite - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mpure-code - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mqdsp6-compat - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mraoint - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrdpid - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrdpru - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrdrnd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrdseed - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrecip - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrecip= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrecord-mcount - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mred-zone - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mreference-types - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mregnames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mregparm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mregparm= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrelax - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrelax-all - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrelax-pic-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrelax-relocations=no - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrelaxed-simd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrestrict-it - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mretpoline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mretpoline-external-thunk - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrop-protect - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrtd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrtm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mrvv-vector-bits= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msahf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msave-reg-params - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msave-restore - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msave-temp-labels - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mscalar-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msecure-plt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mserialize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msgx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msha - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msha512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mshstk - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msign-ext - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msign-return-address= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msign-return-address-key= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msim - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msimd128 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msimd= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msingle-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mskip-rax-setup - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msm3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msm4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msmall-data-limit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msmall-data-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msmall-data-threshold= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msoft-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msoft-quad-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mspe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mspeculative-load-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msse2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msse2avx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msse3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msse4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msse4.1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msse4.2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msse4a - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mssse3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mstack-alignment= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mstack-arg-probe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mstack-probe-size= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mstack-protector-guard= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mstack-protector-guard-offset= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mstack-protector-guard-reg= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mstack-protector-guard-symbol= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mstackrealign - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mstrict-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msve-vector-bits= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -msvr4-struct-return - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mt-migrate-directory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtail-call - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtargetos= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtbm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtgsplit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mthread-model - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mthreads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mthumb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtls-dialect= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtls-direct-seg-refs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtls-size= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtocdata - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtocdata= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtp= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtsxldtrk - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtune= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtvos-simulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mtvos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -muclibc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -muintr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -multi_module - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -multi-lib-config= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -multiply_defined - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -multiply_defined_unused - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -munaligned-access - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -munaligned-symbols - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -municode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -munsafe-fp-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -musermsr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv55 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv60 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv62 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv65 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv66 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv67 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv67t - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv68 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv69 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv71 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv71t - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv73 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mv8plus - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mvaes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mvector-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mvevpu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mvirt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mvis - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mvis2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mvis3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mvpclmulqdq - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mvsx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mvx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mvzeroupper - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mwaitpkg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mwarn-nonportable-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mwatchos-simulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mwatchos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mwatchsimulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mwavefrontsize64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mwbnoinvd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mwide-arithmetic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mwidekl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mwindows - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mx32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mx87 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mxcoff-build-id= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mxcoff-roptr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mxgot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mxop - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mxsave - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mxsavec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mxsaveopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mxsaves - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mzos-hlq-clang= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mzos-hlq-csslib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mzos-hlq-le= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mzos-sys-include= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -mzvector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -n - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -new-struct-path-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no_dead_strip_inits_and_terms - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-canonical-prefixes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-clear-ast-before-backend - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-code-completion-globals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-code-completion-ns-level-decls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-cpp-precomp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-cuda-gpu-arch= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-cuda-include-ptx= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-cuda-noopt-device-debug - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-cuda-version-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fno-c++-static-destructors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-default-config - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-emit-llvm-uselists - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-enable-noundef-analysis - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-gpu-bundle-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-hip-rt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-implicit-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-integrated-cpp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-offload-add-rpath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-offload-arch= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-offload-compress - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-offload-new-driver - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-pedantic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-pie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-pointer-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-round-trip-args - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -no-struct-path-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-system-header-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --no-wasm-opt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nobuiltininc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nodefaultlibs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nodriverkitlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nofixprebinding - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nogpuinc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nohipwrapperinc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nolibc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nomultidefs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nopie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -noprebind - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -noprofilelib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -noseglinkedit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nostartfiles - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nostdinc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nostdinc++ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nostdlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nostdlibinc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nostdlib++ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -nostdsysteminc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --nvptx-arch-tool= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-openacc-macro-override - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fexperimental-openacc-macro-override= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -p - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -pagezero_size - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -pass-exit-codes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -pch-through-hdrstop-create - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -pch-through-hdrstop-use - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -pch-through-header= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -pedantic-errors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -pg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -pie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -pipe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -plugin-arg- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -pointer-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -preamble-bytes= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -prebind - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -prebind_all_twolevel_modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -preload - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-dependency-directives-minimized-source - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-diagnostic-options - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-effective-triple - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-enabled-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-file-name= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-ivar-layout - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-libgcc-file-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-multi-directory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-multi-flags-experimental - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-multi-lib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-multi-os-directory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-preamble - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-prog-name= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-resource-dir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-rocm-search-dirs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-runtime-dir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-search-dirs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-stats - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-library-module-manifest-path - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-supported-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-target-triple - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -print-targets - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -private_bundle - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --product-name= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -pthreads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --ptxas-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -r - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -rdynamic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -read_only_relocs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -reexport_framework - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -reexport-l - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -reexport_library - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -regcall4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -relaxed-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -relocatable-pch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -remap - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -remap-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -rewrite-legacy-objc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -rewrite-macros - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -rewrite-objc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -rewrite-test - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --rocm-device-lib-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --rocm-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -round-trip-args - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -rpath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --rsp-quoting= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -rtlib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -s - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-address-destructor= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -fsanitize-address-use-after-return= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -save-stats - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -save-stats= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -sectalign - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -sectcreate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -sectobjectsymbols - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -sectorder - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -seg1addr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -seg_addr_table - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -seg_addr_table_filename - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -segaddr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -segcreate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -seglinkedit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -segprot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -segs_read_ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -segs_read_only_addr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -segs_read_write_addr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -setup-static-analyzer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -shared - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -shared-libgcc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -shared-libsan - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -show-encoding - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --show-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -show-inst - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -single_module - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -skip-function-bodies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -source-date-epoch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -specs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -specs= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /spirv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -split-dwarf-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -split-dwarf-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -stack-protector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -stack-protector-buffer-size - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -stack-usage-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --start-no-unused-arguments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -startfiles - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -static - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -static-define - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -static-libgcc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -static-libgfortran - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -static-libsan - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -static-libstdc++ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -static-openmp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -static-pie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -stats-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -stats-file-append - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -std-default= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -stdlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -stdlib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -stdlib++-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -sub_library - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -sub_umbrella - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --sycl-link - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -sycl-std= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --symbol-graph-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -sys-header-deps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --system-header-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -t - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --target= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -target-abi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -target - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -target-linker-version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 /T - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -target-sdk-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -templight-dump - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -time - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -traditional - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -traditional-cpp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -trigraphs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -trim-egraph - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -twolevel_namespace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -twolevel_namespace_hints - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -u - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -umbrella - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -undef - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -undefined - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -unexported_symbols_list - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -unwindlib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -v - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -vectorize-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -vectorize-slp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -verify - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -verify= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --verify-debug-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -verify-ignore-unexpected - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -verify-ignore-unexpected= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -verify-pch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -vfsoverlay - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -via-file-asm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -vtordisp-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --warning-suppression-mappings= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 --wasm-opt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -weak_framework - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -weak_library - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -weak_reference_mismatches - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -weak-l - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -whatsloaded - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -why_load - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -whyload - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -working-directory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -working-directory= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -y - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not not --crash %clang --driver-mode=flang -fc1 -z - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
=======
! RUN: not %clang --driver-mode=flang -fc1 -A - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -A- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -B - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -C - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -CC - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -EB - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -EL - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Eonly - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -F - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -faapcs-bitfield-load - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -G - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -G= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -H - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -K - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -L - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -M - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -MD - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -MF - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -MG - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -MJ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -MM - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -MMD - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -MP - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -MQ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -MT - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -MV - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Mach - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Q - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Qn - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Qunused-arguments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Qy - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -T - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -V - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -X - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xanalyzer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xarch_ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xarch_device - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xarch_host - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xassembler - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xclang - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xcuda-fatbinary - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xcuda-ptxas - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xflang - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xlinker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xoffload-linker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xopenmp-target - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xopenmp-target= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xpreprocessor - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Z - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Z-Xlinker-no-demangle - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Z-reserved-lib-cckext - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Z-reserved-lib-stdc++ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Zlinker-input - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --CLASSPATH - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --CLASSPATH= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -### - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /AI - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Brepro - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Brepro- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Bt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Bt+ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /C - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /E - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /EH - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /EP - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /F - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /FA - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /FC - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /FI - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /FR - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /FS - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /FU - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fe: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fi: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fo - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fo: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fp: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /G1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /G2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GA - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GF - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GF- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GH - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GL - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GL- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GR - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GR- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GS - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GS- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GT - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GX - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GX- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /GZ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Ge - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gh - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gm- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gregcall - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gregcall4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gw - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gw- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gy - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gy- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gz - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /H - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /HELP - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /LD - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /LDd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /LN - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /MD - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /MDd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /MP - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /MT - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /MTd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /P - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /QIfist - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /QIntel-jcc-erratum - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /? - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qfast_transcendentals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qimprecise_fwaits - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qpar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qpar-report - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qsafe_fp_loads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qspectre - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qspectre-load - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qspectre-load-cf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qvec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qvec- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qvec-report - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /TC - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /TP - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Tc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Tp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /V - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /X - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Y- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Yc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Yd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Yl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Yu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Z7 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /ZH:MD5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /ZH:SHA1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /ZH:SHA_256 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /ZI - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /ZW - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Za - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:__STDC__ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:__cplusplus - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:alignedNew - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:alignedNew- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:auto - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:char8_t - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:char8_t- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:dllexportInlines - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:dllexportInlines- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:forScope - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:inline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:rvalueCast - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:sizedDealloc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:sizedDealloc- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:strictStrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:ternary - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:threadSafeInit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:threadSafeInit- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:tlsGuards - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:tlsGuards- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:trigraphs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:trigraphs- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:twoPhase - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:twoPhase- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:wchar_t - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zc:wchar_t- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Ze - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zo - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zo- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Zs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /analyze- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /arch: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /arm64EC - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /await - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /await: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /bigobj - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /c - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /cgthreads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /clang: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /clr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /constexpr: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /d1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /d1PP - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /d1reportAllClassLayout - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /d2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /d2FastFail - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /d2Zi+ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /diagnostics:caret - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /diagnostics:classic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /diagnostics:column - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /diasdkdir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /doc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /errorReport - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /execution-charset: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /experimental: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /exportHeader - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /external: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /external:I - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /external:W0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /external:W1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /external:W2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /external:W3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /external:W4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /external:env: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /favor - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /fno-sanitize-address-vcasan-lib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /fp:contract - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /fp:except - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /fp:except- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /fp:fast - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /fp:precise - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /fp:strict - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /fsanitize=address - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /fsanitize-address-use-after-return - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /guard: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /headerUnit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /headerUnit:angle - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /headerUnit:quote - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /headerName: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /help - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /homeparams - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /hotpatch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /imsvc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /kernel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /kernel- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /link - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /nologo - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /permissive - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /permissive- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /reference - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /sdl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /sdl- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /showFilenames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /showFilenames- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /showIncludes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /showIncludes:user - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /sourceDependencies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /sourceDependencies:directives - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /source-charset: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /std: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /translateInclude - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /tune: - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /u - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /utf-8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /validate-charset - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /validate-charset- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /vctoolsdir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /vctoolsversion - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /vd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /vmb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /vmg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /vmm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /vms - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /vmv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /volatile:iso - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /volatile:ms - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /w - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /w - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /wd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /winsdkdir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /winsdkversion - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /winsysroot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --all-warnings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --analyze - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --analyzer-no-default-checks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --analyzer-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --assemble - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --assert - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --assert= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --bootclasspath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --bootclasspath= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --classpath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --classpath= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --comments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --comments-in-macros - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --compile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --constant-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --debug - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --debug= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --define-macro - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --define-macro= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --dyld-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --dyld-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --encoding - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --encoding= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --entry - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --extdirs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --extdirs= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --extra-warnings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --for-linker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --for-linker= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --force-link - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --force-link= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --help-hidden - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --imacros= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-barrier - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-directory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-directory= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-directory-after - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-directory-after= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-with-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-with-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-with-prefix-after - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-with-prefix-after= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-with-prefix-before - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --include-with-prefix-before= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --language - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --language= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --library-directory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --library-directory= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --mhwdiv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --mhwdiv= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --migrate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-line-commands - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-standard-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-standard-libraries - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-undefined - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --param - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --param= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --precompile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --preprocess - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --print-diagnostic-categories - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --print-file-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --print-missing-file-dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --print-prog-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --profile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --resource - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --resource= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --rtlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -serialize-diagnostics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --signed-char - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --std - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --stdlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --sysroot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --sysroot= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --target-help - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --trace-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --undefine-macro - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --undefine-macro= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --unsigned-char - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --user-dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --verbose - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --warn- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --warn-= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --write-dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --write-user-dependencies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -add-plugin - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -alias_list - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -faligned-alloc-unavailable - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -all_load - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -allowable_client - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --amdgpu-arch-tool= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cfg-add-implicit-dtors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -unoptimized-cfg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyze - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyze-function - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyze-function= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-checker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-checker= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-checker-help - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-checker-help-alpha - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-checker-help-developer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-checker-option-help - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-checker-option-help-alpha - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-checker-option-help-developer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-config - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-config-compatibility-mode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-config-compatibility-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-config-help - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-constraints - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-constraints= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-disable-all-checks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-disable-checker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-disable-checker= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-disable-retry-exhausted - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-display-progress - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-dump-egraph - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-dump-egraph= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-inline-max-stack-depth - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-inline-max-stack-depth= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-inlining-mode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-inlining-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-list-enabled-checkers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-max-loop - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-note-analysis-entry-points - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-opt-analyze-headers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-output= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-purge - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-purge= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-stats - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-viz-egraph-graphviz - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -analyzer-werror - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fnew-alignment - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -faligned-new - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-aligned-new - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ftree-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-tree-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ftree-slp-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-tree-slp-vectorize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fterminated-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcuda-rdc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-cuda-rdc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --hip-device-lib-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -grecord-gcc-switches - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-record-gcc-switches - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -miphoneos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -miphonesimulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mllvm= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmacosx-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nocudainc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nocudalib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-multiarch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --system-header-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-system-header-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -integrated-as - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-integrated-as - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -coverage-data-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -coverage-notes-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-cuda-approx-transcendentals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Gs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qgather- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qscatter- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xmicrosoft-visualc-tools-root - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xmicrosoft-visualc-tools-version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xmicrosoft-windows-sdk-root - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xmicrosoft-windows-sdk-version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xmicrosoft-windows-sys-root - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Qembed_debug - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -shared-libasan - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -static-libasan - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fslp-vectorize-aggressive - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-slp-vectorize-aggressive - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -frecord-gcc-switches - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-record-gcc-switches - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xclang= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fexpensive-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-expensive-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdefer-pop - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-defer-pop - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xparser - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -Xcompiler - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-blacklist= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-blacklist - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fhonor-infinites - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-honor-infinites - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -findirect-virtual-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --config - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ansi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -arch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -arch_errors_fatal - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -arch_only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -arcmt-action= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -arcmt-migrate-emit-errors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -arcmt-migrate-report-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -as-secure-log-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ast-dump - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ast-dump= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ast-dump-all - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ast-dump-all= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ast-dump-decl-types - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ast-dump-filter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ast-dump-filter= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ast-dump-lookups - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ast-list - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ast-merge - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ast-print - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ast-view - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --autocomplete= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -aux-target-cpu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -aux-target-feature - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -aux-triple - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -b - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -bind_at_load - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -building-pch-with-obj - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -bundle - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -bundle_loader - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -c - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -c-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -canonical-prefixes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ccc- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ccc-arcmt-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ccc-arcmt-migrate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ccc-arcmt-modify - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ccc-gcc-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ccc-install-dir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ccc-objcmt-migrate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ccc-print-bindings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ccc-print-phases - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cfguard - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cfguard-no-checks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -chain-include - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-denorms-are-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-ext= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-fast-relaxed-math - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-finite-math-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-fp32-correctly-rounded-divide-sqrt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-kernel-arg-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-mad-enable - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-no-signed-zeros - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-no-stdinc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-opt-disable - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-single-precision-constant - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-std= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-strict-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-uniform-work-group-size - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cl-unsafe-math-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -clear-ast-before-backend - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -client_name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -code-completion-at - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -code-completion-at= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -code-completion-brief-comments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -code-completion-macros - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -code-completion-patterns - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -code-completion-with-fixits - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -combine - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -compatibility_version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -compiler-options-dump - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -compress-debug-sections - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -compress-debug-sections= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --config= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --config-system-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --config-user-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -coverage - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -coverage-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --crel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --cuda-compile-host-device - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --cuda-device-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --cuda-feature= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --cuda-gpu-arch= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --cuda-host-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --cuda-include-ptx= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --cuda-noopt-device-debug - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --cuda-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --cuda-path-ignore-env - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cuid= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -current_version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -cxx-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dA - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dD - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dE - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dI - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -d - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -d - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -darwin-target-variant - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -darwin-target-variant-sdk-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -darwin-target-variant-triple - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dead_strip - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -debug-forward-template-params - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -debug-info-macro - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -debugger-tuning= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -default-function-attr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --defsym - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dependency-dot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dependency-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -detailed-preprocessing-record - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -diagnostic-log-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -serialize-diagnostic-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -disable-O0-optnone - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -disable-free - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -disable-lifetime-markers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -disable-llvm-optzns - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -disable-llvm-passes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -disable-llvm-verifier - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -disable-objc-default-synthesize-properties - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -disable-pragma-debug-crash - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -disable-red-zone - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -discard-value-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --driver-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dsym-dir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dump-coverage-mapping - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dump-deserialized-decls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dump-raw-tokens - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dump-tokens - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dumpdir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dumpmachine - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dumpspecs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dumpversion - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dwarf-debug-flags - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dwarf-debug-producer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dwarf-explicit-import - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dwarf-ext-refs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dwarf-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Fo - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /Vd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --E - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /HV - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /hlsl-no-stdinc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --dxv-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /validator-version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dylib_file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dylinker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dylinker_install_name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dynamic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -dynamiclib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -e - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ehcontguard - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --embed-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-ast - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-cir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-codegen-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --emit-extension-symbol-graphs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-header-unit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-html - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-interface-stubs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-llvm-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-llvm-uselists - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-merged-ifs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-module - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-module-interface - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-pch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --pretty-sgf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /emit-pristine-llvm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-reduced-module-interface - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --emit-sgf-symbol-labels-for-testing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --emit-static-lib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -emit-symbol-graph - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /enable-16bit-types - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -enable-noundef-analysis - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -enable-tlsdesc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --end-no-unused-arguments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -error-on-deserialized-decl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -error-on-deserialized-decl= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -exception-model - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -exception-model= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -exported_symbols_list - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -extract-api - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --extract-api-ignores= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -faapcs-bitfield-width - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -faddress-space-map-mangling= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -faggressive-function-elimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -falign-commons - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -falign-jumps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -falign-jumps= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -falign-labels - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -falign-labels= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -falign-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -faligned-new= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fall-intrinsics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fallow-pch-with-different-modules-cache-path - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fallow-pch-with-compiler-errors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fallow-pcm-with-compiler-errors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fkeep-inline-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -funit-at-a-time - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fapinotes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fapinotes-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fapinotes-swift-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fapply-global-visibility-to-externs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fauto-profile= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fautomatic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fbacktrace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fblas-matmul-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fblocks-runtime-optional - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fbounds-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fbracket-depth - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fbranch-count-reg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fbuild-session-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fbuild-session-timestamp= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcall-saved-x10 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcall-saved-x11 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcall-saved-x12 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcall-saved-x13 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcall-saved-x14 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcall-saved-x15 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcall-saved-x18 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcall-saved-x8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcall-saved-x9 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcaller-saves - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /fcgl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcheck= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcheck-array-temporaries - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcheck-new - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fclang-abi-compat= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcoarray= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcomment-block-commands= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcompatibility-qualified-id-block-type-checking - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcomplete-member-pointers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fconst-strings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fconstant-string-class - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fconvergent-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcrash-diagnostics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcrash-diagnostics= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcrash-diagnostics-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcray-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fctor-dtor-return-this - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcuda-allow-variadic-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcuda-flush-denormals-to-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcuda-include-gpubinary - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcuda-is-device - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcuda-short-ptr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcx-fortran-rules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fcx-limited-range - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fc++-abi= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fd-lines-as-code - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fd-lines-as-comments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdebugger-cast-result-to-id - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdebugger-objc-literal - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdebugger-support - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdeclare-opencl-builtins - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdeclspec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdefault-calling-conv= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdefault-inline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdepfile-entry= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdeprecated-macro - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdevirtualize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdevirtualize-speculatively - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdiagnostics-fixit-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdiagnostics-format - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdiagnostics-format= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdiagnostics-parseable-fixits - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdiagnostics-print-source-range-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdiagnostics-show-category - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdiagnostics-show-category= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdisable-module-hash - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdiscard-value-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdollar-ok - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdriver-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdump-fortran-optimized - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdump-fortran-original - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdump-parse-tree - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdump-record-layouts - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdump-record-layouts-canonical - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdump-record-layouts-complete - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdump-record-layouts-simple - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fdump-vtable-layouts - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fembed-bitcode-marker - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fencode-extended-block-signature - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ferror-limit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fexperimental-isel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fexperimental-relative-c++-abi-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fexperimental-sanitize-metadata=atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fexperimental-sanitize-metadata=covered - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fexperimental-sanitize-metadata=uar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fexperimental-strict-floating-point - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fexternal-blas - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fexternc-nounwind - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ff2c - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffake-address-space-map - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fimplicit-modules-use-lock - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffine-grained-bitfield-accesses - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffinite-math-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finline-limit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-a0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-a1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-a2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-a3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-a4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-a5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-a6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-d0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-d1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-d2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-d3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-d4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-d5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-d6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-d7 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-g1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-g2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-g3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-g4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-g5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-g6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-g7 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-i0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-i1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-i2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-i3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-i4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-i5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-l0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-l1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-l2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-l3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-l4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-l5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-l6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-l7 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-o0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-o1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-o2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-o3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-o4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-o5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-r9 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x10 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x11 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x12 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x13 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x14 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x15 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x17 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x18 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x19 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x20 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x21 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x22 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x23 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x24 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x25 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x26 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x27 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x28 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x29 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x30 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x31 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x7 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffixed-x9 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffloat-store - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fforbid-guard-variables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffpe-trap= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffree-line-length- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffrontend-optimize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ffuchsia-api-level= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fgcse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fgcse-after-reload - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fgcse-las - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fgcse-sm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fglobal-isel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fgpu-allow-device-init - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fgpu-default-stream= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fgpu-defer-diag - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fgpu-exclude-wrong-side-overloads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fgpu-flush-denormals-to-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fgpu-inline-threshold= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fgpu-rdc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fgpu-sanitize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fhalf-no-semantic-interposition - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fhip-dump-offload-linker-script - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fhip-emit-relocatable - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fhip-fp32-correctly-rounded-divide-sqrt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fhip-kernel-arg-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fhip-new-launch-api - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fhlsl-strict-availability - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -filelist - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -filetype - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finclude-default-header - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finit-character= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finit-integer= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finit-local-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finit-logical= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finit-real= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finline-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finline-functions-called-once - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finline-hint-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finline-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finline-small-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -finteger-4-integer-8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fipa-cp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fivopts - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fix-only-warnings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fix-what-you-can - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fixit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fixit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fixit-recompile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fixit-to-temporary - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -flat_namespace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -flimit-debug-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -flto-unit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -flto-visibility-public-std - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmax-array-constructor= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmax-errors= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmax-identifier-length - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmax-stack-var-size= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmax-subrecord-length= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmerge-constants - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmerge-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodule-feature - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodule-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodule-file-home-is-cwd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodule-format= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodule-implementation-of - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodule-map-file-home-is-cwd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodule-maps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodule-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodule-output= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodule-private - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-cache-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-codegen - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-debuginfo - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-disable-diagnostic-validation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-embed-all-files - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-embed-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-hash-content - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-local-submodule-visibility - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-prune-after= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-prune-interval= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-strict-context-hash - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-user-build-path - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-validate-once-per-build-session - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodules-validate-system-headers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodulo-sched - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fmodulo-sched-allow-regmoves - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fnative-half-arguments-and-returns - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fnative-half-type - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-aapcs-bitfield-width - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-aggressive-function-elimination - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-align-commons - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-align-jumps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-align-labels - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-align-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-all-intrinsics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-keep-inline-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-unit-at-a-time - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-apinotes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-apinotes-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-backtrace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-bitfield-type-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-bounds-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-branch-count-reg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-caller-saves - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-check-array-temporaries - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-check-new - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-complete-member-pointers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-const-strings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-convergent-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-crash-diagnostics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-cray-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-cuda-flush-denormals-to-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-cuda-host-device-constexpr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-cuda-short-ptr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-cx-fortran-rules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-cx-limited-range - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-d-lines-as-code - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-d-lines-as-comments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-declspec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-default-inline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-deprecated-macro - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-devirtualize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-devirtualize-speculatively - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-diagnostics-use-presumed-location - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-discard-value-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-dllexport-inlines - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-dollar-ok - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-dump-fortran-optimized - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-dump-fortran-original - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-dump-parse-tree - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-experimental-isel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-experimental-relative-c++-abi-vtables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-external-blas - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-f2c - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-implicit-modules-use-lock - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-fine-grained-bitfield-accesses - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-finite-math-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-inline-limit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-float-store - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-frontend-optimize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-gcse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-gcse-after-reload - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-gcse-las - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-gcse-sm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-global-isel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-gpu-allow-device-init - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-gpu-defer-diag - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-gpu-exclude-wrong-side-overloads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-gpu-flush-denormals-to-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-gpu-rdc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-gpu-sanitize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-hip-emit-relocatable - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-hip-fp32-correctly-rounded-divide-sqrt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-hip-kernel-arg-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-hip-new-launch-api - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-init-local-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-inline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-inline-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-inline-functions-called-once - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-inline-small-functions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-integer-4-integer-8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-ipa-cp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-ivopts - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-limit-debug-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-lto-unit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-math-builtin - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-max-identifier-length - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-merge-constants - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-module-maps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-module-private - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-modules-error-recovery - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-modules-global-index - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-modules-share-filemanager - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-modules-validate-system-headers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-modulo-sched - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-modulo-sched-allow-regmoves - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-offload-implicit-host-device-templates - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-offload-via-llvm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-openmp-new-driver - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-pack-derived - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-padding-on-unsigned-fixed-point - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-pch-timestamp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-peel-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-prefetch-loop-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-profile-correction - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-profile-use - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-profile-values - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-range-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-real-4-real-10 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-real-4-real-16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-real-4-real-8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-real-8-real-10 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-real-8-real-16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-real-8-real-4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-realloc-lhs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-recovery-ast - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-recovery-ast-type - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-recursive - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-rename-registers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-reorder-blocks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-repack-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-rtlib-add-rpath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-address-globals-dead-stripping - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-address-outline-instrumentation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-address-poison-custom-array-cookie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-address-use-after-scope - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-address-use-odr-indicator - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-cfi-canonical-jump-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-cfi-cross-dso - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-coverage= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-hwaddress-experimental-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-ignorelist - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-link-c++-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-link-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-memory-track-origins - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-memory-use-after-dtor - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-minimal-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-recover - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-recover= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-stats - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-thread-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-thread-func-entry-exit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-thread-memory-access - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-trap - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-trap= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sanitize-undefined-trap-on-error - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-schedule-insns - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-schedule-insns2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-second-underscore - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sign-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-signaling-nans - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-signed-wchar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-single-precision-constant - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-strength-reduce - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-sycl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-tracer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-tree-dce - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-tree-ter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-tree-vrp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-unroll-all-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-unsafe-loop-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-unsigned-char - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-unswitch-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-use-ctor-homing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-use-linker-plugin - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-validate-pch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-variable-expansion-in-unroller - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-vect-cost-model - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-verify-intermediate-code - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-wchar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-web - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-whole-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fno-whole-program - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fobjc-arc-cxxlib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fobjc-dispatch-method= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fobjc-runtime-has-weak - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fobjc-subscripting-legacy-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -foffload-implicit-host-device-templates - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -foffload-via-llvm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fopenmp-new-driver - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -force_cpusubtype_ALL - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -force_flat_namespace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -force_load - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -foverride-record-layout= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fpack-derived - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fpadding-on-unsigned-fixed-point - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fparse-all-comments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fpatchable-function-entry-offset= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fpeel-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fplugin-arg- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fprebuilt-module-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fprefetch-loop-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fpreserve-vec3-type - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fprofile-correction - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fprofile-instrument= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fprofile-instrument-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fprofile-instrument-use-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fprofile-values - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -framework - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -frandomize-layout-seed= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -frandomize-layout-seed-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -frange-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -freal-4-real-10 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -freal-4-real-16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -freal-4-real-8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -freal-8-real-10 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -freal-8-real-16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -freal-8-real-4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -frealloc-lhs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -frecord-marker= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -frecovery-ast - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -frecovery-ast-type - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -frecursive - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -frename-registers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -freorder-blocks - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -frepack-arrays - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -frtlib-add-rpath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-address-field-padding= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-address-globals-dead-stripping - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-address-outline-instrumentation - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-address-poison-custom-array-cookie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-address-use-after-scope - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-address-use-odr-indicator - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-cfi-canonical-jump-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-cfi-cross-dso - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-cfi-icall-generalize-pointers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-cfi-icall-experimental-normalize-integers - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-8bit-counters - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-allowlist= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-control-flow - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-ignorelist= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-indirect-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-inline-8bit-counters - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-inline-bool-flag - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-no-prune - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-pc-table - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-stack-depth - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-bb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-cmp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-div - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-gep - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-loads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-pc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-pc-guard - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-trace-stores - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-coverage-type= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-hwaddress-abi= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-hwaddress-experimental-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-ignorelist= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-link-c++-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-link-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-memory-track-origins - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-memory-track-origins= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-memory-use-after-dtor - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-memtag-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-minimal-runtime - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-recover - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-recover= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-stats - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-system-ignorelist= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-thread-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-thread-func-entry-exit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-thread-memory-access - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-trap - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-trap= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-undefined-ignore-overflow-pattern= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-undefined-strip-path-components= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-undefined-trap-on-error - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fschedule-insns - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fschedule-insns2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsecond-underscore - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fshow-skipped-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsign-zero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsignaling-nans - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsigned-wchar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsingle-precision-constant - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fspv-target-env= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fstrength-reduce - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsycl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsycl-is-device - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsycl-is-host - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsystem-module - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ftabstop - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ftest-module-file-extension= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ftracer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ftree-dce - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ftree-ter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ftree-vrp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ftype-visibility= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -function-alignment - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -funknown-anytype - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -funroll-all-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -funsafe-loop-optimizations - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -funswitch-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -funwind-tables= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fuse-ctor-homing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fuse-cuid= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fuse-linker-plugin - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fuse-register-sized-bitfield-access - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fvariable-expansion-in-unroller - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fvect-cost-model - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fverify-debuginfo-preserve - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fverify-debuginfo-preserve-export= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fverify-intermediate-code - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fwarn-stack-size= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fwchar-type= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fweb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fwhole-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fwhole-program - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -g0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -g1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -g2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -g3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -g - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --gcc-install-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --gcc-toolchain= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --gcc-triple= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gcodeview - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gcodeview-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gcodeview-ghash - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gcoff - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gcolumn-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gdbx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gdwarf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gdwarf32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gdwarf64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gdwarf-2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gdwarf-3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gdwarf-4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gdwarf-5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gdwarf-aranges - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gembed-source - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gen-cdb-fragment-path - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gen-reproducer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gen-reproducer= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gfull - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ggdb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ggdb0 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ggdb1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ggdb2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ggdb3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ggnu-pubnames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ginline-line-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gline-directives-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gline-tables-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -glldb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gmlt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gmodules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-codeview-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-codeview-ghash - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-column-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-embed-source - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-gnu-pubnames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-inline-line-tables - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-omit-unreferenced-methods - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-pubnames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-record-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-simple-template-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-split-dwarf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-strict-dwarf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gno-template-alias - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gomit-unreferenced-methods - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --gpu-bundle-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --gpu-instrument-lib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --gpu-max-threads-per-block= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --gpu-use-aux-triple-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gpubnames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -grecord-command-line - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gsce - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gsimple-template-names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gsimple-template-names= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gsplit-dwarf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gsplit-dwarf= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gsrc-hash= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gstabs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gstrict-dwarf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gtemplate-alias - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gtoggle - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gused - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gvms - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gxcoff - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gz - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -gz= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -header-include-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -header-include-filtering= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -header-include-format= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -headerpad_max_install_names - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --hip-device-lib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --hip-link - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --hip-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --hip-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --hipspv-pass-plugin= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --hipstdpar - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --hipstdpar-interpose-alloc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --hipstdpar-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --hipstdpar-prim-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --hipstdpar-thrust-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -hlsl-entry - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -iapinotes-modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ibuiltininc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -idirafter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -iframework - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -iframeworkwithsysroot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -imacros - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -image_base - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -imultilib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -include - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -include-pch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -init - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -inline-asm= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -install_name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -interface-stub-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -internal-externc-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -internal-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -iprefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -iquote - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -isysroot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -isystem - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -isystem-after - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -ivfsoverlay - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -iwithprefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -iwithprefixbefore - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -iwithsysroot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -keep_private_externs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -l - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -lazy_framework - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -lazy_library - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --ld-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --libomptarget-amdgcn-bc-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --libomptarget-amdgpu-bc-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --libomptarget-nvptx-bc-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --linker-option= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -llvm-verify-each - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m3dnow - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m3dnowa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m68000 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m68010 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m68020 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m68030 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m68040 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m68060 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m68881 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -m80387 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mseses - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mabicalls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mabs= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -madx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -maes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -main-file-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -maix32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -maix64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -maix-shared-lib-tls-model-opt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -maix-small-local-dynamic-tls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -maix-small-local-exec-tls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -maix-struct-return - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -malign-branch= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -malign-branch-boundary= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -malign-double - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -malign-functions= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -malign-jumps= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -malign-loops= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -maltivec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mamdgpu-ieee - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mamdgpu-precise-memory-op - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mamx-avx512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mamx-bf16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mamx-complex - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mamx-fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mamx-fp8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mamx-int8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mamx-movrs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mamx-tf32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mamx-tile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mamx-transpose - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mannotate-tablejump - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mappletvos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mappletvsimulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mapx-features= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mapx-inline-asm-use-gpr32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mapxf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -march= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -marm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -marm64x - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -masm= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -massembler-fatal-warnings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -massembler-no-warn - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -matomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx10.1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx10.1-256 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx10.1-512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx10.2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx10.2-256 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx10.2-512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512bf16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512bitalg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512bw - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512cd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512dq - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512f - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512ifma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512vbmi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512vbmi2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512vl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512vnni - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512vp2intersect - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavx512vpopcntdq - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavxifma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavxneconvert - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavxvnni - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavxvnniint16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mavxvnniint8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mbackchain - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mbig-endian - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mbmi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mbmi2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mbranch-likely - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mbranch-protection= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mbranch-protection-pauth-lr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mbranch-target-enforce - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mbranches-within-32B-boundaries - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mbulk-memory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mbulk-memory-opt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcabac - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcall-indirect-overlong - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcf-branch-label-scheme= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcheck-zero-division - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcldemote - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mclflushopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mclwb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mclzero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcmpb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcmpccxadd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcmse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcompact-branches= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mconsole - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mconstant-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mconstructor-aliases - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcpu= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcrbits - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcrc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcrc32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcumode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcx16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mdaz-ftz - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mdebug-pass - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mdefault-build-attributes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mdefault-visibility-export-mapping= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mdirect-move - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mdiv32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mdll - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mdouble= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mdouble-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mdsp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mdspr2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mdynamic-no-pic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -meabi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mefpu2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -membedded-data - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -menable-experimental-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -menqcmd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mevex512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mexception-handling - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mexec-model= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mexecute-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mextended-const - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mextern-sdata - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mf16c - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfancy-math-387 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfentry - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfix4300 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfix-and-continue - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfix-cmse-cve-2021-35465 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfix-cortex-a53-835769 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfix-cortex-a57-aes-1742098 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfix-cortex-a72-aes-1655431 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfix-gr712rc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfix-ut700 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfloat128 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfloat-abi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfloat-abi= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfma4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfp16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfp32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfp64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfpmath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfpmath= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfprnd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfpu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfpu= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfpxx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mframe-chain= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfrecipe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfsgsbase - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfsmuld - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfunction-return= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mfxsr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mgeneral-regs-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mgfni - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mginv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mglibc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mglobal-merge - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mgpopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mguard= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mguarded-control-stack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mhard-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mhard-quad-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mharden-sls= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mhvx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mhvx= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mhvx-ieee-fp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mhvx-length= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mhvx-qfloat - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mhreset - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mhtm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mhwdiv= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mhwmult= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -miamcu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mieee-fp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mieee-rnd-near - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mignore-xcoff-visibility - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -migrate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-finalize-removal - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-ns-alloc-error - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mimplicit-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mimplicit-it= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mincremental-linker-compatible - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mindirect-branch-cs-prefix - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mindirect-jump= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -minline-all-stringops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -minvariant-function-descriptors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -minvpcid - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mios-simulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mios-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips32r2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips32r3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips32r5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips32r6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips64r2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips64r3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips64r5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mips64r6 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -misel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mkernel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mkl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlam-bh - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlamcas - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlasx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mld-seq-sa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mldc1-sdc1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlimit-float-precision - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlink-bitcode-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlink-builtin-bitcode-postopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlinker-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlittle-endian - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlocal-sdata - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlong-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlong-double-128 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlong-double-64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlong-double-80 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlongcall - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlr-for-calls-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlsx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlvi-cfi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlvi-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlwp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mlzcnt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmacos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmadd4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmapsyms=implicit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmark-bti-property - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmcu= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmemops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmfcrf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmfocrf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmicromips - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmmx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmovbe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmovdir64b - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmovdiri - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmovrs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmpx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mms-bitfields - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmsa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmultimemory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmultivalue - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmutable-globals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mmwaitx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mnan= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-3dnow - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-3dnowa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-80387 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-abicalls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-adx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-aes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-altivec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-amdgpu-ieee - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-amdgpu-precise-memory-op - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-amx-avx512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-amx-bf16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-amx-complex - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-amx-fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-amx-fp8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-amx-int8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-amx-movrs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-amx-tf32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-amx-tile - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-amx-transpose - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-annotate-tablejump - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-apx-features= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-apxf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx10.1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx10.1-256 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx10.1-512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx10.2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx10.2-256 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx10.2-512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512bf16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512bitalg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512bw - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512cd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512dq - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512f - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512ifma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512vbmi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512vbmi2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512vl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512vnni - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512vp2intersect - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avx512vpopcntdq - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avxifma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avxneconvert - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avxvnni - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avxvnniint16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-avxvnniint8 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-backchain - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-bmi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-bmi2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-branch-likely - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-bti-at-return-twice - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-bulk-memory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-bulk-memory-opt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-call-indirect-overlong - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-check-zero-division - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-cldemote - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-clflushopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-clwb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-clzero - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-cmpb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-cmpccxadd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-constant-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-constructor-aliases - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-crbits - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-crc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-crc32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-cumode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-cx16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-daz-ftz - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-default-build-attributes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-div32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-dsp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-dspr2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-embedded-data - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-enqcmd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-evex512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-exception-handling - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mnoexecstack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-execute-only - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-extended-const - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-extern-sdata - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-f16c - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fix-cmse-cve-2021-35465 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fix-cortex-a53-835769 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fix-cortex-a57-aes-1742098 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fix-cortex-a72-aes-1655431 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-float128 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fma4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fmv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fp16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fp-ret-in-387 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fprnd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fpu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-frecipe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fsgsbase - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fsmuld - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-fxsr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-gather - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-gfni - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-ginv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-global-merge - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-gpopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-hvx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-hvx-ieee-fp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-hvx-qfloat - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-hreset - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-htm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-iamcu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-implicit-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-incremental-linker-compatible - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-inline-all-stringops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-invariant-function-descriptors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-invpcid - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-isel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-kl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-lam-bh - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-lamcas - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-lasx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-ld-seq-sa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-ldc1-sdc1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-link-builtin-bitcode-postopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-local-sdata - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-long-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-longcall - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-lsx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-lvi-cfi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-lvi-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-lwp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-lzcnt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-madd4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-memops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-mfcrf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-mfocrf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-micromips - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-mips16 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-mma - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-mmx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-movbe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-movdir64b - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-movdiri - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-movrs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-movt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-mpx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-ms-bitfields - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-msa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-mt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-multimemory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-multivalue - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-mutable-globals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-mwaitx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-neg-immediates - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-nontrapping-fptoint - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-nvj - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-nvs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-odd-spreg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-omit-leaf-frame-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-outline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-outline-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-packed-stack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-packets - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-pascal-strings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-pclmul - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-pconfig - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-pcrel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-pic-data-is-text-relative - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-pku - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-popc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-popcnt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-popcntd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-power10-vector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-power8-vector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-power9-vector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-prefetchi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-prefixed - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-prfchw - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-ptwrite - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-pure-code - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-raoint - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-rdpid - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-rdpru - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-rdrnd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-rdseed - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-red-zone - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-reference-types - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-regnames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-relax - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-relax-all - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-relax-pic-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-relaxed-simd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-restrict-it - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-retpoline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-retpoline-external-thunk - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-rtd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-rtm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sahf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-save-restore - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-scalar-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-scatter - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-serialize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-seses - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sgx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sha - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sha512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-shstk - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sign-ext - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-simd128 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-skip-rax-setup - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sm3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sm4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-soft-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-spe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-speculative-load-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sse2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sse3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sse4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sse4.1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sse4.2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-sse4a - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-ssse3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-stack-arg-probe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-stackrealign - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-tail-call - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-tbm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-tgsplit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-thumb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-tls-direct-seg-refs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-tocdata - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-tocdata= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-tsxldtrk - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-type-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-uintr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-unaligned-access - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-unaligned-symbols - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-unsafe-fp-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-usermsr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-v8plus - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-vaes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-vector-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-vevpu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-virt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-vis - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-vis2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-vis3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-vpclmulqdq - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-vsx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-vx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-vzeroupper - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-waitpkg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-warn-nonportable-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-wavefrontsize64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-wbnoinvd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-wide-arithmetic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-widekl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-x87 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-xcoff-roptr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-xgot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-xop - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-xsave - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-xsavec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-xsaveopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-xsaves - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-zvector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mnocrc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-direct-move - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mnontrapping-fptoint - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mnop-mcount - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-paired-vector-memops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mno-crypto - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mnvj - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mnvs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -modd-spreg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -module-dependency-dir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -module-file-deps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -module-file-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -momit-leaf-frame-pointer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -moslib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -moutline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -moutline-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpacked-stack - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpackets - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpad-max-prefix-size= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpaired-vector-memops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpascal-strings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpclmul - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpconfig - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpcrel - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpic-data-is-text-relative - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpku - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpopc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpopcnt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpopcntd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpower10-vector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mcrypto - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpower8-vector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpower9-vector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mprefer-vector-width= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mprefetchi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mprefixed - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mprfchw - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mprintf-kind= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mprivileged - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mptwrite - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mpure-code - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mqdsp6-compat - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mraoint - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrdpid - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrdpru - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrdrnd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrdseed - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrecip - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrecip= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrecord-mcount - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mred-zone - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mreference-types - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mregnames - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mregparm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mregparm= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrelax - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrelax-all - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrelax-pic-calls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrelax-relocations=no - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrelaxed-simd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrestrict-it - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mretpoline - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mretpoline-external-thunk - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrop-protect - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrtd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrtm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mrvv-vector-bits= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msahf - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msave-reg-params - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msave-restore - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msave-temp-labels - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mscalar-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msecure-plt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mserialize - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msgx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msha - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msha512 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mshstk - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msign-ext - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msign-return-address= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msign-return-address-key= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msim - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msimd128 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msimd= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msingle-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mskip-rax-setup - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msm3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msm4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msmall-data-limit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msmall-data-limit= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msmall-data-threshold= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msoft-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msoft-quad-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mspe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mspeculative-load-hardening - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msse - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msse2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msse2avx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msse3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msse4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msse4.1 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msse4.2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msse4a - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mssse3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mstack-alignment= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mstack-arg-probe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mstack-probe-size= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mstack-protector-guard= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mstack-protector-guard-offset= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mstack-protector-guard-reg= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mstack-protector-guard-symbol= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mstackrealign - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msve-vector-bits= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -msvr4-struct-return - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mt-migrate-directory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtail-call - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtargetos= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtbm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtgsplit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mthread-model - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mthreads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mthumb - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtls-dialect= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtls-direct-seg-refs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtls-size= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtocdata - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtocdata= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtp= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtsxldtrk - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtune= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtvos-simulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mtvos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -muclibc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -muintr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -multi_module - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -multi-lib-config= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -multiply_defined - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -multiply_defined_unused - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -munaligned-access - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -munaligned-symbols - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -municode - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -munsafe-fp-atomics - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -musermsr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv5 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv55 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv60 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv62 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv65 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv66 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv67 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv67t - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv68 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv69 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv71 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv71t - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv73 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mv8plus - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mvaes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mvector-strict-align - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mvevpu - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mvirt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mvis - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mvis2 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mvis3 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mvpclmulqdq - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mvsx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mvx - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mvzeroupper - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mwaitpkg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mwarn-nonportable-cfstrings - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mwatchos-simulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mwatchos-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mwatchsimulator-version-min= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mwavefrontsize64 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mwbnoinvd - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mwide-arithmetic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mwidekl - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mwindows - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mx32 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mx87 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mxcoff-build-id= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mxcoff-roptr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mxgot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mxop - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mxsave - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mxsavec - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mxsaveopt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mxsaves - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mzos-hlq-clang= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mzos-hlq-csslib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mzos-hlq-le= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mzos-sys-include= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -mzvector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -n - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -new-struct-path-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no_dead_strip_inits_and_terms - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-canonical-prefixes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-clear-ast-before-backend - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-code-completion-globals - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-code-completion-ns-level-decls - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-cuda-gpu-arch= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-cuda-include-ptx= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-cuda-noopt-device-debug - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-cuda-version-check - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-default-config - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-emit-llvm-uselists - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-enable-noundef-analysis - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-gpu-bundle-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-hip-rt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-implicit-float - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-integrated-cpp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-offload-add-rpath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-offload-arch= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-offload-compress - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-offload-new-driver - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-pedantic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-pie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-pointer-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-round-trip-args - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -no-struct-path-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-system-header-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --no-wasm-opt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nobuiltininc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nodefaultlibs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nodriverkitlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nofixprebinding - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nogpuinc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nohipwrapperinc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nolibc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nomultidefs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nopie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -noprebind - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -noprofilelib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -noseglinkedit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nostartfiles - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nostdinc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nostdinc++ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nostdlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nostdlibinc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nostdlib++ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -nostdsysteminc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --nvptx-arch-tool= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fexperimental-openacc-macro-override= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -p - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -pagezero_size - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -pass-exit-codes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -pch-through-hdrstop-create - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -pch-through-hdrstop-use - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -pch-through-header= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -pedantic-errors - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -pg - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -pie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -pipe - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -plugin-arg- - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -pointer-tbaa - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -preamble-bytes= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -prebind - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -prebind_all_twolevel_modules - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -preload - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-dependency-directives-minimized-source - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-diagnostic-options - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-effective-triple - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-enabled-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-file-name= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-ivar-layout - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-libgcc-file-name - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-multi-directory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-multi-flags-experimental - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-multi-lib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-multi-os-directory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-preamble - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-prog-name= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-resource-dir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-rocm-search-dirs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-runtime-dir - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-search-dirs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-stats - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-library-module-manifest-path - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-supported-extensions - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-target-triple - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -print-targets - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -private_bundle - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --product-name= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -pthreads - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --ptxas-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -r - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -rdynamic - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -read_only_relocs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -reexport_framework - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -reexport-l - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -reexport_library - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -regcall4 - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -relaxed-aliasing - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -relocatable-pch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -remap - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -remap-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -resource-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -rewrite-legacy-objc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -rewrite-macros - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -rewrite-objc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -rewrite-test - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --rocm-device-lib-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --rocm-path= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -round-trip-args - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -rpath - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --rsp-quoting= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -rtlib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -s - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-address-destructor= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -fsanitize-address-use-after-return= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -save-stats - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -save-stats= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -sectalign - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -sectcreate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -sectobjectsymbols - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -sectorder - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -seg1addr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -seg_addr_table - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -seg_addr_table_filename - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -segaddr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -segcreate - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -seglinkedit - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -segprot - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -segs_read_ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -segs_read_only_addr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -segs_read_write_addr - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -setup-static-analyzer - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -shared - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -shared-libgcc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -shared-libsan - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -show-encoding - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --show-includes - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -show-inst - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -single_module - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -skip-function-bodies - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -source-date-epoch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -specs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -specs= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /spirv - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -split-dwarf-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -split-dwarf-output - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -stack-protector - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -stack-protector-buffer-size - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -stack-usage-file - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --start-no-unused-arguments - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -startfiles - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -static - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -static-define - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -static-libgcc - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -static-libgfortran - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -static-libsan - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -static-libstdc++ - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -static-openmp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -static-pie - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -stats-file= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -stats-file-append - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -std-default= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -stdlib - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -stdlib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -stdlib++-isystem - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -sub_library - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -sub_umbrella - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --sycl-link - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -sycl-std= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --symbol-graph-dir= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -sys-header-deps - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --system-header-prefix= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -t - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --target= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -target-abi - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -target - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -target-linker-version - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 /T - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -target-sdk-version= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -templight-dump - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -time - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -traditional - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -traditional-cpp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -trigraphs - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -trim-egraph - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -triple= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -twolevel_namespace - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -twolevel_namespace_hints - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -u - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -umbrella - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -undef - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -undefined - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -unexported_symbols_list - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -unwindlib= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -v - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -vectorize-loops - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -vectorize-slp - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -verify - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -verify= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --verify-debug-info - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -verify-ignore-unexpected - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -verify-ignore-unexpected= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -verify-pch - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -vfsoverlay - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -via-file-asm - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -vtordisp-mode= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --warning-suppression-mappings= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 --wasm-opt - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -weak_framework - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -weak_library - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -weak_reference_mismatches - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -weak-l - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -whatsloaded - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -why_load - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -whyload - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -working-directory - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -working-directory= - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -y - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! RUN: not %clang --driver-mode=flang -fc1 -z - < /dev/null 2>&1 | FileCheck -check-prefix=FC1Option %s
! FC1Option: {{(unknown argument|no such file or directory|does not exist)}}
