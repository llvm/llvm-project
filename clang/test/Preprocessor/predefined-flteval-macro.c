// RUN: %clang_cc1 -std=c11  -E -triple=aarch64 -xc  %s | FileCheck %s
// RUN: %clang_cc1 -std=c11  -triple=aarch64 -xc  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple=arm64 -xc  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple=aarch64_be -xc  -fsyntax-only %s
// RUN: %clang_cc1  -triple=arm64 -xc++  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=aarch64-apple-ios7.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=aarch64-windows-msvc -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple=aarch64 -mcmodel=small -xc  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple=aarch64 -mcmodel=tiny -xc  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple=aarch64 -mcmodel=large -xc  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=thumbv7-windows-msvc -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=arm-none-none -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=arm-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple arm-none-none -target-abi apcs-gnu -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=armeb-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=arm-none-linux-gnueabi -target-feature +soft-float -target-feature +soft-float-abi -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=arm-none-linux-gnueabi -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=armv6-unknown-cloudabi-eabihf -fsyntax-only %s
// RUN: %clang -c -ffreestanding -target arm-netbsd-eabi -fsyntax-only %s
// RUN: %clang -c -ffreestanding -target arm-netbsd-eabihf -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=arm-none-eabi -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=arm-none-eabihf -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=aarch64-none-eabi -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=aarch64-none-eabihf -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=aarch64-none-elf -fsyntax-only %s
// RUN: %clang -target x86_64-apple-darwin -arch armv7s -x c -fsyntax-only %s
// RUN: %clang -target x86_64-apple-darwin -arch armv6m -x c -fsyntax-only %s
// RUN: %clang -target x86_64-apple-darwin -arch armv7m -x c -fsyntax-only %s
// RUN: %clang -target x86_64-apple-darwin -arch armv7em -x c -fsyntax-only %s
// RUN: %clang -target x86_64-apple-darwin -arch armv7 -x c -fsyntax-only %s
// RUN: %clang -c -target arm -mhwdiv=arm -x c -fsyntax-only %s
// RUN: %clang -c -target arm -mthumb -mhwdiv=thumb -x c -fsyntax-only %s
// RUN: %clang -c -target arm -x c -fsyntax-only %s
// RUN: %clang -c -target arm -mthumb -x c -fsyntax-only %s
// RUN: %clang -c -target arm -mhwdiv=thumb -x c -fsyntax-only %s
// RUN: %clang -c -target arm -mthumb -mhwdiv=arm -x c -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=armv8-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=armebv8-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=thumbv8 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=thumbebv8 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=thumbv5 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=thumbv6t2 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=thumbv7 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=thumbebv7 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=thumbv7-pc-windows-gnu -exception-model=dwarf -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc-none-none -target-cpu 603e -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=powerpc-none-none -target-cpu 603e -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc-none-none -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix7.2.0.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix7.1.0.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix6.1.0.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix5.3.0.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix5.2.0.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix5.1.0.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix5.0.0.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix4.3.0.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix4.1.0.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix3.2.0.0 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -fno-wchar -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix7.1.0.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -x c -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix7.1.0.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char -pthread -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc-ibm-aix7.1.0.0 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc-unknown-linux-gnu -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc-unknown-linux-gnu -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc-unknown-linux-gnu -target-feature +spe -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc-unknown-linux-gnu -target-cpu 8548 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc-apple-darwin8 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mipsel-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding             -triple=mips64-none-none -target-abi n32 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding             -triple=mips64-none-none -target-abi n32 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding             -triple=mips64el-none-none -target-abi n32 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips64-none-none -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=mips64-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips64el-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips-none-nones             -target-cpu mips32 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips-none-none             -target-cpu mips32r2 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips-none-none             -target-cpu mips32r3 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips-none-none             -target-cpu mips32r5 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips-none-none             -target-cpu mips32r6 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips64-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips64-none-none             -target-cpu mips64 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips64-none-none             -target-cpu mips64r2 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips64-none-none             -target-cpu mips64r3 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips64-none-none             -target-cpu mips64r5 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips64-none-none             -target-cpu mips64r6 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips64-none-none             -target-cpu octeon -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips64-none-none             -target-cpu octeon+ -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +soft-float -ffreestanding    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +single-float -ffreestanding    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +soft-float -target-feature +single-float    -ffreestanding -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +mips16    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature -mips16    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +micromips    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature -micromips    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +dsp    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +dspr2    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +msa    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +nomadd4    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11     -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips32r3 -target-feature +nan2008    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips32r3 -target-feature -nan2008    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips32r3 -target-feature +abs2008    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips32r3 -target-feature -abs2008    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11      -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +fpxx    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips32r6 -target-feature +fpxx    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11      -triple=mips64-none-none -fsyntax-only %s
// RUN: not %clang_cc1 -std=c11  -target-feature -fp64    -triple=mips64-none-none  2>&1 -fsyntax-only %s
// RUN: not %clang_cc1 -std=c11  -target-feature +fpxx    -triple=mips64-none-none  2>&1 -fsyntax-only %s
// RUN: not %clang_cc1 -std=c11  -target-cpu mips64r6 -target-feature +fpxx    -triple=mips64-none-none  2>&1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature -fp64    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +fp64    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +single-float    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature +fp64    -triple=mips64-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-feature -fp64 -target-feature +single-float    -triple=mips64-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips32r6    -triple=mips-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips64r6    -triple=mips64-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips32    -triple=mips-unknown-netbsd -mrelocation-model pic -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips64    -triple=mips64-unknown-netbsd -mrelocation-model pic -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips32    -triple=mips-unknown-freebsd -mrelocation-model pic -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips64    -triple=mips64-unknown-freebsd -mrelocation-model pic -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips32    -triple=mips-unknown-openbsd -mrelocation-model pic -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -target-cpu mips64    -triple=mips64-unknown-openbsd -mrelocation-model pic -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu pwr7 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=powerpc64-none-none -target-cpu pwr7 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64le-none-none -target-cpu pwr7 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu 630 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu pwr3 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu power3 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu pwr4 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu power4 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu pwr5 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu power5 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu pwr5x -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu power5x -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu pwr6 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu power6 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu pwr6x -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu power6x -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu pwr7 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu power7 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu pwr8 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu power8 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64le-none-none -target-cpu ppc64le -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu pwr9 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu power9 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu pwr10 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu power10 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-cpu future -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-feature +mma -target-cpu power10 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-none-none -target-feature +float128 -target-cpu power9 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=powerpc64-ibm-aix7.1.0.0 -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-unknown-linux-gnu -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-unknown-linux-gnu -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-unknown-linux-gnu -target-abi elfv1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-unknown-linux-gnu -target-abi elfv2 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64le-unknown-linux-gnu -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64le-unknown-linux-gnu -target-abi elfv1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64le-unknown-linux-gnu -target-abi elfv2 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-unknown-freebsd11 -target-abi elfv1 -xc  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-unknown-freebsd12 -target-abi elfv1 -xc  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-unknown-freebsd13 -target-abi elfv2 -xc  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64le-unknown-freebsd13 -target-abi elfv2 -xc  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-unknown-openbsd -target-abi elfv2 -xc  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-linux-musl -target-abi elfv2 -xc  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64le-unknown-linux-gnu -target-abi elfv2 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-unknown-freebsd -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64le-unknown-freebsd -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=s390x-none-none -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=s390x-none-none -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=s390x-none-zos -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=gnu++14 -ffreestanding -triple=s390x-none-zos -fno-signed-char -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=arm64_32-apple-ios7.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=thumbv7k-apple-watchos2.0 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=2 -triple=i386-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=i386-pc-linux-gnu -target-cpu pentium4 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=i386-pc-linux-gnu -target-cpu pentium4 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=i386-pc-linux-gnu -target-cpu pentium4 -malign-double -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=2 -triple=i386-netbsd -target-cpu i486 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -DFEM=2 -triple=i386-netbsd -target-cpu i486 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=2 -triple=i386-netbsd -target-cpu i486 -malign-double -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=i386-netbsd -target-feature +sse2 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=i386-netbsd6 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=1 -triple=i386-netbsd6 -target-feature +sse2 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -DFEM=2 -triple=i686-pc-mingw32 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -DFEM=2 -fms-extensions -triple=i686-pc-mingw32 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -DFEM=2 -triple=i686-unknown-cygwin -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -DFEM=2 -fms-extensions -triple=i686-unknown-cygwin -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=x86_64-none-none -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=x86_64-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=x86_64h-none-none -fsyntax-only %s
// RUN: %clang_cc1 -xc -mcmodel=medium -DFEM=2 -triple=i386-unknown-linux -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=x86_64-none-none-gnux32 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=x86_64-none-none-gnux32 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=x86_64-unknown-cloudabi -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=x86_64-pc-linux-gnu -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=x86_64-unknown-freebsd9.1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=x86_64-netbsd -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -x assembler-with-cpp -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -fblocks -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=c++2b -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=c++20 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=c++2a -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=c++17 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=c++1z -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=c++14 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=c++1y -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=c++11 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -fdeprecated-macro -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -std=c99 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -std=c11 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -std=c1x -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -std=iso9899:2011 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -std=iso9899:201x -fsyntax-only %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple=x86_64-pc-win32 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple=x86_64-pc-linux-gnu -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple=x86_64-apple-darwin -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple=armv7a-apple-darwin -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=gnu++2b -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=gnu++20 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=gnu++2a -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=gnu++17 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=gnu++1z -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=gnu++14 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=gnu++1y -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -std=gnu++11 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -std=iso9899:199409 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -DFEM=2 -fms-extensions -triple i686-pc-win32 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -DFEM=2 -fms-extensions -triple i686-pc-win32 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -fno-wchar -DFEM=2 -fms-extensions -triple i686-pc-win32 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -x objective-c -fsyntax-only %s
// RUN: %clang_cc1  -x objective-c++ -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -x objective-c -fobjc-gc -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -x objective-c -fobjc-exceptions -fsyntax-only %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -fno-inline -O3 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -O1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -Og -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -Os -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -Oz -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -fpascal-strings -fsyntax-only %s
// RUN: %clang_cc1 -std=c11 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -fwchar-type=short -fno-signed-wchar -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -fwchar-type=short -fno-signed-wchar -triple=x86_64-w64-mingw32 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -fwchar-type=short -fno-signed-wchar -triple=x86_64-unknown-windows-cygnus  -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -fwchar-type=int -DFEM=2 -triple=i686-unknown-unknown -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -fwchar-type=int -triple=x86_64-unknown-unknown -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=msp430-none-none -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=msp430-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=nvptx-none-none -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=nvptx-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=nvptx64-none-none -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=nvptx64-none-none -fsyntax-only %s
// RUN: %clang_cc1  -x cl -ffreestanding -triple=amdgcn -fsyntax-only %s
// RUN: %clang_cc1  -x cl -ffreestanding -triple=r600 -target-cpu caicos -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=sparc-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=sparc-rtems-elf -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=sparc-none-netbsd -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=sparc-none-openbsd -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=sparc-none-none -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=sparc-none-openbsd -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=tce-none-none -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=tce-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=x86_64-scei-ps4 -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -ffreestanding -triple=x86_64-scei-ps4 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple=x86_64-pc-mingw32 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -fms-extensions -triple=x86_64-unknown-mingw32 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=sparc64-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=sparc64-none-openbsd -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=x86_64-pc-kfreebsd-gnu -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=2 -triple=i686-pc-kfreebsd-gnu -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -DFEM=2 -triple i686-pc-linux-gnu -fobjc-runtime=gcc -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -triple sparc-rtems-elf -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -x objective-c -DFEM=2 -triple i386-unknown-freebsd -fobjc-runtime=gnustep-1.9 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -x objective-c -DFEM=2 -triple i386-unknown-freebsd -fobjc-runtime=gnustep-2.5 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple arm-linux-androideabi -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -DFEM=2 -triple i686-linux-android -fsyntax-only %s
// RUN: %clang_cc1  -x c++ -triple x86_64-linux-android -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple arm-linux-androideabi20 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple lanai-unknown-unknown -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=amd64-unknown-openbsd6.1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=aarch64-unknown-openbsd6.1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=arm-unknown-openbsd6.1-gnueabi -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=2 -triple=i386-unknown-openbsd6.1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc-unknown-openbsd6.1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64-unknown-openbsd6.1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=powerpc64le-unknown-openbsd6.1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips64-unknown-openbsd6.1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=mips64el-unknown-openbsd6.1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=sparc64-unknown-openbsd6.1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=riscv64-unknown-openbsd6.1 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=xcore-none-none -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=wasm32-unknown-unknown -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=wasm64-unknown-unknown -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=wasm32-wasi -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=wasm64-wasi -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -DFEM=2 -triple i686-windows-cygnus -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple x86_64-windows-cygnus -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=avr -fsyntax-only %s
// RUN: %clang_cc1  -ffreestanding     -DFEM=2 -triple i686-windows-msvc -fms-compatibility -x c++ -fsyntax-only %s
// RUN: %clang_cc1  -ffreestanding     -triple x86_64-windows-msvc -fms-compatibility -x c++ -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding                   -triple=aarch64-apple-ios9 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding                   -triple=aarch64-apple-macosx10.12 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -DFEM=2 -triple i386-apple-macosx -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple x86_64-apple-macosx -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -DFEM=2 -triple i386-apple-ios-simulator -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple armv7-apple-ios -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple x86_64-apple-ios-simulator -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple arm64-apple-ios -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -DFEM=2 -triple i386-apple-tvos-simulator -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple armv7-apple-tvos -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple x86_64-apple-tvos-simulator -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple arm64-apple-tvos -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -DFEM=2 -triple i386-apple-watchos-simulator -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple armv7k-apple-watchos -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple x86_64-apple-watchos-simulator -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple arm64-apple-watchos -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple armv7-apple-none-macho -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -triple arm64-apple-none-macho -ffreestanding   -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=riscv32 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=riscv32-unknown-linux -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=riscv32  -fforce-enable-int128 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=riscv64 -fsyntax-only %s
// RUN: %clang_cc1 -std=c11  -ffreestanding -triple=riscv64-unknown-linux -fsyntax-only %s
#ifndef FEM
#define FEM 0
#endif
#ifdef __cplusplus
#define SASSERT static_assert
#else
#define SASSERT _Static_assert
#endif
int getFEM() {
  SASSERT(__FLT_EVAL_METHOD__ == FEM, "Unexpected macro value");
  return __FLT_EVAL_METHOD__;
  // Note, the preprocessor in -E mode no longer expands this macro.
  // CHECK: __FLT_EVAL_METHOD__
}
