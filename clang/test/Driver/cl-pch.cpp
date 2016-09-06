// Note: %s and %S must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// /Yc
// RUN: %clang_cl -Werror /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC %s
// 1. Build .pch file.
// CHECK-YC: cc1
// CHECK-YC: -emit-pch
// CHECK-YC: -o
// CHECK-YC: pchfile.pch
// CHECK-YC: -x
// CHECK-YC: "c++"
// 2. Use .pch file.
// CHECK-YC: cc1
// CHECK-YC: -emit-obj
// CHECK-YC: -include-pch
// CHECK-YC: pchfile.pch

// /Yc /Fo
// /Fo overrides the .obj output filename, but not the .pch filename
// RUN: %clang_cl -Werror /Fomyobj.obj /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YCO %s
// 1. Build .pch file.
// CHECK-YCO: cc1
// CHECK-YCO: -emit-pch
// CHECK-YCO: -o
// CHECK-YCO: pchfile.pch
// 2. Use .pch file.
// CHECK-YCO: cc1
// CHECK-YCO: -emit-obj
// CHECK-YCO: -include-pch
// CHECK-YCO: pchfile.pch
// CHECK-YCO: -o
// CHECK-YCO: myobj.obj

// /Yc /Y-
// /Y- disables pch generation
// RUN: %clang_cl -Werror /Y- /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-Y_ %s
// CHECK-YC-Y_-NOT: -emit-pch
// CHECK-YC-Y_-NOT: -include-pch

// /Yu
// RUN: %clang_cl -Werror /Yupchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU %s
// Use .pch file, but don't build it.
// CHECK-YU-NOT: -emit-pch
// CHECK-YU: cc1
// CHECK-YU: -emit-obj
// CHECK-YU: -include-pch
// CHECK-YU: pchfile.pch

// /Yu /Y-
// RUN: %clang_cl -Werror /Y- /Yupchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU-Y_ %s
// CHECK-YU-Y_-NOT: -emit-pch
// CHECK-YU-Y_-NOT: -include-pch

// /Yc /Yu -- /Yc overrides /Yc if they both refer to the same file
// RUN: %clang_cl -Werror /Ycpchfile.h /Yupchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-YU %s
// 1. Build .pch file.
// CHECK-YC-YU: cc1
// CHECK-YC-YU: -emit-pch
// CHECK-YC-YU: -o
// CHECK-YC-YU: pchfile.pch
// 2. Use .pch file.
// CHECK-YC-YU: cc1
// CHECK-YC-YU: -emit-obj
// CHECK-YC-YU: -include-pch
// CHECK-YC-YU: pchfile.pch

// If /Yc /Yu refer to different files, semantics are pretty wonky.  Since this
// doesn't seem like something that's important in practice, just punt for now.
// RUN: %clang_cl -Werror /Ycfoo1.h /Yufoo2.h /FIfoo1.h /FIfoo2.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-YU-MISMATCH %s
// CHECK-YC-YU-MISMATCH: error: support for '/Yc' and '/Yu' with different filenames not implemented yet; flags ignored

// Similarly, punt on /Yc with more than one input file.
// RUN: %clang_cl -Werror /Ycfoo1.h /FIfoo1.h /c -### -- %s %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-MULTIINPUT %s
// CHECK-YC-MULTIINPUT: error: support for '/Yc' with more than one source file not implemented yet; flag ignored

// /Yc /Yu /Y-
// RUN: %clang_cl -Werror /Ycpchfile.h /Yupchfile.h /FIpchfile.h /Y- /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-YU-Y_ %s
// CHECK-YC-YU-Y_-NOT: -emit-pch
// CHECK-YC-YU-Y_-NOT: -include-pch

// Test computation of pch filename in various cases.

// /Yu /Fpout.pch => out.pch is filename
// RUN: %clang_cl -Werror /Yupchfile.h /FIpchfile.h /Fpout.pch /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YUFP1 %s
// Use .pch file, but don't build it.
// CHECK-YUFP1: -include-pch
// CHECK-YUFP1: out.pch

// /Yu /Fpout => out.pch is filename (.pch gets added if no extension present)
// RUN: %clang_cl -Werror /Yupchfile.h /FIpchfile.h /Fpout.pch /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YUFP2 %s
// Use .pch file, but don't build it.
// CHECK-YUFP2: -include-pch
// CHECK-YUFP2: out.pch

// /Yu /Fpout.bmp => out.bmp is filename (.pch not added when extension present)
// RUN: %clang_cl -Werror /Yupchfile.h /FIpchfile.h /Fpout.bmp /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YUFP3 %s
// Use .pch file, but don't build it.
// CHECK-YUFP3: -include-pch
// CHECK-YUFP3: out.bmp

// /Yusub/dir.h => sub/dir.pch
// RUN: %clang_cl -Werror /Yusub/pchfile.h /FIsub/pchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YUFP4 %s
// Use .pch file, but don't build it.
// CHECK-YUFP4: -include-pch
// CHECK-YUFP4: sub/pchfile.pch

// /Yudir.h /Isub => dir.pch
// RUN: %clang_cl -Werror /Yupchfile.h /FIpchfile.h /Isub /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YUFP5 %s
// Use .pch file, but don't build it.
// CHECK-YUFP5: -include-pch
// CHECK-YUFP5: pchfile.pch

// FIXME: /Fpdir: use dir/VCx0.pch when dir is directory, where x is major MSVS
// version in use.

// Spot-check one use of /Fp with /Yc too, else trust the /Yu test cases above
// also all assume to /Yc.
// RUN: %clang_cl -Werror /Ycpchfile.h /FIpchfile.h /Fpsub/file.pch /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YCFP %s
// 1. Build .pch file.
// CHECK-YCFP: cc1
// CHECK-YCFP: -emit-pch
// CHECK-YCFP: -o
// CHECK-YCFP: sub/file.pch
// 2. Use .pch file.
// CHECK-YCFP: cc1
// CHECK-YCFP: -emit-obj
// CHECK-YCFP: -include-pch
// CHECK-YCFP: sub/file.pch

// /Ycfoo2.h /FIfoo1.h /FIfoo2.h /FIfoo3.h
// => foo1 and foo2 go into pch, foo3 into main compilation
// /Yc
// RUN: %clang_cl -Werror /Ycfoo2.h /FIfoo1.h /FIfoo2.h /FIfoo3.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YCFIFIFI %s
// 1. Build .pch file: Includes foo1.h (but NOT foo3.h) and compiles foo2.h
// CHECK-YCFIFIFI: cc1
// CHECK-YCFIFIFI: -emit-pch
// CHECK-YCFIFIFI: -include
// CHECK-YCFIFIFI: foo1.h
// CHECK-YCFIFIFI-NOT: foo2.h
// CHECK-YCFIFIFI-NOT: foo3.h
// CHECK-YCFIFIFI: -o
// CHECK-YCFIFIFI: foo2.pch
// CHECK-YCFIFIFI: -x
// CHECK-YCFIFIFI: "c++"
// CHECK-YCFIFIFI: foo2.h
// 2. Use .pch file: Inlucdes foo2.pch and foo3.h
// CHECK-YCFIFIFI: cc1
// CHECK-YCFIFIFI: -emit-obj
// CHECK-YCFIFIFI-NOT: foo1.h
// CHECK-YCFIFIFI-NOT: foo2.h
// CHECK-YCFIFIFI: -include-pch
// CHECK-YCFIFIFI: foo2.pch
// CHECK-YCFIFIFI: -include
// CHECK-YCFIFIFI: foo3.h

// /Yucfoo2.h /FIfoo1.h /FIfoo2.h /FIfoo3.h
// => foo1 foo2 filtered out, foo3 into main compilation
// RUN: %clang_cl -Werror /Yufoo2.h /FIfoo1.h /FIfoo2.h /FIfoo3.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YUFIFIFI %s
// Use .pch file, but don't build it.
// CHECK-YUFIFIFI-NOT: -emit-pch
// CHECK-YUFIFIFI: cc1
// CHECK-YUFIFIFI: -emit-obj
// CHECK-YUFIFIFI-NOT: foo1.h
// CHECK-YUFIFIFI-NOT: foo2.h
// CHECK-YUFIFIFI: -include-pch
// CHECK-YUFIFIFI: foo2.pch
// CHECK-YUFIFIFI: -include
// CHECK-YUFIFIFI: foo3.h

// FIXME: Implement support for /Ycfoo.h / /Yufoo.h without /FIfoo.h
// RUN: %clang_cl -Werror /Ycfoo.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-NOFI %s
// CHECK-YC-NOFI: error: support for '/Yc' without a corresponding /FI flag not implemented yet; flag ignored
// RUN: %clang_cl -Werror /Yufoo.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU-NOFI %s
// CHECK-YU-NOFI: error: support for '/Yu' without a corresponding /FI flag not implemented yet; flag ignored

// /Yc and /FI relative to /I paths...
// The rules are:
// Yu/Yc and FI parameter must match exactly, else it's not found
// Must match literally exactly: /FI./foo.h /Ycfoo.h does _not_ work.
// However, the path can be relative to /I paths.
// FIXME: Update the error messages below once /FI is no longer required, but
// these test cases all should stay failures as they fail with cl.exe.

// Check that ./ isn't canonicalized away.
// RUN: %clang_cl -Werror /Ycpchfile.h /FI./pchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-I1 %s
// CHECK-YC-I1: support for '/Yc' without a corresponding /FI flag not implemented yet; flag ignored

// Check that ./ isn't canonicalized away.
// RUN: %clang_cl -Werror /Yc./pchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-I2 %s
// CHECK-YC-I2: support for '/Yc' without a corresponding /FI flag not implemented yet; flag ignored

// With an actual /I argument.
// RUN: %clang_cl -Werror /Ifoo /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-I3 %s
// 1. This writes pchfile.pch into the root dir, even if this will pick up
//    foo/pchfile.h
// CHECK-YC-I3: cc1
// CHECK-YC-I3: -emit-pch
// CHECK-YC-I3: -o
// CHECK-YC-I3: pchfile.pch
// 2. Use .pch file.
// CHECK-YC-I3: cc1
// CHECK-YC-I3: -emit-obj
// CHECK-YC-I3: -include-pch
// CHECK-YC-I3: pchfile.pch

// Check that ./ isn't canonicalized away for /Yu either.
// RUN: %clang_cl -Werror /Yupchfile.h /FI./pchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU-I1 %s
// CHECK-YU-I1: support for '/Yu' without a corresponding /FI flag not implemented yet; flag ignored

// But /FIfoo/bar.h /Ycfoo\bar.h does work, as does /FIfOo.h /Ycfoo.H
// FIXME: This part isn't implemented yet. The following two tests should not
// show an error but do regular /Yu handling.
// RUN: %clang_cl -Werror /YupchFILE.h /FI./pchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU-CASE %s
// CHECK-YU-CASE: support for '/Yu' without a corresponding /FI flag not implemented yet; flag ignored
// RUN: %clang_cl -Werror /Yu./pchfile.h /FI.\pchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU-SLASH %s
// CHECK-YU-SLASH: support for '/Yu' without a corresponding /FI flag not implemented yet; flag ignored

// cl.exe warns on multiple /Yc, /Yu, /Fp arguments, but clang-cl silently just
// uses the last one.  This is true for e.g. /Fo too, so not warning on this
// is self-consistent with clang-cl's flag handling.

// Interaction with /fallback

// /Yc /fallback => /Yc not passed on (but /FI is)
// RUN: %clang_cl -Werror /Ycpchfile.h /FIpchfile.h /Fpfoo.pch /fallback /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-FALLBACK %s
// Note that in /fallback builds, if creation of the pch fails the main compile
// does still run so that /fallback can have an effect (this part is not tested)
// CHECK-YC-FALLBACK: cc1
// CHECK-YC-FALLBACK: -emit-obj
// CHECK-YC-FALLBACK: -include-pch
// CHECK-YC-FALLBACK: foo.pch
// CHECK-YC-FALLBACK: ||
// CHECK-YC-FALLBACK: cl.exe
// CHECK-YC-FALLBACK-NOT: -include-pch
// CHECK-YC-FALLBACK-NOT: /Ycpchfile.h
// CHECK-YC-FALLBACK: /FIpchfile.h
// CHECK-YC-FALLBACK-NOT: /Fpfoo.pch

// /Yu /fallback => /Yu not passed on (but /FI is)
// RUN: %clang_cl -Werror /Yupchfile.h /FIpchfile.h /Fpfoo.pch /fallback /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU-FALLBACK %s
// CHECK-YU-FALLBACK-NOT: -emit-pch
// CHECK-YU-FALLBACK: cc1
// CHECK-YU-FALLBACK: -emit-obj
// CHECK-YU-FALLBACK: -include-pch
// CHECK-YU-FALLBACK: foo.pch
// CHECK-YU-FALLBACK: ||
// CHECK-YU-FALLBACK: cl.exe
// CHECK-YU-FALLBACK-NOT: -include-pch
// CHECK-YU-FALLBACK-NOT: /Yupchfile.h
// CHECK-YU-FALLBACK: /FIpchfile.h
// CHECK-YU-FALLBACK-NOT: /Fpfoo.pch

// /FI without /Yu => pch file not used, even if it exists (different from
// -include, which picks up .gch files if they exist).
// RUN: touch %t.pch
// RUN: %clang_cl -Werror /FI%t.pch /Fp%t.pch /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-FI %s
// CHECK-FI-NOT: -include-pch
// CHECK-FI: -include

// Test interaction of /Yc with language mode flags.

// If /TC changes the input language to C, a c pch file should be produced.
// RUN: %clang_cl /TC -Werror /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YCTC %s
// CHECK-YCTC: cc1
// CHECK-YCTC: -emit-pch
// CHECK-YCTC: -o
// CHECK-YCTC: pchfile.pch
// CHECK-YCTC: -x
// CHECK-YCTC: "c"

// Also check lower-case /Tc variant.
// RUN: %clang_cl -Werror /Ycpchfile.h /FIpchfile.h /c -### /Tc%s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YCTc %s
// CHECK-YCTc: cc1
// CHECK-YCTc: -emit-pch
// CHECK-YCTc: -o
// CHECK-YCTc: pchfile.pch
// CHECK-YCTc: -x
// CHECK-YCTc: "c"

// Don't crash when a non-source file is passed.
// RUN: %clang_cl -Werror /Ycpchfile.h /FIpchfile.h /c -### -- %S/Inputs/file.prof 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NoSource %s
// CHECK-NoSource: file.prof:{{.*}}input unused

// ...but if an explicit file turns the file into a source file, handle it:
// RUN: %clang_cl /TP -Werror /Ycpchfile.h /FIpchfile.h /c -### -- %S/Inputs/file.prof 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NoSourceTP %s
// CHECK-NoSourceTP: cc1
// CHECK-NoSourceTP: -emit-pch
// CHECK-NoSourceTP: -o
// CHECK-NoSourceTP: pchfile.pch
// CHECK-NoSourceTP: -x
// CHECK-NoSourceTP: "c++"
