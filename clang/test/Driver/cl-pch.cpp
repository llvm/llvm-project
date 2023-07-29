// Note: %s and %S must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// /Yc
// RUN: %clang_cl -Werror /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC %s
// 1. Build .pch file.
// CHECK-YC: cc1
// CHECK-YC: -emit-pch
// CHECK-YC: -building-pch-with-obj
// CHECK-YC: -o
// CHECK-YC: pchfile.pch
// CHECK-YC: -x
// CHECK-YC: "c++-header"
// 2. Use .pch file.
// CHECK-YC: cc1
// CHECK-YC: -emit-obj
// CHECK-YC: -building-pch-with-obj
// CHECK-YC: -include-pch
// CHECK-YC: pchfile.pch

// /Yc /Fo
// /Fo overrides the .obj output filename, but not the .pch filename
// RUN: %clang_cl -Werror /Fomyobj.obj /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YCO %s
// 1. Build .pch file.
// CHECK-YCO: cc1
// CHECK-YCO: -emit-pch
// CHECK-YCO: -building-pch-with-obj
// CHECK-YCO: -o
// CHECK-YCO: pchfile.pch
// 2. Use .pch file.
// CHECK-YCO: cc1
// CHECK-YCO: -emit-obj
// CHECK-YCO: -building-pch-with-obj
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
// CHECK-YU-NOT: -building-pch-with-obj
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
// CHECK-YC-YU: -building-pch-with-obj
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
// CHECK-YCFIFIFI: -pch-through-header=foo2.h
// CHECK-YCFIFIFI: -include
// CHECK-YCFIFIFI: foo1.h
// CHECK-YCFIFIFI: -include
// CHECK-YCFIFIFI: foo2.h
// CHECK-YCFIFIFI: -include
// CHECK-YCFIFIFI: foo3.h
// CHECK-YCFIFIFI: -o
// CHECK-YCFIFIFI: foo2.pch
// CHECK-YCFIFIFI: -x
// CHECK-YCFIFIFI: "c++-header"
// CHECK-YCFIFIFI: cl-pch.cpp
// 2. Use .pch file: Inlucdes foo2.pch and foo3.h
// CHECK-YCFIFIFI: cc1
// CHECK-YCFIFIFI: -emit-obj
// CHECK-YCFIFIFI: -include-pch
// CHECK-YCFIFIFI: foo2.pch
// CHECK-YCFIFIFI: -pch-through-header=foo2.h
// CHECK-YCFIFIFI: -include
// CHECK-YCFIFIFI: foo1.h
// CHECK-YCFIFIFI: -include
// CHECK-YCFIFIFI: foo2.h
// CHECK-YCFIFIFI: -include
// CHECK-YCFIFIFI: foo3.h
// CHECK-YCFIFIFI: -o
// CHECK-YCFIFIFI: cl-pch.obj
// CHECK-YCFIFIFI: -x
// CHECK-YCFIFIFI: "c++"
// CHECK-YCFIFIFI: cl-pch.cpp

// /Yufoo2.h /FIfoo1.h /FIfoo2.h /FIfoo3.h
// => foo1 foo2 filtered out, foo3 into main compilation
// RUN: %clang_cl -Werror /Yufoo2.h /FIfoo1.h /FIfoo2.h /FIfoo3.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YUFIFIFI %s
// Use .pch file, but don't build it.
// CHECK-YUFIFIFI-NOT: -emit-pch
// CHECK-YUFIFIFI: cc1
// CHECK-YUFIFIFI: -emit-obj
// CHECK-YUFIFIFI: -include-pch
// CHECK-YUFIFIFI: foo2.pch
// CHECK-YUFIFIFI: -pch-through-header=foo2.h
// CHECK-YUFIFIFI: -include
// CHECK-YUFIFIFI: foo1.h
// CHECK-YUFIFIFI: -include
// CHECK-YUFIFIFI: foo2.h
// CHECK-YUFIFIFI: -include
// CHECK-YUFIFIFI: foo3.h

// Test /Ycfoo.h / /Yufoo.h without /FIfoo.h
// RUN: %clang_cl -Werror /Ycfoo.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-NOFI %s
// 1. Precompile
// CHECK-YC-NOFI: cc1
// CHECK-YC-NOFI: -emit-pch
// CHECK-YC-NOFI: -pch-through-header=foo.h
// CHECK-YC-NOFI: -o
// CHECK-YC-NOFI: foo.pch
// CHECK-YC-NOFI: -x
// CHECK-YC-NOFI: c++-header
// CHECK-YC-NOFI: cl-pch.cpp
// 2. Build PCH object
// CHECK-YC-NOFI: cc1
// CHECK-YC-NOFI: -emit-obj
// CHECK-YC-NOFI: -include-pch
// CHECK-YC-NOFI: foo.pch
// CHECK-YC-NOFI: -pch-through-header=foo.h
// CHECK-YC-NOFI: -x
// CHECK-YC-NOFI: c++
// CHECK-YC-NOFI: cl-pch.cpp
// RUN: %clang_cl -Werror /Yufoo.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU-NOFI %s
// CHECK-YU-NOFI: cc1
// CHECK-YU-NOFI: -emit-obj
// CHECK-YU-NOFI: -include-pch
// CHECK-YU-NOFI: foo.pch
// CHECK-YU-NOFI: -pch-through-header=foo.h
// CHECK-YU-NOFI: -x
// CHECK-YU-NOFI: c++
// CHECK-YU-NOFI: cl-pch.cpp

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

// But /FIfoo/bar.h /Ycfoo\bar.h does work, as does /FIfOo.h /Ycfoo.H
// RUN: %clang_cl -Werror /YupchFILE.h /FI./pchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU-CASE %s
// CHECK-YU-CASE: -pch-through-header=pchFILE.h
// CHECK-YU-CASE: -include
// CHECK-YU-CASE: "./pchfile.h"
// RUN: %clang_cl -Werror /Yu./pchfile.h /FI.\\pchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU-SLASH %s
// CHECK-YU-SLASH: -pch-through-header=./pchfile.h
// CHECK-YU-SLASH: -include
// CHECK-YU-SLASH: ".{{[/\\]+}}pchfile.h"

// /Yc without an argument creates a PCH from the code before #pragma hdrstop.
// /Yu without an argument uses a PCH and starts compiling after the
// #pragma hdrstop.
// RUN: %clang_cl -Werror /Yc /Fpycnoarg.pch /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-NOARG %s
// 1. Create .pch file
// CHECK-YC-NOARG: cc1
// CHECK-YC-NOARG: -emit-pch
// CHECK-YC-NOARG: -pch-through-hdrstop-create
// CHECK-YC-NOARG: -o
// CHECK-YC-NOARG: ycnoarg.pch
// CHECK-YC-NOARG: -x
// CHECK-YC-NOARG: "c++-header"
// CHECK-YC-NOARG: cl-pch.cpp
// 2. Use .pch file: Includes ycnoarg.pch
// CHECK-YC-NOARG: cc1
// CHECK-YC-NOARG: -emit-obj
// CHECK-YC-NOARG: -include-pch
// CHECK-YC-NOARG: ycnoarg.pch
// CHECK-YC-NOARG: -pch-through-hdrstop-create
// CHECK-YC-NOARG: -o
// CHECK-YC-NOARG: cl-pch.obj
// CHECK-YC-NOARG: -x
// CHECK-YC-NOARG: "c++"
// CHECK-YC-NOARG: cl-pch.cpp

// RUN: %clang_cl -Werror /Yu /Fpycnoarg.pch /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU-NOARG %s
// Use .pch file, but don't build it.
// CHECK-YU-NOARG-NOT: -emit-pch
// CHECK-YU-NOARG: cc1
// CHECK-YU-NOARG: -emit-obj
// CHECK-YU-NOARG: -include-pch
// CHECK-YU-NOARG: ycnoarg.pch
// CHECK-YU-NOARG: -pch-through-hdrstop-use
// CHECK-YU-NOARG: -o
// CHECK-YU-NOARG: cl-pch.obj
// CHECK-YU-NOARG: -x
// CHECK-YU-NOARG: "c++"
// CHECK-YU-NOARG: cl-pch.cpp

// /Yc with no argument and no /FP
// RUN: %clang_cl -Werror /Yc /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-NOARG-NOFP %s
// 1. Create .pch file
// CHECK-YC-NOARG-NOFP: cc1
// CHECK-YC-NOARG-NOFP: -emit-pch
// CHECK-YC-NOARG-NOFP: -pch-through-hdrstop-create
// CHECK-YC-NOARG-NOFP: -o
// CHECK-YC-NOARG-NOFP: cl-pch.pch
// CHECK-YC-NOARG-NOFP: -x
// CHECK-YC-NOARG-NOFP: "c++-header"
// CHECK-YC-NOARG-NOFP: cl-pch.cpp
// 2. Use .pch file: Includes cl-pch.pch
// CHECK-YC-NOARG-NOFP: cc1
// CHECK-YC-NOARG-NOFP: -emit-obj
// CHECK-YC-NOARG-NOFP: -include-pch
// CHECK-YC-NOARG-NOFP: cl-pch.pch
// CHECK-YC-NOARG-NOFP: -pch-through-hdrstop-create
// CHECK-YC-NOARG-NOFP: -o
// CHECK-YC-NOARG-NOFP: cl-pch.obj
// CHECK-YC-NOARG-NOFP: -x
// CHECK-YC-NOARG-NOFP: "c++"
// CHECK-YC-NOARG-NOFP: cl-pch.cpp

// cl.exe warns on multiple /Yc, /Yu, /Fp arguments, but clang-cl silently just
// uses the last one.  This is true for e.g. /Fo too, so not warning on this
// is self-consistent with clang-cl's flag handling.

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

// ...but if an explicit flag turns the file into a source file, handle it:
// RUN: %clang_cl /TP -Werror /Ycpchfile.h /FIpchfile.h /c -### -- %S/Inputs/file.prof 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-NoSourceTP %s
// CHECK-NoSourceTP: cc1
// CHECK-NoSourceTP: -emit-pch
// CHECK-NoSourceTP: -o
// CHECK-NoSourceTP: pchfile.pch
// CHECK-NoSourceTP: -x
// CHECK-NoSourceTP: "c++"

// If only preprocessing, PCH options are ignored.
// RUN: %clang_cl /P /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-P %s
// CHECK-YC-P-NOT: -emit-pch
// CHECK-YC-P-NOT: -include-pch

// RUN: %clang_cl /E /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YC-E %s
// CHECK-YC-E-NOT: -emit-pch
// CHECK-YC-E-NOT: -include-pch

// RUN: %clang_cl /P /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU-P %s
// CHECK-YU-P-NOT: -emit-pch
// CHECK-YU-P-NOT: -include-pch

// RUN: %clang_cl /E /Ycpchfile.h /FIpchfile.h /c -### -- %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-YU-E %s
// CHECK-YU-E-NOT: -emit-pch
// CHECK-YU-E-NOT: -include-pch
