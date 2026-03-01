// RUN: rm -rf %t
// RUN: mkdir %t
//
// Build module map with 40 modules; two layers with importing and re-exporting
// the previous layer.
//
// RUN: echo 'module a0 { header "a0.h" export * }' > %t/module.modulemap
// RUN: echo 'module b0 { header "b0.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a1 { header "a1.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b1 { header "b1.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a2 { header "a2.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b2 { header "b2.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a3 { header "a3.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b3 { header "b3.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a4 { header "a4.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b4 { header "b4.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a5 { header "a5.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b5 { header "b5.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a6 { header "a6.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b6 { header "b6.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a7 { header "a7.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b7 { header "b7.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a8 { header "a8.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b8 { header "b8.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a9 { header "a9.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b9 { header "b9.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a10 { header "a10.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b10 { header "b10.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a11 { header "a11.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b11 { header "b11.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a12 { header "a12.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b12 { header "b12.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a13 { header "a13.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b13 { header "b13.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a14 { header "a14.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b14 { header "b14.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a15 { header "a15.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b15 { header "b15.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a16 { header "a16.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b16 { header "b16.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a17 { header "a17.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b17 { header "b17.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a18 { header "a18.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b18 { header "b18.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a19 { header "a19.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b19 { header "b19.h" export * }' >> %t/module.modulemap
// RUN: echo 'module a20 { header "a20.h" export * }' >> %t/module.modulemap
// RUN: echo 'module b20 { header "b20.h" export * }' >> %t/module.modulemap
//
// Build the corresponding headers.
//
// RUN: echo 'template<int> struct X {}; extern X<0> x0;' > %t/x.h
// RUN: echo 'extern int n;' > %t/a0.h
// RUN: cp %t/a0.h %t/b0.h
// RUN: echo '#include "x.h"' > %t/a1.h
// RUN: echo '#include "a0.h"' >> %t/a1.h
// RUN: echo '#include "b0.h"' >> %t/a1.h
// RUN: cp %t/a1.h %t/b1.h
// RUN: echo '#include "x.h"' > %t/a2.h
// RUN: echo '#include "a1.h"' >> %t/a2.h
// RUN: echo '#include "b1.h"' >> %t/a2.h
// RUN: cp %t/a2.h %t/b2.h
// RUN: echo '#include "x.h"' > %t/a3.h
// RUN: echo '#include "a2.h"' >> %t/a3.h
// RUN: echo '#include "b2.h"' >> %t/a3.h
// RUN: cp %t/a3.h %t/b3.h
// RUN: echo '#include "x.h"' > %t/a4.h
// RUN: echo '#include "a3.h"' >> %t/a4.h
// RUN: echo '#include "b3.h"' >> %t/a4.h
// RUN: cp %t/a4.h %t/b4.h
// RUN: echo '#include "x.h"' > %t/a5.h
// RUN: echo '#include "a4.h"' >> %t/a5.h
// RUN: echo '#include "b4.h"' >> %t/a5.h
// RUN: cp %t/a5.h %t/b5.h
// RUN: echo '#include "x.h"' > %t/a6.h
// RUN: echo '#include "a5.h"' >> %t/a6.h
// RUN: echo '#include "b5.h"' >> %t/a6.h
// RUN: cp %t/a6.h %t/b6.h
// RUN: echo '#include "x.h"' > %t/a7.h
// RUN: echo '#include "a6.h"' >> %t/a7.h
// RUN: echo '#include "b6.h"' >> %t/a7.h
// RUN: cp %t/a7.h %t/b7.h
// RUN: echo '#include "x.h"' > %t/a8.h
// RUN: echo '#include "a7.h"' >> %t/a8.h
// RUN: echo '#include "b7.h"' >> %t/a8.h
// RUN: cp %t/a8.h %t/b8.h
// RUN: echo '#include "x.h"' > %t/a9.h
// RUN: echo '#include "a8.h"' >> %t/a9.h
// RUN: echo '#include "b8.h"' >> %t/a9.h
// RUN: cp %t/a9.h %t/b9.h
// RUN: echo '#include "x.h"' > %t/a10.h
// RUN: echo '#include "a9.h"' >> %t/a10.h
// RUN: echo '#include "b9.h"' >> %t/a10.h
// RUN: cp %t/a10.h %t/b10.h
// RUN: echo '#include "x.h"' > %t/a11.h
// RUN: echo '#include "a10.h"' >> %t/a11.h
// RUN: echo '#include "b10.h"' >> %t/a11.h
// RUN: cp %t/a11.h %t/b11.h
// RUN: echo '#include "x.h"' > %t/a12.h
// RUN: echo '#include "a11.h"' >> %t/a12.h
// RUN: echo '#include "b11.h"' >> %t/a12.h
// RUN: cp %t/a12.h %t/b12.h
// RUN: echo '#include "x.h"' > %t/a13.h
// RUN: echo '#include "a12.h"' >> %t/a13.h
// RUN: echo '#include "b12.h"' >> %t/a13.h
// RUN: cp %t/a13.h %t/b13.h
// RUN: echo '#include "x.h"' > %t/a14.h
// RUN: echo '#include "a13.h"' >> %t/a14.h
// RUN: echo '#include "b13.h"' >> %t/a14.h
// RUN: cp %t/a14.h %t/b14.h
// RUN: echo '#include "x.h"' > %t/a15.h
// RUN: echo '#include "a14.h"' >> %t/a15.h
// RUN: echo '#include "b14.h"' >> %t/a15.h
// RUN: cp %t/a15.h %t/b15.h
// RUN: echo '#include "x.h"' > %t/a16.h
// RUN: echo '#include "a15.h"' >> %t/a16.h
// RUN: echo '#include "b15.h"' >> %t/a16.h
// RUN: cp %t/a16.h %t/b16.h
// RUN: echo '#include "x.h"' > %t/a17.h
// RUN: echo '#include "a16.h"' >> %t/a17.h
// RUN: echo '#include "b16.h"' >> %t/a17.h
// RUN: cp %t/a17.h %t/b17.h
// RUN: echo '#include "x.h"' > %t/a18.h
// RUN: echo '#include "a17.h"' >> %t/a18.h
// RUN: echo '#include "b17.h"' >> %t/a18.h
// RUN: cp %t/a18.h %t/b18.h
// RUN: echo '#include "x.h"' > %t/a19.h
// RUN: echo '#include "a18.h"' >> %t/a19.h
// RUN: echo '#include "b18.h"' >> %t/a19.h
// RUN: cp %t/a19.h %t/b19.h
// RUN: echo '#include "x.h"' > %t/a20.h
// RUN: echo '#include "a19.h"' >> %t/a20.h
// RUN: echo '#include "b19.h"' >> %t/a20.h
// RUN: cp %t/a20.h %t/b20.h
//
// Explicitly build all the modules.
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a0 -x c++ -emit-module %t/module.modulemap -o %t/a0.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b0 -x c++ -emit-module %t/module.modulemap -o %t/b0.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a1 -x c++ -emit-module %t/module.modulemap -o %t/a1.pcm -fmodule-file=%t/a0.pcm -fmodule-file=%t/b0.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b1 -x c++ -emit-module %t/module.modulemap -o %t/b1.pcm -fmodule-file=%t/a0.pcm -fmodule-file=%t/b0.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a2 -x c++ -emit-module %t/module.modulemap -o %t/a2.pcm -fmodule-file=%t/a1.pcm -fmodule-file=%t/b1.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b2 -x c++ -emit-module %t/module.modulemap -o %t/b2.pcm -fmodule-file=%t/a1.pcm -fmodule-file=%t/b1.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a3 -x c++ -emit-module %t/module.modulemap -o %t/a3.pcm -fmodule-file=%t/a2.pcm -fmodule-file=%t/b2.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b3 -x c++ -emit-module %t/module.modulemap -o %t/b3.pcm -fmodule-file=%t/a2.pcm -fmodule-file=%t/b2.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a4 -x c++ -emit-module %t/module.modulemap -o %t/a4.pcm -fmodule-file=%t/a3.pcm -fmodule-file=%t/b3.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b4 -x c++ -emit-module %t/module.modulemap -o %t/b4.pcm -fmodule-file=%t/a3.pcm -fmodule-file=%t/b3.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a5 -x c++ -emit-module %t/module.modulemap -o %t/a5.pcm -fmodule-file=%t/a4.pcm -fmodule-file=%t/b4.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b5 -x c++ -emit-module %t/module.modulemap -o %t/b5.pcm -fmodule-file=%t/a4.pcm -fmodule-file=%t/b4.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a6 -x c++ -emit-module %t/module.modulemap -o %t/a6.pcm -fmodule-file=%t/a5.pcm -fmodule-file=%t/b5.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b6 -x c++ -emit-module %t/module.modulemap -o %t/b6.pcm -fmodule-file=%t/a5.pcm -fmodule-file=%t/b5.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a7 -x c++ -emit-module %t/module.modulemap -o %t/a7.pcm -fmodule-file=%t/a6.pcm -fmodule-file=%t/b6.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b7 -x c++ -emit-module %t/module.modulemap -o %t/b7.pcm -fmodule-file=%t/a6.pcm -fmodule-file=%t/b6.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a8 -x c++ -emit-module %t/module.modulemap -o %t/a8.pcm -fmodule-file=%t/a7.pcm -fmodule-file=%t/b7.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b8 -x c++ -emit-module %t/module.modulemap -o %t/b8.pcm -fmodule-file=%t/a7.pcm -fmodule-file=%t/b7.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a9 -x c++ -emit-module %t/module.modulemap -o %t/a9.pcm -fmodule-file=%t/a8.pcm -fmodule-file=%t/b8.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b9 -x c++ -emit-module %t/module.modulemap -o %t/b9.pcm -fmodule-file=%t/a8.pcm -fmodule-file=%t/b8.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a10 -x c++ -emit-module %t/module.modulemap -o %t/a10.pcm -fmodule-file=%t/a9.pcm -fmodule-file=%t/b9.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b10 -x c++ -emit-module %t/module.modulemap -o %t/b10.pcm -fmodule-file=%t/a9.pcm -fmodule-file=%t/b9.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a11 -x c++ -emit-module %t/module.modulemap -o %t/a11.pcm -fmodule-file=%t/a10.pcm -fmodule-file=%t/b10.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b11 -x c++ -emit-module %t/module.modulemap -o %t/b11.pcm -fmodule-file=%t/a10.pcm -fmodule-file=%t/b10.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a12 -x c++ -emit-module %t/module.modulemap -o %t/a12.pcm -fmodule-file=%t/a11.pcm -fmodule-file=%t/b11.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b12 -x c++ -emit-module %t/module.modulemap -o %t/b12.pcm -fmodule-file=%t/a11.pcm -fmodule-file=%t/b11.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a13 -x c++ -emit-module %t/module.modulemap -o %t/a13.pcm -fmodule-file=%t/a12.pcm -fmodule-file=%t/b12.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b13 -x c++ -emit-module %t/module.modulemap -o %t/b13.pcm -fmodule-file=%t/a12.pcm -fmodule-file=%t/b12.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a14 -x c++ -emit-module %t/module.modulemap -o %t/a14.pcm -fmodule-file=%t/a13.pcm -fmodule-file=%t/b13.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b14 -x c++ -emit-module %t/module.modulemap -o %t/b14.pcm -fmodule-file=%t/a13.pcm -fmodule-file=%t/b13.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a15 -x c++ -emit-module %t/module.modulemap -o %t/a15.pcm -fmodule-file=%t/a14.pcm -fmodule-file=%t/b14.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b15 -x c++ -emit-module %t/module.modulemap -o %t/b15.pcm -fmodule-file=%t/a14.pcm -fmodule-file=%t/b14.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a16 -x c++ -emit-module %t/module.modulemap -o %t/a16.pcm -fmodule-file=%t/a15.pcm -fmodule-file=%t/b15.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b16 -x c++ -emit-module %t/module.modulemap -o %t/b16.pcm -fmodule-file=%t/a15.pcm -fmodule-file=%t/b15.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a17 -x c++ -emit-module %t/module.modulemap -o %t/a17.pcm -fmodule-file=%t/a16.pcm -fmodule-file=%t/b16.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b17 -x c++ -emit-module %t/module.modulemap -o %t/b17.pcm -fmodule-file=%t/a16.pcm -fmodule-file=%t/b16.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a18 -x c++ -emit-module %t/module.modulemap -o %t/a18.pcm -fmodule-file=%t/a17.pcm -fmodule-file=%t/b17.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b18 -x c++ -emit-module %t/module.modulemap -o %t/b18.pcm -fmodule-file=%t/a17.pcm -fmodule-file=%t/b17.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a19 -x c++ -emit-module %t/module.modulemap -o %t/a19.pcm -fmodule-file=%t/a18.pcm -fmodule-file=%t/b18.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b19 -x c++ -emit-module %t/module.modulemap -o %t/b19.pcm -fmodule-file=%t/a18.pcm -fmodule-file=%t/b18.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=a20 -x c++ -emit-module %t/module.modulemap -o %t/a20.pcm -fmodule-file=%t/a19.pcm -fmodule-file=%t/b19.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fmodule-name=b20 -x c++ -emit-module %t/module.modulemap -o %t/b20.pcm -fmodule-file=%t/a19.pcm -fmodule-file=%t/b19.pcm
//
// Build, using all the modules.
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%t -fsyntax-only %s \
// RUN:            -fmodule-file=%t/a0.pcm -fmodule-file=%t/b0.pcm \
// RUN:            -fmodule-file=%t/a1.pcm -fmodule-file=%t/b1.pcm \
// RUN:            -fmodule-file=%t/a2.pcm -fmodule-file=%t/b2.pcm \
// RUN:            -fmodule-file=%t/a3.pcm -fmodule-file=%t/b3.pcm \
// RUN:            -fmodule-file=%t/a4.pcm -fmodule-file=%t/b4.pcm \
// RUN:            -fmodule-file=%t/a5.pcm -fmodule-file=%t/b5.pcm \
// RUN:            -fmodule-file=%t/a6.pcm -fmodule-file=%t/b6.pcm \
// RUN:            -fmodule-file=%t/a7.pcm -fmodule-file=%t/b7.pcm \
// RUN:            -fmodule-file=%t/a8.pcm -fmodule-file=%t/b8.pcm \
// RUN:            -fmodule-file=%t/a9.pcm -fmodule-file=%t/b9.pcm \
// RUN:            -fmodule-file=%t/a10.pcm -fmodule-file=%t/b10.pcm \
// RUN:            -fmodule-file=%t/a11.pcm -fmodule-file=%t/b11.pcm \
// RUN:            -fmodule-file=%t/a12.pcm -fmodule-file=%t/b12.pcm \
// RUN:            -fmodule-file=%t/a13.pcm -fmodule-file=%t/b13.pcm \
// RUN:            -fmodule-file=%t/a14.pcm -fmodule-file=%t/b14.pcm \
// RUN:            -fmodule-file=%t/a15.pcm -fmodule-file=%t/b15.pcm \
// RUN:            -fmodule-file=%t/a16.pcm -fmodule-file=%t/b16.pcm \
// RUN:            -fmodule-file=%t/a17.pcm -fmodule-file=%t/b17.pcm \
// RUN:            -fmodule-file=%t/a18.pcm -fmodule-file=%t/b18.pcm \
// RUN:            -fmodule-file=%t/a19.pcm -fmodule-file=%t/b19.pcm \
// RUN:            -fmodule-file=%t/a20.pcm -fmodule-file=%t/b20.pcm

#include "a20.h"
#include "b20.h"
int k = n;
