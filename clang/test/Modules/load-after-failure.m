// RUN: rm -rf %t
// RUN: mkdir -p %t

// RUN: echo '@import B;' > %t/A.h
// RUN: echo '@import C;' > %t/B.h
// RUN: echo '@import D;' >> %t/B.h
// RUN: echo '// C.h' > %t/C.h
// RUN: echo '// D.h' > %t/D.h
// RUN: echo 'module A { header "A.h" }' > %t/module.modulemap
// RUN: echo 'module B { header "B.h" }' >> %t/module.modulemap
// RUN: echo 'module C { header "C.h" }' >> %t/module.modulemap
// RUN: echo 'module D { header "D.h" }' >> %t/module.modulemap

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %t %s -verify
// RUN: echo " " >> %t/D.h
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %t %s -verify
// expected-no-diagnostics


@import C;
@import A;
@import C;
// When compiling A, C will be be loaded then removed when D fails. Ensure
// this does not cause problems importing C again later.
