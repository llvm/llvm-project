// REQUIRES: zlib || zstd
// REQUIRES: shell
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo '//////////////////////////////////////////////////////////////////////' > %t/a.h
// RUN: cat %t/a.h %t/a.h %t/a.h %t/a.h > %t/b.h
// RUN: cat %t/b.h %t/b.h %t/b.h %t/b.h > %t/a.h
// RUN: cat %t/a.h %t/a.h %t/a.h %t/a.h > %t/b.h
// RUN: cat %t/b.h %t/b.h %t/b.h %t/b.h > %t/a.h
// RUN: cat %t/a.h %t/a.h %t/a.h %t/a.h > %t/b.h
// RUN: cat %t/b.h %t/b.h %t/b.h %t/b.h > %t/a.h
// RUN: cat %t/a.h %t/a.h %t/a.h %t/a.h > %t/b.h
// RUN: cat %t/b.h %t/b.h %t/b.h %t/b.h > %t/a.h
// RUN: echo 'module a { header "a.h" }' > %t/modulemap
//
// RUN: %clang_cc1 -fmodules -I%t -fmodules-cache-path=%t -fmodule-name=a -emit-module %t/modulemap -fmodules-embed-all-files -o %t/a.pcm
//
// The above embeds ~4.5MB of highly-predictable /s and \ns into the pcm file.
// Check that the resulting file is under 40KB:
//
// RUN: wc -c %t/a.pcm | FileCheck --check-prefix=CHECK-SIZE %s
// CHECK-SIZE: {{(^|[^0-9])[123][0-9][0-9][0-9][0-9]($|[^0-9])}}
