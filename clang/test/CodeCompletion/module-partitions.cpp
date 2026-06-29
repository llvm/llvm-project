// RUN: rm -rf %t && mkdir %t

// Set up a partition of module M.
// RUN: echo 'export module M:Part; export int mpart_func();' > %t/m-part.cppm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/m-part.cppm -o %t/M-Part.pcm

// Set up module N with a partition.
// RUN: echo 'export module N:OtherPart; export int npart_func();' > %t/n-otherpart.cppm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/n-otherpart.cppm -o %t/N-OtherPart.pcm
// RUN: printf 'export module N;\nexport import :OtherPart;\n' > %t/n.cppm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/n.cppm -fmodule-file=N:OtherPart=%t/N-OtherPart.pcm -o %t/N.pcm

// Complete at the module name position in an "import" inside module M.
// Own partition (M:Part) should be suggested; another module's partition
// (N:OtherPart) should be filtered out.
// RUN: %clang_cc1 -std=c++20 -code-completion-at=%s:%(line+6):8 %s \
// RUN:   -fmodule-file=M:Part=%t/M-Part.pcm \
// RUN:   -fmodule-file=N=%t/N.pcm \
// RUN:   -fmodule-file=N:OtherPart=%t/N-OtherPart.pcm | FileCheck %s

export module M;
import ;

// CHECK-NOT: OtherPart
// CHECK: M:Part
// CHECK-NOT: OtherPart
