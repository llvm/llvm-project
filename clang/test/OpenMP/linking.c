// Test the that the driver produces reasonable linker invocations with
// -fopenmp or -fopenmp|libgomp.
//
// FIXME: Replace DEFAULT_OPENMP_LIB below with the value chosen at configure time.
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp -target i386-unknown-linux -rtlib=platform \
// RUN:   | FileCheck --check-prefix=CHECK-LD-32 %s
// CHECK-LD-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-32: "-l[[DEFAULT_OPENMP_LIB:[^"]*]]"
// CHECK-LD-32: "-lpthread" "-latomic" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp -target x86_64-unknown-linux -rtlib=platform \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64 %s
// CHECK-LD-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-64: "-l[[DEFAULT_OPENMP_LIB:[^"]*]]"
// CHECK-LD-64: "-lpthread" "-latomic" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp=libgomp -target i386-unknown-linux -rtlib=platform \
// RUN:   | FileCheck --check-prefix=CHECK-GOMP-LD-32 %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 -fopenmp-simd -target i386-unknown-linux -rtlib=platform | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: lgomp
// SIMD-ONLY2-NOT: lomp
// SIMD-ONLY2-NOT: liomp
// CHECK-GOMP-LD-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-GOMP-LD-32: "-lgomp" "-lrt"
// CHECK-GOMP-LD-32: "-lpthread" "-latomic" "-lc"

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 -fopenmp-simd -target i386-unknown-linux -rtlib=platform | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: lgomp
// SIMD-ONLY2-NOT: lomp
// SIMD-ONLY2-NOT: liomp
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp=libgomp -target x86_64-unknown-linux -rtlib=platform \
// RUN:   | FileCheck --check-prefix=CHECK-GOMP-LD-64 %s
// CHECK-GOMP-LD-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-GOMP-LD-64: "-lgomp" "-lrt"
// CHECK-GOMP-LD-64: "-lpthread" "-latomic" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp -target i386-unknown-linux -rtlib=platform \
// RUN:   | FileCheck --check-prefix=CHECK-IOMP5-LD-32 %s
// CHECK-IOMP5-LD-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-IOMP5-LD-32: "-l[[DEFAULT_OPENMP_LIB:[^"]*]]"
// CHECK-IOMP5-LD-32: "-lpthread" "-latomic" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp -target x86_64-unknown-linux -rtlib=platform \
// RUN:   | FileCheck --check-prefix=CHECK-IOMP5-LD-64 %s
// CHECK-IOMP5-LD-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-IOMP5-LD-64: "-l[[DEFAULT_OPENMP_LIB:[^"]*]]"
// CHECK-IOMP5-LD-64: "-lpthread" "-latomic" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp=lib -target i386-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-LIB-LD-32 %s
// CHECK-LIB-LD-32: error: unsupported argument 'lib' to option 'fopenmp='
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp=lib -target x86_64-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-LIB-LD-64 %s
// CHECK-LIB-LD-64: error: unsupported argument 'lib' to option 'fopenmp='
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp -fopenmp=libgomp -target i386-unknown-linux \
// RUN:     -rtlib=platform \
// RUN:   | FileCheck --check-prefix=CHECK-LD-OVERRIDE-32 %s
// CHECK-LD-OVERRIDE-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-OVERRIDE-32: "-lgomp" "-lrt"
// CHECK-LD-OVERRIDE-32: "-lpthread" "-latomic" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp -fopenmp=libgomp -target x86_64-unknown-linux \
// RUN:     -rtlib=platform \
// RUN:   | FileCheck --check-prefix=CHECK-LD-OVERRIDE-64 %s
// CHECK-LD-OVERRIDE-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-OVERRIDE-64: "-lgomp" "-lrt"
// CHECK-LD-OVERRIDE-64: "-lpthread" "-latomic" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp=libomp -target x86_64-msvc-win32 -rtlib=platform \
// RUN:   | FileCheck --check-prefix=CHECK-MSVC-LINK-64 %s
// CHECK-MSVC-LINK-64: link.exe
// CHECK-MSVC-LINK-64-SAME: -nodefaultlib:vcomp.lib
// CHECK-MSVC-LINK-64-SAME: -nodefaultlib:vcompd.lib
// CHECK-MSVC-LINK-64-SAME: -libpath:{{.+}}/../lib
// CHECK-MSVC-LINK-64-SAME: -defaultlib:libomp.lib

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 -fopenmp-simd -target x86_64-msvc-win32 -rtlib=platform | FileCheck --check-prefix SIMD-ONLY11 %s
// SIMD-ONLY11-NOT: libiomp
// SIMD-ONLY11-NOT: libomp
// SIMD-ONLY11-NOT: libgomp
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp=libiomp5 -target x86_64-msvc-win32 -rtlib=platform \
// RUN:   | FileCheck --check-prefix=CHECK-MSVC-ILINK-64 %s

// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 -fopenmp-simd -target x86_64-msvc-win32 -rtlib=platform | FileCheck --check-prefix SIMD-ONLY11 %s
// SIMD-ONLY11-NOT: libiomp
// SIMD-ONLY11-NOT: libomp
// SIMD-ONLY11-NOT: libgomp
// CHECK-MSVC-ILINK-64: link.exe
// CHECK-MSVC-ILINK-64-SAME: -nodefaultlib:vcomp.lib
// CHECK-MSVC-ILINK-64-SAME: -nodefaultlib:vcompd.lib
// CHECK-MSVC-ILINK-64-SAME: -libpath:{{.+}}/../lib
// CHECK-MSVC-ILINK-64-SAME: -defaultlib:libiomp5md.lib
//
