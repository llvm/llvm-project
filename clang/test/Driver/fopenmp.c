// RUN: %clang -target x86_64-linux-gnu -fopenmp=libomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libgomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-OPENMP
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libiomp5 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang -target x86_64-apple-darwin -fopenmp=libomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang -target x86_64-apple-darwin -fopenmp=libgomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-OPENMP
// RUN: %clang -target x86_64-apple-darwin -fopenmp=libiomp5 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang -target x86_64-freebsd -fopenmp=libomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang -target x86_64-freebsd -fopenmp=libgomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-OPENMP
// RUN: %clang -target x86_64-freebsd -fopenmp=libiomp5 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang -target x86_64-netbsd -fopenmp=libomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang -target x86_64-netbsd -fopenmp=libgomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-OPENMP
// RUN: %clang -target x86_64-netbsd -fopenmp=libiomp5 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang -target x86_64-openbsd -fopenmp=libomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang -target x86_64-openbsd -fopenmp=libgomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-OPENMP
// RUN: %clang -target x86_64-openbsd -fopenmp=libiomp5 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang -target x86_64-windows-gnu -fopenmp=libomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang -target x86_64-windows-gnu -fopenmp=libgomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-OPENMP
// RUN: %clang -target x86_64-windows-gnu -fopenmp=libiomp5 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang_cl --target=x86_64-windows-msvc /clang:-fopenmp=libomp /openmp -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang_cl --target=x86_64-windows-msvc /clang:-fopenmp=libgomp /openmp -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-OPENMP
// RUN: %clang_cl --target=x86_64-windows-msvc /clang:-fopenmp=libiomp5 /openmp -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang_cl --target=x86_64-windows-msvc /clang:-fopenmp=libomp /openmp:experimental -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang_cl --target=x86_64-windows-msvc /clang:-fopenmp=libgomp /openmp:experimental -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-OPENMP
// RUN: %clang_cl --target=x86_64-windows-msvc /clang:-fopenmp=libiomp5 /openmp:experimental -### -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang_cl --target=x86_64-windows-msvc /openmp- -### -- %s 2>&1 | FileCheck --check-prefix=CHECK-CC1-NO-OPENMP %s
//
// CHECK-CC1-OPENMP: "-cc1"
// CHECK-CC1-OPENMP: "-fopenmp"
//
// CHECK-CC1-NO-OPENMP: "-cc1"
// CHECK-CC1-NO-OPENMP-NOT: "-fopenmp"
//
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-OMP
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-GOMP --check-prefix=CHECK-LD-GOMP-RT
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-IOMP5
//
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libomp -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-OMP
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libgomp -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-GOMP --check-prefix=CHECK-LD-STATIC-GOMP-RT
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libiomp5 -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-IOMP5
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libiomp5 -static -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-IOMP5-NO-BDYNAMIC
//
// RUN: %clang -nostdlib -target x86_64-linux-gnu -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OMP
// RUN: %clang -nostdlib -target x86_64-linux-gnu -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-GOMP
// RUN: %clang -nostdlib -target x86_64-linux-gnu -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-IOMP5
//
// RUN: %clang -target x86_64-darwin -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-OMP
// RUN: %clang -target x86_64-darwin -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-GOMP --check-prefix=CHECK-LD-GOMP-NO-RT
// RUN: %clang -target x86_64-darwin -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-IOMP5
//
// RUN: %clang -nostdlib -target x86_64-darwin -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OMP
// RUN: %clang -nostdlib -target x86_64-darwin -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-GOMP
// RUN: %clang -nostdlib -target x86_64-darwin -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-IOMP5
//
// RUN: %clang -target x86_64-freebsd -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-OMP
// RUN: %clang -target x86_64-freebsd -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-GOMP --check-prefix=CHECK-LD-GOMP-NO-RT
// RUN: %clang -target x86_64-freebsd -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-IOMP5
//
// RUN: %clang -target x86_64-freebsd -fopenmp=libomp -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-OMP
// RUN: %clang -target x86_64-freebsd -fopenmp=libgomp -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-GOMP --check-prefix=CHECK-LD-STATIC-GOMP-NO-RT
// RUN: %clang -target x86_64-freebsd -fopenmp=libiomp5 -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-IOMP5
// RUN: %clang -target x86_64-freebsd -fopenmp=libiomp5 -static -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-IOMP5-NO-BDYNAMIC
//
// RUN: %clang -nostdlib -target x86_64-freebsd -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OMP
// RUN: %clang -nostdlib -target x86_64-freebsd -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-GOMP
// RUN: %clang -nostdlib -target x86_64-freebsd -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-IOMP5
//
// RUN: %clang -target x86_64-netbsd -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-OMP
// RUN: %clang -target x86_64-netbsd -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-GOMP --check-prefix=CHECK-LD-GOMP-NO-RT
// RUN: %clang -target x86_64-netbsd -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-IOMP5
//
// RUN: %clang -target x86_64-netbsd -fopenmp=libomp -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-OMP
// RUN: %clang -target x86_64-netbsd -fopenmp=libgomp -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-GOMP --check-prefix=CHECK-LD-STATIC-GOMP-NO-RT
// RUN: %clang -target x86_64-netbsd -fopenmp=libiomp5 -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-IOMP5
// RUN: %clang -target x86_64-netbsd -fopenmp=libiomp5 -static -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-IOMP5-NO-BDYNAMIC
//
// RUN: %clang -nostdlib -target x86_64-netbsd -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OMP
// RUN: %clang -nostdlib -target x86_64-netbsd -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-GOMP
// RUN: %clang -nostdlib -target x86_64-netbsd -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-IOMP5
//
// RUN: %clang -target x86_64-openbsd -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-OMP
// RUN: %clang -target x86_64-openbsd -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-GOMP --check-prefix=CHECK-LD-GOMP-NO-RT
// RUN: %clang -target x86_64-openbsd -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-IOMP5
//
// RUN: %clang -target x86_64-openbsd -fopenmp=libomp -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-OMP
// RUN: %clang -target x86_64-openbsd -fopenmp=libgomp -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-GOMP --check-prefix=CHECK-LD-STATIC-GOMP-NO-RT
// RUN: %clang -target x86_64-openbsd -fopenmp=libiomp5 -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-IOMP5
// RUN: %clang -target x86_64-openbsd -fopenmp=libiomp5 -static -static-openmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-STATIC-IOMP5-NO-BDYNAMIC
//
// RUN: %clang -nostdlib -target x86_64-openbsd -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OMP
// RUN: %clang -nostdlib -target x86_64-openbsd -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-GOMP
// RUN: %clang -nostdlib -target x86_64-openbsd -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-IOMP5
//
// RUN: %clang -target x86_64-windows-gnu -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-OMP
// RUN: %clang -target x86_64-windows-gnu -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-GOMP --check-prefix=CHECK-LD-GOMP-NO-RT
// RUN: %clang -target x86_64-windows-gnu -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-IOMP5MD
//
// RUN: %clang -nostdlib -target x86_64-windows-gnu -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-OMP
// RUN: %clang -nostdlib -target x86_64-windows-gnu -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-GOMP
// RUN: %clang -nostdlib -target x86_64-windows-gnu -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-NO-IOMP5MD
//
// CHECK-LD-OMP: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-OMP: "-lomp"
//
// CHECK-LD-GOMP: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-GOMP: "-lgomp"
// CHECK-LD-GOMP-RT: "-lrt"
// CHECK-LD-GOMP-NO-RT-NOT: "-lrt"
//
// CHECK-LD-IOMP5: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-IOMP5: "-liomp5"
//
// CHECK-LD-IOMP5MD: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-IOMP5MD: "-liomp5md"
//
// CHECK-NO-OMP: "{{.*}}ld{{(.exe)?}}"
// CHECK-NO-OMP-NOT: "-lomp"
//
// CHECK-NO-GOMP: "{{.*}}ld{{(.exe)?}}"
// CHECK-NO-GOMP-NOT: "-lgomp"
//
// CHECK-NO-IOMP5: "{{.*}}ld{{(.exe)?}}"
// CHECK-NO-IOMP5-NOT: "-liomp5"
//
// CHECK-NO-IOMP5MD: "{{.*}}ld{{(.exe)?}}"
// CHECK-NO-IOMP5MD-NOT: "-liomp5md"
//
// CHECK-LD-STATIC-OMP: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-STATIC-OMP: "-Bstatic" "-lomp" "-Bdynamic"
//
// CHECK-LD-STATIC-GOMP: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-STATIC-GOMP: "-Bstatic" "-lgomp" "-Bdynamic"
// CHECK-LD-STATIC-GOMP-RT: "-lrt"
// CHECK-LD-STATIC-GOMP-NO-RT-NOT: "-lrt"
//
// CHECK-LD-STATIC-IOMP5: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-STATIC-IOMP5: "-Bstatic" "-liomp5" "-Bdynamic"
//
// CHECK-LD-STATIC-IOMP5-NO-BDYNAMIC: "{{.*}}ld{{(.exe)?}}"
// For x86 Gnu, the driver passes -static, while FreeBSD, NetBSD and OpenBSD pass -Bstatic
// CHECK-LD-STATIC-IOMP5-NO-BDYNAMIC: "-{{B?}}static" {{.*}} "-liomp5"
// CHECK-LD-STATIC-IOMP5-NO-BDYNAMIC-NOT: "-Bdynamic"
//
// RUN: %clang -target x86_64-linux-gnu -fopenmp -fopenmp-enable-irbuilder -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMPIRBUILDER
//
// CHECK-CC1-OPENMPIRBUILDER: "-cc1"
// CHECK-CC1-OPENMPIRBUILDER-SAME: "-fopenmp"
// CHECK-CC1-OPENMPIRBUILDER-SAME: "-fopenmp-enable-irbuilder"
//
// We'd like to check that the default is sane, but until we have the ability
// to *always* semantically analyze OpenMP without always generating runtime
// calls (in the event of an unsupported runtime), we don't have a good way to
// test the CC1 invocation. Instead, just ensure we do eventually link *some*
// OpenMP runtime.
//
// RUN: %clang -target x86_64-linux-gnu -fopenmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-ANY
// RUN: %clang -target x86_64-darwin -fopenmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-ANY
// RUN: %clang -target x86_64-freebsd -fopenmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-ANY
// RUN: %clang -target x86_64-netbsd -fopenmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-ANY
// RUN: %clang -target x86_64-openbsd -fopenmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-ANY
// RUN: %clang -target x86_64-windows-gnu -fopenmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-ANYMD
//
// CHECK-LD-ANY: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-ANY: "-l{{(omp|gomp|iomp5)}}"
//
// CHECK-LD-ANYMD: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-ANYMD: "-l{{(omp|gomp|iomp5md)}}"
