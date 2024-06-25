! RUN: %flang -target x86_64-linux-gnu -fopenmp=libomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-OPENMP
! RUN: %flang -target x86_64-linux-gnu -fopenmp=libgomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-NO-OPENMP
! RUN: %flang -target x86_64-linux-gnu -fopenmp=libiomp5 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-OPENMP
! RUN: %flang -target x86_64-apple-darwin -fopenmp=libomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-OPENMP
! RUN: %flang -target x86_64-apple-darwin -fopenmp=libgomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-NO-OPENMP
! RUN: %flang -target x86_64-apple-darwin -fopenmp=libiomp5 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-OPENMP
! RUN: %flang -target x86_64-freebsd -fopenmp=libomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-OPENMP
! RUN: %flang -target x86_64-freebsd -fopenmp=libgomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-NO-OPENMP
! RUN: %flang -target x86_64-freebsd -fopenmp=libiomp5 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-OPENMP
! RUN: %flang -target x86_64-windows-gnu -fopenmp=libomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-OPENMP
! RUN: %flang -target x86_64-windows-gnu -fopenmp=libgomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-NO-OPENMP --check-prefix=CHECK-WARNING
! RUN: %flang -target x86_64-windows-gnu -fopenmp=libiomp5 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-FC1-OPENMP

! CHECK-FC1-OPENMP: "-fc1"
! CHECK-FC1-OPENMP: "-fopenmp"
!
! CHECK-WARNING: warning: the library '-fopenmp=={{.*}}' is not supported, OpenMP will not be enabled
! CHECK-FC1-NO-OPENMP: "-fc1"
! CHECK-FC1-NO-OPENMP-NOT: "-fopenmp"
!
! RUN: %flang -target x86_64-linux-gnu -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-OMP
! RUN: %flang -target x86_64-linux-gnu -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-GOMP --check-prefix=CHECK-LD-GOMP-RT
! RUN: %flang -target x86_64-linux-gnu -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-IOMP5
!
! RUN: %flang -target x86_64-darwin -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-OMP
! RUN: %flang -target x86_64-darwin -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-GOMP --check-prefix=CHECK-LD-GOMP-NO-RT
! RUN: %flang -target x86_64-darwin -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-IOMP5
!
! RUN: %flang -target x86_64-freebsd -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-OMP
! RUN: %flang -target x86_64-freebsd -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-GOMP --check-prefix=CHECK-LD-GOMP-NO-RT
! RUN: %flang -target x86_64-freebsd -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-IOMP5
!
! RUN: %flang -target x86_64-windows-gnu -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-OMP
! RUN: %flang -target x86_64-windows-gnu -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-GOMP --check-prefix=CHECK-LD-GOMP-NO-RT
! RUN: %flang -target x86_64-windows-gnu -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-IOMP5MD
!
! CHECK-LD-OMP: "{{.*}}ld{{(.exe)?}}"
! CHECK-LD-OMP: "-lomp"
!
! CHECK-LD-GOMP: "{{.*}}ld{{(.exe)?}}"
! CHECK-LD-GOMP: "-lgomp"
! CHECK-LD-GOMP-RT: "-lrt"
! CHECK-LD-GOMP-NO-RT-NOT: "-lrt"
!
! CHECK-LD-IOMP5: "{{.*}}ld{{(.exe)?}}"
! CHECK-LD-IOMP5: "-liomp5"
!
! CHECK-LD-IOMP5MD: "{{.*}}ld{{(.exe)?}}"
! CHECK-LD-IOMP5MD: "-liomp5md"
!
! We'd like to check that the default is sane, but until we have the ability
! to *always* semantically analyze OpenMP without always generating runtime
! calls (in the event of an unsupported runtime), we don't have a good way to
! test the FC1 invocation. Instead, just ensure we do eventually link *some*
! OpenMP runtime.
!
! RUN: %flang -target x86_64-linux-gnu -fopenmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-ANY
! RUN: %flang -target x86_64-darwin -fopenmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-ANY
! RUN: %flang -target x86_64-freebsd -fopenmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-ANY
! RUN: %flang -target x86_64-windows-gnu -fopenmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-ANYMD
!
! CHECK-LD-ANY: "{{.*}}ld{{(.exe)?}}"
! CHECK-LD-ANY: "-l{{(omp|gomp|iomp5)}}"
!
! CHECK-LD-ANYMD: "{{.*}}ld{{(.exe)?}}"
! CHECK-LD-ANYMD: "-l{{(omp|gomp|iomp5md)}}"
