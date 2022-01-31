// UNSUPPORTED: system-windows

// RUN: %clang -fopenmp %s -L%S/Inputs -o %t && llvm-readelf --dynamic-table %t | FileCheck %s --check-prefixes=CHECK-DEFAULT
// RUN: %clang -fopenmp -fopenmp-implicit-rpath %s -L%S/Inputs -o %t && llvm-readelf --dynamic-table %t | FileCheck %s --check-prefixes=CHECK-EXPLICIT
// RUN: %clang -fopenmp -fno-openmp-implicit-rpath %s -L%S/Inputs -o %t && llvm-readelf --dynamic-table %t | FileCheck %s --check-prefixes=CHECK-DISABLED

// RUN: %clang -fopenmp -Wl,--disable-new-dtags %s -L%S/Inputs -o %t && llvm-readelf --dynamic-table %t | FileCheck %s --check-prefixes=CHECK-DEFAULT-RPATH
// RUN: %clang -fopenmp -fopenmp-implicit-rpath -Wl,--disable-new-dtags %s -L%S/Inputs -o %t && llvm-readelf --dynamic-table %t | FileCheck %s --check-prefixes=CHECK-EXPLICIT-RPATH
// RUN: %clang -fopenmp -fno-openmp-implicit-rpath -Wl,--disable-new-dtags %s -L%S/Inputs -o %t && llvm-readelf --dynamic-table %t | FileCheck %s --check-prefixes=CHECK-DISABLED-RPATH

// RUN: %clang -fopenmp -Wl,--enable-new-dtags %s -L%S/Inputs -o %t && llvm-readelf --dynamic-table %t | FileCheck %s --check-prefixes=CHECK-DEFAULT-RUNPATH
// RUN: %clang -fopenmp -fopenmp-implicit-rpath -Wl,--enable-new-dtags %s -L%S/Inputs -o %t && llvm-readelf --dynamic-table %t | FileCheck %s --check-prefixes=CHECK-EXPLICIT-RUNPATH
// RUN: %clang -fopenmp -fno-openmp-implicit-rpath -Wl,--enable-new-dtags %s -L%S/Inputs -o %t && llvm-readelf --dynamic-table %t | FileCheck %s --check-prefixes=CHECK-DISABLED-RUNPATH

// RUN: %clang -Wl,-rpath=early -fopenmp %s -L%S/Inputs -o %t -Wl,-rpath=late && llvm-readelf --dynamic-table %t | FileCheck %s --check-prefixes=CHECK-COMPOSABLE

// CHECK-DEFAULT:      ({{R|RUN}}PATH) Library {{r|run}}path: [{{.*}}lib{{.*}}]
// CHECK-EXPLICIT:     ({{R|RUN}}PATH) Library {{r|run}}path: [{{.*}}lib{{.*}}]
// CHECK-DISABLED-NOT: ({{R|RUN}}PATH)

// CHECK-DEFAULT-RPATH:      (RPATH) Library rpath: [{{.*}}lib{{.*}}]
// CHECK-EXPLICIT-RPATH:     (RPATH) Library rpath: [{{.*}}lib{{.*}}]
// CHECK-DISABLED-RPATH-NOT: (RPATH)

// CHECK-DEFAULT-RUNPATH:      (RUNPATH) Library runpath: [{{.*}}lib{{.*}}]
// CHECK-EXPLICIT-RUNPATH:     (RUNPATH) Library runpath: [{{.*}}lib{{.*}}]
// CHECK-DISABLED-RUNPATH-NOT: (RUNPATH)

// CHECK-COMPOSABLE: ({{R|RUN}}PATH) Library {{r|run}}path: [early:late:{{.*}}lib{{.*}}]

int main() {}
