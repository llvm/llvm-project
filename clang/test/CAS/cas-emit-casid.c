// REQUIRES: aarch64-registered-target
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -target arm64-apple-macosx12.0.0 -c -Xclang -fcas-backend -Xclang -fcas-path -Xclang %t/cas -Xclang -fcas-backend-mode=native -Xclang -fcas-emit-casid-file %s -o %t/test.o 
// RUN: cat %t/test.o.casid | FileCheck %s --check-prefix=NATIVE_FILENAME
// NATIVE_FILENAME: CASID:Jllvmcas://{{.*}}
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -target arm64-apple-macosx12.0.0 -c -Xclang -fcas-backend -Xclang -fcas-path -Xclang %t/cas -Xclang -fcas-backend-mode=verify -Xclang -fcas-emit-casid-file %s -o %t/test.o 
// RUN: cat %t/test.o.casid | FileCheck %s --check-prefix=VERIFY_FILENAME
// VERIFY_FILENAME: CASID:Jllvmcas://{{.*}}
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -target arm64-apple-macosx12.0.0 -c -Xclang -fcas-backend -Xclang -fcas-path -Xclang %t/cas -Xclang -fcas-backend-mode=casid -Xclang -fcas-emit-casid-file %s -o %t/test.o 
// RUN: not cat %t/test.o.casid
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -target arm64-apple-macosx12.0.0 -c -Xclang -fcas-backend -Xclang -fcas-path -Xclang %t/cas -Xclang -fcas-backend-mode=native -Xclang -fcas-emit-casid-file %s -o -
// RUN: not cat %t/test.o.casid
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -target arm64-apple-macosx12.0.0 -c -Xclang -fcas-backend -Xclang -fcas-path -Xclang %t/cas -Xclang -fcas-backend-mode=verify -Xclang -fcas-emit-casid-file %s -o -
// RUN: not cat %t/test.o.casid
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -target arm64-apple-macosx12.0.0 -c -Xclang -fcas-backend -Xclang -fcas-path -Xclang %t/cas -Xclang -fcas-backend-mode=casid -Xclang -fcas-emit-casid-file %s -o -
// RUN: not cat %t/test.o.casid

void test(void) {}

int test1(void) {
  return 0;
}
