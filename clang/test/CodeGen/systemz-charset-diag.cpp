// RUN: %clang_cc1 -triple s390x-none-zos -fexec-charset IBM-1047 %s -std=c++17 -emit-llvm -o - -verify

const char* Computer = "🖥️"; // expected-error-re {{conversion to literal encoding failed: {{.*}}}}
