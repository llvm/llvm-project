// RUN: not %clang_cc1 -triple x86_64-apple-macos11 -fcompilation-caching-service-path %t -fcas-backend -fcasid-output -emit-obj %s -o %t.o 2>&1 | FileCheck %s
// CHECK: error: '-fcas-backend' is incompatible with remote caching backend
// CHECK: error: '-fcasid-output' is incompatible with remote caching backend
