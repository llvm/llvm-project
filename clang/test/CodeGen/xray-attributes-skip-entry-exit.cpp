// RUN: %clang_cc1 -fxray-instrument \
// RUN:     -fxray-instrumentation-bundle=function-entry -x c++ \
// RUN:     -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s \
// RUN:     | FileCheck --check-prefixes CHECK,SKIPEXIT %s
// RUN: %clang_cc1 -fxray-instrument \
// RUN:     -fxray-instrumentation-bundle=function-exit -x c++ \
// RUN:     -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s \
// RUN:     | FileCheck --check-prefixes CHECK,SKIPENTRY %s
// RUN: %clang_cc1 -fxray-instrument \
// RUN:     -fxray-instrumentation-bundle=function-entry,function-exit -x c++ \
// RUN:     -std=c++11 -triple x86_64-unknown-unknown -emit-llvm -o - %s \
// RUN:     | FileCheck --check-prefixes CHECK,NOSKIPENTRY,NOSKIPEXIT %s

// CHECK: define{{.*}} void @_Z13justAFunctionv() #[[ATTR:[0-9]+]] {
void justAFunction() {
}

// SKIPENTRY: attributes #[[ATTR]] = {{.*}} "xray-skip-entry" {{.*}}
// SKIPEXIT: attributes #[[ATTR]] = {{.*}} "xray-skip-exit" {{.*}}

// NOSKIPENTRY-NOT: attributes #[[ATTR]] = {{.*}} "xray-skip-entry" {{.*}}
// NOSKIPEXIT-NOT: attributes #[[ATTR]] = {{.*}} "xray-skip-exit" {{.*}}
