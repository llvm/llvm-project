// RUN: %clang_cc1 -fxray-instrument -fxray-instruction-threshold=1 -fxray-function-groups=3 -fxray-selected-function-group=0 \
// RUN:            -emit-llvm -o - %s -triple x86_64-unknown-linux-gnu | FileCheck --check-prefix=GROUP0 %s

// RUN: %clang_cc1 -fxray-instrument -fxray-instruction-threshold=1 -fxray-function-groups=3 -fxray-selected-function-group=1 \
// RUN:            -emit-llvm -o - %s -triple x86_64-unknown-linux-gnu | FileCheck --check-prefix=GROUP1 %s

// RUN: %clang_cc1 -fxray-instrument -fxray-instruction-threshold=1 -fxray-function-groups=3 -fxray-selected-function-group=2 \
// RUN:            -emit-llvm -o - %s -triple x86_64-unknown-linux-gnu | FileCheck --check-prefix=GROUP2 %s

static int foo() { // part of group 0
  return 1;
}

int bar() { // part of group 2
  return 1;
}

int yarr() { // part of group 1
  foo();
  return 1;
}

[[clang::xray_always_instrument]] int always() { // part of group 0
  return 1;
}

[[clang::xray_never_instrument]] int never() { // part of group 1
  return 1;
}

// GROUP0: define{{.*}} i32 @_Z3barv() #[[ATTRS_BAR:[0-9]+]] {
// GROUP0: define{{.*}} i32 @_Z4yarrv() #[[ATTRS_BAR]] {
// GROUP0: define{{.*}} i32 @_ZL3foov() #[[ATTRS_FOO:[0-9]+]] {
// GROUP0: define{{.*}} i32 @_Z6alwaysv() #[[ATTRS_ALWAYS:[0-9]+]] {
// GROUP0: define{{.*}} i32 @_Z5neverv() #[[ATTRS_NEVER:[0-9]+]] {
// GROUP0-DAG: attributes #[[ATTRS_BAR]] = {{.*}} "function-instrument"="xray-never" {{.*}}
// GROUP0-DAG: attributes #[[ATTRS_ALWAYS]] = {{.*}} "function-instrument"="xray-always" {{.*}}
// GROUP0-DAG: attributes #[[ATTRS_NEVER]] = {{.*}} "function-instrument"="xray-never" {{.*}}

// GROUP1: define{{.*}} i32 @_Z3barv() #[[ATTRS_BAR:[0-9]+]] {
// GROUP1: define{{.*}} i32 @_Z4yarrv() #[[ATTRS_YARR:[0-9]+]] {
// GROUP1: define{{.*}} i32 @_ZL3foov() #[[ATTRS_BAR]] {
// GROUP1: define{{.*}} i32 @_Z6alwaysv() #[[ATTRS_ALWAYS:[0-9]+]] {
// GROUP1: define{{.*}} i32 @_Z5neverv() #[[ATTRS_NEVER:[0-9]+]] {
// GROUP1-DAG: attributes #[[ATTRS_BAR]] = {{.*}} "function-instrument"="xray-never" {{.*}}
// GROUP1-DAG: attributes #[[ATTRS_ALWAYS]] = {{.*}} "function-instrument"="xray-always" {{.*}}
// GROUP1-DAG: attributes #[[ATTRS_NEVER]] = {{.*}} "function-instrument"="xray-never" {{.*}}

// GROUP2: define{{.*}} i32 @_Z3barv() #[[ATTRS_BAR:[0-9]+]] {
// GROUP2: define{{.*}} i32 @_Z4yarrv() #[[ATTRS_YARR:[0-9]+]] {
// GROUP2: define{{.*}} i32 @_ZL3foov() #[[ATTRS_YARR]] {
// GROUP2: define{{.*}} i32 @_Z6alwaysv() #[[ATTRS_ALWAYS:[0-9]+]] {
// GROUP2: define{{.*}} i32 @_Z5neverv() #[[ATTRS_NEVER:[0-9]+]] {
// GROUP2-DAG: attributes #[[ATTRS_YARR]] = {{.*}} "function-instrument"="xray-never" {{.*}}
// GROUP2-DAG: attributes #[[ATTRS_ALWAYS]] = {{.*}} "function-instrument"="xray-always" {{.*}}
// GROUP2-DAG: attributes #[[ATTRS_NEVER]] = {{.*}} "function-instrument"="xray-never" {{.*}}
