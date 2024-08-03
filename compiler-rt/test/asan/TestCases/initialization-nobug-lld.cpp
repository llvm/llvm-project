// RUN: %clangxx_asan -O3 %p/initialization-nobug.cpp %p/Helpers/initialization-nobug-extra.cpp -fuse-ld=lld -o %t && %env_asan_opts=check_initialization_order=true:report_globals=3 %run %t 2>&1 | FileCheck %s --implicit-check-not "DynInit"

// Same as initialization-nobug.cpp, but with lld we expect just one
// `DynInitUnpoison` executed after `AfterDynamicInit` at the end.
// REQUIRES: lld-available

// With dynamic runtimes `AfterDynamicInit` will called before `executable`
// contructors, with constructors of dynamic runtime.
// XFAIL: asan-dynamic-runtime

// CHECK: DynInitPoison module: {{.*}}initialization-nobug.cpp
// CHECK: DynInitPoison module: {{.*}}initialization-nobug-extra.cpp
// CHECK: AfterDynamicInit
// CHECK: DynInitUnpoison
