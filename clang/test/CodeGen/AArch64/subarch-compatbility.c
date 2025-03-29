// REQUIRES: aarch64-registered-target
// RUN: %clang -target aarch64-none-linux -march=armv9.3-a -o %t -c %s 2>&1 | FileCheck --allow-empty %s

// Successor targets should be ableto call predecessor target functions.
__attribute__((__always_inline__,target("v8a")))
int armv80(int i) {
    return i + 42;
}

__attribute__((__always_inline__,target("v8.1a")))
int armv81(int i) {
    return armv80(i);
}

__attribute__((__always_inline__,target("v8.2a")))
int armv82(int i) {
    return armv81(i);
}

__attribute__((__always_inline__,target("v8.3a")))
int armv83(int i) {
    return armv82(i);
}

__attribute__((__always_inline__,target("v8.4a")))
int armv84(int i) {
    return armv83(i);
}

__attribute__((__always_inline__,target("v8.5a")))
int armv85(int i) {
    return armv84(i);
}

__attribute__((__always_inline__,target("v8.6a")))
int armv86(int i) {
    return armv85(i);
}

__attribute__((__always_inline__,target("v8.7a")))
int armv87(int i) {
    return armv86(i);
}

__attribute__((__always_inline__,target("v8.8a")))
int armv88(int i) {
    return armv87(i);
}

__attribute__((__always_inline__,target("v9a")))
int armv9(int i) {
    return armv85(i);
}

__attribute__((__always_inline__,target("v9.1a")))
int armv91(int i) {
    return armv9(i);
}

__attribute__((__always_inline__,target("v9.2a")))
int armv92(int i) {
    return armv91(i);
}

__attribute__((__always_inline__,target("v9.3a")))
int armv93(int i) {
    return armv92(i);
}

// CHECK-NOT: always_inline function {{.*}} requires target feature {{.*}}, but would be inlined into function {{.*}} that is compiled without support for {{.*}}
// CHECK-NOT: {{.*}} is not a recognized feature for this target
