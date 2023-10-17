// RUN: touch %t.o
//
// RUN: %clang --target=x86_64-unknown-linux -### %t.o -flto=thin \
// RUN:   -fprofile-use 2>&1 | FileCheck %s

// CHECK: -plugin-opt=cs-profile-path=default.profdata

// RUN: %clang --target=apple-arm64-ios -### %t.o -flto=thin -fuse-ld=lld -B%S/Inputs/lld -fprofile-use 2>&1 | FileCheck %s --check-prefix=DARWIN-USE1
// RUN: %clang --target=apple-arm64-ios -### %t.o -flto=thin -fuse-ld=lld -B%S/Inputs/lld -fprofile-use=a.profdata 2>&1 | FileCheck %s --check-prefix=DARWIN-USE2

// DARWIN-USE1: "--cs-profile-path=default.profdata"
// DARWIN-USE2: "--cs-profile-path=a.profdata"

// RUN: %clang --target=apple-arm64-ios -### %t.o -flto=thin -fuse-ld=lld -B%S/Inputs/lld -fcs-profile-generate 2>&1 | FileCheck %s --check-prefix=DARWIN-GEN1
// RUN: %clang --target=apple-arm64-ios -### %t.o -flto=thin -fuse-ld=lld -B%S/Inputs/lld -fcs-profile-generate=directory 2>&1 | FileCheck %s --check-prefix=DARWIN-GEN2

// DARWIN-GEN1: "--cs-profile-generate"
// DARWIN-GEN1-SAME: "--cs-profile-path=default_%m.profraw"
// DARWIN-GEN2: "--cs-profile-generate"
// DARWIN-GEN2-SAME: "--cs-profile-path=directory{{(/|\\\\)}}default_%m.profraw"
