// RUN: rm -f %t.o && touch %t.o
//
// RUN: %clang --target=arm64-apple-macos -### %t.o -flto=thin -fuse-ld=ld -fprofile-sample-use=a.profdata 2>&1 | FileCheck %s --check-prefix=DARWIN-LD-USE
// RUN: %clang --target=arm64-apple-macos -### %t.o -flto      -fuse-ld=ld -fprofile-sample-use=a.profdata 2>&1 | FileCheck %s --check-prefix=DARWIN-LD-USE

// DARWIN-LD-USE: "-mllvm" "-sample-profile-file=a.profdata"

// RUN: %clang --target=apple-arm64-ios -### %t.o -flto=thin -fuse-ld=lld -B%S/Inputs/lld -fprofile-sample-use=a.profdata 2>&1 | FileCheck %s --check-prefix=DARWIN-LLD-USE

// DARWIN-LLD-USE: "-mllvm" "-sample-profile-file=a.profdata"

// -fno-profile-sample-use suppresses the linker flag.
// RUN: %clang --target=arm64-apple-macos -### %t.o -flto=thin -fuse-ld=ld -fprofile-sample-use=a.profdata -fno-profile-sample-use 2>&1 | FileCheck %s --check-prefix=DARWIN-LD-NONE

// DARWIN-LD-NONE-NOT: "-sample-profile-file=
