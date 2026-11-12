// RUN: %clang_cc1 -mframe-pointer=all -triple x86_64-apple-darwin10 -target-cpu core2 -S -o - %s | FileCheck %s --check-prefix=ALWAYS-X86
// RUN: %clang_cc1 -mframe-pointer=all -triple x86_64-apple-darwin12 -target-cpu core2 -S -o - %s | FileCheck %s --check-prefix=ALWAYS-X86
// RUN: %clang_cc1 -fswift-async-fp=never -mframe-pointer=all -triple x86_64-apple-darwin10 -target-cpu core2 -S -o - %s | FileCheck %s --check-prefix=NEVER-X86
// RUN: %clang_cc1 -fswift-async-fp=never -mframe-pointer=all -triple x86_64-apple-darwin12 -target-cpu core2 -S -o - %s | FileCheck %s --check-prefix=NEVER-X86
// RUN: %clang_cc1 -fswift-async-fp=auto -mframe-pointer=all -triple x86_64-apple-darwin10 -target-cpu core2 -S -o - %s | FileCheck %s --check-prefix=AUTO-X86
// RUN: %clang_cc1 -fswift-async-fp=auto -mframe-pointer=all -triple x86_64-apple-darwin12 -target-cpu core2 -S -o - %s | FileCheck %s --check-prefix=ALWAYS-X86
// RUN: %clang_cc1 -fswift-async-fp=always -mframe-pointer=all -triple x86_64-apple-darwin10 -target-cpu core2 -S -o - %s | FileCheck %s --check-prefix=ALWAYS-X86
// RUN: %clang_cc1 -fswift-async-fp=always -mframe-pointer=all -triple x86_64-apple-darwin12 -target-cpu core2 -S -o - %s | FileCheck %s --check-prefix=ALWAYS-X86

// RUN: %clang_cc1 -mframe-pointer=all -triple arm64-apple-ios9 -target-cpu cyclone -S -o - %s | FileCheck %s --check-prefix=ALWAYS-ARM64
// RUN: %clang_cc1 -mframe-pointer=all -triple arm64-apple-ios15 -target-cpu cyclone -S -o - %s | FileCheck %s --check-prefix=ALWAYS-ARM64
// RUN: %clang_cc1 -fswift-async-fp=auto -mframe-pointer=all -triple arm64-apple-ios9 -target-cpu cyclone -S -o - %s | FileCheck %s --check-prefix=AUTO-ARM64
// RUN: %clang_cc1 -fswift-async-fp=auto -mframe-pointer=all -triple arm64-apple-ios15 -target-cpu cyclone -S -o - %s | FileCheck %s --check-prefix=ALWAYS-ARM64
// RUN: %clang_cc1 -fswift-async-fp=never -mframe-pointer=all -triple arm64-apple-ios9 -target-cpu cyclone -S -o - %s | FileCheck %s --check-prefix=NEVER-ARM64
// RUN: %clang_cc1 -fswift-async-fp=never -mframe-pointer=all -triple arm64-apple-ios15 -target-cpu cyclone -S -o - %s | FileCheck %s --check-prefix=NEVER-ARM64
// RUN: %clang_cc1 -fswift-async-fp=always -mframe-pointer=all -triple arm64-apple-ios9 -target-cpu cyclone -S -o - %s | FileCheck %s --check-prefix=ALWAYS-ARM64
// RUN: %clang_cc1 -fswift-async-fp=always -mframe-pointer=all -triple arm64-apple-ios15 -target-cpu cyclone -S -o - %s | FileCheck %s --check-prefix=ALWAYS-ARM64

// REQUIRES: aarch64-registered-target,x86-registered-target

#define SWIFTASYNCCALL __attribute__((swiftasynccall))
#define ASYNC_CONTEXT __attribute__((swift_async_context))

SWIFTASYNCCALL void async_context_1(ASYNC_CONTEXT void *ctx) {}

// AUTO-X86: _async_context_1:
// AUTO-X86:	_swift_async_extendedFramePointerFlags

// ALWAYS-X86: _async_context_1:
// ALWAYS-X86: btsq	$60

// NEVER-X86: _async_context_1:
// NEVER-X86-NOT:	_swift_async_extendedFramePointerFlags
// NEVER-X86-NOT: btsq	$60

// AUTO-ARM64: _async_context_1
// AUTO-ARM64: _swift_async_extendedFramePointerFlags

// ALWAYS-ARM64: _async_context_1
// ALWAYS-ARM64: orr x29, x29, #0x1000000000000000

// NEVER-ARM64: _async_context_1:
// NEVER-ARM64-NOT:	_swift_async_extendedFramePointerFlags
// NEVER-ARM64-NOT: orr x29, x29, #0x1000000000000000
