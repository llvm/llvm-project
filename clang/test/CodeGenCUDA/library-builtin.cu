// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | \
// RUN:  FileCheck --check-prefixes=HOST,BOTH %s
// RUN: %clang_cc1 -fcuda-is-device -triple nvptx64-nvidia-cuda \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=DEVICE,BOTH

// BOTH-LABEL: define{{.*}} float @logf(float

// logf() should be calling itself recursively as we don't have any standard
// library on device side.
// DEVICE: call contract float @logf(float
extern "C" __attribute__((device)) float logf(float __x) { return logf(__x); }

// NOTE: this case is to illustrate the expected differences in behavior between
// the host and device. In general we do not mess with host-side standard
// library.
//
// Host is assumed to have standard library, so logf() calls LLVM intrinsic.
// HOST: call contract float @llvm.log.f32(float
extern "C" float logf(float __x) { return logf(__x); }
