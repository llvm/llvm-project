// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-feature +altivec -target-cpu pwr8 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-aix -target-feature +altivec -target-cpu pwr8 -emit-llvm %s -o - | FileCheck %s
vector float foo1(vector float x) { return x; }
// CHECK:  define <4 x float> @foo1(<4 x float> %x) [[ATTR:#[0-9]+]] {
// CHECK:  entry:
// CHECK:    %x.addr = alloca <4 x float>, align 16
// CHECK:    store <4 x float> %x, <4 x float>* %x.addr, align 16
// CHECK:    %0 = load <4 x float>, <4 x float>* %x.addr, align 16
// CHECK:    ret <4 x float> %0
// CHECK:  }
vector double foo2(vector double x) { return x; }
// CHECK:  define <2 x double> @foo2(<2 x double> %x) [[ATTR]] {
// CHECK:  entry:
// CHECK:    %x.addr = alloca <2 x double>, align 16
// CHECK:    store <2 x double> %x, <2 x double>* %x.addr, align 16
// CHECK:    %0 = load <2 x double>, <2 x double>* %x.addr, align 16
// CHECK:    ret <2 x double> %0
// CHECK:  }
vector int foo3(vector int x) { return x; }
// CHECK:  define <4 x i32> @foo3(<4 x i32> %x) [[ATTR]] {
// CHECK:  entry:
// CHECK:    %x.addr = alloca <4 x i32>, align 16
// CHECK:    store <4 x i32> %x, <4 x i32>* %x.addr, align 16
// CHECK:    %0 = load <4 x i32>, <4 x i32>* %x.addr, align 16
// CHECK:    ret <4 x i32> %0
// CHECK:  }
vector short int foo4(vector short int x) { return x; }
// CHECK:  define <8 x i16> @foo4(<8 x i16> %x) [[ATTR]] {
// CHECK:  entry:
// CHECK:    %x.addr = alloca <8 x i16>, align 16
// CHECK:    store <8 x i16> %x, <8 x i16>* %x.addr, align 16
// CHECK:    %0 = load <8 x i16>, <8 x i16>* %x.addr, align 16
// CHECK:    ret <8 x i16> %0
// CHECK:  }
vector char foo5(vector char x) { return x; }
// CHECK:  define <16 x i8> @foo5(<16 x i8> %x) [[ATTR]] {
// CHECK:  entry:
// CHECK:    %x.addr = alloca <16 x i8>, align 16
// CHECK:    store <16 x i8> %x, <16 x i8>* %x.addr, align 16
// CHECK:    %0 = load <16 x i8>, <16 x i8>* %x.addr, align 16
// CHECK:    ret <16 x i8> %0
// CHECK:  }

