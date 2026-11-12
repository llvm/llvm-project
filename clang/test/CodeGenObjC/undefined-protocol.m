// RUN: %clang_cc1 -emit-llvm-only -fobjc-runtime=gcc %s

// PR13820
// REQUIRES: LP64

@protocol MadeUpProtocol;

@interface Object <MadeUpProtocol> @end
@implementation Object @end
