// RUN: %clang_cc1 -x objective-c++ -fblocks -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 -o - %s

extern "C" {
        short foo() { } 
}
typedef unsigned char Boolean;

