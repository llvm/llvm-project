// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -fobjc-exceptions %s -o - | FileCheck %s


extern int printf(const char*, ...);

int main()
{
    @try {
        @throw @"foo";
    } @catch (id e) {
        @try {
// CHECK: call void @objc_exception_throw
           @throw;
        } @catch (id e) {
            if (e) {
                printf("caught \n");
            } else {
                printf("caught (WRONG)\n");
            }
        } @catch (...) {
            printf("caught nothing (WRONG)\n");
        }
    }
}

