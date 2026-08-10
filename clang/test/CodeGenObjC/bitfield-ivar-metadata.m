// RUN: %clang_cc1 -emit-llvm -o %t %s

@interface INTF
{
    unsigned ivar1;
    unsigned ivar2;
    unsigned char BDIVAR3:1;
    unsigned char BDIVAR4:1;
}
@end

@implementation INTF
@end


