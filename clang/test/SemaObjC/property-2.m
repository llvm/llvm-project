// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface Tester 
@property char PropertyAtomic_char;
@property short PropertyAtomic_short;
@property int PropertyAtomic_int;
@property long PropertyAtomic_long;
@property long long PropertyAtomic_longlong;
@property float PropertyAtomic_float;
@property double PropertyAtomic_double;
@property(assign) id PropertyAtomic_id;
@property(retain) id PropertyAtomicRetained_id;
@property(copy) id PropertyAtomicRetainedCopied_id;
@property(retain) id PropertyAtomicRetainedGCOnly_id;
@property(copy) id PropertyAtomicRetainedCopiedGCOnly_id;
@end

@implementation Tester
@dynamic PropertyAtomic_char;
@dynamic PropertyAtomic_short;
@dynamic PropertyAtomic_int;
@dynamic PropertyAtomic_long;
@dynamic PropertyAtomic_longlong;
@dynamic PropertyAtomic_float;
@dynamic PropertyAtomic_double;
@dynamic PropertyAtomic_id;
@dynamic PropertyAtomicRetained_id;
@dynamic PropertyAtomicRetainedCopied_id;
@dynamic PropertyAtomicRetainedGCOnly_id;
@dynamic PropertyAtomicRetainedCopiedGCOnly_id;
@end

@interface SubClass : Tester
{
    char PropertyAtomic_char;
    short PropertyAtomic_short;
    int PropertyAtomic_int;
    long PropertyAtomic_long;
    long long PropertyAtomic_longlong;
    float PropertyAtomic_float;
    double PropertyAtomic_double;
    id PropertyAtomic_id;
    id PropertyAtomicRetained_id;
    id PropertyAtomicRetainedCopied_id;
    id PropertyAtomicRetainedGCOnly_id;
    id PropertyAtomicRetainedCopiedGCOnly_id;
}
@end

@implementation SubClass
@synthesize PropertyAtomic_char;
@synthesize PropertyAtomic_short;
@synthesize PropertyAtomic_int;
@synthesize PropertyAtomic_long;
@synthesize PropertyAtomic_longlong;
@synthesize PropertyAtomic_float;
@synthesize PropertyAtomic_double;
@synthesize PropertyAtomic_id;
@synthesize PropertyAtomicRetained_id;
@synthesize PropertyAtomicRetainedCopied_id;
@synthesize PropertyAtomicRetainedGCOnly_id;
@synthesize PropertyAtomicRetainedCopiedGCOnly_id;
@end
