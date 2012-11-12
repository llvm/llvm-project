typedef int Int;

@interface I1 
@end

// Matching category
@interface I1 (Cat1)
- (Int)method0;
@end

// Matching class extension
@interface I1 ()
- (Int)method1;
@end

// Mismatched category
@interface I1 (Cat2)
- (float)method2;
@end

@interface I2
@end

// Mismatched class extension
@interface I2 ()
- (float)method3;
@end

// Category with implementation
@interface I2 (Cat3)
@end

@implementation I2 (Cat3)
@end

// Category with implementation
@interface I2 (Cat5)
@end

@implementation I2 (Cat5)
@end

// Category with mismatched implementation
@interface I2 (Cat6)
@end

@implementation I2 (Cat6)
- (int)blah { return 0; }
@end
