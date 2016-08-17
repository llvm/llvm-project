#ifndef SOMEKIT_H
#define SOMEKIT_H

__attribute__((objc_root_class))
@interface A
-(A*)transform:(A*)input;
-(A*)transform:(A*)input integer:(int)integer;

@property (nonatomic, readonly, retain) A* someA;
@property (nonatomic, retain) A* someOtherA;

@property (nonatomic) int intValue;
@end

@interface B : A
@end

@interface C : A
- (instancetype)init;
- (instancetype)initWithA:(A*)a;
@end

@interface ProcessInfo : A
+(instancetype)processInfo;
@end

#endif
