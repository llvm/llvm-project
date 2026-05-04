#ifndef SOMEKIT_PRIVATE_H
#define SOMEKIT_PRIVATE_H

#import <SomeKit/SomeKitForNullAnnotation.h>

@interface A(Private)
-(A*)privateTransform:(A*)input;

@property (nonatomic) A* internalProperty;
@end

@protocol InternalProtocol
- (id) MomeMethod;
@end

#endif

