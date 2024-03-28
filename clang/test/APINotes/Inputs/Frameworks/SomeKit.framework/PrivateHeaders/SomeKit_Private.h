#ifndef SOMEKIT_PRIVATE_H
#define SOMEKIT_PRIVATE_H

#import <SomeKit/SomeKit.h>

@interface A(Private)
-(A*)privateTransform:(A*)input;

@property (nonatomic) A* internalProperty;
@end

@protocol InternalProtocol
@end

#endif

