#ifndef SOMEKIT_H
#define SOMEKIT_H

#define ROOT_CLASS __attribute__((objc_root_class))

ROOT_CLASS
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


@interface MyClass : A
- Inst;
+ Clas;
@end

struct CGRect {
  float origin;
  float size;
};
typedef struct CGRect NSRect;

@interface I
- (void) Meth : (NSRect[4])exposedRects;
- (void) Meth1 : (const  I*)exposedRects;
- (void) Meth2 : (const I*)exposedRects;
- (void) Meth3 : (I*)exposedRects;
- (const I*) Meth4;
- (const I*) Meth5 : (int) Arg1 : (const I*)Arg2 : (double)Arg3 :   (const I*) Arg4 :(const  volatile id) Arg5;
- (volatile const I*) Meth6 : (const char *)Arg1 : (const char *)Arg2 : (double)Arg3 :   (const I*) Arg4 :(const  volatile id) Arg5;
@end

@class NSURL, NSArray, NSError;
@interface INTF_BLOCKS
  + (void)getNonLocalVersionsOfItemAtURL:(NSURL *)url completionHandler:(void (^)(NSArray *nonLocalFileVersions, NSError *error))completionHandler;
  + (void *)getNonLocalVersionsOfItemAtURL2:(NSURL *)url completionHandler:(void (^)(NSArray *nonLocalFileVersions, NSError *error))completionHandler;
  + (NSError **)getNonLocalVersionsOfItemAtURL3:(int)url completionHandler:(void (^)(NSArray *nonLocalFileVersions, NSError *error))completionHandler;
  + (id)getNonLocalVersionsOfItemAtURL4:(NSURL *)url completionHandler:(void (^)(int nonLocalFileVersions, NSError *error, NSURL*))completionHandler;
@end

#endif
