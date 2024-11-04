@interface A
-(void)m0;
@end

@interface B
-(void)m0;
+(void)m1;
@end

#define CDef \
@interface C \
-(void)m4; \
@end

CDef

@protocol P0
-(void)m5;
@end
