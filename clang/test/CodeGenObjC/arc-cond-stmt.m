// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -o - %s
// rdar://10327068

@class NSString;

@interface NSAssertionHandler {
}

+ (NSAssertionHandler *)currentHandler;

- (void)handleFailureInMethod:(SEL)selector object:(id)object file:(NSString *)fileName lineNumber:(int)line ,...;

@end

typedef enum
{
 MWRaceOrder_MeetName,
 MWRaceOrder_MeetPosition,
 MWRaceOrder_MeetDistance,
 MWRaceOrder_Name,
 MWRaceOrder_Position,
 MWRaceOrder_Distance,
 MWRaceOrder_Default = MWRaceOrder_Name,
 MWRaceOrder_MeetDefault = MWRaceOrder_MeetName,
} MWRaceOrder;

@interface MWViewMeetController
@property (nonatomic, assign) MWRaceOrder raceOrder;
@end

@implementation MWViewMeetController

- (int)orderSegment
{
 switch (self.raceOrder)
 {

  default:
  { [(NSAssertionHandler *)0 handleFailureInMethod:_cmd object:self file:(NSString*)0 lineNumber:192 ]; };
   break;
 }

 return 0;
}

@synthesize raceOrder;

@end
