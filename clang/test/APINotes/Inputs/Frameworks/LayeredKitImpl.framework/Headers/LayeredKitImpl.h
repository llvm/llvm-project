@protocol UpwardProto;
@class UpwardClass;

@interface PerfectlyNormalClass
@end

void doImplementationThings(UpwardClass *first, id <UpwardProto> second) __attribute((unavailable));
