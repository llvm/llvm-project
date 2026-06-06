@interface NSObject 
@end

typedef unsigned char BOOL; 
#ifndef NS_AVAILABLE
#define NS_AVAILABLE(x,y) __attribute__((availability(macosx,introduced=x)))
#endif 
#ifndef NS_UNAVAILABLE
#define NS_UNAVAILABLE  __attribute__((unavailable))
#endif 
#ifndef NS_DEPRECATED_MAC
#define NS_DEPRECATED_MAC(x,y) __attribute__((availability(macosx,introduced=x,deprecated=y,message="" )));
#endif 

@interface NSManagedObject
@end 

@interface NSSet 
@end 
