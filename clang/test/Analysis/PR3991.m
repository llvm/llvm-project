// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core -analyzer-store=region -verify -triple x86_64-apple-darwin9 -Wno-incomplete-implementation %s
// expected-no-diagnostics

//===----------------------------------------------------------------------===//
// Delta-debugging produced forward declarations.
//===----------------------------------------------------------------------===//

typedef signed char BOOL;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object;
@end  @protocol NSCopying  - (id)copyWithZone:(NSZone *)zone;
@end  @protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end  @protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end    @interface NSObject <NSObject> {
}
@end    extern id <NSObject> NSAllocateObject(Class aClass, unsigned extraBytes, NSZone *zone);
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding>  - (unsigned)count;
@end   @class NSTimer, NSPort, NSArray;
@class NSURLHandle, NSMutableArray, NSMutableData, NSData, NSURL;
@interface NSResponder : NSObject <NSCoding> {
}
@end      @class NSBitmapImageRep, NSCursor, NSGraphicsContext, NSImage, NSPasteboard, NSScrollView, NSWindow, NSAttributedString;
@interface NSView : NSResponder {
  struct __VFlags2 {
  }
  _vFlags2;
}
@end @class NSTextField, NSPanel, NSArray, NSWindow, NSImage, NSButton, NSError;
@interface NSBox : NSView {
}
@end @class GDataFeedDocList, GDataServiceTicket, GDataServiceTicket, IHGoogleDocsAdapter;
@protocol IHGoogleDocsAdapterDelegate  - (void)googleDocsAdapter:(IHGoogleDocsAdapter*)inGoogleDocsAdapter accountVerifyIsValid:(BOOL)inIsValid error:(NSError *)inError;
@end   @interface IHGoogleDocsAdapter : NSObject {
}
- (NSArray *)entries;
@end extern Class const kGDataUseRegisteredClass ;
@interface IHGoogleDocsAdapter ()  - (GDataFeedDocList *)feedDocList;
- (NSArray *)directoryPathComponents;
- (unsigned int)currentPathComponentIndex;
- (void)setCurrentPathComponentIndex:(unsigned int)aCurrentPathComponentIndex;
- (NSURL *)folderFeedURL;
@end  

@implementation IHGoogleDocsAdapter    - (id)initWithUsername:(NSString *)inUsername password:(NSString *)inPassword owner:(NSObject <IHGoogleDocsAdapterDelegate> *)owner {
  return 0;
}

//===----------------------------------------------------------------------===//
// Actual test case:
//
// The analyzer currently doesn't reason about ObjCKVCRefExpr.  Have both
// GRExprEngine::Visit and GRExprEngine::VisitLValue have such expressions
// evaluate to UnknownVal.
//===----------------------------------------------------------------------===//

- (void)docListListFetchTicket:(GDataServiceTicket *)ticket               finishedWithFeed:(GDataFeedDocList *)feed {
  BOOL doGetDir = self.directoryPathComponents != 0 && self.currentPathComponentIndex < [self.directoryPathComponents count];
  if (doGetDir)  {
    BOOL isDirExisting = [[self.feedDocList entries] count] > 0;
    if (isDirExisting)   {
      if (self.folderFeedURL != 0)    {
        if (++self.currentPathComponentIndex == [self.directoryPathComponents count])     {
        }
      }
    }
  }
}
@end
