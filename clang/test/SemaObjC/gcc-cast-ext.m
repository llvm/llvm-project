// RUN: %clang_cc1 -verify -Wno-pointer-to-int-cast -Wno-objc-root-class %s
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
typedef struct _NSRange { } NSRange;

@class PBXFileReference;

@interface PBXDocBookmark
+ alloc;	// expected-note {{method 'alloc' declared here}}
- autorelease;	// expected-note {{method 'autorelease' declared here}}
@end

// GCC allows pointer expressions in integer constant expressions.
struct {
  char control[((int)(char *)2)]; // expected-warning {{extension}}
} xx;

@implementation PBXDocBookmark // expected-warning {{method definition for 'autorelease' not found}}\
                               // expected-warning {{method definition for 'alloc' not found}}

+ (id)bookmarkWithFileReference:(PBXFileReference *)fileRef gylphRange:(NSRange)range anchor:(NSString *)htmlAnchor
{
    NSRange r = (NSRange)range;
    return [[[self alloc] initWithFileReference:fileRef gylphRange:(NSRange)range anchor:(NSString *)htmlAnchor] autorelease];  // expected-warning {{method '-initWithFileReference:gylphRange:anchor:' not found (return type defaults to 'id')}}
}
@end
