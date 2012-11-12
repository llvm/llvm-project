// RUN: %clang_cc1 -verify -fms-extensions -Wno-objc-root-class %s
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
typedef struct _NSRange { } NSRange;

@class PBXFileReference;

@interface PBXDocBookmark
+ alloc;	// expected-note {{method definition for 'alloc' not found}}
- autorelease;	// expected-note {{method definition for 'autorelease' not found}}
@end

// GCC allows pointer expressions in integer constant expressions.
struct {
  char control[((int)(char *)2)];
} xx;

@implementation PBXDocBookmark  // expected-warning {{incomplete implementation}}

+ (id)bookmarkWithFileReference:(PBXFileReference *)fileRef gylphRange:(NSRange)range anchor:(NSString *)htmlAnchor
{
    NSRange r = (NSRange)range;
    return [[[self alloc] initWithFileReference:fileRef gylphRange:(NSRange)range anchor:(NSString *)htmlAnchor] autorelease];  // expected-warning {{method '-initWithFileReference:gylphRange:anchor:' not found (return type defaults to 'id')}}
}
@end
