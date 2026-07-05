// RUN: %clang_cc1  -fsyntax-only -verify %s
// expected-no-diagnostics

@interface SStoreNodeInfo 

@property(nonatomic,readonly,retain) id descriptionShort;

- (id)stringByAppendingFormat:(int)format, ... ;

@end

@interface SStoreNodeInfo_iDisk : SStoreNodeInfo
{
@private
 id _etag;
}
@end

@implementation SStoreNodeInfo_iDisk
- (id) X { return [super.descriptionShort stringByAppendingFormat:1, _etag]; }

@end
