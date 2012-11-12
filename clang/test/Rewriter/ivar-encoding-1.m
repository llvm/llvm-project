// RUN: %clang_cc1 -rewrite-objc -fobjc-runtime=macosx-fragile-10.5  %s -o -

@interface Intf
{
  id ivar;
  id ivar1[12];

  id **ivar3;

  id (*ivar4) (id, id);
}
@end

@implementation Intf
@end
