// RUN: %clang_cc1 -triple=i686-apple-darwin9 -emit-llvm -o %t %s
// RUN: grep -e "T@\\\\22NSString\\\\22" %t
@interface NSString @end

typedef NSString StoreVersionID ;

@interface Parent 
  @property(retain) StoreVersionID* foo;
@end

@implementation Parent
@dynamic foo;
@end
