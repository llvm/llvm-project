// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

@interface SSyncCEList
{
	id _list;
}
@end

@implementation SSyncCEList

- (id) list { return 0; }
@end

@interface SSyncConflictList : SSyncCEList
@end

@implementation SSyncConflictList

- (id)Meth : (SSyncConflictList*)other
  {
    return other.list;
  }
@end

