// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: not grep objc_assign_ivar %t
// RUN: grep objc_assign_strongCast %t | count 8
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: not grep objc_assign_ivar %t
// RUN: grep objc_assign_strongCast %t | count 8

@interface TestUnarchiver 
{
	void  *allUnarchivedObjects;
}
@end

@implementation TestUnarchiver

struct unarchive_list {
    int ifield;
    id *list;
};

- (id)init {
    (*((struct unarchive_list *)allUnarchivedObjects)).list = 0;
    ((struct unarchive_list *)allUnarchivedObjects)->list = 0;
    (**((struct unarchive_list **)allUnarchivedObjects)).list = 0;
    (*((struct unarchive_list **)allUnarchivedObjects))->list = 0;
    return 0;
}

@end

// rdar://10191569
@interface I
{
  struct S {
    id _timer;
  } *p_animationState;
}
@end

@implementation I
- (void) Meth {
  p_animationState->_timer = 0;
  (*p_animationState)._timer = 0;
  (&(*p_animationState))->_timer = 0;
}
@end
