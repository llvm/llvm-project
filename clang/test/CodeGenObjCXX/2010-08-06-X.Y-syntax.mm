// RUN: %clang_cc1 -emit-llvm %s -o -
struct TFENode {
  TFENode(const TFENode& inNode);
};

@interface TIconViewController
- (const TFENode&) target;
@end

void sortAllChildrenForNode(const TFENode&node);

@implementation TIconViewController
- (void) setArrangeBy {
  sortAllChildrenForNode(self.target);
}
@end
