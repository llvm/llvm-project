@interface ExplicitIVarsInInterface {
  int _requiresImplementationTU;
}

@property int requiresImplementationTU;

@end

// RUN: clang-refactor-test rename-initiate -at=%s:2:7 -new-name=foo -implementation-tu="%S/Inputs/ObjCImplementationTURequestsImplementation.m" -dump-symbols %s | FileCheck --check-prefix=CHECK1 %s
// CHECK1: Implementation TU USR: 'c:objc(cs)ExplicitIVarsInInterface@_requiresImplementationTU'

// RUN: not clang-refactor-test rename-initiate -at=%s:2:7 -new-name=foo -implementation-tu="%S/MissingFile.m" -dump-symbols %s 2>&1 | FileCheck --check-prefix=CHECK-ERR1 %s
// CHECK-ERR1: failed to load implementation TU

@interface NoNeedForImplementationTUs {
  int _p1;
}

@property int p1;
@property int p2;

@end

@implementation NoNeedForImplementationTUs {
  int _p2;
}

@end

// RUN: clang-refactor-test rename-initiate -at=%s:16:7 -new-name=foo %s | FileCheck --check-prefix=CHECK-NO %s
// RUN: clang-refactor-test rename-initiate -at=%s:25:7 -new-name=foo %s | FileCheck --check-prefix=CHECK-NO %s
// CHECK-NO-NOT: Implementation TU USR
