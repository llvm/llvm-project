// RUN: rm -rf %t
// RUN: split-file %s %t

// Build first header file
// RUN: echo "#define FIRST" >> %t/include/first.h
// RUN: cat %t/test.m        >> %t/include/first.h
// RUN: echo "#undef FIRST"  >> %t/include/first.h

// Build second header file
// RUN: echo "#define SECOND" >> %t/include/second.h
// RUN: cat %t/test.m         >> %t/include/second.h
// RUN: echo "#undef SECOND"  >> %t/include/second.h

// Test that each header can compile
// RUN: %clang_cc1 -fsyntax-only -x objective-c %t/include/first.h -fblocks -fobjc-arc
// RUN: %clang_cc1 -fsyntax-only -x objective-c %t/include/second.h -fblocks -fobjc-arc

// Run test
// RUN: %clang_cc1 -I%t/include -verify %t/test.m -fblocks -fobjc-arc \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// In non-modular case we ignore protocol redefinitions. But with modules
// previous definition can come from a hidden [sub]module. And in this case we
// allow a new definition if it is equivalent to the hidden one.
//
// This test case is to verify equivalence checks.

//--- include/common.h
#ifndef COMMON_H
#define COMMON_H
@protocol CommonProtocol @end
@protocol ExtraProtocol @end
#endif

//--- include/first-empty.h
//--- include/module.modulemap
module Common {
  header "common.h"
  export *
}
module First {
  module Empty {
    header "first-empty.h"
  }
  module Hidden {
    header "first.h"
    export *
  }
}
module Second {
  header "second.h"
  export *
}

//--- test.m
#if defined(FIRST) || defined(SECOND)
# include "common.h"
#endif

#if !defined(FIRST) && !defined(SECOND)
# include "first-empty.h"
# include "second.h"
#endif

#if defined(FIRST)
@protocol CompareForwardDeclaration1;
@protocol CompareForwardDeclaration2<CommonProtocol> @end
#elif defined(SECOND)
@protocol CompareForwardDeclaration1<CommonProtocol> @end
@protocol CompareForwardDeclaration2;
#else
id<CompareForwardDeclaration1> compareForwardDeclaration1;
id<CompareForwardDeclaration2> compareForwardDeclaration2;
#endif

#if defined(FIRST)
@protocol CompareMatchingConformingProtocols<CommonProtocol> @end
@protocol ForwardProtocol;
@protocol CompareMatchingConformingForwardProtocols<ForwardProtocol> @end

@protocol CompareProtocolPresence1<CommonProtocol> @end
@protocol CompareProtocolPresence2 @end

@protocol CompareDifferentProtocols<CommonProtocol> @end
@protocol CompareProtocolOrder<CommonProtocol, ExtraProtocol> @end
#elif defined(SECOND)
@protocol CompareMatchingConformingProtocols<CommonProtocol> @end
@protocol ForwardProtocol @end
@protocol CompareMatchingConformingForwardProtocols<ForwardProtocol> @end

@protocol CompareProtocolPresence1 @end
@protocol CompareProtocolPresence2<CommonProtocol> @end

@protocol CompareDifferentProtocols<ExtraProtocol> @end
@protocol CompareProtocolOrder<ExtraProtocol, CommonProtocol> @end
#else
id<CompareMatchingConformingProtocols> compareMatchingConformingProtocols;
id<CompareMatchingConformingForwardProtocols> compareMatchingConformingForwardProtocols;

id<CompareProtocolPresence1> compareProtocolPresence1;
// expected-error@first.h:* {{'CompareProtocolPresence1' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 1 referenced protocol}}
// expected-note@second.h:* {{but in 'Second' found 0 referenced protocols}}
id<CompareProtocolPresence2> compareProtocolPresence2;
// expected-error@first.h:* {{'CompareProtocolPresence2' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 0 referenced protocols}}
// expected-note@second.h:* {{but in 'Second' found 1 referenced protocol}}

id<CompareDifferentProtocols> compareDifferentProtocols;
// expected-error@first.h:* {{'CompareDifferentProtocols' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 1st referenced protocol with name 'CommonProtocol'}}
// expected-note@second.h:* {{but in 'Second' found 1st referenced protocol with different name 'ExtraProtocol'}}
id<CompareProtocolOrder> compareProtocolOrder;
// expected-error@first.h:* {{'CompareProtocolOrder' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 1st referenced protocol with name 'CommonProtocol'}}
// expected-note@second.h:* {{but in 'Second' found 1st referenced protocol with different name 'ExtraProtocol'}}
#endif

#if defined(FIRST)
@protocol CompareMatchingMethods
- (float)matchingMethod:(int)arg;
@end

@protocol CompareMethodPresence1
- (void)presenceMethod1;
@end
@protocol CompareMethodPresence2
@end

@protocol CompareMethodName
- (void)methodNameA;
@end

@protocol CompareMethodArgCount
- (void)methodArgCount:(int)arg0 :(int)arg1;
@end
@protocol CompareMethodArgName
- (void)methodArgName:(int)argNameA;
@end
@protocol CompareMethodArgType
- (void)methodArgType:(int)argType;
@end

@protocol CompareMethodReturnType
- (int)methodReturnType;
@end

@protocol CompareMethodOrder
- (void)methodOrderFirst;
- (void)methodOrderSecond;
@end

@protocol CompareMethodClassInstance
- (void)methodClassInstance;
@end

@protocol CompareMethodRequirednessExplicit
@optional
- (void)methodRequiredness;
@end
@protocol CompareMethodRequirednessDefault
// @required is default
- (void)methodRequiredness;
@end
#elif defined(SECOND)
@protocol CompareMatchingMethods
- (float)matchingMethod:(int)arg;
@end

@protocol CompareMethodPresence1
@end
@protocol CompareMethodPresence2
- (void)presenceMethod2;
@end

@protocol CompareMethodName
- (void)methodNameB;
@end

@protocol CompareMethodArgCount
- (void)methodArgCount:(int)arg0;
@end
@protocol CompareMethodArgName
- (void)methodArgName:(int)argNameB;
@end
@protocol CompareMethodArgType
- (void)methodArgType:(float)argType;
@end

@protocol CompareMethodReturnType
- (float)methodReturnType;
@end

@protocol CompareMethodOrder
- (void)methodOrderSecond;
- (void)methodOrderFirst;
@end

@protocol CompareMethodClassInstance
+ (void)methodClassInstance;
@end

@protocol CompareMethodRequirednessExplicit
@required
- (void)methodRequiredness;
@end
@protocol CompareMethodRequirednessDefault
@required
- (void)methodRequiredness;
@end
#else
id<CompareMatchingMethods> compareMatchingMethods; // no error
id<CompareMethodPresence1> compareMethodPresence1;
// expected-error@first.h:* {{'CompareMethodPresence1' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method}}
// expected-note@second.h:* {{but in 'Second' found end of class}}
id<CompareMethodPresence2> compareMethodPresence2;
// expected-error@first.h:* {{'CompareMethodPresence2' has different definitions in different modules; first difference is definition in module 'First.Hidden' found end of class}}
// expected-note@second.h:* {{but in 'Second' found method}}
id<CompareMethodName> compareMethodName;
// expected-error@first.h:* {{'CompareMethodName' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodNameA'}}
// expected-note@second.h:* {{but in 'Second' found different method 'methodNameB'}}

id<CompareMethodArgCount> compareMethodArgCount;
// expected-error@first.h:* {{'CompareMethodArgCount' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodArgCount::' that has 2 parameters}}
// expected-note@second.h:* {{but in 'Second' found method 'methodArgCount:' that has 1 parameter}}
id<CompareMethodArgName> compareMethodArgName;
// expected-error@first.h:* {{'CompareMethodArgName' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodArgName:' with 1st parameter named 'argNameA'}}
// expected-note@second.h:* {{but in 'Second' found method 'methodArgName:' with 1st parameter named 'argNameB'}}
id<CompareMethodArgType> compareMethodArgType;
// expected-error@first.h:* {{'CompareMethodArgType' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodArgType:' with 1st parameter of type 'int'}}
// expected-note@second.h:* {{but in 'Second' found method 'methodArgType:' with 1st parameter of type 'float'}}

id<CompareMethodReturnType> compareMethodReturnType;
// expected-error@first.h:* {{'CompareMethodReturnType' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodReturnType' with return type 'int'}}
// expected-note@second.h:* {{but in 'Second' found method 'methodReturnType' with different return type 'float'}}

id<CompareMethodOrder> compareMethodOrder;
// expected-error@first.h:* {{'CompareMethodOrder' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodOrderFirst'}}
// expected-note@second.h:* {{but in 'Second' found different method 'methodOrderSecond'}}
id<CompareMethodClassInstance> compareMethodClassInstance;
// expected-error@first.h:* {{'CompareMethodClassInstance' has different definitions in different modules; first difference is definition in module 'First.Hidden' found instance method 'methodClassInstance'}}
// expected-note@second.h:* {{but in 'Second' found method 'methodClassInstance' as class method}}

id<CompareMethodRequirednessExplicit> compareMethodRequirednessExplicit;
// expected-error@first.h:* {{'CompareMethodRequirednessExplicit' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 'optional' method control}}
// expected-note@second.h:* {{but in 'Second' found 'required' method control}}
id<CompareMethodRequirednessDefault> compareMethodRequirednessDefault; // no error
#endif
