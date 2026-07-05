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

// Run the same test with second.h being modular
// RUN: cat %t/include/second.modulemap >> %t/include/module.modulemap
// RUN: %clang_cc1 -I%t/include -verify %t/test.m -fblocks -fobjc-arc -DTEST_MODULAR=1 \
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
//--- include/second.modulemap
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
#ifdef TEST_MODULAR
// expected-note@second.h:* {{but in 'Second' found 0 referenced protocols}}
#else
// expected-note@second.h:* {{but in definition here found 0 referenced protocols}}
#endif
id<CompareProtocolPresence2> compareProtocolPresence2;
// expected-error@first.h:* {{'CompareProtocolPresence2' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 0 referenced protocols}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found 1 referenced protocol}}

id<CompareDifferentProtocols> compareDifferentProtocols;
// expected-error@first.h:* {{'CompareDifferentProtocols' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 1st referenced protocol with name 'CommonProtocol'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found 1st referenced protocol with different name 'ExtraProtocol'}}
id<CompareProtocolOrder> compareProtocolOrder;
// expected-error@first.h:* {{'CompareProtocolOrder' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 1st referenced protocol with name 'CommonProtocol'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found 1st referenced protocol with different name 'ExtraProtocol'}}
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
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found end of class}}
id<CompareMethodPresence2> compareMethodPresence2;
// expected-error@first.h:* {{'CompareMethodPresence2' has different definitions in different modules; first difference is definition in module 'First.Hidden' found end of class}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found method}}
id<CompareMethodName> compareMethodName;
// expected-error@first.h:* {{'CompareMethodName' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodNameA'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found different method 'methodNameB'}}

id<CompareMethodArgCount> compareMethodArgCount;
// expected-error@first.h:* {{'CompareMethodArgCount' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodArgCount::' that has 2 parameters}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found method 'methodArgCount:' that has 1 parameter}}
id<CompareMethodArgName> compareMethodArgName;
// expected-error@first.h:* {{'CompareMethodArgName' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodArgName:' with 1st parameter named 'argNameA'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found method 'methodArgName:' with 1st parameter named 'argNameB'}}
id<CompareMethodArgType> compareMethodArgType;
// expected-error@first.h:* {{'CompareMethodArgType' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodArgType:' with 1st parameter of type 'int'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found method 'methodArgType:' with 1st parameter of type 'float'}}

id<CompareMethodReturnType> compareMethodReturnType;
// expected-error@first.h:* {{'CompareMethodReturnType' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodReturnType' with return type 'int'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found method 'methodReturnType' with different return type 'float'}}

id<CompareMethodOrder> compareMethodOrder;
// expected-error@first.h:* {{'CompareMethodOrder' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodOrderFirst'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found different method 'methodOrderSecond'}}
id<CompareMethodClassInstance> compareMethodClassInstance;
// expected-error@first.h:* {{'CompareMethodClassInstance' has different definitions in different modules; first difference is definition in module 'First.Hidden' found instance method 'methodClassInstance'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found method 'methodClassInstance' as class method}}

id<CompareMethodRequirednessExplicit> compareMethodRequirednessExplicit;
// expected-error@first.h:* {{'CompareMethodRequirednessExplicit' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 'optional' method control}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found 'required' method control}}
id<CompareMethodRequirednessDefault> compareMethodRequirednessDefault; // no error
#endif

#if defined(FIRST)
@protocol CompareMatchingProperties
@property int matchingPropName;
@end

@protocol ComparePropertyPresence1
@property int propPresence1;
@end
@protocol ComparePropertyPresence2
@end

@protocol ComparePropertyName
@property int propNameA;
@end

@protocol ComparePropertyType
@property int propType;
@end

@protocol ComparePropertyOrder
@property int propOrderX;
@property int propOrderY;
@end

@protocol CompareMatchingPropertyAttributes
@property (nonatomic, assign) int matchingProp;
@end
@protocol ComparePropertyAttributes
@property (nonatomic) int propAttributes;
@end
// Edge cases.
@protocol CompareFirstImplAttribute
@property int firstImplAttribute;
@end
@protocol CompareLastImplAttribute
// Cannot test with protocols 'direct' attribute because it's not allowed.
@property (class) int lastImplAttribute;
@end
#elif defined(SECOND)
@protocol CompareMatchingProperties
@property int matchingPropName;
@end

@protocol ComparePropertyPresence1
@end
@protocol ComparePropertyPresence2
@property int propPresence2;
@end

@protocol ComparePropertyName
@property int propNameB;
@end

@protocol ComparePropertyType
@property float propType;
@end

@protocol ComparePropertyOrder
@property int propOrderY;
@property int propOrderX;
@end

@protocol CompareMatchingPropertyAttributes
@property (assign, nonatomic) int matchingProp;
@end
@protocol ComparePropertyAttributes
@property (atomic) int propAttributes;
@end
// Edge cases.
@protocol CompareFirstImplAttribute
@property (readonly) int firstImplAttribute;
@end
@protocol CompareLastImplAttribute
@property int lastImplAttribute;
@end
#else
id<CompareMatchingProperties> compareMatchingProperties;
id<ComparePropertyPresence1> comparePropertyPresence1;
// expected-error@first.h:* {{'ComparePropertyPresence1' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found end of class}}
id<ComparePropertyPresence2> comparePropertyPresence2;
// expected-error@first.h:* {{'ComparePropertyPresence2' has different definitions in different modules; first difference is definition in module 'First.Hidden' found end of class}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property}}
id<ComparePropertyName> comparePropertyName;
// expected-error@first.h:* {{'ComparePropertyName' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property 'propNameA'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property 'propNameB'}}
id<ComparePropertyType> comparePropertyType;
// expected-error@first.h:* {{'ComparePropertyType' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property 'propType' with type 'int'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property 'propType' with type 'float'}}
id<ComparePropertyOrder> comparePropertyOrder;
// expected-error@first.h:* {{'ComparePropertyOrder' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property 'propOrderX'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property 'propOrderY'}}

id<CompareMatchingPropertyAttributes> compareMatchingPropertyAttributes;
id<ComparePropertyAttributes> comparePropertyAttributes;
// expected-error@first.h:* {{'ComparePropertyAttributes' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property 'propAttributes' with 'nonatomic' attribute}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property 'propAttributes' with different 'nonatomic' attribute}}
id<CompareFirstImplAttribute> compareFirstImplAttribute;
// expected-error@first.h:* {{'CompareFirstImplAttribute' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property 'firstImplAttribute' with default 'readonly' attribute}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property 'firstImplAttribute' with different 'readonly' attribute}}
id<CompareLastImplAttribute> compareLastImplAttribute;
// expected-error@first.h:* {{'CompareLastImplAttribute' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property 'lastImplAttribute' with 'class' attribute}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property 'lastImplAttribute' with different 'class' attribute}}
#endif
