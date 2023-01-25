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

// Test that we don't accept different class definitions with the same name
// from multiple modules but detect mismatches and provide actionable
// diagnostic.

//--- include/common.h
#ifndef COMMON_H
#define COMMON_H
@interface NSObject @end
@protocol CommonProtocol @end
@protocol ExtraProtocol @end
#endif

//--- include/first-empty.h
//--- include/module.modulemap
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
@class CompareForwardDeclaration1;
@interface CompareForwardDeclaration2: NSObject @end
#elif defined(SECOND)
@interface CompareForwardDeclaration1: NSObject @end
@class CompareForwardDeclaration2;
#else
CompareForwardDeclaration1 *compareForwardDeclaration1;
CompareForwardDeclaration2 *compareForwardDeclaration2;
#endif

#if defined(FIRST)
@interface CompareMatchingSuperclass: NSObject @end

@interface CompareSuperclassPresence1: NSObject @end
@interface CompareSuperclassPresence2 @end

@interface CompareDifferentSuperclass: NSObject @end
#elif defined(SECOND)
@interface CompareMatchingSuperclass: NSObject @end

@interface CompareSuperclassPresence1 @end
@interface CompareSuperclassPresence2: NSObject @end

@interface DifferentSuperclass: NSObject @end
@interface CompareDifferentSuperclass: DifferentSuperclass @end
#else
CompareMatchingSuperclass *compareMatchingSuperclass;
CompareSuperclassPresence1 *compareSuperclassPresence1;
// expected-error@first.h:* {{'CompareSuperclassPresence1' has different definitions in different modules; first difference is definition in module 'First.Hidden' found super class with type 'NSObject'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found no super class}}
CompareSuperclassPresence2 *compareSuperclassPresence2;
// expected-error@first.h:* {{'CompareSuperclassPresence2' has different definitions in different modules; first difference is definition in module 'First.Hidden' found no super class}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found super class with type 'NSObject'}}
CompareDifferentSuperclass *compareDifferentSuperclass;
// expected-error@first.h:* {{'CompareDifferentSuperclass' has different definitions in different modules; first difference is definition in module 'First.Hidden' found super class with type 'NSObject'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found super class with type 'DifferentSuperclass'}}
#endif

#if defined(FIRST)
@interface CompareMatchingConformingProtocols: NSObject<CommonProtocol> @end
@protocol ForwardProtocol;
@interface CompareMatchingConformingForwardProtocols: NSObject<ForwardProtocol> @end

@interface CompareProtocolPresence1: NSObject<CommonProtocol> @end
@interface CompareProtocolPresence2: NSObject @end

@interface CompareDifferentProtocols: NSObject<CommonProtocol> @end
@interface CompareProtocolOrder: NSObject<CommonProtocol, ExtraProtocol> @end
#elif defined(SECOND)
@interface CompareMatchingConformingProtocols: NSObject<CommonProtocol> @end
@protocol ForwardProtocol @end
@interface CompareMatchingConformingForwardProtocols: NSObject<ForwardProtocol> @end

@interface CompareProtocolPresence1: NSObject @end
@interface CompareProtocolPresence2: NSObject<CommonProtocol> @end

@interface CompareDifferentProtocols: NSObject<ExtraProtocol> @end
@interface CompareProtocolOrder: NSObject<ExtraProtocol, CommonProtocol> @end
#else
CompareMatchingConformingProtocols *compareMatchingConformingProtocols;
CompareMatchingConformingForwardProtocols *compareMatchingConformingForwardProtocols;

CompareProtocolPresence1 *compareProtocolPresence1;
// expected-error@first.h:* {{'CompareProtocolPresence1' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 1 referenced protocol}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found 0 referenced protocols}}
CompareProtocolPresence2 *compareProtocolPresence2;
// expected-error@first.h:* {{'CompareProtocolPresence2' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 0 referenced protocols}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found 1 referenced protocol}}

CompareDifferentProtocols *compareDifferentProtocols;
// expected-error@first.h:* {{'CompareDifferentProtocols' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 1st referenced protocol with name 'CommonProtocol'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found 1st referenced protocol with different name 'ExtraProtocol'}}
CompareProtocolOrder *compareProtocolOrder;
// expected-error@first.h:* {{'CompareProtocolOrder' has different definitions in different modules; first difference is definition in module 'First.Hidden' found 1st referenced protocol with name 'CommonProtocol'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found 1st referenced protocol with different name 'ExtraProtocol'}}
#endif

#if defined(FIRST)
@interface CompareMatchingIVars: NSObject { int ivarName; } @end

@interface CompareIVarPresence1: NSObject @end
@interface CompareIVarPresence2: NSObject { int ivarPresence2; } @end

@interface CompareIVarName: NSObject { int ivarName; } @end
@interface CompareIVarType: NSObject { int ivarType; } @end
@interface CompareIVarOrder: NSObject {
  int ivarNameInt;
  float ivarNameFloat;
}
@end

@interface CompareIVarVisibilityExplicit: NSObject {
@public
  int ivarVisibility;
}
@end
@interface CompareIVarVisibilityDefault: NSObject {
  int ivarVisibilityDefault;
}
@end
#elif defined(SECOND)
@interface CompareMatchingIVars: NSObject { int ivarName; } @end

@interface CompareIVarPresence1: NSObject { int ivarPresence1; } @end
@interface CompareIVarPresence2: NSObject @end

@interface CompareIVarName: NSObject { int differentIvarName; } @end
@interface CompareIVarType: NSObject { float ivarType; } @end
@interface CompareIVarOrder: NSObject {
  float ivarNameFloat;
  int ivarNameInt;
}
@end

@interface CompareIVarVisibilityExplicit: NSObject {
@private
  int ivarVisibility;
}
@end
@interface CompareIVarVisibilityDefault: NSObject {
@public
  int ivarVisibilityDefault;
}
@end
#else
CompareMatchingIVars *compareMatchingIVars;

CompareIVarPresence1 *compareIVarPresence1;
#ifdef TEST_MODULAR
// expected-error@second.h:* {{'CompareIVarPresence1::ivarPresence1' from module 'Second' is not present in definition of 'CompareIVarPresence1' in module 'First.Hidden'}}
// expected-note@first.h:* {{definition has no member 'ivarPresence1'}}
#else
// expected-error@first.h:* {{'CompareIVarPresence1' has different definitions in different modules; first difference is definition in module 'First.Hidden' found end of class}}
// expected-note@second.h:* {{but in definition here found instance variable}}
#endif
CompareIVarPresence2 *compareIVarPresence2;
// expected-error@first.h:* {{'CompareIVarPresence2' has different definitions in different modules; first difference is definition in module 'First.Hidden' found instance variable}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found end of class}}

CompareIVarName *compareIVarName;
#ifdef TEST_MODULAR
// expected-error@second.h:* {{'CompareIVarName::differentIvarName' from module 'Second' is not present in definition of 'CompareIVarName' in module 'First.Hidden'}}
// expected-note@first.h:* {{definition has no member 'differentIvarName'}}
#else
// expected-error@first.h:* {{'CompareIVarName' has different definitions in different modules; first difference is definition in module 'First.Hidden' found field 'ivarName'}}
// expected-note@second.h:* {{but in definition here found field 'differentIvarName'}}
#endif
CompareIVarType *compareIVarType;
#ifdef TEST_MODULAR
// expected-error@second.h:* {{'CompareIVarType::ivarType' from module 'Second' is not present in definition of 'CompareIVarType' in module 'First.Hidden'}}
// expected-note@first.h:* {{declaration of 'ivarType' does not match}}
#else
// expected-error@first.h:* {{'CompareIVarType' has different definitions in different modules; first difference is definition in module 'First.Hidden' found field 'ivarType' with type 'int'}}
// expected-note@second.h:* {{but in definition here found field 'ivarType' with type 'float'}}
#endif
CompareIVarOrder *compareIVarOrder;
// expected-error@first.h:* {{'CompareIVarOrder' has different definitions in different modules; first difference is definition in module 'First.Hidden' found field 'ivarNameInt'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found field 'ivarNameFloat'}}

CompareIVarVisibilityExplicit *compareIVarVisibilityExplicit;
// expected-error@first.h:* {{'CompareIVarVisibilityExplicit' has different definitions in different modules; first difference is definition in module 'First.Hidden' found instance variable 'ivarVisibility' access control is @public}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found instance variable 'ivarVisibility' access control is @private}}
CompareIVarVisibilityDefault *compareIVarVisibilityDefault;
// expected-error@first.h:* {{'CompareIVarVisibilityDefault' has different definitions in different modules; first difference is definition in module 'First.Hidden' found instance variable 'ivarVisibilityDefault' access control is @protected}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found instance variable 'ivarVisibilityDefault' access control is @public}}
#endif

#if defined(FIRST)
@interface CompareMatchingMethods: NSObject
- (float)matchingMethod:(int)arg;
@end

@interface CompareMethodPresence1: NSObject
- (void)presenceMethod1;
@end
@interface CompareMethodPresence2: NSObject
@end

@interface CompareMethodName: NSObject
- (void)methodNameA;
@end

@interface CompareMethodArgCount: NSObject
- (void)methodArgCount:(int)arg0 :(int)arg1;
@end
@interface CompareMethodArgName: NSObject
- (void)methodArgName:(int)argNameA;
@end
@interface CompareMethodArgType: NSObject
- (void)methodArgType:(int)argType;
@end

@interface CompareMethodReturnType: NSObject
- (int)methodReturnType;
@end

@interface CompareMethodOrder: NSObject
- (void)methodOrderFirst;
- (void)methodOrderSecond;
@end

@interface CompareMethodClassInstance: NSObject
+ (void)methodClassInstance;
@end
#elif defined(SECOND)
@interface CompareMatchingMethods: NSObject
- (float)matchingMethod:(int)arg;
@end

@interface CompareMethodPresence1: NSObject
@end
@interface CompareMethodPresence2: NSObject
- (void)presenceMethod2;
@end

@interface CompareMethodName: NSObject
- (void)methodNameB;
@end

@interface CompareMethodArgCount: NSObject
- (void)methodArgCount:(int)arg0;
@end
@interface CompareMethodArgName: NSObject
- (void)methodArgName:(int)argNameB;
@end
@interface CompareMethodArgType: NSObject
- (void)methodArgType:(float)argType;
@end

@interface CompareMethodReturnType: NSObject
- (float)methodReturnType;
@end

@interface CompareMethodOrder: NSObject
- (void)methodOrderSecond;
- (void)methodOrderFirst;
@end

@interface CompareMethodClassInstance: NSObject
- (void)methodClassInstance;
@end
#else
CompareMatchingMethods *compareMatchingMethods;
CompareMethodPresence1 *compareMethodPresence1;
// expected-error@first.h:* {{'CompareMethodPresence1' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found end of class}}
CompareMethodPresence2 *compareMethodPresence2;
// expected-error@first.h:* {{'CompareMethodPresence2' has different definitions in different modules; first difference is definition in module 'First.Hidden' found end of class}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found method}}
CompareMethodName *compareMethodName;
// expected-error@first.h:* {{'CompareMethodName' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodNameA'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found different method 'methodNameB'}}

CompareMethodArgCount *compareMethodArgCount;
// expected-error@first.h:* {{'CompareMethodArgCount' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodArgCount::' that has 2 parameters}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found method 'methodArgCount:' that has 1 parameter}}
CompareMethodArgName *compareMethodArgName;
// expected-error@first.h:* {{'CompareMethodArgName' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodArgName:' with 1st parameter named 'argNameA'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found method 'methodArgName:' with 1st parameter named 'argNameB'}}
CompareMethodArgType *compareMethodArgType;
// expected-error@first.h:* {{'CompareMethodArgType' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodArgType:' with 1st parameter of type 'int'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found method 'methodArgType:' with 1st parameter of type 'float'}}

CompareMethodReturnType *compareMethodReturnType;
// expected-error@first.h:* {{'CompareMethodReturnType' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodReturnType' with return type 'int'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found method 'methodReturnType' with different return type 'float'}}

CompareMethodOrder *compareMethodOrder;
// expected-error@first.h:* {{'CompareMethodOrder' has different definitions in different modules; first difference is definition in module 'First.Hidden' found method 'methodOrderFirst'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found different method 'methodOrderSecond'}}
CompareMethodClassInstance *compareMethodClassInstance;
// expected-error@first.h:* {{'CompareMethodClassInstance' has different definitions in different modules; first difference is definition in module 'First.Hidden' found class method 'methodClassInstance'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found method 'methodClassInstance' as instance method}}
#endif

#if defined(FIRST)
@interface CompareMatchingProperties: NSObject
@property int matchingPropName;
@end

@interface ComparePropertyPresence1: NSObject
@property int propPresence1;
@end
@interface ComparePropertyPresence2: NSObject
@end

@interface ComparePropertyName: NSObject
@property int propNameA;
@end

@interface ComparePropertyType: NSObject
@property int propType;
@end

@interface ComparePropertyOrder: NSObject
@property int propOrderX;
@property int propOrderY;
@end

@interface CompareMatchingPropertyAttributes: NSObject
@property (nonatomic, assign) int matchingProp;
@end
@interface ComparePropertyAttributes: NSObject
@property (readonly) int propAttributes;
@end
// Edge cases.
@interface CompareFirstImplAttribute: NSObject
@property int firstImplAttribute;
@end
@interface CompareLastImplAttribute: NSObject
@property (direct) int lastImplAttribute;
@end
#elif defined(SECOND)
@interface CompareMatchingProperties: NSObject
@property int matchingPropName;
@end

@interface ComparePropertyPresence1: NSObject
@end
@interface ComparePropertyPresence2: NSObject
@property int propPresence2;
@end

@interface ComparePropertyName: NSObject
@property int propNameB;
@end

@interface ComparePropertyType: NSObject
@property float propType;
@end

@interface ComparePropertyOrder: NSObject
@property int propOrderY;
@property int propOrderX;
@end

@interface CompareMatchingPropertyAttributes: NSObject
@property (assign, nonatomic) int matchingProp;
@end
@interface ComparePropertyAttributes: NSObject
@property (readwrite) int propAttributes;
@end
// Edge cases.
@interface CompareFirstImplAttribute: NSObject
@property (readonly) int firstImplAttribute;
@end
@interface CompareLastImplAttribute: NSObject
@property int lastImplAttribute;
@end
#else
CompareMatchingProperties *compareMatchingProperties;
ComparePropertyPresence1 *comparePropertyPresence1;
// expected-error@first.h:* {{'ComparePropertyPresence1' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found end of class}}
ComparePropertyPresence2 *comparePropertyPresence2;
// expected-error@first.h:* {{'ComparePropertyPresence2' has different definitions in different modules; first difference is definition in module 'First.Hidden' found end of class}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property}}

ComparePropertyName *comparePropertyName;
// expected-error@first.h:* {{'ComparePropertyName' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property 'propNameA'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property 'propNameB'}}
ComparePropertyType *comparePropertyType;
// expected-error@first.h:* {{'ComparePropertyType' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property 'propType' with type 'int'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property 'propType' with type 'float'}}
ComparePropertyOrder *comparePropertyOrder;
// expected-error@first.h:* {{'ComparePropertyOrder' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property 'propOrderX'}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property 'propOrderY'}}

CompareMatchingPropertyAttributes *compareMatchingPropertyAttributes;
ComparePropertyAttributes *comparePropertyAttributes;
// expected-error@first.h:* {{'ComparePropertyAttributes' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property 'propAttributes' with 'readonly' attribute}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property 'propAttributes' with different 'readonly' attribute}}
CompareFirstImplAttribute *compareFirstImplAttribute;
// expected-error@first.h:* {{'CompareFirstImplAttribute' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property 'firstImplAttribute' with default 'readonly' attribute}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property 'firstImplAttribute' with different 'readonly' attribute}}
CompareLastImplAttribute *compareLastImplAttribute;
// expected-error@first.h:* {{'CompareLastImplAttribute' has different definitions in different modules; first difference is definition in module 'First.Hidden' found property 'lastImplAttribute' with 'direct' attribute}}
// expected-note-re@second.h:* {{but in {{'Second'|definition here}} found property 'lastImplAttribute' with different 'direct' attribute}}
#endif
