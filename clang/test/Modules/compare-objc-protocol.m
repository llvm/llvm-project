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
