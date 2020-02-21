// Clear and create directories
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: mkdir %t/cache
// RUN: mkdir %t/Inputs

// Build first header file
// RUN: echo "#define FIRST" >> %t/Inputs/first.h
// RUN: cat %s               >> %t/Inputs/first.h

// Build second header file
// RUN: echo "#define SECOND" >> %t/Inputs/second.h
// RUN: cat %s                >> %t/Inputs/second.h

// Test that each header can compile
// RUN: %clang_cc1 -fsyntax-only -x objective-c %t/Inputs/first.h -fblocks -fobjc-arc
// RUN: %clang_cc1 -fsyntax-only -x objective-c %t/Inputs/second.h -fblocks -fobjc-arc

// Build module map file
// RUN: echo "module FirstModule {"     >> %t/Inputs/module.map
// RUN: echo "    header \"first.h\""   >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map
// RUN: echo "module SecondModule {"    >> %t/Inputs/module.map
// RUN: echo "    header \"second.h\""  >> %t/Inputs/module.map
// RUN: echo "}"                        >> %t/Inputs/module.map

// Run test
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x objective-c -I%t/Inputs -verify %s -fblocks -fobjc-arc

#if !defined(FIRST) && !defined(SECOND)
#include "first.h"
#include "second.h"
#endif

#if defined(FIRST) || defined(SECOND)
@interface Jazz
@end
@protocol X
@end
@protocol Y
@end
#endif

#if defined(FIRST)
@interface Jazz (Smooth)
-(void)play;
-(void)stopA;
@end
#elif defined(SECOND)
@interface Jazz (Smooth)
-(void)improvise;
-(void)stopB;
@end
#else
@implementation Jazz (Smooth) // expected-warning {{method definition for 'stopA' not found}}
-(void)play { return; }
-(void)improvise { return; }
@end
// expected-error@first.h:* {{category 'Smooth' on interface 'Jazz' has different definitions in different modules; first difference is definition in module 'FirstModule' found method name 'play'}}
// expected-note@second.h:* {{but in 'SecondModule' found method name 'improvise'}}
// expected-note@first.h:* {{method 'stopA' declared here}}
#endif

#if defined(FIRST)
@interface Jazz (Bebop) <X>
@end
#elif defined(SECOND)
@interface Jazz (Bebop) <Y>
@end
#else
@implementation Jazz (Bebop)
-(void)play { return; }
@end
// expected-error@first.h:* {{category 'Bebop' on interface 'Jazz' has different definitions in different modules; first difference is definition in module 'FirstModule' found with 1st protocol named 'X'}}
// expected-note@second.h:* {{but in 'SecondModule' found with 1st protocol named 'Y'}}
#endif

#if defined(FIRST)
@interface Jazz (BossaNova)
-(void)play;
@end
#elif defined(SECOND)
@interface Jazz (BossaNova)
-(void)stop;
@end
#else
@implementation Jazz (BossaNova) // expected-warning {{method definition for 'play' not found}}
-(void)pause { return; }
@end
// expected-error@first.h:* {{category 'BossaNova' on interface 'Jazz' has different definitions in different modules; first difference is definition in module 'FirstModule' found method name 'play'}}
// expected-note@second.h:* {{but in 'SecondModule' found method name 'stop'}}
// expected-note@first.h:* {{method 'play' declared here}}
#endif

#if defined(FIRST)
@interface Jazz (Free)
@property (nonatomic, assign) float f;
@end
#elif defined(SECOND)
@interface Jazz (Free)
@property (nonatomic, assign) float d;
@end
#else
@implementation Jazz (Free)
-(void)play { return; }
@end
// expected-error@first.h:* {{category 'Free' on interface 'Jazz' has different definitions in different modules; first difference is definition in module 'FirstModule' found property name 'f'}}
// expected-note@second.h:* {{but in 'SecondModule' found property name 'd'}}
#endif

// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif

#ifdef SECOND
#undef SECOND
#endif
