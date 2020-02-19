// Clear and create directories
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: mkdir %t/cache
// RUN: mkdir -p %t/Inputs/Frameworks/First.framework/Headers
// RUN: mkdir -p %t/Inputs/Frameworks/Second.framework/Headers
// RUN: mkdir -p %t/Inputs/Frameworks/First.framework/Modules
// RUN: mkdir -p %t/Inputs/Frameworks/Second.framework/Modules

// Build first header file
// RUN: echo "#define FIRST" >> %t/Inputs/Frameworks/First.framework/Headers/first.h
// RUN: cat %s               >> %t/Inputs/Frameworks/First.framework/Headers/first.h

// Build second header file
// RUN: echo "#define SECOND" >> %t/Inputs/Frameworks/Second.framework/Headers/second.h
// RUN: cat %s                >> %t/Inputs/Frameworks/Second.framework/Headers/second.h

// Test that each header can compile
// RUN: %clang_cc1 -fsyntax-only -x objective-c %t/Inputs/Frameworks/First.framework/Headers/first.h -fblocks -fobjc-arc
// RUN: %clang_cc1 -fsyntax-only -x objective-c %t/Inputs/Frameworks/Second.framework/Headers/second.h -fblocks -fobjc-arc

// Build module map file
// RUN: echo "framework module First {"     >> %t/Inputs/Frameworks/First.framework/Modules/module.modulemap
// RUN: echo "    header \"first.h\""   >> %t/Inputs/Frameworks/First.framework/Modules/module.modulemap
// RUN: echo "}"                        >> %t/Inputs/Frameworks/First.framework/Modules/module.modulemap
// RUN: echo "framework module Second {"    >> %t/Inputs/Frameworks/Second.framework/Modules/module.modulemap
// RUN: echo "    header \"second.h\""  >> %t/Inputs/Frameworks/Second.framework/Modules/module.modulemap
// RUN: echo "}"                        >> %t/Inputs/Frameworks/Second.framework/Modules/module.modulemap

// Build APINotes file
// RUN: echo "---" >> %t/Inputs/Frameworks/First.framework/Headers/First.apinotes
// RUN: echo "Name: First" >> %t/Inputs/Frameworks/First.framework/Headers/First.apinotes
// RUN: echo "Classes:" >> %t/Inputs/Frameworks/First.framework/Headers/First.apinotes
// RUN: echo "- Name: IP" >> %t/Inputs/Frameworks/First.framework/Headers/First.apinotes
// RUN: echo "  Properties:" >> %t/Inputs/Frameworks/First.framework/Headers/First.apinotes
// RUN: echo "  - Name: ws" >> %t/Inputs/Frameworks/First.framework/Headers/First.apinotes
// RUN: echo "    Nullability: O" >> %t/Inputs/Frameworks/First.framework/Headers/First.apinotes

// Run test
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -x objective-c -F%t/Inputs/Frameworks -fapinotes-modules -verify %s -fblocks -fobjc-arc

// expected-no-diagnostics

#if !defined(FIRST) && !defined(SECOND)
#include <First/first.h>
#include <Second/second.h>
#endif

#if defined(FIRST) || defined(SECOND)
@interface WS
- (void)sayHello;
@end
#endif

#if defined(FIRST)
@interface IP
@property (nonatomic, readonly, strong) WS *ws;
@end
#elif defined(SECOND)
@interface IP
@property (nonatomic, readonly, strong) WS *ws;
@end
#else
IP *ip;
#endif

// Keep macros contained to one file.
#ifdef FIRST
#undef FIRST
#endif

#ifdef SECOND
#undef SECOND
#endif
