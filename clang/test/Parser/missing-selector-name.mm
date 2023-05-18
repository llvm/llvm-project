// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface PodiumWalkerController
@property (assign) id PROP;
- (void) // expected-error {{expected ';' after method prototype}}
@end // expected-error {{expected selector for Objective-C method}}


id GVAR;

id StopProgressAnimation()
{

    PodiumWalkerController *controller;
    return controller.PROP;
}

@interface P1
@property (assign) id PROP;
- (void); // expected-error {{expected selector for Objective-C method}}
@end

id GG=0;

id Stop1()
{

    PodiumWalkerController *controller;
    return controller.PROP;
}

@interface P2
@property (assign) id PROP;
- (void)Meth {} // expected-error {{expected ';' after method prototype}}
@end

@interface P3
@property (assign) id PROP;
- (void)
- (void)Meth {} // expected-error {{expected selector for Objective-C method}} \
                // expected-error {{expected ';' after method prototype}}
@end

id HH=0;

id Stop2()
{

    PodiumWalkerController *controller;
    return controller.PROP;
}
