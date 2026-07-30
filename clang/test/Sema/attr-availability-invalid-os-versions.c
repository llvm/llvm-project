/// This test verifies diagnostic reporting for OS versions within an invalid range.
// RUN: %clang_cc1 -triple x86_64-apple-darwin25 -fsyntax-only -verify %s

 __attribute__((availability(macosx,introduced=16, deprecated=20))) 
void invalid_dep(void); // expected-warning@-1 {{invalid macOS version '20' in availability attribute}}
                        // expected-note@-2 {{implicitly treating version as '30'}}
 
__attribute__((availability(macosx,introduced=15,  obsoleted=19)))
void invalid_obsolete(void); // expected-warning@-1 {{invalid macOS version '19' in availability attribute}}
                             // expected-note@-2 {{implicitly treating version as '29'}}

__attribute__((availability(macosx,introduced=24)))
void invalid_intro(void); // expected-warning@-1 {{invalid macOS version '24' in availability attribute}}
                          // expected-note@-2 {{implicitly treating version as '34'}}

__attribute__((availability(watchos,introduced=24)))
void invalid_watch_intro(void); // expected-warning@-1 {{invalid watchOS version '24' in availability attribute}}
                                // expected-note@-2 {{implicitly treating version as '38'}}

__attribute__((availability(maccatalyst, introduced=18.1, deprecated=21.1)))
void invalid_catalyst_intro(void); // expected-warning@-1 {{invalid macCatalyst version '21.1' in availability attribute}}
                                   // expected-note@-2 {{implicitly treating version as '28.1'}}

__attribute__((availability(macos, introduced=15, deprecated=26)))
__attribute__((availability(watchos, introduced=15)))
__attribute__((availability(ios, introduced=15, deprecated=18)))
void invalid_intro_multiple(void); // expected-warning@-2 {{invalid watchOS version '15' in availability attribute}}
                                   // expected-note@-3 {{implicitly treating version as '29'}}
 
__attribute__((availability(macCatalyst, introduced=19.2, deprecated=22.2))) 
void invalid_dep(void); // expected-warning@-1 {{invalid macCatalyst version '19.2' in availability attribute}}
                        // expected-note@-2 {{implicitly treating version as '26.2'}}
                        // expected-warning@-3 {{invalid macCatalyst version '22.2' in availability attribute}}
                        // expected-note@-4 {{implicitly treating version as '29.2'}}

