//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#import <Foundation/Foundation.h>

#if defined(__APPLE__)
#if defined(__arm__) || defined(__arm64__) || defined(__aarch64__)
#define IOS
#endif
#endif

#if defined(IOS)
#import <Foundation/NSGeometry.h>
#else
#import <Carbon/Carbon.h>
#endif

int main (int argc, const char * argv[])
{
    
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];

	    NSString *str0 = [[NSNumber numberWithUnsignedLongLong:0xFF] stringValue];
	    NSString *str1 = [NSString stringWithCString:"A rather short ASCII NSString object is here" encoding:NSASCIIStringEncoding];
	    NSString *str2 = [NSString stringWithUTF8String:"A rather short UTF8 NSString object is here"];
	    NSString *str3 = @"A string made with the at sign is here";
	    NSString *str4 = [NSString stringWithFormat:@"This is string number %ld right here", (long)4];
	    NSRect ns_rect_4str = {{1,1},{5,5}};
	    NSString* str5 = NSStringFromRect(ns_rect_4str);
	    NSString* str6 = [@"/usr/doc/README.1ST" pathExtension];
	    const unichar myCharacters[] = {0x03C3,'x','x'};
	    NSString *str7 = [NSString stringWithCharacters: myCharacters
	                                             length: sizeof myCharacters / sizeof *myCharacters];
	    NSString* str8 = [@"/usr/doc/file.hasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTime" pathExtension];
	    const unichar myOtherCharacters[] = {'a',' ', 'v','e','r','y',' ',
	        'm','u','c','h',' ','b','o','r','i','n','g',' ','t','a','s','k',
	        ' ','t','o',' ','w','r','i','t','e', ' ', 'a', ' ', 's', 't', 'r', 'i', 'n', 'g', ' ',
	        't','h','i','s',' ','w','a','y','!','!',0x03C3, 0};
	    NSString *str9 = [NSString stringWithCharacters: myOtherCharacters
	                                             length: sizeof myOtherCharacters / sizeof *myOtherCharacters];
	    const unichar myNextCharacters[] = {0x03C3, 0x0000};
	    NSString *str10 = [NSString stringWithFormat:@"This is a Unicode string %S number %ld right here", myNextCharacters, (long)4];
	    NSString *str11 = NSStringFromClass([str10 class]);
	    NSString *label1 = @"Process Name: ";
	    NSString *label2 = @"Process Id: ";
	    NSString *processName = [[NSProcessInfo processInfo] processName];
	    NSString *processID = [NSString stringWithFormat:@"%d", [[NSProcessInfo processInfo] processIdentifier]];
	    NSString *str12 = [NSString stringWithFormat:@"%@ %@ %@ %@", label1, processName, label2, processID];
	    NSString *eAcute = [NSString stringWithFormat: @"%C", 0x00E9];
	    NSString *randomHaziChar = [NSString stringWithFormat: @"%C", 0x9DC5];
	    NSString *japanese = @"色は匂へど散りぬるを";
	    NSString *italian = @"L'Italia è una Repubblica democratica, fondata sul lavoro. La sovranità appartiene al popolo, che la esercita nelle forme e nei limiti della Costituzione.";
	    NSString* french = @"Que veut cette horde d'esclaves, De traîtres, de rois conjurés?";
	    NSString* german = @"Über-Ich und aus den Ansprüchen der sozialen Umwelt";
	    void* data_set[3] = {str1,str2,str3};
      NSString *hebrew = [NSString stringWithString:@"לילה טוב"];

	    NSAttributedString* attrString = [[NSAttributedString alloc] initWithString:@"hello world from foo" attributes:[NSDictionary new]];
	    [attrString isEqual:nil];
	    NSAttributedString* mutableAttrString = [[NSMutableAttributedString alloc] initWithString:@"hello world from foo" attributes:[NSDictionary new]];
	    [mutableAttrString isEqual:nil];

	    NSString* mutableString = [[NSMutableString alloc] initWithString:@"foo"];
	    [mutableString insertString:@"foo said this string needs to be very long so much longer than whatever other string has been seen ever before by anyone of the mankind that of course this is still not long enough given what foo our friend foo our lovely dearly friend foo desired of us so i am adding more stuff here for the sake of it and for the joy of our friend who is named guess what just foo. hence, dear friend foo, stay safe, your string is now  long enough to accommodate your testing need and I will make sure that if not we extend it with even more fuzzy random meaningless words pasted one after the other from a long tiresome friday evening spent working in my office. my office mate went home but I am still randomly typing just for the fun of seeing what happens of the length of a Mutable String in Cocoa if it goes beyond one byte.. so be it, dear " atIndex:0];

	    NSString* mutableGetConst = [NSString stringWithCString:[mutableString cString]];

	    [mutableGetConst length];
	    CFMutableStringRef mutable_string_ref = CFStringCreateMutable(NULL,100);
	    CFStringAppend(mutable_string_ref, CFSTR("Wish ya knew"));
	    CFStringRef cfstring_ref = CFSTR("HELLO WORLD");

	NSArray *components = @[@"usr", @"blah", @"stuff"];
	NSString *path = [NSString pathWithComponents: components];

  const unichar someOfTheseAreNUL[] = {'a',' ', 'v','e','r','y',' ',
      'm','u','c','h',' ','b','o','r','i','n','g',' ','t','a','s','k',
      ' ','t','o',' ','w','r','i','t','e', 0, 'a', ' ', 's', 't', 'r', 'i', 'n', 'g', ' ',
      't','h','i','s',' ','w','a','y','!','!', 0x03C3, 0};
  NSString *strwithNULs = [NSString stringWithCharacters: someOfTheseAreNUL
                                           length: sizeof someOfTheseAreNUL / sizeof *someOfTheseAreNUL];

  const unichar someOfTheseAreNUL2[] = {'a',' ', 'v','e','r','y',' ',
      'm','u','c','h',' ','b','o','r','i','n','g',' ','t','a','s','k',
      ' ','t','o',' ','w','r','i','t','e', 0, 'a', ' ', 's', 't', 'r', 'i', 'n', 'g', ' ',
      't','h','i','s',' ','w','a','y','!','!'};
  NSString *strwithNULs2 = [NSString stringWithCharacters: someOfTheseAreNUL2
                                           length: sizeof someOfTheseAreNUL2 / sizeof *someOfTheseAreNUL2];

    [pool drain]; // break here
    return 0;
}

