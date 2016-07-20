//===-- main.m ------------------------------------------------*- ObjC -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

@interface MyClass : NSObject
{
    int i;
    char c;
    float f; 
}

- (id)initWithInt: (int)x andFloat:(float)y andChar:(char)z;
- (int)doIncrementByInt: (int)x;

@end

@interface MyOtherClass : MyClass
{
    int i2;
    MyClass *backup;
}
- (id)initWithInt: (int)x andFloat:(float)y andChar:(char)z andOtherInt:(int)q;

@end

@implementation MyClass

- (id)initWithInt: (int)x andFloat:(float)y andChar:(char)z
{
    self = [super init];
    if (self) {
        self->i = x;
        self->f = y;
        self->c = z;
    }    
    return self;
}

- (int)doIncrementByInt: (int)x
{
    self->i += x;
    return self->i;
}

@end

@implementation MyOtherClass

- (id)initWithInt: (int)x andFloat:(float)y andChar:(char)z andOtherInt:(int)q
{
    self = [super initWithInt:x andFloat:y andChar:z];
    if (self) {
        self->i2 = q;
        self->backup = [[MyClass alloc] initWithInt:x andFloat:y andChar:z];
    }    
    return self;
}

@end

@interface Atom : NSObject {
    float mass;
}
-(void)setMass:(float)newMass;
-(float)mass;
@end

@interface Molecule : NSObject {
    NSArray *atoms;
}
-(void)setAtoms:(NSArray *)newAtoms;
-(NSArray *)atoms;
@end

@implementation  Atom

-(void)setMass:(float)newMass
{
    mass = newMass;
}
-(float)mass
{
    return mass;
}

@end

@implementation Molecule

-(void)setAtoms:(NSArray *)newAtoms
{
    atoms = newAtoms;
}
-(NSArray *)atoms
{
    return atoms;
}
@end

@interface My_KVO_Observer : NSObject
-(void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change 
	context:(void *)context;
- (id) init;
- (void) dealloc;
@end

@implementation My_KVO_Observer
-(void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change 
                      context:(void *)context {
	// we do not really care about KVO'ing - do nothing
	return;
}
- (id) init
{
    self = [super init]; 
    return self;
}

- (void) dealloc
{
    [super dealloc];
}
@end

int main (int argc, const char * argv[])
{
    
    NSAutoreleasePool * pool = [[NSAutoreleasePool alloc] init];
    
    MyClass *object = [[MyClass alloc] initWithInt:1 andFloat:3.14 andChar: 'E'];
    
    [object doIncrementByInt:3];
    
    MyOtherClass *object2 = [[MyOtherClass alloc] initWithInt:2 andFloat:6.28 andChar: 'G' andOtherInt:-1];
    
    [object2 doIncrementByInt:3];
    
	    NSNumber* num1 = [NSNumber numberWithInt:5];
	    NSNumber* num2 = [NSNumber numberWithFloat:3.14];
	    NSNumber* num3 = [NSNumber numberWithDouble:3.14];
	    NSNumber* num4 = [NSNumber numberWithUnsignedLongLong:0xFFFFFFFFFFFFFFFE];
	    NSNumber* num5 = [NSNumber numberWithChar:'A'];
	    NSNumber* num6 = [NSNumber numberWithUnsignedLongLong:0xFF];
	    NSNumber* num7 = [NSNumber numberWithLong:0x1E8480];
	    NSNumber* num8_Y = [NSNumber numberWithBool:YES];
	    NSNumber* num8_N = [NSNumber numberWithBool:NO];
	    NSNumber* num9 = [NSNumber numberWithShort:0x1E8480];
	    NSNumber* num_at1 = @12;
	    NSNumber* num_at2 = @-12;
	    NSNumber* num_at3 = @12.5;
	    NSNumber* num_at4 = @-12.5;

		NSDecimalNumber* decimal_one = [NSDecimalNumber one];

	    NSString *str0 = [num6 stringValue];

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

	    NSString *strA1 = [NSString stringWithCString:"A rather short ASCII NSString object is here" encoding:NSASCIIStringEncoding];

	    NSString *strA2 = [NSString stringWithUTF8String:"A rather short UTF8 NSString object is here"];

	    NSString *strA3 = @"A string made with the at sign is here";

	    NSString *strA4 = [NSString stringWithFormat:@"This is string number %ld right here", (long)4];

	    NSString* strA5 = NSStringFromRect(ns_rect_4str);

	    NSString* strA6 = [@"/usr/doc/README.1ST" pathExtension];

	    NSString *strA7 = [NSString stringWithCharacters: myCharacters
	                                             length: sizeof myCharacters / sizeof *myCharacters];

	    NSString* strA8 = [@"/usr/doc/file.hasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTime" pathExtension];

	    NSString *strA9 = [NSString stringWithCharacters: myOtherCharacters
	                                             length: sizeof myOtherCharacters / sizeof *myOtherCharacters];

	    NSString *strA10 = [NSString stringWithFormat:@"This is a Unicode string %S number %ld right here", myNextCharacters, (long)4];

	    NSString *strA11 = NSStringFromClass([str10 class]);

	    NSString *strA12 = [NSString stringWithFormat:@"%@ %@ %@ %@", label1, processName, label2, processID];

	    NSString *strB1 = [NSString stringWithCString:"A rather short ASCII NSString object is here" encoding:NSASCIIStringEncoding];

	    NSString *strB2 = [NSString stringWithUTF8String:"A rather short UTF8 NSString object is here"];

	    NSString *strB3 = @"A string made with the at sign is here";

	    NSString *strB4 = [NSString stringWithFormat:@"This is string number %ld right here", (long)4];

	    NSString* strB5 = NSStringFromRect(ns_rect_4str);

	    NSString* strB6 = [@"/usr/doc/README.1ST" pathExtension];

	    NSString *strB7 = [NSString stringWithCharacters: myCharacters
	                                              length: sizeof myCharacters / sizeof *myCharacters];

	    NSString* strB8 = [@"/usr/doc/file.hasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTime" pathExtension];

	    NSString *strB9 = [NSString stringWithCharacters: myOtherCharacters
	                                              length: sizeof myOtherCharacters / sizeof *myOtherCharacters];

	    NSString *strB10 = [NSString stringWithFormat:@"This is a Unicode string %S number %ld right here", myNextCharacters, (long)4];

	    NSString *strB11 = NSStringFromClass([str10 class]);

	    NSString *strB12 = [NSString stringWithFormat:@"%@ %@ %@ %@", label1, processName, label2, processID];

	    NSString *strC11 = NSStringFromClass([str10 class]);

	    NSString *strC12 = [NSString stringWithFormat:@"%@ %@ %@ %@", label1, processName, label2, processID];

	    NSString *strC1 = [NSString stringWithCString:"A rather short ASCII NSString object is here" encoding:NSASCIIStringEncoding];

	    NSString *strC2 = [NSString stringWithUTF8String:"A rather short UTF8 NSString object is here"];

	    NSString *strC3 = @"A string made with the at sign is here";

	    NSString *strC4 = [NSString stringWithFormat:@"This is string number %ld right here", (long)4];

	    NSString* strC5 = NSStringFromRect(ns_rect_4str);

	    NSString* strC6 = [@"/usr/doc/README.1ST" pathExtension];

	    NSString *strC7 = [NSString stringWithCharacters: myCharacters
	                                              length: sizeof myCharacters / sizeof *myCharacters];

	    NSString* strC8 = [@"/usr/doc/file.hasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTimehasVeryLongExtensionThisTime" pathExtension];

	    NSString *strC9 = [NSString stringWithCharacters: myOtherCharacters
	                                              length: sizeof myOtherCharacters / sizeof *myOtherCharacters];

	    NSString *strC10 = [NSString stringWithFormat:@"This is a Unicode string %S number %ld right here", myNextCharacters, (long)4];

	    NSString *strD11 = NSStringFromClass([str10 class]);

	    NSString *strD12 = [NSString stringWithFormat:@"%@ %@ %@ %@", label1, processName, label2, processID];

	    NSString *eAcute = [NSString stringWithFormat: @"%C", 0x00E9];
	    NSString *randomHaziChar = [NSString stringWithFormat: @"%C", 0x9DC5];
	    NSString *japanese = @"色は匂へど散りぬるを";
	    NSString *italian = @"L'Italia è una Repubblica democratica, fondata sul lavoro. La sovranità appartiene al popolo, che la esercita nelle forme e nei limiti della Costituzione.";
	    NSString* french = @"Que veut cette horde d'esclaves, De traîtres, de rois conjurés?";
	    NSString* german = @"Über-Ich und aus den Ansprüchen der sozialen Umwelt";

	    void* data_set[3] = {str1,str2,str3};
	
		NSString *hebrew = [NSString stringWithString:@"לילה טוב"];

	    NSArray* newArray = [[NSMutableArray alloc] init];
	    [newArray addObject:str1];
	    [newArray addObject:str2];
	    [newArray addObject:str3];
	    [newArray addObject:str4];
	    [newArray addObject:str5];
	    [newArray addObject:str6];
	    [newArray addObject:str7];
	    [newArray addObject:str8];
	    [newArray addObject:str9];
	    [newArray addObject:str10];
	    [newArray addObject:str11];
	    [newArray addObject:str12];
	    [newArray addObject:strA1];
	    [newArray addObject:strA2];
	    [newArray addObject:strA3];
	    [newArray addObject:strA4];
	    [newArray addObject:strA5];
	    [newArray addObject:strA6];
	    [newArray addObject:strA7];
	    [newArray addObject:strA8];
	    [newArray addObject:strA9];
	    [newArray addObject:strA10];
	    [newArray addObject:strA11];
	    [newArray addObject:strA12];
	    [newArray addObject:strB1];
	    [newArray addObject:strB2];
	    [newArray addObject:strB3];
	    [newArray addObject:strB4];
	    [newArray addObject:strB5];
	    [newArray addObject:strB6];
	    [newArray addObject:strB7];
	    [newArray addObject:strB8];
	    [newArray addObject:strB9];
	    [newArray addObject:strB10];
	    [newArray addObject:strB11];
	    [newArray addObject:strB12];
	    [newArray addObject:strC1];
	    [newArray addObject:strC2];
	    [newArray addObject:strC3];
	    [newArray addObject:strC4];
	    [newArray addObject:strC5];
	    [newArray addObject:strC6];
	    [newArray addObject:strC7];
	    [newArray addObject:strC8];
	    [newArray addObject:strC9];
	    [newArray addObject:strC10];
	    [newArray addObject:strC11];
	    [newArray addObject:strC12];
	    [newArray addObject:strD11];
	    [newArray addObject:strD12];

	    NSDictionary* newDictionary = [[NSDictionary alloc] initWithObjects:newArray forKeys:newArray];
	    NSDictionary *newMutableDictionary = [[NSMutableDictionary alloc] init];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar0"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar1"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar2"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar3"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar4"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar5"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar6"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar7"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar8"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar9"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar10"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar11"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar12"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar13"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar14"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar15"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar16"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar17"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar18"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar19"];
	    [newMutableDictionary setObject:@"foo" forKey:@"bar20"];

	    NSAttributedString* attrString = [[NSAttributedString alloc] initWithString:@"hello world from foo" attributes:newDictionary];
	    [attrString isEqual:nil];
	    NSAttributedString* mutableAttrString = [[NSMutableAttributedString alloc] initWithString:@"hello world from foo" attributes:newDictionary];
	    [mutableAttrString isEqual:nil];

	    NSString* mutableString = [[NSMutableString alloc] initWithString:@"foo"];
	    [mutableString insertString:@"foo said this string needs to be very long so much longer than whatever other string has been seen ever before by anyone of the mankind that of course this is still not long enough given what foo our friend foo our lovely dearly friend foo desired of us so i am adding more stuff here for the sake of it and for the joy of our friend who is named guess what just foo. hence, dear friend foo, stay safe, your string is now  long enough to accommodate your testing need and I will make sure that if not we extend it with even more fuzzy random meaningless words pasted one after the other from a long tiresome friday evening spent working in my office. my office mate went home but I am still randomly typing just for the fun of seeing what happens of the length of a Mutable String in Cocoa if it goes beyond one byte.. so be it, dear " atIndex:0];

	    NSString* mutableGetConst = [NSString stringWithCString:[mutableString cString]];

	    [mutableGetConst length];

	    NSData *immutableData = [[NSData alloc] initWithBytes:"HELLO" length:4];
	    NSData *mutableData = [[NSMutableData alloc] initWithBytes:"NODATA" length:6];

	    [mutableData appendBytes:"MOREDATA" length:8];

	    [immutableData length];
	    [mutableData length];

	    NSSet* nsset = [[NSSet alloc] initWithObjects:str1,str2,str3,nil];
	    NSSet *nsmutableset = [[NSMutableSet alloc] initWithObjects:str1,str2,str3,nil];
	    [nsmutableset addObject:str4];

	    CFDataRef data_ref = CFDataCreate(kCFAllocatorDefault, [immutableData bytes], 5);

	    CFMutableDataRef mutable_data_ref = CFDataCreateMutable(kCFAllocatorDefault, 8);
	    CFDataAppendBytes(mutable_data_ref, [mutableData bytes], 5);

	    CFMutableStringRef mutable_string_ref = CFStringCreateMutable(NULL,100);
	    CFStringAppend(mutable_string_ref, CFSTR("Wish ya knew"));

	    CFStringRef cfstring_ref = CFSTR("HELLO WORLD");

	    CFArrayRef cfarray_ref = CFArrayCreate(NULL, data_set, 3, NULL);
	    CFMutableArrayRef mutable_array_ref = CFArrayCreateMutable(NULL, 16, NULL);

	    CFArraySetValueAtIndex(mutable_array_ref, 0, str1);
	    CFArraySetValueAtIndex(mutable_array_ref, 1, str2);
	    CFArraySetValueAtIndex(mutable_array_ref, 2, str3);
	    CFArraySetValueAtIndex(mutable_array_ref, 3, str4);
	    CFArraySetValueAtIndex(mutable_array_ref, 0, str5); // replacing value at 0!!
	    CFArraySetValueAtIndex(mutable_array_ref, 4, str6);
	    CFArraySetValueAtIndex(mutable_array_ref, 5, str7);
	    CFArraySetValueAtIndex(mutable_array_ref, 6, str8);
	    CFArraySetValueAtIndex(mutable_array_ref, 7, str9);
	    CFArraySetValueAtIndex(mutable_array_ref, 8, str10);
	    CFArraySetValueAtIndex(mutable_array_ref, 9, str11);
	    CFArraySetValueAtIndex(mutable_array_ref, 10, str12);

	    CFBinaryHeapRef binheap_ref = CFBinaryHeapCreate(NULL, 15, &kCFStringBinaryHeapCallBacks, NULL);
	    CFBinaryHeapAddValue(binheap_ref, str1);
	    CFBinaryHeapAddValue(binheap_ref, str2);
	    CFBinaryHeapAddValue(binheap_ref, str3);
	    CFBinaryHeapAddValue(binheap_ref, str4);
	    CFBinaryHeapAddValue(binheap_ref, str5);
	    CFBinaryHeapAddValue(binheap_ref, str6);
	    CFBinaryHeapAddValue(binheap_ref, str7);
	    CFBinaryHeapAddValue(binheap_ref, str8);
	    CFBinaryHeapAddValue(binheap_ref, str9);
	    CFBinaryHeapAddValue(binheap_ref, str10);
	    CFBinaryHeapAddValue(binheap_ref, str11);
	    CFBinaryHeapAddValue(binheap_ref, str12);
	    CFBinaryHeapAddValue(binheap_ref, strA1);
	    CFBinaryHeapAddValue(binheap_ref, strB1);
	    CFBinaryHeapAddValue(binheap_ref, strC1);
	    CFBinaryHeapAddValue(binheap_ref, strA11);
	    CFBinaryHeapAddValue(binheap_ref, strB11);
	    CFBinaryHeapAddValue(binheap_ref, strC11);
	    CFBinaryHeapAddValue(binheap_ref, strB12);
	    CFBinaryHeapAddValue(binheap_ref, strC12);
	    CFBinaryHeapAddValue(binheap_ref, strA12);

	    CFURLRef cfurl_ref = CFURLCreateWithString(NULL, CFSTR("http://www.foo.bar/"), NULL);
	    CFURLRef cfchildurl_ref = CFURLCreateWithString(NULL, CFSTR("page.html"), cfurl_ref);
	    CFURLRef cfgchildurl_ref = CFURLCreateWithString(NULL, CFSTR("?whatever"), cfchildurl_ref);

	    NSDictionary *error_userInfo = @{@"a": @1, @"b" : @2};
	    NSError *nserror = [[NSError alloc] initWithDomain:@"Foobar" code:12 userInfo:error_userInfo];
	    NSError **nserrorptr = &nserror;

	    NSBundle* bundle_string = [[NSBundle alloc] initWithPath:@"/System/Library/Frameworks/Accelerate.framework"];
	    NSBundle* bundle_url = [[NSBundle alloc] initWithURL:[[NSURL alloc] initWithString:@"file://localhost/System/Library/Frameworks/Foundation.framework"]];

	    NSBundle* main_bundle = [NSBundle mainBundle];

	    NSArray* bundles = [NSBundle allBundles];

	    NSURL *nsurl0;

	    for (NSBundle* bundle in bundles)
	    {
	        nsurl0 = [bundle bundleURL];
	    }

	    NSException* except0 = [[NSException alloc] initWithName:@"TheGuyWhoHasNoName" reason:@"cuz it's funny" userInfo:nil];
	    NSException* except1 = [[NSException alloc] initWithName:@"TheGuyWhoHasNoName~1" reason:@"cuz it's funny" userInfo:nil];
	    NSException* except2 = [[NSException alloc] initWithName:@"TheGuyWhoHasNoName`2" reason:@"cuz it's funny" userInfo:nil];
	    NSException* except3 = [[NSException alloc] initWithName:@"TheGuyWhoHasNoName/3" reason:@"cuz it's funny" userInfo:nil];

	    NSURL *nsurl = [[NSURL alloc] initWithString:@"http://www.foo.bar"];
	    NSURL *nsurl2 = [NSURL URLWithString:@"page.html" relativeToURL:nsurl];
	    NSURL *nsurl3 = [NSURL URLWithString:@"?whatever" relativeToURL:nsurl2];
    
		NSDate *date1 = [NSDate dateWithNaturalLanguageString:@"6pm April 10, 1985"];
		NSDate *date2 = [NSDate dateWithNaturalLanguageString:@"12am January 1, 2011"];
		NSDate *date3 = [NSDate date];
		NSDate *date4 = [NSDate dateWithTimeIntervalSince1970:24*60*60];
    NSDate *date5 = [NSDate dateWithTimeIntervalSinceReferenceDate: floor([[NSDate date] timeIntervalSinceReferenceDate])];

		CFAbsoluteTime date1_abs = CFDateGetAbsoluteTime(date1);
		CFAbsoluteTime date2_abs = CFDateGetAbsoluteTime(date2);
		CFAbsoluteTime date3_abs = CFDateGetAbsoluteTime(date3);
		CFAbsoluteTime date4_abs = CFDateGetAbsoluteTime(date4);
		CFAbsoluteTime date5_abs = CFDateGetAbsoluteTime(date5);

	    NSIndexSet *iset1 = [[NSIndexSet alloc] initWithIndexesInRange:NSMakeRange(1, 4)];
	    NSIndexSet *iset2 = [[NSIndexSet alloc] initWithIndexesInRange:NSMakeRange(1, 512)];

	    NSMutableIndexSet *imset = [[NSMutableIndexSet alloc] init];
	    [imset addIndex:1936];
	    [imset addIndex:7];
	    [imset addIndex:9];
	    [imset addIndex:11];
	    [imset addIndex:24];
	    [imset addIndex:41];
	    [imset addIndex:58];
	    [imset addIndex:61];
	    [imset addIndex:62];
	    [imset addIndex:63];

	    CFTimeZoneRef cupertino = CFTimeZoneCreateWithName (
	                                            NULL,
	                                            CFSTR("PST"),
	                                            YES);
	    CFTimeZoneRef home = CFTimeZoneCreateWithName (
	                                            NULL,
	                                            CFSTR("Europe/Rome"),
	                                            YES);
	    CFTimeZoneRef europe = CFTimeZoneCreateWithName (
	                                            NULL,
	                                            CFSTR("CET"),
	                                            YES);

		NSTimeZone *cupertino_ns = [NSTimeZone timeZoneWithAbbreviation:@"PST"];
		NSTimeZone *home_ns = [NSTimeZone timeZoneWithName:@"Europe/Rome"];
		NSTimeZone *europe_ns = [NSTimeZone timeZoneWithAbbreviation:@"CET"];

	CFGregorianUnits cf_greg_units = {1,3,5,12,5,7};
	CFGregorianDate cf_greg_date = CFAbsoluteTimeGetGregorianDate(CFDateGetAbsoluteTime(date1), NULL);
	CFRange cf_range = {4,4};
	NSPoint ns_point = {4,4};
	NSRange ns_range = {4,4};
		
	NSRect ns_rect = {{1,1},{5,5}};
	NSRect* ns_rect_ptr = &ns_rect;
	NSRectArray ns_rect_arr = &ns_rect;
	NSSize ns_size = {5,7};
	NSSize* ns_size_ptr = &ns_size;
	
	CGSize cg_size = {1,6};
	CGPoint cg_point = {2,7};
	CGRect cg_rect = {{1,2}, {7,7}};
	
#ifndef IOS
	RGBColor rgb_color = {3,56,35};
	RGBColor* rgb_color_ptr = &rgb_color;
#endif
	
	Rect rect = {4,8,4,7};
	Rect* rect_ptr = &rect;
	
	Point point = {7,12};
	Point* point_ptr = &point;
	
#ifndef IOS
	HIPoint hi_point = {7,12};
	HIRect hi_rect = {{3,5},{4,6}};
#endif
	
	SEL foo_selector = @selector(foo_selector_impl);
	
	CFMutableBitVectorRef mut_bv = CFBitVectorCreateMutable(NULL, 64);
	CFBitVectorSetCount(mut_bv, 50);
    CFBitVectorSetBitAtIndex(mut_bv, 0, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 1, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 2, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 5, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 6, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 8, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 10, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 11, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 16, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 17, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 19, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 20, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 22, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 24, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 28, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 29, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 30, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 30, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 31, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 34, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 35, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 37, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 39, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 40, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 41, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 43, 1);
    CFBitVectorSetBitAtIndex(mut_bv, 47, 1);

	Molecule *molecule = [Molecule new];

	Class myclass = NSClassFromString(@"NSValue");
	Class myclass2 = [str0 class];
	Class myclass3 = [molecule class];
	Class myclass4 = NSClassFromString(@"NSMutableArray");
	Class myclass5 = [nil class];

	NSArray *components = @[@"usr", @"blah", @"stuff"];
	NSString *path = [NSString pathWithComponents: components];

    [molecule addObserver:[My_KVO_Observer new] forKeyPath:@"atoms" options:0 context:NULL];     // Set break point at this line.
    [newMutableDictionary addObserver:[My_KVO_Observer new] forKeyPath:@"weirdKeyToKVO" options:NSKeyValueObservingOptionNew context:NULL];

    [molecule setAtoms:nil];
    [molecule setAtoms:[NSMutableArray new]];

    [pool drain];
    return 0;
}

