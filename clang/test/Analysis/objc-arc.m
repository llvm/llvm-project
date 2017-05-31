// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -analyzer-checker=core,osx.cocoa.RetainCount,deadcode -verify -fblocks -analyzer-opt-analyze-nested-blocks -fobjc-arc -analyzer-config path-diagnostics-alternate=true -analyzer-output=plist-multi-file -o %t.plist %s
// RUN: FileCheck --input-file=%t.plist %s

typedef signed char BOOL;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
typedef unsigned long NSUInteger;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSCoding;
@protocol NSMutableCopying;
@protocol NSFastEnumeration
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone;
@end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
+ (id)alloc;
- (id)init;
- (NSString *)description;
@end
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
- (id)initWithObjects:(const id [])objects count:(NSUInteger)cnt;
+ (id)arrayWithObject:(id)anObject;
+ (id)arrayWithObjects:(const id [])objects count:(NSUInteger)cnt;
+ (id)arrayWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
- (id)initWithObjects:(id)firstObj, ... __attribute__((sentinel(0,1)));
- (id)initWithArray:(NSArray *)array;
@end

typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;
typedef double CFTimeInterval;
typedef CFTimeInterval CFAbsoluteTime;
extern CFAbsoluteTime CFAbsoluteTimeGetCurrent(void);
typedef const struct __CFDate * CFDateRef;
extern CFDateRef CFDateCreate(CFAllocatorRef allocator, CFAbsoluteTime at);

typedef const void* objc_objectptr_t;
__attribute__((ns_returns_retained)) id objc_retainedObject(objc_objectptr_t __attribute__((cf_consumed)) pointer);
__attribute__((ns_returns_not_retained)) id objc_unretainedObject(objc_objectptr_t pointer);

// Test the analyzer is working at all.
void test_working() {
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning {{null}}
}

// Test that in ARC mode that blocks are correctly automatically copied
// and not flagged as warnings by the analyzer.
typedef void (^Block)(void);
void testblock_bar(int x);

Block testblock_foo(int x) {
  Block b = ^{ testblock_bar(x); };
  return b; // no-warning
}

Block testblock_baz(int x) {
  return ^{ testblock_bar(x); }; // no-warning
}

Block global_block;

void testblock_qux(int x) {
  global_block = ^{ testblock_bar(x); }; // no-warning
}

// Test that Objective-C pointers are null initialized.
void test_nil_initialized() {
  id x;
  if (x == 0)
    return;
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
}

// Test that we don't flag leaks of Objective-C objects.
void test_alloc() {
  [NSObject alloc]; // no-warning
}

// Test that CF allocations are still caught as leaks.
void test_cf_leak() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t); // expected-warning {{Potential leak}}
  (void) date;
}

// Test that 'init' methods do not try to claim ownerhip of an *unowned* allocated object
// in ARC mode.
@interface RDar9424890_A :  NSObject
- (id)initWithCleaner:(int)pop mop:(NSString *)mop ;
- (RDar9424890_A *)rdar9424890:(NSString *)identifier;
@end
@interface RDar9424890_B : NSObject
@end
@implementation RDar9424890_B
- (RDar9424890_A *)obj:(RDar9424890_A *)obj {
  static NSString *WhizFiz = @"WhizFiz";
  RDar9424890_A *cell = [obj rdar9424890:WhizFiz];
  if (cell == ((void*)0)) {
    cell = [[RDar9424890_A alloc] initWithCleaner:0 mop:WhizFiz]; // no-warning
  }
  return cell;
}
@end

// Test that dead store checking works in the prescence of "cleanups" in the AST.
void rdar9424882() {
  id x = [NSObject alloc]; // expected-warning {{Value stored to 'x' during its initialization is never read}}
}

// Test 
typedef const void *CFTypeRef;
typedef const struct __CFString *CFStringRef;

@interface NSString
- (id) self;
@end

CFTypeRef CFCreateSomething();
CFStringRef CFCreateString();
CFTypeRef CFGetSomething();
CFStringRef CFGetString();

id CreateSomething();
NSString *CreateNSString();

void from_cf() {
  id obj1 = (__bridge_transfer id)CFCreateSomething(); // expected-warning{{never read}}
  id obj2 = (__bridge_transfer NSString*)CFCreateString();
  [obj2 self]; // Add a use, to show we can use the object after it has been transferred.
  id obj3 = (__bridge id)CFGetSomething();
  [obj3 self]; // Add a use, to show we can use the object after it has been bridged.
  id obj4 = (__bridge NSString*)CFGetString(); // expected-warning{{never read}}
  id obj5 = (__bridge id)CFCreateSomething(); // expected-warning{{never read}} expected-warning{{leak}}
  id obj6 = (__bridge NSString*)CFCreateString(); // expected-warning{{never read}} expected-warning{{leak}}
}

void to_cf(id obj) {
  CFTypeRef cf1 = (__bridge_retained CFTypeRef)CreateSomething(); // expected-warning{{never read}}
  CFStringRef cf2 = (__bridge_retained CFStringRef)CreateNSString(); // expected-warning{{never read}}
  CFTypeRef cf3 = (__bridge CFTypeRef)CreateSomething(); // expected-warning{{never read}}
  CFStringRef cf4 = (__bridge CFStringRef)CreateNSString();  // expected-warning{{never read}}
}

void test_objc_retainedObject() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);
  id x = objc_retainedObject(date);
  (void) x;
}

void test_objc_unretainedObject() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(0, t);  // expected-warning {{Potential leak}}
  id x = objc_unretainedObject(date);
  (void) x;
}

// Previously this resulted in a "return of stack address" warning.
id test_return() {
  id x = (__bridge_transfer id) CFCreateString();
  return x; // no-warning
}

void test_objc_arrays() {
    { // CASE ONE -- OBJECT IN ARRAY CREATED DIRECTLY
        NSObject *o = [[NSObject alloc] init];
        NSArray *a = [[NSArray alloc] initWithObjects:o, (void*)0];
        [a description];
        [o description];
    }

    { // CASE TWO -- OBJECT IN ARRAY CREATED BY DUPING AUTORELEASED ARRAY
        NSObject *o = [[NSObject alloc] init];
        NSArray *a1 = [NSArray arrayWithObjects:o, (void*)0];
        NSArray *a2 = [[NSArray alloc] initWithArray:a1];
        [a2 description];
        [o description];
    }

    { // CASE THREE -- OBJECT IN RETAINED @[]
        NSObject *o = [[NSObject alloc] init];
        NSArray *a3 = @[o];
        [a3 description];
        [o description];
    }
    {
      // CASE 4, verify analyzer still working.
      CFCreateString(); // expected-warning {{leak}}
    }
}

// <rdar://problem/11059275> - dispatch_set_context and ARC.
__attribute__((cf_returns_retained)) CFTypeRef CFBridgingRetain(id X);
typedef void* dispatch_object_t;
void dispatch_set_context(dispatch_object_t object, const void *context);

void rdar11059275(dispatch_object_t object) {
  NSObject *o = [[NSObject alloc] init];
  dispatch_set_context(object, CFBridgingRetain(o)); // no-warning  
}
void rdar11059275_positive() {
  NSObject *o = [[NSObject alloc] init]; // expected-warning {{leak}}
  CFBridgingRetain(o);
}
void rdar11059275_negative() {
  NSObject *o = [[NSObject alloc] init]; // no-warning
  (void) o;
}

__attribute__((ns_returns_retained)) id rdar14061675_helper() {
  return [[NSObject alloc] init];
}

id rdar14061675() {
  // ARC produces an implicit cast here. We need to make sure the combination
  // of that and the inlined call don't produce a spurious edge cycle.
  id result = rdar14061675_helper();
  *(volatile int *)0 = 1; // expected-warning{{Dereference of null pointer}}
  return result;
}

// CHECK:  <key>diagnostics</key>
// CHECK-NEXT:  <array>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>53</integer>
// CHECK-NEXT:       <key>col</key><integer>3</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>53</integer>
// CHECK-NEXT:          <key>col</key><integer>3</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>53</integer>
// CHECK-NEXT:          <key>col</key><integer>8</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>&apos;p&apos; initialized to a null pointer value</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>&apos;p&apos; initialized to a null pointer value</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>53</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>53</integer>
// CHECK-NEXT:            <key>col</key><integer>5</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>54</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>54</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>54</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>54</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>54</integer>
// CHECK-NEXT:            <key>col</key><integer>6</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>54</integer>
// CHECK-NEXT:            <key>col</key><integer>6</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>54</integer>
// CHECK-NEXT:       <key>col</key><integer>6</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>54</integer>
// CHECK-NEXT:          <key>col</key><integer>4</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>54</integer>
// CHECK-NEXT:          <key>col</key><integer>4</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Dereference of null pointer (loaded from variable &apos;p&apos;)</string>
// CHECK-NEXT:    <key>category</key><string>Logic error</string>
// CHECK-NEXT:    <key>type</key><string>Dereference of null pointer</string>
// CHECK-NEXT:    <key>check_name</key><string>core.NullDereference</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>b41b510812f2a0d8735c67dfff6a9a5c</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>test_working</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>2</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>54</integer>
// CHECK-NEXT:    <key>col</key><integer>6</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>93</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>93</integer>
// CHECK-NEXT:            <key>col</key><integer>16</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>94</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>94</integer>
// CHECK-NEXT:            <key>col</key><integer>11</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>94</integer>
// CHECK-NEXT:       <key>col</key><integer>20</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>94</integer>
// CHECK-NEXT:          <key>col</key><integer>20</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>94</integer>
// CHECK-NEXT:          <key>col</key><integer>37</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Call to function &apos;CFDateCreate&apos; returns a Core Foundation object with a +1 retain count</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Call to function &apos;CFDateCreate&apos; returns a Core Foundation object with a +1 retain count</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>94</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>94</integer>
// CHECK-NEXT:            <key>col</key><integer>11</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>95</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>95</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>95</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>95</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>96</integer>
// CHECK-NEXT:            <key>col</key><integer>1</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>96</integer>
// CHECK-NEXT:            <key>col</key><integer>1</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>96</integer>
// CHECK-NEXT:       <key>col</key><integer>1</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Object leaked: object allocated and stored into &apos;date&apos; is not referenced later in this execution path and has a retain count of +1</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Object leaked: object allocated and stored into &apos;date&apos; is not referenced later in this execution path and has a retain count of +1</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Potential leak of an object stored into &apos;date&apos;</string>
// CHECK-NEXT:    <key>category</key><string>Memory (Core Foundation/Objective-C)</string>
// CHECK-NEXT:    <key>type</key><string>Leak</string>
// CHECK-NEXT:    <key>check_name</key><string>osx.cocoa.RetainCount</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>61d185b2522d15fb327f6784e0217adf</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>test_cf_leak</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>2</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>96</integer>
// CHECK-NEXT:    <key>col</key><integer>1</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>119</integer>
// CHECK-NEXT:       <key>col</key><integer>6</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>119</integer>
// CHECK-NEXT:          <key>col</key><integer>6</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>119</integer>
// CHECK-NEXT:          <key>col</key><integer>6</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>119</integer>
// CHECK-NEXT:          <key>col</key><integer>10</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>119</integer>
// CHECK-NEXT:          <key>col</key><integer>25</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Value stored to &apos;x&apos; during its initialization is never read</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Value stored to &apos;x&apos; during its initialization is never read</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Value stored to &apos;x&apos; during its initialization is never read</string>
// CHECK-NEXT:    <key>category</key><string>Dead store</string>
// CHECK-NEXT:    <key>type</key><string>Dead initialization</string>
// CHECK-NEXT:    <key>check_name</key><string>deadcode.DeadStores</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>63df5972efc8a7acccdbd2aca10c9e9e</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>rdar9424882</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>119</integer>
// CHECK-NEXT:    <key>col</key><integer>6</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>139</integer>
// CHECK-NEXT:       <key>col</key><integer>6</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>139</integer>
// CHECK-NEXT:          <key>col</key><integer>6</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>139</integer>
// CHECK-NEXT:          <key>col</key><integer>9</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>139</integer>
// CHECK-NEXT:          <key>col</key><integer>13</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>139</integer>
// CHECK-NEXT:          <key>col</key><integer>53</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Value stored to &apos;obj1&apos; during its initialization is never read</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Value stored to &apos;obj1&apos; during its initialization is never read</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Value stored to &apos;obj1&apos; during its initialization is never read</string>
// CHECK-NEXT:    <key>category</key><string>Dead store</string>
// CHECK-NEXT:    <key>type</key><string>Dead initialization</string>
// CHECK-NEXT:    <key>check_name</key><string>deadcode.DeadStores</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>125290ceaf2b55f5778c262d87b2b6d5</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>from_cf</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>139</integer>
// CHECK-NEXT:    <key>col</key><integer>6</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>144</integer>
// CHECK-NEXT:       <key>col</key><integer>6</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>144</integer>
// CHECK-NEXT:          <key>col</key><integer>6</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>144</integer>
// CHECK-NEXT:          <key>col</key><integer>9</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>144</integer>
// CHECK-NEXT:          <key>col</key><integer>13</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>144</integer>
// CHECK-NEXT:          <key>col</key><integer>45</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Value stored to &apos;obj4&apos; during its initialization is never read</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Value stored to &apos;obj4&apos; during its initialization is never read</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Value stored to &apos;obj4&apos; during its initialization is never read</string>
// CHECK-NEXT:    <key>category</key><string>Dead store</string>
// CHECK-NEXT:    <key>type</key><string>Dead initialization</string>
// CHECK-NEXT:    <key>check_name</key><string>deadcode.DeadStores</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>54627578ee3b8520400ae899bc32b3d6</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>from_cf</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>6</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>144</integer>
// CHECK-NEXT:    <key>col</key><integer>6</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>145</integer>
// CHECK-NEXT:       <key>col</key><integer>6</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>145</integer>
// CHECK-NEXT:          <key>col</key><integer>6</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>145</integer>
// CHECK-NEXT:          <key>col</key><integer>9</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>145</integer>
// CHECK-NEXT:          <key>col</key><integer>13</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>145</integer>
// CHECK-NEXT:          <key>col</key><integer>44</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Value stored to &apos;obj5&apos; during its initialization is never read</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Value stored to &apos;obj5&apos; during its initialization is never read</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Value stored to &apos;obj5&apos; during its initialization is never read</string>
// CHECK-NEXT:    <key>category</key><string>Dead store</string>
// CHECK-NEXT:    <key>type</key><string>Dead initialization</string>
// CHECK-NEXT:    <key>check_name</key><string>deadcode.DeadStores</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>27c31d7f5825a1613c0206b5be082800</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>from_cf</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>7</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>145</integer>
// CHECK-NEXT:    <key>col</key><integer>6</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>146</integer>
// CHECK-NEXT:       <key>col</key><integer>6</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>146</integer>
// CHECK-NEXT:          <key>col</key><integer>6</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>146</integer>
// CHECK-NEXT:          <key>col</key><integer>9</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>146</integer>
// CHECK-NEXT:          <key>col</key><integer>13</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>146</integer>
// CHECK-NEXT:          <key>col</key><integer>48</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Value stored to &apos;obj6&apos; during its initialization is never read</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Value stored to &apos;obj6&apos; during its initialization is never read</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Value stored to &apos;obj6&apos; during its initialization is never read</string>
// CHECK-NEXT:    <key>category</key><string>Dead store</string>
// CHECK-NEXT:    <key>type</key><string>Dead initialization</string>
// CHECK-NEXT:    <key>check_name</key><string>deadcode.DeadStores</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>a59b85c1e38300cb17eaeedcf193f94b</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>from_cf</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>8</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>146</integer>
// CHECK-NEXT:    <key>col</key><integer>6</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>139</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>139</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>145</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>145</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>145</integer>
// CHECK-NEXT:       <key>col</key><integer>26</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>145</integer>
// CHECK-NEXT:          <key>col</key><integer>26</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>145</integer>
// CHECK-NEXT:          <key>col</key><integer>44</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Call to function &apos;CFCreateSomething&apos; returns a Core Foundation object with a +1 retain count</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Call to function &apos;CFCreateSomething&apos; returns a Core Foundation object with a +1 retain count</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>145</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>145</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>146</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>146</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>146</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>146</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>146</integer>
// CHECK-NEXT:            <key>col</key><integer>33</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>146</integer>
// CHECK-NEXT:            <key>col</key><integer>46</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>146</integer>
// CHECK-NEXT:       <key>col</key><integer>33</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>146</integer>
// CHECK-NEXT:          <key>col</key><integer>33</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>146</integer>
// CHECK-NEXT:          <key>col</key><integer>48</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Object leaked: object allocated and stored into &apos;obj5&apos; is not referenced later in this execution path and has a retain count of +1</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Object leaked: object allocated and stored into &apos;obj5&apos; is not referenced later in this execution path and has a retain count of +1</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Potential leak of an object stored into &apos;obj5&apos;</string>
// CHECK-NEXT:    <key>category</key><string>Memory (Core Foundation/Objective-C)</string>
// CHECK-NEXT:    <key>type</key><string>Leak</string>
// CHECK-NEXT:    <key>check_name</key><string>osx.cocoa.RetainCount</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>5baa7d5f38420d0a035aa61607675f3e</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>from_cf</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>7</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>146</integer>
// CHECK-NEXT:    <key>col</key><integer>33</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>139</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>139</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>146</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>146</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>146</integer>
// CHECK-NEXT:       <key>col</key><integer>33</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>146</integer>
// CHECK-NEXT:          <key>col</key><integer>33</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>146</integer>
// CHECK-NEXT:          <key>col</key><integer>48</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Call to function &apos;CFCreateString&apos; returns a Core Foundation object with a +1 retain count</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Call to function &apos;CFCreateString&apos; returns a Core Foundation object with a +1 retain count</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>146</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>146</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>147</integer>
// CHECK-NEXT:            <key>col</key><integer>1</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>147</integer>
// CHECK-NEXT:            <key>col</key><integer>1</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>147</integer>
// CHECK-NEXT:       <key>col</key><integer>1</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Object leaked: object allocated and stored into &apos;obj6&apos; is not referenced later in this execution path and has a retain count of +1</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Object leaked: object allocated and stored into &apos;obj6&apos; is not referenced later in this execution path and has a retain count of +1</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Potential leak of an object stored into &apos;obj6&apos;</string>
// CHECK-NEXT:    <key>category</key><string>Memory (Core Foundation/Objective-C)</string>
// CHECK-NEXT:    <key>type</key><string>Leak</string>
// CHECK-NEXT:    <key>check_name</key><string>osx.cocoa.RetainCount</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>4665e04694fd55e7c4ed7a67860b3b74</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>from_cf</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>8</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>147</integer>
// CHECK-NEXT:    <key>col</key><integer>1</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>150</integer>
// CHECK-NEXT:       <key>col</key><integer>13</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>150</integer>
// CHECK-NEXT:          <key>col</key><integer>13</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>150</integer>
// CHECK-NEXT:          <key>col</key><integer>15</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>150</integer>
// CHECK-NEXT:          <key>col</key><integer>19</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>150</integer>
// CHECK-NEXT:          <key>col</key><integer>64</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Value stored to &apos;cf1&apos; during its initialization is never read</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Value stored to &apos;cf1&apos; during its initialization is never read</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Value stored to &apos;cf1&apos; during its initialization is never read</string>
// CHECK-NEXT:    <key>category</key><string>Dead store</string>
// CHECK-NEXT:    <key>type</key><string>Dead initialization</string>
// CHECK-NEXT:    <key>check_name</key><string>deadcode.DeadStores</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>70a67cc8fefa3ad133a35c3ffb579b9e</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>to_cf</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>150</integer>
// CHECK-NEXT:    <key>col</key><integer>13</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>151</integer>
// CHECK-NEXT:       <key>col</key><integer>15</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>151</integer>
// CHECK-NEXT:          <key>col</key><integer>15</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>151</integer>
// CHECK-NEXT:          <key>col</key><integer>17</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>151</integer>
// CHECK-NEXT:          <key>col</key><integer>21</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>151</integer>
// CHECK-NEXT:          <key>col</key><integer>67</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Value stored to &apos;cf2&apos; during its initialization is never read</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Value stored to &apos;cf2&apos; during its initialization is never read</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Value stored to &apos;cf2&apos; during its initialization is never read</string>
// CHECK-NEXT:    <key>category</key><string>Dead store</string>
// CHECK-NEXT:    <key>type</key><string>Dead initialization</string>
// CHECK-NEXT:    <key>check_name</key><string>deadcode.DeadStores</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>afc41cacb8f32bfbe4a4152f20cec5bc</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>to_cf</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>2</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>151</integer>
// CHECK-NEXT:    <key>col</key><integer>15</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>152</integer>
// CHECK-NEXT:       <key>col</key><integer>13</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>152</integer>
// CHECK-NEXT:          <key>col</key><integer>13</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>152</integer>
// CHECK-NEXT:          <key>col</key><integer>15</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>152</integer>
// CHECK-NEXT:          <key>col</key><integer>19</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>152</integer>
// CHECK-NEXT:          <key>col</key><integer>55</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Value stored to &apos;cf3&apos; during its initialization is never read</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Value stored to &apos;cf3&apos; during its initialization is never read</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Value stored to &apos;cf3&apos; during its initialization is never read</string>
// CHECK-NEXT:    <key>category</key><string>Dead store</string>
// CHECK-NEXT:    <key>type</key><string>Dead initialization</string>
// CHECK-NEXT:    <key>check_name</key><string>deadcode.DeadStores</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>e512cd0ada59beca5acfa53e2a632670</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>to_cf</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>3</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>152</integer>
// CHECK-NEXT:    <key>col</key><integer>13</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>153</integer>
// CHECK-NEXT:       <key>col</key><integer>15</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>153</integer>
// CHECK-NEXT:          <key>col</key><integer>15</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>153</integer>
// CHECK-NEXT:          <key>col</key><integer>17</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>153</integer>
// CHECK-NEXT:          <key>col</key><integer>21</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>153</integer>
// CHECK-NEXT:          <key>col</key><integer>58</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Value stored to &apos;cf4&apos; during its initialization is never read</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Value stored to &apos;cf4&apos; during its initialization is never read</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Value stored to &apos;cf4&apos; during its initialization is never read</string>
// CHECK-NEXT:    <key>category</key><string>Dead store</string>
// CHECK-NEXT:    <key>type</key><string>Dead initialization</string>
// CHECK-NEXT:    <key>check_name</key><string>deadcode.DeadStores</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>a43ae78dcef14395931eeb452f81819f</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>to_cf</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>4</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>153</integer>
// CHECK-NEXT:    <key>col</key><integer>15</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>164</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>164</integer>
// CHECK-NEXT:            <key>col</key><integer>16</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>165</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>165</integer>
// CHECK-NEXT:            <key>col</key><integer>11</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>165</integer>
// CHECK-NEXT:       <key>col</key><integer>20</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>165</integer>
// CHECK-NEXT:          <key>col</key><integer>20</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>165</integer>
// CHECK-NEXT:          <key>col</key><integer>37</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Call to function &apos;CFDateCreate&apos; returns a Core Foundation object with a +1 retain count</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Call to function &apos;CFDateCreate&apos; returns a Core Foundation object with a +1 retain count</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>165</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>165</integer>
// CHECK-NEXT:            <key>col</key><integer>11</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>166</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>166</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>166</integer>
// CHECK-NEXT:       <key>col</key><integer>3</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>166</integer>
// CHECK-NEXT:          <key>col</key><integer>3</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>166</integer>
// CHECK-NEXT:          <key>col</key><integer>6</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Object leaked: object allocated and stored into &apos;date&apos; is not referenced later in this execution path and has a retain count of +1</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Object leaked: object allocated and stored into &apos;date&apos; is not referenced later in this execution path and has a retain count of +1</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Potential leak of an object stored into &apos;date&apos;</string>
// CHECK-NEXT:    <key>category</key><string>Memory (Core Foundation/Objective-C)</string>
// CHECK-NEXT:    <key>type</key><string>Leak</string>
// CHECK-NEXT:    <key>check_name</key><string>osx.cocoa.RetainCount</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>798e65f80df0526369f9bb240e3d91fd</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>test_objc_unretainedObject</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>2</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>166</integer>
// CHECK-NEXT:    <key>col</key><integer>3</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>178</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>178</integer>
// CHECK-NEXT:            <key>col</key><integer>16</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>181</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>181</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>181</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>181</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>185</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>185</integer>
// CHECK-NEXT:            <key>col</key><integer>16</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>185</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>185</integer>
// CHECK-NEXT:            <key>col</key><integer>16</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>189</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>189</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>189</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>189</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>193</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>193</integer>
// CHECK-NEXT:            <key>col</key><integer>16</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>193</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>193</integer>
// CHECK-NEXT:            <key>col</key><integer>16</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>196</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>196</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>196</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>196</integer>
// CHECK-NEXT:            <key>col</key><integer>9</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>200</integer>
// CHECK-NEXT:            <key>col</key><integer>7</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>200</integer>
// CHECK-NEXT:            <key>col</key><integer>20</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>200</integer>
// CHECK-NEXT:       <key>col</key><integer>7</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>200</integer>
// CHECK-NEXT:          <key>col</key><integer>7</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>200</integer>
// CHECK-NEXT:          <key>col</key><integer>22</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Call to function &apos;CFCreateString&apos; returns a Core Foundation object with a +1 retain count</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Call to function &apos;CFCreateString&apos; returns a Core Foundation object with a +1 retain count</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>200</integer>
// CHECK-NEXT:            <key>col</key><integer>7</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>200</integer>
// CHECK-NEXT:            <key>col</key><integer>20</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>202</integer>
// CHECK-NEXT:            <key>col</key><integer>1</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>202</integer>
// CHECK-NEXT:            <key>col</key><integer>1</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>202</integer>
// CHECK-NEXT:       <key>col</key><integer>1</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Object leaked: allocated object is not referenced later in this execution path and has a retain count of +1</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Potential leak of an object</string>
// CHECK-NEXT:    <key>category</key><string>Memory (Core Foundation/Objective-C)</string>
// CHECK-NEXT:    <key>type</key><string>Leak</string>
// CHECK-NEXT:    <key>check_name</key><string>osx.cocoa.RetainCount</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>e1fbcc142b678b3c2c43737ee35b64d9</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>test_objc_arrays</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>24</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>202</integer>
// CHECK-NEXT:    <key>col</key><integer>1</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>214</integer>
// CHECK-NEXT:       <key>col</key><integer>17</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>214</integer>
// CHECK-NEXT:          <key>col</key><integer>17</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>214</integer>
// CHECK-NEXT:          <key>col</key><integer>39</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Method returns an Objective-C object with a +0 retain count</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Method returns an Objective-C object with a +0 retain count</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>214</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>214</integer>
// CHECK-NEXT:            <key>col</key><integer>10</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>215</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>215</integer>
// CHECK-NEXT:            <key>col</key><integer>18</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>215</integer>
// CHECK-NEXT:       <key>col</key><integer>3</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>215</integer>
// CHECK-NEXT:          <key>col</key><integer>3</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>215</integer>
// CHECK-NEXT:          <key>col</key><integer>21</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>215</integer>
// CHECK-NEXT:          <key>col</key><integer>20</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>215</integer>
// CHECK-NEXT:          <key>col</key><integer>20</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Reference count incremented. The object now has a +1 retain count</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Reference count incremented. The object now has a +1 retain count</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>215</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>215</integer>
// CHECK-NEXT:            <key>col</key><integer>18</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>216</integer>
// CHECK-NEXT:            <key>col</key><integer>1</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>216</integer>
// CHECK-NEXT:            <key>col</key><integer>1</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>216</integer>
// CHECK-NEXT:       <key>col</key><integer>1</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Object leaked: object allocated and stored into &apos;o&apos; is not referenced later in this execution path and has a retain count of +1</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Object leaked: object allocated and stored into &apos;o&apos; is not referenced later in this execution path and has a retain count of +1</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Potential leak of an object stored into &apos;o&apos;</string>
// CHECK-NEXT:    <key>category</key><string>Memory (Core Foundation/Objective-C)</string>
// CHECK-NEXT:    <key>type</key><string>Leak</string>
// CHECK-NEXT:    <key>check_name</key><string>osx.cocoa.RetainCount</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>e300a279615a384d2b310329651d3978</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>rdar11059275_positive</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>1</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>216</integer>
// CHECK-NEXT:    <key>col</key><integer>1</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>path</key>
// CHECK-NEXT:    <array>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>229</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>229</integer>
// CHECK-NEXT:            <key>col</key><integer>4</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>230</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>230</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>control</string>
// CHECK-NEXT:      <key>edges</key>
// CHECK-NEXT:       <array>
// CHECK-NEXT:        <dict>
// CHECK-NEXT:         <key>start</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>230</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>230</integer>
// CHECK-NEXT:            <key>col</key><integer>3</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:         <key>end</key>
// CHECK-NEXT:          <array>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>230</integer>
// CHECK-NEXT:            <key>col</key><integer>22</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:           <dict>
// CHECK-NEXT:            <key>line</key><integer>230</integer>
// CHECK-NEXT:            <key>col</key><integer>22</integer>
// CHECK-NEXT:            <key>file</key><integer>0</integer>
// CHECK-NEXT:           </dict>
// CHECK-NEXT:          </array>
// CHECK-NEXT:        </dict>
// CHECK-NEXT:       </array>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:     <dict>
// CHECK-NEXT:      <key>kind</key><string>event</string>
// CHECK-NEXT:      <key>location</key>
// CHECK-NEXT:      <dict>
// CHECK-NEXT:       <key>line</key><integer>230</integer>
// CHECK-NEXT:       <key>col</key><integer>22</integer>
// CHECK-NEXT:       <key>file</key><integer>0</integer>
// CHECK-NEXT:      </dict>
// CHECK-NEXT:      <key>ranges</key>
// CHECK-NEXT:      <array>
// CHECK-NEXT:        <array>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>230</integer>
// CHECK-NEXT:          <key>col</key><integer>3</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:         <dict>
// CHECK-NEXT:          <key>line</key><integer>230</integer>
// CHECK-NEXT:          <key>col</key><integer>24</integer>
// CHECK-NEXT:          <key>file</key><integer>0</integer>
// CHECK-NEXT:         </dict>
// CHECK-NEXT:        </array>
// CHECK-NEXT:      </array>
// CHECK-NEXT:      <key>depth</key><integer>0</integer>
// CHECK-NEXT:      <key>extended_message</key>
// CHECK-NEXT:      <string>Dereference of null pointer</string>
// CHECK-NEXT:      <key>message</key>
// CHECK-NEXT:      <string>Dereference of null pointer</string>
// CHECK-NEXT:     </dict>
// CHECK-NEXT:    </array>
// CHECK-NEXT:    <key>description</key><string>Dereference of null pointer</string>
// CHECK-NEXT:    <key>category</key><string>Logic error</string>
// CHECK-NEXT:    <key>type</key><string>Dereference of null pointer</string>
// CHECK-NEXT:    <key>check_name</key><string>core.NullDereference</string>
// CHECK-NEXT:    <!-- This hash is experimental and going to change! -->
// CHECK-NEXT:    <key>issue_hash_content_of_line_in_context</key><string>37efdaee587b783c6d49a228ff5ba49f</string>
// CHECK-NEXT:   <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT:   <key>issue_context</key><string>rdar14061675</string>
// CHECK-NEXT:   <key>issue_hash_function_offset</key><string>4</string>
// CHECK-NEXT:   <key>location</key>
// CHECK-NEXT:   <dict>
// CHECK-NEXT:    <key>line</key><integer>230</integer>
// CHECK-NEXT:    <key>col</key><integer>22</integer>
// CHECK-NEXT:    <key>file</key><integer>0</integer>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:   </dict>
// CHECK-NEXT:  </array>
