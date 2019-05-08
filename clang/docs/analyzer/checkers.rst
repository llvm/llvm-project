==================
Available Checkers
==================

The analyzer performs checks that are categorized into families or "checkers".

The default set of checkers covers a variety of checks targeted at finding security and API usage bugs,
dead code, and other logic errors. See the :ref:`default-checkers` checkers list below.

In addition to these, the analyzer contains a number of :ref:`alpha-checkers` (aka *alpha* checkers).
These checkers are under development and are switched off by default. They may crash or emit a higher number of false positives.

The :ref:`debug-checkers` package contains checkers for analyzer developers for debugging purposes.

.. contents:: Table of Contents
   :depth: 4


.. _default-checkers:

Default Checkers
----------------

.. _core-checkers:

core
^^^^
Models core language features and contains general-purpose checkers such as division by zero,
null pointer dereference, usage of uninitialized values, etc.
*These checkers must be always switched on as other checker rely on them.*

core.CallAndMessage (C, C++, ObjC)
""""""""""""""""""""""""""""""""""
 Check for logical errors for function calls and Objective-C message expressions (e.g., uninitialized arguments, null function pointers).

.. literalinclude:: checkers/callandmessage_example.c
    :language: objc

core.DivideZero (C, C++, ObjC)
""""""""""""""""""""""""""""""
 Check for division by zero.

.. literalinclude:: checkers/dividezero_example.c
    :language: c

core.NonNullParamChecker (C, C++, ObjC)
"""""""""""""""""""""""""""""""""""""""
Check for null pointers passed as arguments to a function whose arguments are references or marked with the 'nonnull' attribute.

.. code-block:: cpp

 int f(int *p) __attribute__((nonnull));

 void test(int *p) {
   if (!p)
     f(p); // warn
 }

core.NullDereference (C, C++, ObjC)
"""""""""""""""""""""""""""""""""""
Check for dereferences of null pointers.

.. code-block:: objc

 // C
 void test(int *p) {
   if (p)
     return;

   int x = p[0]; // warn
 }

 // C
 void test(int *p) {
   if (!p)
     *p = 0; // warn
 }

 // C++
 class C {
 public:
   int x;
 };

 void test() {
   C *pc = 0;
   int k = pc->x; // warn
 }

 // Objective-C
 @interface MyClass {
 @public
   int x;
 }
 @end

 void test() {
   MyClass *obj = 0;
   obj->x = 1; // warn
 }

core.StackAddressEscape (C)
"""""""""""""""""""""""""""
Check that addresses to stack memory do not escape the function.

.. code-block:: c

 char const *p;

 void test() {
   char const str[] = "string";
   p = str; // warn
 }

 void* test() {
    return __builtin_alloca(12); // warn
 }

 void test() {
   static int *x;
   int y;
   x = &y; // warn
 }


core.UndefinedBinaryOperatorResult (C)
""""""""""""""""""""""""""""""""""""""
Check for undefined results of binary operators.

.. code-block:: c

 void test() {
   int x;
   int y = x + 1; // warn: left operand is garbage
 }

core.VLASize (C)
""""""""""""""""
Check for declarations of Variable Length Arrays of undefined or zero size.

 Check for declarations of VLA of undefined or zero size.

.. code-block:: c

 void test() {
   int x;
   int vla1[x]; // warn: garbage as size
 }

 void test() {
   int x = 0;
   int vla2[x]; // warn: zero size
 }

core.uninitialized.ArraySubscript (C)
"""""""""""""""""""""""""""""""""""""
Check for uninitialized values used as array subscripts.

.. code-block:: c

 void test() {
   int i, a[10];
   int x = a[i]; // warn: array subscript is undefined
 }

core.uninitialized.Assign (C)
"""""""""""""""""""""""""""""
Check for assigning uninitialized values.

.. code-block:: c

 void test() {
   int x;
   x |= 1; // warn: left expression is uninitialized
 }

core.uninitialized.Branch (C)
"""""""""""""""""""""""""""""
Check for uninitialized values used as branch conditions.

.. code-block:: c

 void test() {
   int x;
   if (x) // warn
     return;
 }

core.uninitialized.CapturedBlockVariable (C)
""""""""""""""""""""""""""""""""""""""""""""
Check for blocks that capture uninitialized values.

.. code-block:: c

 void test() {
   int x;
   ^{ int y = x; }(); // warn
 }

core.uninitialized.UndefReturn (C)
""""""""""""""""""""""""""""""""""
Check for uninitialized values being returned to the caller.

.. code-block:: c

 int test() {
   int x;
   return x; // warn
 }

.. _cplusplus-checkers:


cpluslus
^^^^^^^^

C++ Checkers.

cplusplus.InnerPointer
""""""""""""""""""""""
Check for inner pointers of C++ containers used after re/deallocation.

cplusplus.NewDelete (C++)
"""""""""""""""""""""""""
Check for double-free and use-after-free problems. Traces memory managed by new/delete.

.. literalinclude:: checkers/newdelete_example.cpp
    :language: cpp

cplusplus.NewDeleteLeaks (C++)
""""""""""""""""""""""""""""""
Check for memory leaks. Traces memory managed by new/delete.

.. code-block:: cpp

 void test() {
   int *p = new int;
 } // warn


cplusplus.SelfAssignment (C++)
""""""""""""""""""""""""""""""
Checks C++ copy and move assignment operators for self assignment.

.. _deadcode-checkers:

deadcode
^^^^^^^^

Dead Code Checkers.

deadcode.DeadStores (C)
"""""""""""""""""""""""
Check for values stored to variables that are never read afterwards.

.. code-block:: c

 void test() {
   int x;
   x = 1; // warn
 }

.. _nullability-checkers:

nullability
^^^^^^^^^^^

Objective C checkers that warn for null pointer passing and dereferencing errors.

nullability.NullPassedToNonnull (ObjC)
""""""""""""""""""""""""""""""""""""""
Warns when a null pointer is passed to a pointer which has a _Nonnull type.

.. code-block:: objc

 if (name != nil)
   return;
 // Warning: nil passed to a callee that requires a non-null 1st parameter
 NSString *greeting = [@"Hello " stringByAppendingString:name];

nullability.NullReturnedFromNonnull (ObjC)
""""""""""""""""""""""""""""""""""""""""""
Warns when a null pointer is returned from a function that has _Nonnull return type.

.. code-block:: objc

 - (nonnull id)firstChild {
   id result = nil;
   if ([_children count] > 0)
     result = _children[0];

   // Warning: nil returned from a method that is expected
   // to return a non-null value
   return result;
 }

nullability.NullableDereferenced (ObjC)
"""""""""""""""""""""""""""""""""""""""
Warns when a nullable pointer is dereferenced.

.. code-block:: objc

 struct LinkedList {
   int data;
   struct LinkedList *next;
 };

 struct LinkedList * _Nullable getNext(struct LinkedList *l);

 void updateNextData(struct LinkedList *list, int newData) {
   struct LinkedList *next = getNext(list);
   // Warning: Nullable pointer is dereferenced
   next->data = 7;
 }

nullability.NullablePassedToNonnull (ObjC)
""""""""""""""""""""""""""""""""""""""""""
Warns when a nullable pointer is passed to a pointer which has a _Nonnull type.

.. code-block:: objc

 typedef struct Dummy { int val; } Dummy;
 Dummy *_Nullable returnsNullable();
 void takesNonnull(Dummy *_Nonnull);

 void test() {
   Dummy *p = returnsNullable();
   takesNonnull(p); // warn
 }

nullability.NullableReturnedFromNonnull (ObjC)
""""""""""""""""""""""""""""""""""""""""""""""
Warns when a nullable pointer is returned from a function that has _Nonnull return type.

.. _optin-checkers:

optin
^^^^^

Checkers for portability, performance or coding style specific rules.

optin.cplusplus.UninitializedObject (C++)
"""""""""""""""""""""""""""""""""""

This checker reports uninitialized fields in objects created after a constructor
call. It doesn't only find direct uninitialized fields, but rather makes a deep
inspection of the object, analyzing all of it's fields subfields.
The checker regards inherited fields as direct fields, so one will recieve
warnings for uninitialized inherited data members as well.

.. code-block:: cpp

 // With Pedantic and CheckPointeeInitialization set to true

 struct A {
   struct B {
     int x; // note: uninitialized field 'this->b.x'
     // note: uninitialized field 'this->bptr->x'
     int y; // note: uninitialized field 'this->b.y'
     // note: uninitialized field 'this->bptr->y'
   };
   int *iptr; // note: uninitialized pointer 'this->iptr'
   B b;
   B *bptr;
   char *cptr; // note: uninitialized pointee 'this->cptr'

   A (B *bptr, char *cptr) : bptr(bptr), cptr(cptr) {}
 };

 void f() {
   A::B b;
   char c;
   A a(&b, &c); // warning: 6 uninitialized fields
  //          after the constructor call
 }

 // With Pedantic set to false and
 // CheckPointeeInitialization set to true
 // (every field is uninitialized)

 struct A {
   struct B {
     int x;
     int y;
   };
   int *iptr;
   B b;
   B *bptr;
   char *cptr;

   A (B *bptr, char *cptr) : bptr(bptr), cptr(cptr) {}
 };

 void f() {
   A::B b;
   char c;
   A a(&b, &c); // no warning
 }

 // With Pedantic set to true and
 // CheckPointeeInitialization set to false
 // (pointees are regarded as initialized)

 struct A {
   struct B {
     int x; // note: uninitialized field 'this->b.x'
     int y; // note: uninitialized field 'this->b.y'
   };
   int *iptr; // note: uninitialized pointer 'this->iptr'
   B b;
   B *bptr;
   char *cptr;

   A (B *bptr, char *cptr) : bptr(bptr), cptr(cptr) {}
 };

 void f() {
   A::B b;
   char c;
   A a(&b, &c); // warning: 3 uninitialized fields
  //          after the constructor call
 }


**Options**

This checker has several options which can be set from command line (e.g.
``-analyzer-config optin.cplusplus.UninitializedObject:Pedantic=true``):

* ``Pedantic`` (boolean). If to false, the checker won't emit warnings for
  objects that don't have at least one initialized field. Defaults to false.

* ``NotesAsWarnings``  (boolean). If set to true, the checker will emit a
  warning for each uninitalized field, as opposed to emitting one warning per
  constructor call, and listing the uninitialized fields that belongs to it in
  notes. *Defaults to false*.

* ``CheckPointeeInitialization`` (boolean). If set to false, the checker will
  not analyze the pointee of pointer/reference fields, and will only check
  whether the object itself is initialized. *Defaults to false*.

* ``IgnoreRecordsWithField`` (string). If supplied, the checker will not analyze
  structures that have a field with a name or type name that matches  the given
  pattern. *Defaults to ""*.

optin.cplusplus.VirtualCall (C++)
"""""""""""""""""""""""""""""""""
Check virtual function calls during construction or destruction.

.. code-block:: cpp

 class A {
 public:
   A() {
     f(); // warn
   }
   virtual void f();
 };

 class A {
 public:
   ~A() {
     this->f(); // warn
   }
   virtual void f();
 };

optin.mpi.MPI-Checker (C)
"""""""""""""""""""""""""
Checks MPI code.

.. code-block:: c

 void test() {
   double buf = 0;
   MPI_Request sendReq1;
   MPI_Ireduce(MPI_IN_PLACE, &buf, 1, MPI_DOUBLE, MPI_SUM,
       0, MPI_COMM_WORLD, &sendReq1);
 } // warn: request 'sendReq1' has no matching wait.

 void test() {
   double buf = 0;
   MPI_Request sendReq;
   MPI_Isend(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq);
   MPI_Irecv(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq); // warn
   MPI_Isend(&buf, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &sendReq); // warn
   MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
 }

 void missingNonBlocking() {
   int rank = 0;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Request sendReq1[10][10][10];
   MPI_Wait(&sendReq1[1][7][9], MPI_STATUS_IGNORE); // warn
 }

optin.osx.cocoa.localizability.EmptyLocalizationContextChecker (ObjC)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Check that NSLocalizedString macros include a comment for context.

.. code-block:: objc

 - (void)test {
   NSString *string = NSLocalizedString(@"LocalizedString", nil); // warn
   NSString *string2 = NSLocalizedString(@"LocalizedString", @" "); // warn
   NSString *string3 = NSLocalizedStringWithDefaultValue(
     @"LocalizedString", nil, [[NSBundle alloc] init], nil,@""); // warn
 }

optin.osx.cocoa.localizability.NonLocalizedStringChecker (ObjC)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Warns about uses of non-localized NSStrings passed to UI methods expecting localized NSStrings.

.. code-block:: objc

 NSString *alarmText =
   NSLocalizedString(@"Enabled", @"Indicates alarm is turned on");
 if (!isEnabled) {
   alarmText = @"Disabled";
 }
 UILabel *alarmStateLabel = [[UILabel alloc] init];

 // Warning: User-facing text should use localized string macro
 [alarmStateLabel setText:alarmText];

optin.performance.GCDAntipattern
""""""""""""""""""""""""""""""""
Check for performance anti-patterns when using Grand Central Dispatch.

optin.performance.Padding
"""""""""""""""""""""""""
Check for excessively padded structs.

optin.portability.UnixAPI
"""""""""""""""""""""""""
Finds implementation-defined behavior in UNIX/Posix functions.


.. _security-checkers:

security
^^^^^^^^

Security related checkers.

security.FloatLoopCounter (C)
"""""""""""""""""""""""""""""
Warn on using a floating point value as a loop counter (CERT: FLP30-C, FLP30-CPP).

.. code-block:: c

 void test() {
   for (float x = 0.1f; x <= 1.0f; x += 0.1f) {} // warn
 }

security.insecureAPI.UncheckedReturn (C)
""""""""""""""""""""""""""""""""""""""""
Warn on uses of functions whose return values must be always checked.

.. code-block:: c

 void test() {
   setuid(1); // warn
 }

security.insecureAPI.bcmp (C)
"""""""""""""""""""""""""""""
Warn on uses of the 'bcmp' function.

.. code-block:: c

 void test() {
   bcmp(ptr0, ptr1, n); // warn
 }

security.insecureAPI.bcopy (C)
""""""""""""""""""""""""""""""
Warn on uses of the 'bcopy' function.

.. code-block:: c

 void test() {
   bcopy(src, dst, n); // warn
 }

security.insecureAPI.bzero (C)
""""""""""""""""""""""""""""""
Warn on uses of the 'bzero' function.

.. code-block:: c

 void test() {
   bzero(ptr, n); // warn
 }

security.insecureAPI.getpw (C)
""""""""""""""""""""""""""""""
Warn on uses of the 'getpw' function.

.. code-block:: c

 void test() {
   char buff[1024];
   getpw(2, buff); // warn
 }

security.insecureAPI.gets (C)
"""""""""""""""""""""""""""""
Warn on uses of the 'gets' function.

.. code-block:: c

 void test() {
   char buff[1024];
   gets(buff); // warn
 }

security.insecureAPI.mkstemp (C)
""""""""""""""""""""""""""""""""
Warn when 'mkstemp' is passed fewer than 6 X's in the format string.

.. code-block:: c

 void test() {
   mkstemp("XX"); // warn
 }

security.insecureAPI.mktemp (C)
"""""""""""""""""""""""""""""""
Warn on uses of the ``mktemp`` function.

.. code-block:: c

 void test() {
   char *x = mktemp("/tmp/zxcv"); // warn: insecure, use mkstemp
 }

security.insecureAPI.rand (C)
"""""""""""""""""""""""""""""
Warn on uses of inferior random number generating functions (only if arc4random function is available):
``drand48, erand48, jrand48, lcong48, lrand48, mrand48, nrand48, random, rand_r``.

.. code-block:: c

 void test() {
   random(); // warn
 }

security.insecureAPI.strcpy (C)
"""""""""""""""""""""""""""""""
Warn on uses of the ``strcpy`` and ``strcat`` functions.

.. code-block:: c

 void test() {
   char x[4];
   char *y = "abcd";

   strcpy(x, y); // warn
 }


security.insecureAPI.vfork (C)
""""""""""""""""""""""""""""""
 Warn on uses of the 'vfork' function.

.. code-block:: c

 void test() {
   vfork(); // warn
 }

security.insecureAPI.DeprecatedOrUnsafeBufferHandling (C)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 Warn on occurrences of unsafe or deprecated buffer handling functions, which now have a secure variant: ``sprintf, vsprintf, scanf, wscanf, fscanf, fwscanf, vscanf, vwscanf, vfscanf, vfwscanf, sscanf, swscanf, vsscanf, vswscanf, swprintf, snprintf, vswprintf, vsnprintf, memcpy, memmove, strncpy, strncat, memset``

.. code-block:: c

 void test() {
   char buf [5];
   strncpy(buf, "a", 1); // warn
 }

.. _unix-checkers:

unix
^^^^
POSIX/Unix checkers.

unix.API (C)
""""""""""""
Check calls to various UNIX/Posix functions: ``open, pthread_once, calloc, malloc, realloc, alloca``.

.. literalinclude:: checkers/unix_api_example.c
    :language: c

unix.Malloc (C)
"""""""""""""""
Check for memory leaks, double free, and use-after-free problems. Traces memory managed by malloc()/free().

.. literalinclude:: checkers/unix_malloc_example.c
    :language: c

unix.MallocSizeof (C)
"""""""""""""""""""""
Check for dubious ``malloc`` arguments involving ``sizeof``.

.. code-block:: c

 void test() {
   long *p = malloc(sizeof(short));
     // warn: result is converted to 'long *', which is
     // incompatible with operand type 'short'
   free(p);
 }

unix.MismatchedDeallocator (C, C++)
"""""""""""""""""""""""""""""""""""
Check for mismatched deallocators.

.. literalinclude:: checkers/mismatched_deallocator_example.cpp
    :language: c

unix.Vfork (C)
""""""""""""""
Check for proper usage of ``vfork``.

.. code-block:: c

 int test(int x) {
   pid_t pid = vfork(); // warn
   if (pid != 0)
     return 0;

   switch (x) {
   case 0:
     pid = 1;
     execl("", "", 0);
     _exit(1);
     break;
   case 1:
     x = 0; // warn: this assignment is prohibited
     break;
   case 2:
     foo(); // warn: this function call is prohibited
     break;
   default:
     return 0; // warn: return is prohibited
   }

   while(1);
 }

unix.cstring.BadSizeArg (C)
"""""""""""""""""""""""""""
Check the size argument passed into C string functions for common erroneous patterns. Use ``-Wno-strncat-size`` compiler option to mute other ``strncat``-related compiler warnings.

.. code-block:: c

 void test() {
   char dest[3];
   strncat(dest, """""""""""""""""""""""""*", sizeof(dest));
     // warn: potential buffer overflow
 }

unix.cstrisng.NullArg (C)
"""""""""""""""""""""""""
Check for null pointers being passed as arguments to C string functions:
``strlen, strnlen, strcpy, strncpy, strcat, strncat, strcmp, strncmp, strcasecmp, strncasecmp``.

.. code-block:: c

 int test() {
   return strlen(0); // warn
 }

.. _osx-checkers:

osx
^^^
OS X checkers.

osx.API (C)
"""""""""""
Check for proper uses of various Apple APIs.

.. code-block:: objc

 void test() {
   dispatch_once_t pred = 0;
   dispatch_once(&pred, ^(){}); // warn: dispatch_once uses local
 }

osx.NumberObjectConversion (C, C++, ObjC)
"""""""""""""""""""""""""""""""""""""""""
Check for erroneous conversions of objects representing numbers into numbers.

.. code-block:: objc

 NSNumber *photoCount = [albumDescriptor objectForKey:@"PhotoCount"];
 // Warning: Comparing a pointer value of type 'NSNumber *'
 // to a scalar integer value
 if (photoCount > 0) {
   [self displayPhotos];
 }

osx.ObjCProperty (ObjC)
"""""""""""""""""""""""
Check for proper uses of Objective-C properties.

.. code-block:: objc

 NSNumber *photoCount = [albumDescriptor objectForKey:@"PhotoCount"];
 // Warning: Comparing a pointer value of type 'NSNumber *'
 // to a scalar integer value
 if (photoCount > 0) {
   [self displayPhotos];
 }


osx.SecKeychainAPI (C)
""""""""""""""""""""""
Check for proper uses of Secure Keychain APIs.

.. literalinclude:: checkers/seckeychainapi_example.m
    :language: objc

osx.cocoa.AtSync (ObjC)
"""""""""""""""""""""""
Check for nil pointers used as mutexes for @synchronized.

.. code-block:: objc

 void test(id x) {
   if (!x)
     @synchronized(x) {} // warn: nil value used as mutex
 }

 void test() {
   id y;
   @synchronized(y) {} // warn: uninitialized value used as mutex
 }

osx.cocoa.AutoreleaseWrite
""""""""""""""""""""""""""
Warn about potentially crashing writes to autoreleasing objects from different autoreleasing pools in Objective-C.

osx.cocoa.ClassRelease (ObjC)
"""""""""""""""""""""""""""""
Check for sending 'retain', 'release', or 'autorelease' directly to a Class.

.. code-block:: objc

 @interface MyClass : NSObject
 @end

 void test(void) {
   [MyClass release]; // warn
 }

osx.cocoa.Dealloc (ObjC)
""""""""""""""""""""""""
Warn about Objective-C classes that lack a correct implementation of -dealloc

.. literalinclude:: checkers/dealloc_example.m
    :language: objc

osx.cocoa.IncompatibleMethodTypes (ObjC)
""""""""""""""""""""""""""""""""""""""""
Warn about Objective-C method signatures with type incompatibilities.

.. code-block:: objc

 @interface MyClass1 : NSObject
 - (int)foo;
 @end

 @implementation MyClass1
 - (int)foo { return 1; }
 @end

 @interface MyClass2 : MyClass1
 - (float)foo;
 @end

 @implementation MyClass2
 - (float)foo { return 1.0; } // warn
 @end

osx.cocoa.Loops
"""""""""""""""
Improved modeling of loops using Cocoa collection types.

osx.cocoa.MissingSuperCall (ObjC)
"""""""""""""""""""""""""""""""""
Warn about Objective-C methods that lack a necessary call to super.

.. code-block:: objc

 @interface Test : UIViewController
 @end
 @implementation test
 - (void)viewDidLoad {} // warn
 @end


osx.cocoa.NSAutoreleasePool (ObjC)
""""""""""""""""""""""""""""""""""
Warn for suboptimal uses of NSAutoreleasePool in Objective-C GC mode.

.. code-block:: objc

 void test() {
   NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
   [pool release]; // warn
 }

osx.cocoa.NSError (ObjC)
""""""""""""""""""""""""
Check usage of NSError parameters.

.. code-block:: objc

 @interface A : NSObject
 - (void)foo:(NSError """""""""""""""""""""""")error;
 @end

 @implementation A
 - (void)foo:(NSError """""""""""""""""""""""")error {
   // warn: method accepting NSError"""""""""""""""""""""""" should have a non-void
   // return value
 }
 @end

 @interface A : NSObject
 - (BOOL)foo:(NSError """""""""""""""""""""""")error;
 @end

 @implementation A
 - (BOOL)foo:(NSError """""""""""""""""""""""")error {
   *error = 0; // warn: potential null dereference
   return 0;
 }
 @end

osx.cocoa.NilArg (ObjC)
"""""""""""""""""""""""
Check for prohibited nil arguments to ObjC method calls.

 - caseInsensitiveCompare:
 - compare:
 - compare:options:
 - compare:options:range:
 - compare:options:range:locale:
 - componentsSeparatedByCharactersInSet:
 - initWithFormat:

.. code-block:: objc

 NSComparisonResult test(NSString *s) {
   NSString *aString = nil;
   return [s caseInsensitiveCompare:aString];
     // warn: argument to 'NSString' method
     // 'caseInsensitiveCompare:' cannot be nil
 }


osx.cocoa.NonNilReturnValue
"""""""""""""""""""""""""""
Models the APIs that are guaranteed to return a non-nil value.

osx.cocoa.ObjCGenerics (ObjC)
"""""""""""""""""""""""""""""
Check for type errors when using Objective-C generics.

.. code-block:: objc

 NSMutableArray *names = [NSMutableArray array];
 NSMutableArray *birthDates = names;

 // Warning: Conversion from value of type 'NSDate *'
 // to incompatible type 'NSString *'
 [birthDates addObject: [NSDate date]];

osx.cocoa.RetainCount (ObjC)
""""""""""""""""""""""""""""
Check for leaks and improper reference count management

.. code-block:: objc

 void test() {
   NSString *s = [[NSString alloc] init]; // warn
 }

 CFStringRef test(char *bytes) {
   return CFStringCreateWithCStringNoCopy(
            0, bytes, NSNEXTSTEPStringEncoding, 0); // warn
 }


osx.cocoa.RunLoopAutoreleaseLeak
""""""""""""""""""""""""""""""""
Check for leaked memory in autorelease pools that will never be drained.

osx.cocoa.SelfInit (ObjC)
"""""""""""""""""""""""""
Check that 'self' is properly initialized inside an initializer method.

.. code-block:: objc

 @interface MyObj : NSObject {
   id x;
 }
 - (id)init;
 @end

 @implementation MyObj
 - (id)init {
   [super init];
   x = 0; // warn: instance variable used while 'self' is not
          // initialized
   return 0;
 }
 @end

 @interface MyObj : NSObject
 - (id)init;
 @end

 @implementation MyObj
 - (id)init {
   [super init];
   return self; // warn: returning uninitialized 'self'
 }
 @end

osx.cocoa.SuperDealloc (ObjC)
"""""""""""""""""""""""""""""
Warn about improper use of '[super dealloc]' in Objective-C.

.. code-block:: objc

 @interface SuperDeallocThenReleaseIvarClass : NSObject {
   NSObject *_ivar;
 }
 @end

 @implementation SuperDeallocThenReleaseIvarClass
 - (void)dealloc {
   [super dealloc];
   [_ivar release]; // warn
 }
 @end

osx.cocoa.UnusedIvars (ObjC)
""""""""""""""""""""""""""""
Warn about private ivars that are never used.

.. code-block:: objc

 @interface MyObj : NSObject {
 @private
   id x; // warn
 }
 @end

 @implementation MyObj
 @end

osx.cocoa.VariadicMethodTypes (ObjC)
""""""""""""""""""""""""""""""""""""
Check for passing non-Objective-C types to variadic collection
initialization methods that expect only Objective-C types.

.. code-block:: objc

 void test() {
   [NSSet setWithObjects:@"Foo", "Bar", nil];
     // warn: argument should be an ObjC pointer type, not 'char *'
 }

osx.coreFoundation.CFError (C)
""""""""""""""""""""""""""""""
Check usage of CFErrorRef* parameters

.. code-block:: c

 void test(CFErrorRef *error) {
   // warn: function accepting CFErrorRef* should have a
   // non-void return
 }

 int foo(CFErrorRef *error) {
   *error = 0; // warn: potential null dereference
   return 0;
 }

osx.coreFoundation.CFNumber (C)
"""""""""""""""""""""""""""""""
Check for proper uses of CFNumber APIs.

.. code-block:: c

 CFNumberRef test(unsigned char x) {
   return CFNumberCreate(0, kCFNumberSInt16Type, &x);
    // warn: 8 bit integer is used to initialize a 16 bit integer
 }

osx.coreFoundation.CFRetainRelease (C)
""""""""""""""""""""""""""""""""""""""
Check for null arguments to CFRetain/CFRelease/CFMakeCollectable.

.. code-block:: c

 void test(CFTypeRef p) {
   if (!p)
     CFRetain(p); // warn
 }

 void test(int x, CFTypeRef p) {
   if (p)
     return;

   CFRelease(p); // warn
 }

osx.coreFoundation.containers.OutOfBounds (C)
"""""""""""""""""""""""""""""""""""""""""""""
Checks for index out-of-bounds when using 'CFArray' API.

.. code-block:: c

 void test() {
   CFArrayRef A = CFArrayCreate(0, 0, 0, &kCFTypeArrayCallBacks);
   CFArrayGetValueAtIndex(A, 0); // warn
 }

osx.coreFoundation.containers.PointerSizedValues (C)
""""""""""""""""""""""""""""""""""""""""""""""""""""
Warns if 'CFArray', 'CFDictionary', 'CFSet' are created with non-pointer-size values.

.. code-block:: c

 void test() {
   int x[] = { 1 };
   CFArrayRef A = CFArrayCreate(0, (const void """""""""""""""""""""""")x, 1,
                                &kCFTypeArrayCallBacks); // warn
 }


.. _alpha-checkers:

Experimental Checkers
---------------------

*These are checkers with known issues or limitations that keep them from being on by default. They are likely to have false positives. Bug reports and especially patches are welcome.*

alpha.clone
^^^^^^^^^^^

alpha.clone.CloneChecker (C, C++, ObjC)
"""""""""""""""""""""""""""""""""""""""
Reports similar pieces of code.

.. code-block:: c

 void log();

 int max(int a, int b) { // warn
   log();
   if (a > b)
     return a;
   return b;
 }

 int maxClone(int x, int y) { // similar code here
   log();
   if (x > y)
     return x;
   return y;
 }

alpha.core.BoolAssignment (ObjC)
""""""""""""""""""""""""""""""""
Warn about assigning non-{0,1} values to boolean variables.

.. code-block:: objc

 void test() {
   BOOL b = -1; // warn
 }

alpha.core
^^^^^^^^^^

alpha.core.CallAndMessageUnInitRefArg (C,C++, ObjC)
"""""""""""""""""""""""""""""""""""""""""""""""""""
Check for logical errors for function calls and Objective-C
message expressions (e.g., uninitialized arguments, null function pointers, and pointer to undefined variables).

.. code-block:: c

 void test(void) {
   int t;
   int &p = t;
   int &s = p;
   int &q = s;
   foo(q); // warn
 }

 void test(void) {
   int x;
   foo(&x); // warn
 }

alpha.core.CastSize (C)
"""""""""""""""""""""""
Check when casting a malloc'ed type ``T``, whether the size is a multiple of the size of ``T``.

.. code-block:: c

 void test() {
   int *x = (int *) malloc(11); // warn
 }

alpha.core.CastToStruct (C, C++)
""""""""""""""""""""""""""""""""
Check for cast from non-struct pointer to struct pointer.

.. code-block:: cpp

 // C
 struct s {};

 void test(int *p) {
   struct s *ps = (struct s *) p; // warn
 }

 // C++
 class c {};

 void test(int *p) {
   c *pc = (c *) p; // warn
 }

alpha.core.Conversion (C, C++, ObjC)
""""""""""""""""""""""""""""""""""""
Loss of sign/precision in implicit conversions.

.. code-block:: c

 void test(unsigned U, signed S) {
   if (S > 10) {
     if (U < S) {
     }
   }
   if (S < -10) {
     if (U < S) { // warn (loss of sign)
     }
   }
 }

 void test() {
   long long A = 1LL << 60;
   short X = A; // warn (loss of precision)
 }

alpha.core.DynamicTypeChecker (ObjC)
""""""""""""""""""""""""""""""""""""
Check for cases where the dynamic and the static type of an object are unrelated.


.. code-block:: objc

 id date = [NSDate date];

 // Warning: Object has a dynamic type 'NSDate *' which is
 // incompatible with static type 'NSNumber *'"
 NSNumber *number = date;
 [number doubleValue];

alpha.core.FixedAddr (C)
""""""""""""""""""""""""
Check for assignment of a fixed address to a pointer.

.. code-block:: c

 void test() {
   int *p;
   p = (int *) 0x10000; // warn
 }

alpha.core.IdenticalExpr (C, C++)
"""""""""""""""""""""""""""""""""
Warn about unintended use of identical expressions in operators.

.. code-block:: cpp

 // C
 void test() {
   int a = 5;
   int b = a | 4 | a; // warn: identical expr on both sides
 }

 // C++
 bool f(void);

 void test(bool b) {
   int i = 10;
   if (f()) { // warn: true and false branches are identical
     do {
       i--;
     } while (f());
   } else {
     do {
       i--;
     } while (f());
   }
 }

alpha.core.PointerArithm (C)
""""""""""""""""""""""""""""
Check for pointer arithmetic on locations other than array elements.

.. code-block:: c

 void test() {
   int x;
   int *p;
   p = &x + 1; // warn
 }

alpha.core.PointerSub (C)
"""""""""""""""""""""""""
Check for pointer subtractions on two pointers pointing to different memory chunks.

.. code-block:: c

 void test() {
   int x, y;
   int d = &y - &x; // warn
 }

alpha.core.SizeofPtr (C)
""""""""""""""""""""""""
Warn about unintended use of ``sizeof()`` on pointer expressions.

.. code-block:: c

 struct s {};

 int test(struct s *p) {
   return sizeof(p);
     // warn: sizeof(ptr) can produce an unexpected result
 }

alpha.core.StackAddressAsyncEscape (C)
""""""""""""""""""""""""""""""""""""""
Check that addresses to stack memory do not escape the function that involves dispatch_after or dispatch_async.
This checker is a part of ``core.StackAddressEscape``, but is temporarily disabled until some false positives are fixed.

.. code-block:: c

 dispatch_block_t test_block_inside_block_async_leak() {
   int x = 123;
   void (^inner)(void) = ^void(void) {
     int y = x;
     ++y;
   };
   void (^outer)(void) = ^void(void) {
     int z = x;
     ++z;
     inner();
   };
   return outer; // warn: address of stack-allocated block is captured by a
                 //       returned block
 }

alpha.core.TestAfterDivZero (C)
"""""""""""""""""""""""""""""""
Check for division by variable that is later compared against 0.
Either the comparison is useless or there is division by zero.

.. code-block:: c

 void test(int x) {
   var = 77 / x;
   if (x == 0) { } // warn
 }

alpha.cplusplus
^^^^^^^^^^^^^^^

alpha.cplusplus.DeleteWithNonVirtualDtor (C++)
""""""""""""""""""""""""""""""""""""""""""""""
Reports destructions of polymorphic objects with a non-virtual destructor in their base class.

.. code-block:: cpp

 NonVirtual *create() {
   NonVirtual *x = new NVDerived(); // note: conversion from derived to base
                                    //       happened here
   return x;
 }

 void sink(NonVirtual *x) {
   delete x; // warn: destruction of a polymorphic object with no virtual
             //       destructor
 }

alpha.cplusplus.EnumCastOutOfRange (C++)
""""""""""""""""""""""""""""""""""""""""
Check for integer to enumeration casts that could result in undefined values.

.. code-block:: cpp

 enum TestEnum {
   A = 0
 };

 void foo() {
   TestEnum t = static_cast(-1);
       // warn: the value provided to the cast expression is not in
                the valid range of values for the enum

alpha.cplusplus.InvalidatedIterator (C++)
"""""""""""""""""""""""""""""""""""""""""
Check for use of invalidated iterators.

.. code-block:: cpp

 void bad_copy_assign_operator_list1(std::list &L1,
                                     const std::list &L2) {
   auto i0 = L1.cbegin();
   L1 = L2;
   *i0; // warn: invalidated iterator accessed
 }


alpha.cplusplus.IteratorRange (C++)
"""""""""""""""""""""""""""""""""""
Check for iterators used outside their valid ranges.

.. code-block:: cpp

 void simple_bad_end(const std::vector &v) {
   auto i = v.end();
   *i; // warn: iterator accessed outside of its range
 }

alpha.cplusplus.MismatchedIterator (C++)
""""""""""""""""""""""""""""""""""""""""
Check for use of iterators of different containers where iterators of the same container are expected.

.. code-block:: cpp

 void bad_insert3(std::vector &v1, std::vector &v2) {
   v2.insert(v1.cbegin(), v2.cbegin(), v2.cend()); // warn: container accessed
                                                   //       using foreign
                                                   //       iterator argument
   v1.insert(v1.cbegin(), v1.cbegin(), v2.cend()); // warn: iterators of
                                                   //       different containers
                                                   //       used where the same
                                                   //       container is
                                                   //       expected
   v1.insert(v1.cbegin(), v2.cbegin(), v1.cend()); // warn: iterators of
                                                   //       different containers
                                                   //       used where the same
                                                   //       container is
                                                   //       expected
 }

alpha.cplusplus.MisusedMovedObject (C++)
""""""""""""""""""""""""""""""""""""""""
Method calls on a moved-from object and copying a moved-from object will be reported.


.. code-block:: cpp

  struct A {
   void foo() {}
 };

 void f() {
   A a;
   A b = std::move(a); // note: 'a' became 'moved-from' here
   a.foo();            // warn: method call on a 'moved-from' object 'a'
 }

alpha.deadcode
^^^^^^^^^^^^^^
alpha.deadcode.UnreachableCode (C, C++)
"""""""""""""""""""""""""""""""""""""""
Check unreachable code.

.. code-block:: cpp

 // C
 int test() {
   int x = 1;
   while(x);
   return x; // warn
 }

 // C++
 void test() {
   int a = 2;

   while (a > 1)
     a--;

   if (a > 1)
     a++; // warn
 }

 // Objective-C
 void test(id x) {
   return;
   [x retain]; // warn
 }

alpha.llvm
^^^^^^^^^^

alpha.llvm.Conventions
""""""""""""""""""""""

Check code for LLVM codebase conventions:

* A StringRef should not be bound to a temporary std::string whose lifetime is shorter than the StringRef's.
* Clang AST nodes should not have fields that can allocate memory.


alpha.osx
^^^^^^^^^

alpha.osx.cocoa.DirectIvarAssignment (ObjC)
"""""""""""""""""""""""""""""""""""""""""""
Check for direct assignments to instance variables.


.. code-block:: objc

 @interface MyClass : NSObject {}
 @property (readonly) id A;
 - (void) foo;
 @end

 @implementation MyClass
 - (void) foo {
   _A = 0; // warn
 }
 @end

alpha.osx.cocoa.DirectIvarAssignmentForAnnotatedFunctions (ObjC)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Check for direct assignments to instance variables in
the methods annotated with ``objc_no_direct_instance_variable_assignment``.

.. code-block:: objc

 @interface MyClass : NSObject {}
 @property (readonly) id A;
 - (void) fAnnotated __attribute__((
     annotate("objc_no_direct_instance_variable_assignment")));
 - (void) fNotAnnotated;
 @end

 @implementation MyClass
 - (void) fAnnotated {
   _A = 0; // warn
 }
 - (void) fNotAnnotated {
   _A = 0; // no warn
 }
 @end


alpha.osx.cocoa.InstanceVariableInvalidation (ObjC)
"""""""""""""""""""""""""""""""""""""""""""""""""""
Check that the invalidatable instance variables are
invalidated in the methods annotated with objc_instance_variable_invalidator.

.. code-block:: objc

 @protocol Invalidation <NSObject>
 - (void) invalidate
   __attribute__((annotate("objc_instance_variable_invalidator")));
 @end

 @interface InvalidationImpObj : NSObject <Invalidation>
 @end

 @interface SubclassInvalidationImpObj : InvalidationImpObj {
   InvalidationImpObj *var;
 }
 - (void)invalidate;
 @end

 @implementation SubclassInvalidationImpObj
 - (void) invalidate {}
 @end
 // warn: var needs to be invalidated or set to nil

alpha.osx.cocoa.MissingInvalidationMethod (ObjC)
""""""""""""""""""""""""""""""""""""""""""""""""
Check that the invalidation methods are present in classes that contain invalidatable instance variables.

.. code-block:: objc

 @protocol Invalidation <NSObject>
 - (void)invalidate
   __attribute__((annotate("objc_instance_variable_invalidator")));
 @end

 @interface NeedInvalidation : NSObject <Invalidation>
 @end

 @interface MissingInvalidationMethodDecl : NSObject {
   NeedInvalidation *Var; // warn
 }
 @end

 @implementation MissingInvalidationMethodDecl
 @end

alpha.osx.cocoa.localizability.PluralMisuseChecker (ObjC)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Warns against using one vs. many plural pattern in code when generating localized strings.

.. code-block:: objc

 NSString *reminderText =
   NSLocalizedString(@"None", @"Indicates no reminders");
 if (reminderCount == 1) {
   // Warning: Plural cases are not supported across all languages.
   // Use a .stringsdict file instead
   reminderText =
     NSLocalizedString(@"1 Reminder", @"Indicates single reminder");
 } else if (reminderCount >= 2) {
   // Warning: Plural cases are not supported across all languages.
   // Use a .stringsdict file instead
   reminderText =
     [NSString stringWithFormat:
       NSLocalizedString(@"%@ Reminders", @"Indicates multiple reminders"),
         reminderCount];
 }

alpha.security
^^^^^^^^^^^^^^
alpha.security.ArrayBound (C)
"""""""""""""""""""""""""""""
Warn about buffer overflows (older checker).

.. code-block:: c

 void test() {
   char *s = "";
   char c = s[1]; // warn
 }

 struct seven_words {
   int c[7];
 };

 void test() {
   struct seven_words a, *p;
   p = &a;
   p[0] = a;
   p[1] = a;
   p[2] = a; // warn
 }

 // note: requires unix.Malloc or
 // alpha.unix.MallocWithAnnotations checks enabled.
 void test() {
   int *p = malloc(12);
   p[3] = 4; // warn
 }

 void test() {
   char a[2];
   int *b = (int*)a;
   b[1] = 3; // warn
 }

alpha.security.ArrayBoundV2 (C)
"""""""""""""""""""""""""""""""
Warn about buffer overflows (newer checker).

.. code-block:: c

 void test() {
   char *s = "";
   char c = s[1]; // warn
 }

 void test() {
   int buf[100];
   int *p = buf;
   p = p + 99;
   p[1] = 1; // warn
 }

 // note: compiler has internal check for this.
 // Use -Wno-array-bounds to suppress compiler warning.
 void test() {
   int buf[100][100];
   buf[0][-1] = 1; // warn
 }

 // note: requires alpha.security.taint check turned on.
 void test() {
   char s[] = "abc";
   int x = getchar();
   char c = s[x]; // warn: index is tainted
 }

alpha.security.MallocOverflow (C)
"""""""""""""""""""""""""""""""""
Check for overflows in the arguments to malloc().

.. code-block:: c

 void test(int n) {
   void *p = malloc(n * sizeof(int)); // warn
 }

alpha.security.MmapWriteExec (C)
""""""""""""""""""""""""""""""""
Warn on mmap() calls that are both writable and executable.

.. code-block:: c

 void test(int n) {
   void *c = mmap(NULL, 32, PROT_READ | PROT_WRITE | PROT_EXEC,
                  MAP_PRIVATE | MAP_ANON, -1, 0);
   // warn: Both PROT_WRITE and PROT_EXEC flags are set. This can lead to
   //       exploitable memory regions, which could be overwritten with malicious
   //       code
 }

alpha.security.ReturnPtrRange (C)
"""""""""""""""""""""""""""""""""
Check for an out-of-bound pointer being returned to callers.

.. code-block:: c

 static int A[10];

 int *test() {
   int *p = A + 10;
   return p; // warn
 }

 int test(void) {
   int x;
   return x; // warn: undefined or garbage returned
 }

alpha.security.taint.TaintPropagation (C, C++)
""""""""""""""""""""""""""""""""""""""""""""""
Generate taint information used by other checkers.
A data is tainted when it comes from an unreliable source.

.. code-block:: c

 void test() {
   char x = getchar(); // 'x' marked as tainted
   system(&x); // warn: untrusted data is passed to a system call
 }

 // note: compiler internally checks if the second param to
 // sprintf is a string literal or not.
 // Use -Wno-format-security to suppress compiler warning.
 void test() {
   char s[10], buf[10];
   fscanf(stdin, "%s", s); // 's' marked as tainted

   sprintf(buf, s); // warn: untrusted data as a format string
 }

 void test() {
   size_t ts;
   scanf("%zd", &ts); // 'ts' marked as tainted
   int *p = (int *)malloc(ts * sizeof(int));
     // warn: untrusted data as buffer size
 }

alpha.unix
^^^^^^^^^^^

alpha.unix.BlockInCriticalSection (C)
"""""""""""""""""""""""""""""""""""""
Check for calls to blocking functions inside a critical section.
Applies to: ``lock, unlock, sleep, getc, fgets, read, recv, pthread_mutex_lock,``
`` pthread_mutex_unlock, mtx_lock, mtx_timedlock, mtx_trylock, mtx_unlock, lock_guard, unique_lock``

.. code-block:: c

 void test() {
   std::mutex m;
   m.lock();
   sleep(3); // warn: a blocking function sleep is called inside a critical
             //       section
   m.unlock();
 }

alpha.unix.Chroot (C)
"""""""""""""""""""""
Check improper use of chroot.

.. code-block:: c

 void f();

 void test() {
   chroot("/usr/local");
   f(); // warn: no call of chdir("/") immediately after chroot
 }

alpha.unix.PthreadLock (C)
""""""""""""""""""""""""""
Simple lock -> unlock checker.
Applies to: ``pthread_mutex_lock, pthread_rwlock_rdlock, pthread_rwlock_wrlock, lck_mtx_lock, lck_rw_lock_exclusive``
``lck_rw_lock_shared, pthread_mutex_trylock, pthread_rwlock_tryrdlock, pthread_rwlock_tryrwlock, lck_mtx_try_lock,
lck_rw_try_lock_exclusive, lck_rw_try_lock_shared, pthread_mutex_unlock, pthread_rwlock_unlock, lck_mtx_unlock, lck_rw_done``.


.. code-block:: c

 pthread_mutex_t mtx;

 void test() {
   pthread_mutex_lock(&mtx);
   pthread_mutex_lock(&mtx);
     // warn: this lock has already been acquired
 }

 lck_mtx_t lck1, lck2;

 void test() {
   lck_mtx_lock(&lck1);
   lck_mtx_lock(&lck2);
   lck_mtx_unlock(&lck1);
     // warn: this was not the most recently acquired lock
 }

 lck_mtx_t lck1, lck2;

 void test() {
   if (lck_mtx_try_lock(&lck1) == 0)
     return;

   lck_mtx_lock(&lck2);
   lck_mtx_unlock(&lck1);
     // warn: this was not the most recently acquired lock
 }

alpha.unix.SimpleStream (C)
"""""""""""""""""""""""""""
Check for misuses of stream APIs. Check for misuses of stream APIs: ``fopen, fclose``
(demo checker, the subject of the demo (`Slides <http://llvm.org/devmtg/2012-11/Zaks-Rose-Checker24Hours.pdf>`_ ,
`Video <https://youtu.be/kdxlsP5QVPw>`_) by Anna Zaks and Jordan Rose presented at the
`2012 LLVM Developers' Meeting <http://llvm.org/devmtg/2012-11/>`_).

.. code-block:: c

 void test() {
   FILE *F = fopen("myfile.txt", "w");
 } // warn: opened file is never closed

 void test() {
   FILE *F = fopen("myfile.txt", "w");

   if (F)
     fclose(F);

   fclose(F); // warn: closing a previously closed file stream
 }

alpha.unix.Stream (C)
"""""""""""""""""""""
Check stream handling functions: ``fopen, tmpfile, fclose, fread, fwrite, fseek, ftell, rewind, fgetpos,``
``fsetpos, clearerr, feof, ferror, fileno``.

.. code-block:: c

 void test() {
   FILE *p = fopen("foo", "r");
 } // warn: opened file is never closed

 void test() {
   FILE *p = fopen("foo", "r");
   fseek(p, 1, SEEK_SET); // warn: stream pointer might be NULL
   fclose(p);
 }

 void test() {
   FILE *p = fopen("foo", "r");

   if (p)
     fseek(p, 1, 3);
      // warn: third arg should be SEEK_SET, SEEK_END, or SEEK_CUR

   fclose(p);
 }

 void test() {
   FILE *p = fopen("foo", "r");
   fclose(p);
   fclose(p); // warn: already closed
 }

 void test() {
   FILE *p = tmpfile();
   ftell(p); // warn: stream pointer might be NULL
   fclose(p);
 }


alpha.unix.cstring.BufferOverlap (C)
""""""""""""""""""""""""""""""""""""
Checks for overlap in two buffer arguments. Applies to:  ``memcpy, mempcpy``.

.. code-block:: c

 void test() {
   int a[4] = {0};
   memcpy(a + 2, a + 1, 8); // warn
 }

alpha.unix.cstring.NotNullTerminated (C)
""""""""""""""""""""""""""""""""""""""""
Check for arguments which are not null-terminated strings; applies to: ``strlen, strnlen, strcpy, strncpy, strcat, strncat``.

.. code-block:: c

 void test() {
   int y = strlen((char *)&test); // warn
 }

alpha.unix.cstring.OutOfBounds (C)
""""""""""""""""""""""""""""""""""
Check for out-of-bounds access in string functions; applies to:`` strncopy, strncat``.


.. code-block:: c

 void test() {
   int y = strlen((char *)&test); // warn
 }

alpha.nondeterminism.PointerSorting (C++)
"""""""""""""""""""""""""""""""""""""""""
Check for non-determinism caused by sorting of pointers.

.. code-block:: c

 void test() {
  int a = 1, b = 2;
  std::vector<int *> V = {&a, &b};
  std::sort(V.begin(), V.end()); // warn
 }


Debug Checkers
---------------

.. _debug-checkers:


debug
^^^^^

Checkers used for debugging the analyzer.
:doc:`developer-docs/DebugChecks` page contains a detailed description.

debug.AnalysisOrder
"""""""""""""""""""
Print callbacks that are called during analysis in order.

debug.ConfigDumper
""""""""""""""""""
Dump config table.

debug.DumpCFG Display
"""""""""""""""""""""
Control-Flow Graphs.

debug.DumpCallGraph
"""""""""""""""""""
Display Call Graph.

debug.DumpCalls
"""""""""""""""
Print calls as they are traversed by the engine.

debug.DumpDominators
""""""""""""""""""""
Print the dominance tree for a given CFG.

debug.DumpLiveVars
""""""""""""""""""
Print results of live variable analysis.

debug.DumpTraversal
"""""""""""""""""""
Print branch conditions as they are traversed by the engine.

debug.ExprInspection
""""""""""""""""""""
Check the analyzer's understanding of expressions.

debug.Stats
"""""""""""
Emit warnings with analyzer statistics.

debug.TaintTest
"""""""""""""""
Mark tainted symbols as such.

debug.ViewCFG
"""""""""""""
View Control-Flow Graphs using GraphViz.

debug.ViewCallGraph
"""""""""""""""""""
View Call Graph using GraphViz.

debug.ViewExplodedGraph
"""""""""""""""""""""""
View Exploded Graphs using GraphViz.

