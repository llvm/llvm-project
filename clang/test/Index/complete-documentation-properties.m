// Note: the run lines follow their respective tests, since line/column numbers
// matter in this test.
// This test is for when property accessors do not have their own code 
// completion comments. Use those in their properties in this case. 
// rdar://12791315

@interface AppDelegate
/**
  \brief This is ReadonlyProperty
*/
@property (readonly, getter = ReadonlyGetter) id MyProperty;

/**
  \brief This is GeneralProperty
*/
@property int GeneralProperty;

/**
  \brief This is PropertyInPrimaryClass
*/
@property (copy, nonatomic) id PropertyInPrimaryClass;

- (void) setThisRecord : (id)arg;
- (id) Record;
@end


@interface AppDelegate()
- (id) GetterInClassExtension;
/**
  \brief This is Record
*/
@property (copy, setter = setThisRecord:) id Record;
@end

@interface AppDelegate()
/**
  \brief This is PropertyInClassExtension
*/
@property (copy, getter = GetterInClassExtension) id PropertyInClassExtension;

- (id) PropertyInPrimaryClass;
@end
  
@implementation AppDelegate
- (id) PropertyInPrimaryClass { 
  id p = [self ReadonlyGetter];
  p = [self GetterInClassExtension];
  p = [self PropertyInPrimaryClass];
  p = [self Record];
  [self setThisRecord : (id)0 ];
  p = self.GetterInClassExtension;
  return 0; 
}
@end
// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:47:16 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: {TypedText ReadonlyGetter}{{.*}}(brief comment: This is ReadonlyProperty)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:48:13 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: {TypedText GetterInClassExtension}{{.*}}(brief comment: This is PropertyInClassExtension) 

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:49:13 %s | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: {TypedText PropertyInPrimaryClass}{{.*}}(brief comment: This is PropertyInPrimaryClass)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:50:13 %s | FileCheck -check-prefix=CHECK-CC4 %s
// CHECK-CC4: {TypedText Record}{{.*}}(brief comment: This is Record)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:51:9 %s | FileCheck -check-prefix=CHECK-CC5 %s
// CHECK-CC5: {TypedText setThisRecord:}{Placeholder (id)}{{.*}}(brief comment: This is Record)

// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:52:12 %s | FileCheck -check-prefix=CHECK-CC6 %s
// CHECK-CC6: {TypedText GetterInClassExtension}{{.*}}(brief comment: This is PropertyInClassExtension) 

@interface AnotherAppDelegate
/**
  \brief This is ReadonlyProperty
*/
@property (getter = ReadonlyGetter) int MyProperty;
/**
  \brief This is getter = ReadonlyGetter
*/
- (int) ReadonlyGetter;
@end

@implementation AnotherAppDelegate
- (int) PropertyInPrimaryClass { 
self.ReadonlyGetter;
}
@end
// RUN: env CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 c-index-test -code-completion-at=%s:87:6 %s | FileCheck -check-prefix=CHECK-CC7 %s
// CHECK-CC7: {TypedText ReadonlyGetter}{{.*}}(brief comment: This is getter = ReadonlyGetter) 

