// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface A
 -(int) x;
@property (readonly) int x;
@property int ok;
@end

@interface B
 -(void) setOk:(int)arg;
 -(int) x;
 -(int) ok;
@end

void f0(A *a, B* b) {
  a.x = 10;  // expected-error {{assignment to readonly property}}
  a.ok = 20;
  b.x = 10;  // expected-error {{no setter method 'setX:' for assignment to property}}
  b.ok = 20;
}

typedef struct {
  int i1, i2;
} NSRect;

NSRect NSMakeRect(void);

@interface NSWindow 
{
    NSRect _frame;
}
- (NSRect)frame;
@end

@interface NSWindow (Category)
-(void)methodToMakeClangCrash;
@end

@implementation NSWindow (Category)
-(void)methodToMakeClangCrash
{
 self.frame =  NSMakeRect(); // expected-error {{no setter method 'setFrame:' for assignment to property}}
}
@end
