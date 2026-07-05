// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-unreachable-code
// expected-no-diagnostics

@interface NSObject 
+ alloc;
- init;
@end

@protocol Test
  @property int required;

@optional
  @property int optional;
  @property int optional1;
  @property int optional_preexisting_setter_getter;
  @property (setter = setOptional_preexisting_setter_getter: ,
	     getter = optional_preexisting_setter_getter) int optional_with_setter_getter_attr;
@required
  @property int required1;
@optional
  @property int optional_to_be_defined;
  @property (readonly, getter = optional_preexisting_setter_getter) int optional_getter_attr;
@end

@interface Test : NSObject <Test> {
  int ivar;
  int ivar1;
  int ivar2;
}
@property int required;
@property int optional_to_be_defined;
- (int) optional_preexisting_setter_getter;
- (void) setOptional_preexisting_setter_getter:(int)value;
@end

@implementation Test
@synthesize required = ivar;
@synthesize required1 = ivar1;
@synthesize optional_to_be_defined = ivar2;
- (int) optional_preexisting_setter_getter { return ivar; }
- (void) setOptional_preexisting_setter_getter:(int)value
	   {
		ivar = value;
	   }
- (void) setOptional_getter_attr:(int)value { ivar = value; }
@end

void abort(void);
int main (void)
{
	Test *x = [[Test alloc] init];
	/* 1. Test of a required property */
	x.required1 = 100;
  	if (x.required1 != 100)
	  abort ();

	/* 2. Test of a synthesize optional property */
  	x.optional_to_be_defined = 123;
	if (x.optional_to_be_defined != 123)
	  abort ();

	/* 3. Test of optional property with pre-sxisting defined setter/getter */
	x.optional_preexisting_setter_getter = 200;
	if (x.optional_preexisting_setter_getter != 200)
	  abort ();

	/* 4. Test of optional property with setter/getter attribute */
	if (x.optional_with_setter_getter_attr != 200)
	  abort ();
	return 0;

	/* 5. Test of optional property with getter attribute and default setter method. */
	x.optional_getter_attr = 1000;
        if (x.optional_getter_attr != 1000)
	  abort ();

	return 0;
}

