// RUN: %clang_cc1 %s -fsyntax-only -fblocks -verify -Wno-unreachable-code

// rdar://6597252
Class test1(Class X) {
  return 1 ? X : X;
}


// rdar://6079877
void test2(void) {
  id str = @"foo" 
          "bar\0"    // no-warning
          @"baz"  " blarg";
  id str2 = @"foo" 
            "bar"
           @"baz"
           " b\0larg";  // no-warning

  
  if (@encode(int) == "foo") { }  // expected-warning {{result of comparison against @encode is unspecified}}
}

#define MAX(A,B) ({ __typeof__(A) __a = (A); __typeof__(B) __b = (B); __a < __b ? __b : __a; })
void (^foo)(int, int) = ^(int x, int y) { int z = MAX(x, y); };



// rdar://8445858
@class Object;
static Object *g;
void test3(Object *o) {
  // this is ok.
  __sync_bool_compare_and_swap(&g, 0, o);
}

@class Incomplete_ObjC_class; // expected-note{{forward declaration of class here}}
struct Incomplete_struct; // expected-note {{forward declaration}}

void test_encode(void) {
  (void)@encode(Incomplete_ObjC_class); // expected-error {{incomplete type}}
  (void)@encode(struct Incomplete_struct); // expected-error {{incomplete type}}
  (void)@encode(Incomplete_ObjC_class*);
  (void)@encode(id);
}
