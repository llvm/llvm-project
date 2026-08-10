// RUN: %clang_cc1 -fsyntax-only %s -verify -fblocks
// FIXME: should compile
// Test for blocks with explicit return type specified.

typedef float * PF;
float gf;

@interface NSView
  - (id) some_method_that_returns_id;
@end

NSView *some_object;

void some_func (NSView * (^) (id));

typedef struct dispatch_item_s *dispatch_item_t;
typedef void (^completion_block_t)(void);

typedef double (^myblock)(int);
double test(myblock I);

int main(void) {
  __block int x = 1;
  __block int y = 2;

  (void)^void *{ return 0; };

  (void)^float(float y){ return y; };

  (void)^double (float y, double d) {
    if (y)
      return d;
    else
      return y;
  };

  const char * (^chb) (int flag, const char *arg, char *arg1) = ^ const char * (int flag, const char *arg, char *arg1) {
    if (flag)
      return 0;
    if (flag == 1)
      return arg;
    else if (flag == 2)
      return "";
    return arg1; 
  };

  (void)^PF { return &gf; };

  some_func(^ NSView * (id whatever) { return [some_object some_method_that_returns_id]; });

  double res = test(^(int z){x = y+z; return (double)z; });
}

void func(void) {
  completion_block_t X;

  completion_block_t (^blockx)(dispatch_item_t) = ^completion_block_t (dispatch_item_t item) {
    return X;
  };

  completion_block_t (^blocky)(dispatch_item_t) = ^(dispatch_item_t item) {
    return X;
  };

  blockx = blocky;
}


// intent: block taking int returning block that takes char,int and returns int
int (^(^block)(double x))(char, short);

void foo(void) {
   int one = 1;
   block = ^(double x){ return ^(char c, short y) { return one + c + y; };};  // expected-error {{returning block that lives on the local stack}}
   // or:
   block = ^(double x){ return ^(char c, short y) { return one + (int)c + y; };};  // expected-error {{returning block that lives on the local stack}}
}
