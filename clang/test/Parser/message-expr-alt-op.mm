// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface WeirdInterface
-(void)allOfThem:(int)a
             and:(int)b
          and_eq:(int)c
          bitand:(int)d
           bitor:(int)e
           compl:(int)f
             not:(int)g
          not_eq:(int)h
              or:(int)i
           or_eq:(int)j
             xor:(int)k
          xor_eq:(int)l;

-(void)justAnd:(int)x and:(int)y;
-(void)and;
-(void)and:(int)x;
@end

void call_it(WeirdInterface *x) {
  [x allOfThem:0
           and:0
        and_eq:0
        bitand:0
         bitor:0
         compl:0
           not:0
        not_eq:0
            or:0
         or_eq:0
           xor:0
        xor_eq:0];

  [x and];
  [x and:0];
  [x &&:0]; // expected-error{{expected expression}};
  [x justAnd:0 and:1];
  [x and: 0 ? : 1];
}
