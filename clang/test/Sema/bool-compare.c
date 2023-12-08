// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-logical-not-parentheses


void f(int x, int y, int z) {
  int a,b;


  if ((a > 2) > 1) {} // expected-warning {{comparison of constant 1 with boolean expression is always false}}

  if (a > b)      {} // no warning
  if (a < b)      {} // no warning
  if (a >= b)     {} // no warning
  if (a <= b)     {} // no warning
  if (a == b)     {} // no warning
  if (a != b)     {} // no warning

  if (a > 0) {} // no warning
  if (a > 1) {} // no warning
  if (a > 2) {} // no warning

  if (a >= 0) {} // no warning
  if (a >= 1) {} // no warning
  if (a >= 2) {} // no warning
  if (a >= -1) {} // no warning

  if (a <= 0) {} // no warning
  if (a <= 1) {} // no warning
  if (a <= 2) {} // no warning
  if (a <= -1) {} // no warning


  if (!a > 0) {}  // no warning
  if (!a > 1)     {} // expected-warning {{comparison of constant 1 with boolean expression is always false}}
  if (!a > 2)     {} // expected-warning {{comparison of constant 2 with boolean expression is always false}}
  if (!a > y)     {} // no warning
  if (!a > b)     {} // no warning
  if (!a > -1)    {} // expected-warning {{comparison of constant -1 with boolean expression is always true}}

  if (!a < 0)     {} // expected-warning {{comparison of constant 0 with boolean expression is always false}}
  if (!a < 1)     {} // no warning
  if (!a < 2)     {} // expected-warning {{comparison of constant 2 with boolean expression is always true}}
  if (!a < y)     {} // no warning
  if (!a < b)     {} // no warning
  if (!a < -1)    {} // expected-warning {{comparison of constant -1 with boolean expression is always false}}

  if (!a >= 0)    {} // expected-warning {{comparison of constant 0 with boolean expression is always true}}
  if (!a >= 1)    {} // no warning
  if (!a >= 2)    {} // expected-warning {{comparison of constant 2 with boolean expression is always false}}
  if (!a >= y)    {} // no warning
  if (!a >= b)    {} // no warning
  if (!a >= -1)   {} // expected-warning {{comparison of constant -1 with boolean expression is always true}}

  if (!a <= 0)    {} // no warning
  if (!a <= 1)    {} // expected-warning {{comparison of constant 1 with boolean expression is always true}}
  if (!a <= 2)    {} // expected-warning {{comparison of constant 2 with boolean expression is always true}}
  if (!a <= y)    {} // no warning
  if (!a <= b)    {} // no warning
  if (!a <= -1)   {} // expected-warning {{comparison of constant -1 with boolean expression is always false}}

  if ((a||b) > 0) {} // no warning
  if ((a||b) > 1) {} // expected-warning {{comparison of constant 1 with boolean expression is always false}}
  if ((a||b) > 4) {} // expected-warning {{comparison of constant 4 with boolean expression is always false}}
  if ((a||b) > -1) {}// expected-warning {{comparison of constant -1 with boolean expression is always true}}

  if ((a&&b) > 0) {} // no warning
  if ((a&&b) > 1) {} // expected-warning {{comparison of constant 1 with boolean expression is always false}}
  if ((a&&b) > 4) {} // expected-warning {{comparison of constant 4 with boolean expression is always false}}

  if ((a<y) > 0)  {} // no warning
  if ((a<y) > 1)  {} // expected-warning {{comparison of constant 1 with boolean expression is always false}}
  if ((a<y) > 4)  {} // expected-warning {{comparison of constant 4 with boolean expression is always false}}
  if ((a<y) > z)  {} // no warning
  if ((a<y) > -1) {} // expected-warning {{comparison of constant -1 with boolean expression is always true}}

  if ((a<y) == 0) {} // no warning
  if ((a<y) == 1) {} // no warning
  if ((a<y) == 2) {} // expected-warning {{comparison of constant 2 with boolean expression is always false}}
  if ((a<y) == z) {} // no warning
  if ((a<y) == -1) {}// expected-warning {{comparison of constant -1 with boolean expression is always false}}

  if ((a<y) != 0) {} // no warning
  if ((a<y) != 1) {} // no warning
  if ((a<y) != 2) {} // expected-warning {{comparison of constant 2 with boolean expression is always true}}
  if ((a<y) != z) {} // no warning
  if ((a<y) != -1) {}// expected-warning {{comparison of constant -1 with boolean expression is always true}}

  if ((a<y) == z) {} // no warning
  if (a>y<z)      {} // no warning
  if ((a<y) > z)  {} // no warning
  if((a<y)>(z<y)) {} // no warning
  if((a<y)==(z<y)){} // no warning
  if((a<y)!=(z<y)){} // no warning
  if((z==x)<(y==z)){}// no warning
  if((a<y)!=((z==x)<(y==z))){} //no warning


  if (0 > !a)     {} // expected-warning {{comparison of constant 0 with boolean expression is always false}}
  if (1 > !a)     {} // no warning
  if (2 > !a)     {} // expected-warning {{comparison of constant 2 with boolean expression is always true}}
  if (y > !a)     {} // no warning
  if (-1 > !a)    {} // expected-warning {{comparison of constant -1 with boolean expression is always false}}

  if (0 < !a)     {} // no warning
  if (1 < !a)     {} // expected-warning {{comparison of constant 1 with boolean expression is always false}}
  if (2 < !a)     {} // expected-warning {{comparison of constant 2 with boolean expression is always false}}
  if (y < !a)     {} // no warning
  if (-1 < !a)    {} // expected-warning {{comparison of constant -1 with boolean expression is always true}}

  if (0 >= !a)    {} // no warning
  if (1 >= !a)    {} // expected-warning {{comparison of constant 1 with boolean expression is always true}}
  if (2 >= !a)    {} // expected-warning {{comparison of constant 2 with boolean expression is always true}}
  if (y >= !a)    {} // no warning
  if (-1 >= !a)   {} // expected-warning {{comparison of constant -1 with boolean expression is always false}}

  if (0 <= !a)    {} // expected-warning {{comparison of constant 0 with boolean expression is always true}}
  if (1 <= !a)    {} // no warning
  if (2 <= !a)    {} // expected-warning {{comparison of constant 2 with boolean expression is always false}}
  if (y <= !a)    {} // no warning
  if (-1 <= !a)   {} // expected-warning {{comparison of constant -1 with boolean expression is always true}}

  if (0 > (a||b)) {} // expected-warning {{comparison of constant 0 with boolean expression is always false}}
  if (1 > (a||b)) {} // no warning
  if (4 > (a||b)) {} // expected-warning {{comparison of constant 4 with boolean expression is always true}}

  if (0 > (a&&b)) {} // expected-warning {{comparison of constant 0 with boolean expression is always false}}
  if (1 > (a&&b)) {} // no warning
  if (4 > (a&&b)) {} // expected-warning {{comparison of constant 4 with boolean expression is always true}}

  if (0 > (a<y))  {} // expected-warning {{comparison of constant 0 with boolean expression is always false}}
  if (1 > (a<y))  {} // no warning
  if (4 > (a<y))  {} // expected-warning {{comparison of constant 4 with boolean expression is always true}}
  if (z > (a<y))  {} // no warning
  if (-1 > (a<y)) {} // expected-warning {{comparison of constant -1 with boolean expression is always false}}

  if (0 == (a<y)) {} // no warning
  if (1 == (a<y)) {} // no warning
  if (2 == (a<y)) {} // expected-warning {{comparison of constant 2 with boolean expression is always false}}
  if (z == (a<y)) {} // no warning
  if (-1 == (a<y)){} // expected-warning {{comparison of constant -1 with boolean expression is always false}}

  if (0 !=(a<y))  {} // no warning
  if (1 !=(a<y))  {} // no warning
  if (2 !=(a<y))  {} // expected-warning {{comparison of constant 2 with boolean expression is always true}}
  if (z !=(a<y))  {} // no warning
  if (-1 !=(a<y)) {} // expected-warning {{comparison of constant -1 with boolean expression is always true}}

  if (z ==(a<y))  {}    // no warning
  if (z<a>y)      {}        // no warning
  if (z > (a<y))  {}    // no warning
  if((z<y)>(a<y)) {}   // no warning
  if((z<y)==(a<y)){}  // no warning
  if((z<y)!=(a<y)){}  // no warning
  if((y==z)<(z==x)){}  // no warning
  if(((z==x)<(y==z))!=(a<y)){}  // no warning

  if(((z==x)<(-1==z))!=(a<y)){} // no warning
  if(((z==x)<(z==-1))!=(a<y)){} // no warning
  if(((z==x)<-1)!=(a<y)){} // expected-warning {{comparison of constant -1 with boolean expression is always false}}
  if(((z==x)< 2)!=(a<y)){} // expected-warning {{comparison of constant 2 with boolean expression is always true}}
  if(((z==x)<(z>2))!=(a<y)){} // no warning

}
