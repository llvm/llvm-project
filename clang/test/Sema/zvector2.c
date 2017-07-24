// RUN: %clang_cc1 -triple s390x-linux-gnu -fzvector -target-cpu z14 \
// RUN:  -fno-lax-vector-conversions -W -Wall -Wconversion \
// RUN:  -Werror -fsyntax-only -verify %s

vector signed char sc, sc2;
vector unsigned char uc, uc2;
vector bool char bc, bc2;

vector signed short ss, ss2;
vector unsigned short us, us2;
vector bool short bs, bs2;

vector signed int si, si2;
vector unsigned int ui, ui2;
vector bool int bi, bi2;

vector signed long long sl, sl2;
vector unsigned long long ul, ul2;
vector bool long long bl, bl2;

vector double fd, fd2;

vector float ff, ff2;

void foo(void)
{
  // -------------------------------------------------------------------------
  // Test assignment.
  // -------------------------------------------------------------------------

  ff = ff2;

  sc = ff2; // expected-error {{incompatible type}}
  ff = sc2; // expected-error {{incompatible type}}

  uc = ff2; // expected-error {{incompatible type}}
  ff = uc2; // expected-error {{incompatible type}}

  bc = ff2; // expected-error {{incompatible type}}
  ff = bc2; // expected-error {{incompatible type}}

  fd = ff2; // expected-error {{incompatible type}}
  ff = fd2; // expected-error {{incompatible type}}

  // -------------------------------------------------------------------------
  // Test casts to same element width.
  // -------------------------------------------------------------------------

  ui = (vector unsigned int)ff2;
  ff = (vector float)si2;

  // -------------------------------------------------------------------------
  // Test casts to different element width.
  // -------------------------------------------------------------------------

  uc = (vector unsigned char)ff2;
  us = (vector unsigned short)ff2;
  ul = (vector unsigned long long)ff2;

  ff = (vector float)sc2;
  ff = (vector float)ss2;
  ff = (vector float)sl2;

  // -------------------------------------------------------------------------
  // Test unary operators.
  // -------------------------------------------------------------------------

  ++ff2;
  ff++;

  --ff2;
  ff--;

  ff = +ff2;

  ff = -ff2;

  ff = ~ff2; // expected-error {{invalid argument}}

  // -------------------------------------------------------------------------
  // Test binary arithmetic operators.
  // -------------------------------------------------------------------------

  ff = ff + ff2;
  ff = ff + ui2; // expected-error {{cannot convert}}
  ff = si + ff2; // expected-error {{cannot convert}}
  ff = fd + ff2; // expected-error {{cannot convert}}
  ff += ff2;
  ff += fd2; // expected-error {{cannot convert}}
  sc += ff2; // expected-error {{cannot convert}}

  ff = ff - ff2;
  ff = ff - ui2; // expected-error {{cannot convert}}
  ff = si - ff2; // expected-error {{cannot convert}}
  ff = fd - ff2; // expected-error {{cannot convert}}
  ff -= ff2;
  ff -= fd2; // expected-error {{cannot convert}}
  sc -= ff2; // expected-error {{cannot convert}}

  ff = ff * ff2;
  ff = ff * ui2; // expected-error {{cannot convert}}
  ff = si * ff2; // expected-error {{cannot convert}}
  ff = fd * ff2; // expected-error {{cannot convert}}
  ff *= ff2;
  ff *= fd2; // expected-error {{cannot convert}}
  sc *= ff2; // expected-error {{cannot convert}}

  ff = ff / ff2;
  ff = ff / ui2; // expected-error {{cannot convert}}
  ff = si / ff2; // expected-error {{cannot convert}}
  ff = fd / ff2; // expected-error {{cannot convert}}
  ff /= ff2;
  ff /= fd2; // expected-error {{cannot convert}}
  sc /= ff2; // expected-error {{cannot convert}}

  ff = ff % ff2; // expected-error {{invalid operands}}
  ff = ff % ui2; // expected-error {{invalid operands}}
  ff = si % ff2; // expected-error {{invalid operands}}
  ff = fd % ff2; // expected-error {{invalid operands}}
  ff %= ff2; // expected-error {{invalid operands}}
  ff %= fd2; // expected-error {{invalid operands}}
  sc %= ff2; // expected-error {{invalid operands}}

  // -------------------------------------------------------------------------
  // Test bitwise binary operators.
  // -------------------------------------------------------------------------

  ff = ff & ff2; // expected-error {{invalid operands}}
  ff = bi & ff2; // expected-error {{invalid operands}}
  ff = fd & ff2; // expected-error {{invalid operands}}
  ff = ff & bi2; // expected-error {{invalid operands}}
  ff = ff & si2; // expected-error {{invalid operands}}
  ff = ff & ui2; // expected-error {{invalid operands}}
  sc &= ff2; // expected-error {{invalid operands}}
  ff &= bc2; // expected-error {{invalid operands}}
  ff &= fd2; // expected-error {{invalid operands}}

  ff = ff | ff2; // expected-error {{invalid operands}}
  ff = bi | ff2; // expected-error {{invalid operands}}
  ff = fd | ff2; // expected-error {{invalid operands}}
  ff = ff | bi2; // expected-error {{invalid operands}}
  ff = ff | si2; // expected-error {{invalid operands}}
  ff = ff | ui2; // expected-error {{invalid operands}}
  sc |= ff2; // expected-error {{invalid operands}}
  ff |= bc2; // expected-error {{invalid operands}}
  ff |= fd2; // expected-error {{invalid operands}}

  ff = ff ^ ff2; // expected-error {{invalid operands}}
  ff = bi ^ ff2; // expected-error {{invalid operands}}
  ff = fd ^ ff2; // expected-error {{invalid operands}}
  ff = ff ^ bi2; // expected-error {{invalid operands}}
  ff = ff ^ si2; // expected-error {{invalid operands}}
  ff = ff ^ ui2; // expected-error {{invalid operands}}
  sc ^= ff2; // expected-error {{invalid operands}}
  ff ^= bc2; // expected-error {{invalid operands}}
  ff ^= fd2; // expected-error {{invalid operands}}

  // -------------------------------------------------------------------------
  // Test shift operators.
  // -------------------------------------------------------------------------

  ff = ff << ff2; // expected-error {{integer is required}}
  ff = ff << fd2; // expected-error {{integer is required}}
  ff = ff << ui2; // expected-error {{integer is required}}
  ff = sl << ff2; // expected-error {{integer is required}}
  sc <<= ff2; // expected-error {{integer is required}}
  ff <<= ff2; // expected-error {{integer is required}}
  fd <<= ff2; // expected-error {{integer is required}}

  ff = ff >> ff2; // expected-error {{integer is required}}
  ff = ff >> fd2; // expected-error {{integer is required}}
  ff = ff >> ui2; // expected-error {{integer is required}}
  ff = sl >> ff2; // expected-error {{integer is required}}
  sc >>= ff2; // expected-error {{integer is required}}
  ff >>= ff2; // expected-error {{integer is required}}
  fd >>= ff2; // expected-error {{integer is required}}

  // -------------------------------------------------------------------------
  // Test comparison operators.
  // -------------------------------------------------------------------------

  (void)(ff == ff2);
  (void)(ff == fd2); // expected-error {{cannot convert}}
  (void)(ff == ui2); // expected-error {{cannot convert}}
  (void)(ui == ff2); // expected-error {{cannot convert}}

  (void)(ff != ff2);
  (void)(ff != fd2); // expected-error {{cannot convert}}
  (void)(ff != ui2); // expected-error {{cannot convert}}
  (void)(ui != ff2); // expected-error {{cannot convert}}

  (void)(ff <= ff2);
  (void)(ff <= fd2); // expected-error {{cannot convert}}
  (void)(ff <= ui2); // expected-error {{cannot convert}}
  (void)(ui <= ff2); // expected-error {{cannot convert}}

  (void)(ff >= ff2);
  (void)(ff >= fd2); // expected-error {{cannot convert}}
  (void)(ff >= ui2); // expected-error {{cannot convert}}
  (void)(ui >= ff2); // expected-error {{cannot convert}}

  (void)(ff < ff2);
  (void)(ff < fd2); // expected-error {{cannot convert}}
  (void)(ff < ui2); // expected-error {{cannot convert}}
  (void)(ui < ff2); // expected-error {{cannot convert}}

  (void)(ff > ff2);
  (void)(ff > fd2); // expected-error {{cannot convert}}
  (void)(ff > ui2); // expected-error {{cannot convert}}
  (void)(ui > ff2); // expected-error {{cannot convert}}
}
