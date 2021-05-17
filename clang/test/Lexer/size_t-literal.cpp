// RUN: %clang_cc1 -std=c++2b -fsyntax-only -verify %s

#if 1z != 1
#error "z suffix must be recognized by preprocessor"
#endif
#if 1uz != 1
#error "uz suffix must be recognized by preprocessor"
#endif
#if !(-1z < 0)
#error "z suffix must be interpreted as signed"
#endif
#if !(-1uz > 0)
#error "uz suffix must be interpreted as unsigned"
#endif

void ValidSuffix() {
  // Decimal literals.
  {
    auto a1 = 1z;
    auto a2 = 1Z;

    auto a3 = 1uz;
    auto a4 = 1uZ;
    auto a5 = 1Uz;
    auto a6 = 1UZ;

    auto a7 = 1zu;
    auto a8 = 1Zu;
    auto a9 = 1zU;
    auto a10 = 1ZU;

    auto a11 = 1'2z;
    auto a12 = 1'2Z;
  }
  // Hexadecimal literals.
  {
    auto a1 = 0x1z;
    auto a2 = 0x1Z;

    auto a3 = 0x1uz;
    auto a4 = 0x1uZ;
    auto a5 = 0x1Uz;
    auto a6 = 0x1UZ;

    auto a7 = 0x1zu;
    auto a8 = 0x1Zu;
    auto a9 = 0x1zU;
    auto a10 = 0x1ZU;

    auto a11 = 0x1'2z;
    auto a12 = 0x1'2Z;
  }
  // Binary literals.
  {
    auto a1 = 0b1z;
    auto a2 = 0b1Z;

    auto a3 = 0b1uz;
    auto a4 = 0b1uZ;
    auto a5 = 0b1Uz;
    auto a6 = 0b1UZ;

    auto a7 = 0b1zu;
    auto a8 = 0b1Zu;
    auto a9 = 0b1zU;
    auto a10 = 0b1ZU;

    auto a11 = 0b1'1z;
    auto a12 = 0b1'1Z;
  }
  // Octal literals.
  {
    auto a1 = 01z;
    auto a2 = 01Z;

    auto a3 = 01uz;
    auto a4 = 01uZ;
    auto a5 = 01Uz;
    auto a6 = 01UZ;

    auto a7 = 01zu;
    auto a8 = 01Zu;
    auto a9 = 01zU;
    auto a10 = 01ZU;

    auto a11 = 0'1z;
    auto a12 = 0'1Z;
  }
}

void InvalidSuffix() {
  // Long.
  {
    auto a1 = 1lz; // expected-error {{invalid suffix}}
    auto a2 = 1lZ; // expected-error {{invalid suffix}}
    auto a3 = 1Lz; // expected-error {{invalid suffix}}
    auto a4 = 1LZ; // expected-error {{invalid suffix}}

    auto a5 = 1zl; // expected-error {{invalid suffix}}
    auto a6 = 1Zl; // expected-error {{invalid suffix}}
    auto a7 = 1zL; // expected-error {{invalid suffix}}
    auto a8 = 1ZL; // expected-error {{invalid suffix}}

    auto a9 = 1ulz;  // expected-error {{invalid suffix}}
    auto a10 = 1ulZ; // expected-error {{invalid suffix}}
    auto a11 = 1uLz; // expected-error {{invalid suffix}}
    auto a12 = 1uLZ; // expected-error {{invalid suffix}}

    auto a13 = 1uzl; // expected-error {{invalid suffix}}
    auto a14 = 1uZl; // expected-error {{invalid suffix}}
    auto a15 = 1uzL; // expected-error {{invalid suffix}}
    auto a16 = 1uZL; // expected-error {{invalid suffix}}
  }
  // Long long.
  {
    auto a1 = 1llz; // expected-error {{invalid suffix}}
    auto a2 = 1llZ; // expected-error {{invalid suffix}}
    auto a3 = 1LLz; // expected-error {{invalid suffix}}
    auto a4 = 1LLZ; // expected-error {{invalid suffix}}

    auto a5 = 1zll; // expected-error {{invalid suffix}}
    auto a6 = 1Zll; // expected-error {{invalid suffix}}
    auto a7 = 1zLL; // expected-error {{invalid suffix}}
    auto a8 = 1ZLL; // expected-error {{invalid suffix}}

    auto a9 = 1ullz;  // expected-error {{invalid suffix}}
    auto a10 = 1ullZ; // expected-error {{invalid suffix}}
    auto a11 = 1uLLz; // expected-error {{invalid suffix}}
    auto a12 = 1uLLZ; // expected-error {{invalid suffix}}

    auto a13 = 1uzll; // expected-error {{invalid suffix}}
    auto a14 = 1uZll; // expected-error {{invalid suffix}}
    auto a15 = 1uzLL; // expected-error {{invalid suffix}}
    auto a16 = 1uZLL; // expected-error {{invalid suffix}}
  }
  // Floating point.
  {
    auto a1 = 0.1z;   // expected-error {{invalid suffix}}
    auto a2 = 0.1Z;   // expected-error {{invalid suffix}}
    auto a3 = 0.1uz;  // expected-error {{invalid suffix}}
    auto a4 = 0.1uZ;  // expected-error {{invalid suffix}}
    auto a5 = 0.1Uz;  // expected-error {{invalid suffix}}
    auto a6 = 0.1UZ;  // expected-error {{invalid suffix}}
    auto a7 = 0.1zu;  // expected-error {{invalid suffix}}
    auto a8 = 0.1Zu;  // expected-error {{invalid suffix}}
    auto a9 = 0.1zU;  // expected-error {{invalid suffix}}
    auto a10 = 0.1ZU; // expected-error {{invalid suffix}}

    auto a11 = 0.1fz;   // expected-error {{invalid suffix}}
    auto a12 = 0.1fZ;   // expected-error {{invalid suffix}}
    auto a13 = 0.1fuz;  // expected-error {{invalid suffix}}
    auto a14 = 0.1fuZ;  // expected-error {{invalid suffix}}
    auto a15 = 0.1fUz;  // expected-error {{invalid suffix}}
    auto a16 = 0.1fUZ;  // expected-error {{invalid suffix}}
    auto a17 = 0.1fzu;  // expected-error {{invalid suffix}}
    auto a18 = 0.1fZu;  // expected-error {{invalid suffix}}
    auto a19 = 0.1fzU;  // expected-error {{invalid suffix}}
    auto a110 = 0.1fZU; // expected-error {{invalid suffix}}
  }
  // Repetitive suffix.
  {
    auto a1 = 1zz; // expected-error {{invalid suffix}}
    auto a2 = 1zZ; // expected-error {{invalid suffix}}
    auto a3 = 1Zz; // expected-error {{invalid suffix}}
    auto a4 = 1ZZ; // expected-error {{invalid suffix}}
  }
}
