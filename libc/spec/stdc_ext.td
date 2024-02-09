def StdcExt : StandardSpec<"stdc_ext"> {
  // From N1169 standard:
  // https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1169.pdf
  HeaderSpec StdFix = HeaderSpec<
      "stdfix.h",
      [],  // macros
      [],  // types
      [],  // enums
      [    // functions
      ]
  >;

  let Headers = [
    StdFix,
  ];
}
