def StdcExt : StandardSpec<"stdc_ext"> {
  // From ISO/IEC TR 18037:2008 standard:
  // https://standards.iso.org/ittf/PubliclyAvailableStandards/c051126_ISO_IEC_TR_18037_2008.zip
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
