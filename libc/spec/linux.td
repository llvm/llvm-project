def Linux : StandardSpec<"Linux"> {
  HeaderSpec Errno = HeaderSpec<
      "errno.h",
      [
        Macro<"ENOMEDIUM">,
        Macro<"ENOTBLK">,
        Macro<"EMEDIUMTYPE">,
        Macro<"EBADSLT">,
        Macro<"ECHRNG">,
        Macro<"ERFKILL">,
        Macro<"EUSERS">,
        Macro<"EBADR">,
        Macro<"EL3HLT">,
        Macro<"ENOTUNIQ">,
        Macro<"EXFULL">,
        Macro<"EHOSTDOWN">,
        Macro<"EL3RST">,
        Macro<"ENOPKG">,
        Macro<"ENOCSI">,
        Macro<"EUNATCH">,
        Macro<"EREMCHG">,
        Macro<"ETOOMANYREFS">,
        Macro<"EL2HLT">,
        Macro<"EBADFD">,
        Macro<"EREMOTEIO">,
        Macro<"ENAVAIL">,
        Macro<"ELIBEXEC">,
        Macro<"ESHUTDOWN">,
        Macro<"ENOKEY">,
        Macro<"ESTRPIPE">,
        Macro<"EKEYREJECTED">,
        Macro<"ESRMNT">,
        Macro<"EKEYREVOKED">,
        Macro<"EBADE">,
        Macro<"ELIBBAD">,
        Macro<"EISNAM">,
        Macro<"EBFONT">,
        Macro<"EPFNOSUPPORT">,
        Macro<"EREMOTE">,
        Macro<"EDEADLOCK">,
        Macro<"ENONET">,
        Macro<"EDOTDOT">,
        Macro<"EKEYEXPIRED">,
        Macro<"ELIBSCN">,
        Macro<"ERESTART">,
        Macro<"EBADRQC">,
        Macro<"EUCLEAN">,
        Macro<"ENOANO">,
        Macro<"ELIBACC">,
        Macro<"EHWPOISON">,
        Macro<"ELIBMAX">,
        Macro<"ESOCKTNOSUPPORT">,
        Macro<"ENOTNAM">,
        Macro<"ELNRNG">,
        Macro<"EL2NSYNC">,
        Macro<"EADV">,
        Macro<"ECOMM">,
      ],
      [], // Types
      []  // Functions
  >;

  HeaderSpec SysMMan = HeaderSpec<
      "sys/mman.h",
      [Macro<"MAP_ANONYMOUS">],
      [], // Types
      []  // Functions
  >;

  let Headers = [
    Errno,
    SysMMan,
  ];
}
