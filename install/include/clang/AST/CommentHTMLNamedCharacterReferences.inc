/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* HTML named character reference to UTF-8 translation                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: CommentHTMLNamedCharacterReferences.td                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

StringRef translateHTMLNamedCharacterReferenceToUTF8(
                                             StringRef Name) {
  switch (Name.size()) {
  default: break;
  case 2:	 // 15 strings to match.
    switch (Name[0]) {
    default: break;
    case 'G':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'T':	 // 1 string to match.
        return "\x3e";	 // "GT"
      case 't':	 // 1 string to match.
        return "\x3e";	 // "Gt"
      }
      break;
    case 'L':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'T':	 // 1 string to match.
        return "\x3c";	 // "LT"
      case 't':	 // 1 string to match.
        return "\x3c";	 // "Lt"
      }
      break;
    case 'P':	 // 1 string to match.
      if (Name[1] != 'i')
        break;
      return "\xce\xa0";	 // "Pi"
    case 'X':	 // 1 string to match.
      if (Name[1] != 'i')
        break;
      return "\xce\x9e";	 // "Xi"
    case 'g':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        return "\xe2\x89\xa5";	 // "ge"
      case 't':	 // 1 string to match.
        return "\x3e";	 // "gt"
      }
      break;
    case 'l':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        return "\xe2\x89\xa4";	 // "le"
      case 't':	 // 1 string to match.
        return "\x3c";	 // "lt"
      }
      break;
    case 'm':	 // 1 string to match.
      if (Name[1] != 'u')
        break;
      return "\xce\xbc";	 // "mu"
    case 'n':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        return "\xe2\x89\xa0";	 // "ne"
      case 'u':	 // 1 string to match.
        return "\xce\xbd";	 // "nu"
      }
      break;
    case 'p':	 // 1 string to match.
      if (Name[1] != 'i')
        break;
      return "\xcf\x80";	 // "pi"
    case 'x':	 // 1 string to match.
      if (Name[1] != 'i')
        break;
      return "\xce\xbe";	 // "xi"
    }
    break;
  case 3:	 // 16 strings to match.
    switch (Name[0]) {
    default: break;
    case 'A':	 // 1 string to match.
      if (memcmp(Name.data()+1, "MP", 2) != 0)
        break;
      return "\x26";	 // "AMP"
    case 'P':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'h':	 // 1 string to match.
        if (Name[2] != 'i')
          break;
        return "\xce\xa6";	 // "Phi"
      case 's':	 // 1 string to match.
        if (Name[2] != 'i')
          break;
        return "\xce\xa8";	 // "Psi"
      }
      break;
    case 'R':	 // 1 string to match.
      if (memcmp(Name.data()+1, "EG", 2) != 0)
        break;
      return "\xc2\xae";	 // "REG"
    case 'a':	 // 1 string to match.
      if (memcmp(Name.data()+1, "mp", 2) != 0)
        break;
      return "\x26";	 // "amp"
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "hi", 2) != 0)
        break;
      return "\xcf\x87";	 // "chi"
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+1, "eg", 2) != 0)
        break;
      return "\xc2\xb0";	 // "deg"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ta", 2) != 0)
        break;
      return "\xce\xb7";	 // "eta"
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+1, "nt", 2) != 0)
        break;
      return "\xe2\x88\xab";	 // "int"
    case 'p':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'h':	 // 1 string to match.
        if (Name[2] != 'i')
          break;
        return "\xcf\x86";	 // "phi"
      case 's':	 // 1 string to match.
        if (Name[2] != 'i')
          break;
        return "\xcf\x88";	 // "psi"
      }
      break;
    case 'r':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        if (Name[2] != 'g')
          break;
        return "\xc2\xae";	 // "reg"
      case 'h':	 // 1 string to match.
        if (Name[2] != 'o')
          break;
        return "\xcf\x81";	 // "rho"
      }
      break;
    case 's':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'i':	 // 1 string to match.
        if (Name[2] != 'm')
          break;
        return "\xe2\x88\xbc";	 // "sim"
      case 'u':	 // 1 string to match.
        if (Name[2] != 'm')
          break;
        return "\xe2\x88\x91";	 // "sum"
      }
      break;
    case 't':	 // 1 string to match.
      if (memcmp(Name.data()+1, "au", 2) != 0)
        break;
      return "\xcf\x84";	 // "tau"
    }
    break;
  case 4:	 // 30 strings to match.
    switch (Name[0]) {
    default: break;
    case 'A':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uml", 3) != 0)
        break;
      return "\xc3\x84";	 // "Auml"
    case 'C':	 // 1 string to match.
      if (memcmp(Name.data()+1, "OPY", 3) != 0)
        break;
      return "\xc2\xa9";	 // "COPY"
    case 'E':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uml", 3) != 0)
        break;
      return "\xc3\x8b";	 // "Euml"
    case 'I':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uml", 3) != 0)
        break;
      return "\xc3\x8f";	 // "Iuml"
    case 'O':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uml", 3) != 0)
        break;
      return "\xc3\x96";	 // "Ouml"
    case 'Q':	 // 1 string to match.
      if (memcmp(Name.data()+1, "UOT", 3) != 0)
        break;
      return "\x22";	 // "QUOT"
    case 'U':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uml", 3) != 0)
        break;
      return "\xc3\x9c";	 // "Uuml"
    case 'Y':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uml", 3) != 0)
        break;
      return "\xc5\xb8";	 // "Yuml"
    case 'a':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'p':	 // 1 string to match.
        if (memcmp(Name.data()+2, "os", 2) != 0)
          break;
        return "\x27";	 // "apos"
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ml", 2) != 0)
          break;
        return "\xc3\xa4";	 // "auml"
      }
      break;
    case 'b':	 // 1 string to match.
      if (memcmp(Name.data()+1, "eta", 3) != 0)
        break;
      return "\xce\xb2";	 // "beta"
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "opy", 3) != 0)
        break;
      return "\xc2\xa9";	 // "copy"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uml", 3) != 0)
        break;
      return "\xc3\xab";	 // "euml"
    case 'i':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ta", 2) != 0)
          break;
        return "\xce\xb9";	 // "iota"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "in", 2) != 0)
          break;
        return "\xe2\x88\x88";	 // "isin"
      case 'u':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ml", 2) != 0)
          break;
        return "\xc3\xaf";	 // "iuml"
      }
      break;
    case 'l':	 // 1 string to match.
      if (memcmp(Name.data()+1, "arr", 3) != 0)
        break;
      return "\xe2\x86\x90";	 // "larr"
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "bsp", 3) != 0)
        break;
      return "\xc2\xa0";	 // "nbsp"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uml", 3) != 0)
        break;
      return "\xc3\xb6";	 // "ouml"
    case 'p':	 // 4 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rt", 2) != 0)
          break;
        return "\xe2\x88\x82";	 // "part"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rp", 2) != 0)
          break;
        return "\xe2\x8a\xa5";	 // "perp"
      case 'r':	 // 2 strings to match.
        if (Name[2] != 'o')
          break;
        switch (Name[3]) {
        default: break;
        case 'd':	 // 1 string to match.
          return "\xe2\x88\x8f";	 // "prod"
        case 'p':	 // 1 string to match.
          return "\xe2\x88\x9d";	 // "prop"
        }
        break;
      }
      break;
    case 'q':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uot", 3) != 0)
        break;
      return "\x22";	 // "quot"
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "arr", 3) != 0)
        break;
      return "\xe2\x86\x92";	 // "rarr"
    case 's':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'd':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ot", 2) != 0)
          break;
        return "\xe2\x8b\x85";	 // "sdot"
      case 'e':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ct", 2) != 0)
          break;
        return "\xc2\xa7";	 // "sect"
      }
      break;
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uml", 3) != 0)
        break;
      return "\xc3\xbc";	 // "uuml"
    case 'y':	 // 1 string to match.
      if (memcmp(Name.data()+1, "uml", 3) != 0)
        break;
      return "\xc3\xbf";	 // "yuml"
    case 'z':	 // 1 string to match.
      if (memcmp(Name.data()+1, "eta", 3) != 0)
        break;
      return "\xce\xb6";	 // "zeta"
    }
    break;
  case 5:	 // 48 strings to match.
    switch (Name[0]) {
    default: break;
    case 'A':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+2, "irc", 3) != 0)
          break;
        return "\xc3\x82";	 // "Acirc"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ing", 3) != 0)
          break;
        return "\xc3\x85";	 // "Aring"
      }
      break;
    case 'D':	 // 1 string to match.
      if (memcmp(Name.data()+1, "elta", 4) != 0)
        break;
      return "\xce\x94";	 // "Delta"
    case 'E':	 // 1 string to match.
      if (memcmp(Name.data()+1, "circ", 4) != 0)
        break;
      return "\xc3\x8a";	 // "Ecirc"
    case 'G':	 // 1 string to match.
      if (memcmp(Name.data()+1, "amma", 4) != 0)
        break;
      return "\xce\x93";	 // "Gamma"
    case 'I':	 // 1 string to match.
      if (memcmp(Name.data()+1, "circ", 4) != 0)
        break;
      return "\xc3\x8e";	 // "Icirc"
    case 'O':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+2, "irc", 3) != 0)
          break;
        return "\xc3\x94";	 // "Ocirc"
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ega", 3) != 0)
          break;
        return "\xce\xa9";	 // "Omega"
      }
      break;
    case 'P':	 // 1 string to match.
      if (memcmp(Name.data()+1, "rime", 4) != 0)
        break;
      return "\xe2\x80\xb3";	 // "Prime"
    case 'S':	 // 1 string to match.
      if (memcmp(Name.data()+1, "igma", 4) != 0)
        break;
      return "\xce\xa3";	 // "Sigma"
    case 'T':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'R':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ADE", 3) != 0)
          break;
        return "\xe2\x84\xa2";	 // "TRADE"
      case 'h':	 // 1 string to match.
        if (memcmp(Name.data()+2, "eta", 3) != 0)
          break;
        return "\xce\x98";	 // "Theta"
      }
      break;
    case 'U':	 // 1 string to match.
      if (memcmp(Name.data()+1, "circ", 4) != 0)
        break;
      return "\xc3\x9b";	 // "Ucirc"
    case 'Y':	 // 1 string to match.
      if (memcmp(Name.data()+1, "circ", 4) != 0)
        break;
      return "\xc5\xb6";	 // "Ycirc"
    case 'a':	 // 4 strings to match.
      switch (Name[1]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+2, "irc", 3) != 0)
          break;
        return "\xc3\xa2";	 // "acirc"
      case 'l':	 // 1 string to match.
        if (memcmp(Name.data()+2, "pha", 3) != 0)
          break;
        return "\xce\xb1";	 // "alpha"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ing", 3) != 0)
          break;
        return "\xc3\xa5";	 // "aring"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ymp", 3) != 0)
          break;
        return "\xe2\x89\x88";	 // "asymp"
      }
      break;
    case 'd':	 // 1 string to match.
      if (memcmp(Name.data()+1, "elta", 4) != 0)
        break;
      return "\xce\xb4";	 // "delta"
    case 'e':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+2, "irc", 3) != 0)
          break;
        return "\xc3\xaa";	 // "ecirc"
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+2, "pty", 3) != 0)
          break;
        return "\xe2\x88\x85";	 // "empty"
      case 'q':	 // 1 string to match.
        if (memcmp(Name.data()+2, "uiv", 3) != 0)
          break;
        return "\xe2\x89\xa1";	 // "equiv"
      }
      break;
    case 'g':	 // 1 string to match.
      if (memcmp(Name.data()+1, "amma", 4) != 0)
        break;
      return "\xce\xb3";	 // "gamma"
    case 'i':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+2, "irc", 3) != 0)
          break;
        return "\xc3\xae";	 // "icirc"
      case 'n':	 // 1 string to match.
        if (memcmp(Name.data()+2, "fin", 3) != 0)
          break;
        return "\xe2\x88\x9e";	 // "infin"
      }
      break;
    case 'k':	 // 1 string to match.
      if (memcmp(Name.data()+1, "appa", 4) != 0)
        break;
      return "\xce\xba";	 // "kappa"
    case 'l':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+2, "eil", 3) != 0)
          break;
        return "\xe2\x8c\x88";	 // "lceil"
      case 'd':	 // 1 string to match.
        if (memcmp(Name.data()+2, "quo", 3) != 0)
          break;
        return "\xe2\x80\x9c";	 // "ldquo"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "quo", 3) != 0)
          break;
        return "\xe2\x80\x98";	 // "lsquo"
      }
      break;
    case 'm':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'd':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ash", 3) != 0)
          break;
        return "\xe2\x80\x94";	 // "mdash"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+2, "nus", 3) != 0)
          break;
        return "\xe2\x88\x92";	 // "minus"
      }
      break;
    case 'n':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "bla", 3) != 0)
          break;
        return "\xe2\x88\x87";	 // "nabla"
      case 'd':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ash", 3) != 0)
          break;
        return "\xe2\x80\x93";	 // "ndash"
      case 'o':	 // 1 string to match.
        if (memcmp(Name.data()+2, "tin", 3) != 0)
          break;
        return "\xe2\x88\x89";	 // "notin"
      }
      break;
    case 'o':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+2, "irc", 3) != 0)
          break;
        return "\xc3\xb4";	 // "ocirc"
      case 'm':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ega", 3) != 0)
          break;
        return "\xcf\x89";	 // "omega"
      }
      break;
    case 'p':	 // 1 string to match.
      if (memcmp(Name.data()+1, "rime", 4) != 0)
        break;
      return "\xe2\x80\xb2";	 // "prime"
    case 'r':	 // 4 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "dic", 3) != 0)
          break;
        return "\xe2\x88\x9a";	 // "radic"
      case 'c':	 // 1 string to match.
        if (memcmp(Name.data()+2, "eil", 3) != 0)
          break;
        return "\xe2\x8c\x89";	 // "rceil"
      case 'd':	 // 1 string to match.
        if (memcmp(Name.data()+2, "quo", 3) != 0)
          break;
        return "\xe2\x80\x9d";	 // "rdquo"
      case 's':	 // 1 string to match.
        if (memcmp(Name.data()+2, "quo", 3) != 0)
          break;
        return "\xe2\x80\x99";	 // "rsquo"
      }
      break;
    case 's':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+2, "gma", 3) != 0)
          break;
        return "\xcf\x83";	 // "sigma"
      case 'z':	 // 1 string to match.
        if (memcmp(Name.data()+2, "lig", 3) != 0)
          break;
        return "\xc3\x9f";	 // "szlig"
      }
      break;
    case 't':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'h':	 // 1 string to match.
        if (memcmp(Name.data()+2, "eta", 3) != 0)
          break;
        return "\xce\xb8";	 // "theta"
      case 'i':	 // 1 string to match.
        if (memcmp(Name.data()+2, "mes", 3) != 0)
          break;
        return "\xc3\x97";	 // "times"
      case 'r':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ade", 3) != 0)
          break;
        return "\xe2\x84\xa2";	 // "trade"
      }
      break;
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+1, "circ", 4) != 0)
        break;
      return "\xc3\xbb";	 // "ucirc"
    case 'y':	 // 1 string to match.
      if (memcmp(Name.data()+1, "circ", 4) != 0)
        break;
      return "\xc5\xb7";	 // "ycirc"
    }
    break;
  case 6:	 // 37 strings to match.
    switch (Name[0]) {
    default: break;
    case 'A':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cute", 4) != 0)
          break;
        return "\xc3\x81";	 // "Aacute"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rave", 4) != 0)
          break;
        return "\xc3\x80";	 // "Agrave"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ilde", 4) != 0)
          break;
        return "\xc3\x83";	 // "Atilde"
      }
      break;
    case 'C':	 // 1 string to match.
      if (memcmp(Name.data()+1, "cedil", 5) != 0)
        break;
      return "\xc3\x87";	 // "Ccedil"
    case 'E':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cute", 4) != 0)
          break;
        return "\xc3\x89";	 // "Eacute"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rave", 4) != 0)
          break;
        return "\xc3\x88";	 // "Egrave"
      }
      break;
    case 'I':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cute", 4) != 0)
          break;
        return "\xc3\x8d";	 // "Iacute"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rave", 4) != 0)
          break;
        return "\xc3\x8c";	 // "Igrave"
      }
      break;
    case 'L':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ambda", 5) != 0)
        break;
      return "\xce\x9b";	 // "Lambda"
    case 'N':	 // 1 string to match.
      if (memcmp(Name.data()+1, "tilde", 5) != 0)
        break;
      return "\xc3\x91";	 // "Ntilde"
    case 'O':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cute", 4) != 0)
          break;
        return "\xc3\x93";	 // "Oacute"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rave", 4) != 0)
          break;
        return "\xc3\x92";	 // "Ograve"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ilde", 4) != 0)
          break;
        return "\xc3\x95";	 // "Otilde"
      }
      break;
    case 'U':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cute", 4) != 0)
          break;
        return "\xc3\x9a";	 // "Uacute"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rave", 4) != 0)
          break;
        return "\xc3\x99";	 // "Ugrave"
      }
      break;
    case 'Y':	 // 1 string to match.
      if (memcmp(Name.data()+1, "acute", 5) != 0)
        break;
      return "\xc3\x9d";	 // "Yacute"
    case 'a':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cute", 4) != 0)
          break;
        return "\xc3\xa1";	 // "aacute"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rave", 4) != 0)
          break;
        return "\xc3\xa0";	 // "agrave"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ilde", 4) != 0)
          break;
        return "\xc3\xa3";	 // "atilde"
      }
      break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "cedil", 5) != 0)
        break;
      return "\xc3\xa7";	 // "ccedil"
    case 'e':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cute", 4) != 0)
          break;
        return "\xc3\xa9";	 // "eacute"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rave", 4) != 0)
          break;
        return "\xc3\xa8";	 // "egrave"
      }
      break;
    case 'i':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cute", 4) != 0)
          break;
        return "\xc3\xad";	 // "iacute"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rave", 4) != 0)
          break;
        return "\xc3\xac";	 // "igrave"
      }
      break;
    case 'l':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "mbda", 4) != 0)
          break;
        return "\xce\xbb";	 // "lambda"
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+2, "loor", 4) != 0)
          break;
        return "\xe2\x8c\x8a";	 // "lfloor"
      }
      break;
    case 'n':	 // 1 string to match.
      if (memcmp(Name.data()+1, "tilde", 5) != 0)
        break;
      return "\xc3\xb1";	 // "ntilde"
    case 'o':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cute", 4) != 0)
          break;
        return "\xc3\xb3";	 // "oacute"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rave", 4) != 0)
          break;
        return "\xc3\xb2";	 // "ograve"
      case 't':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ilde", 4) != 0)
          break;
        return "\xc3\xb5";	 // "otilde"
      }
      break;
    case 'p':	 // 1 string to match.
      if (memcmp(Name.data()+1, "lusmn", 5) != 0)
        break;
      return "\xc2\xb1";	 // "plusmn"
    case 'r':	 // 1 string to match.
      if (memcmp(Name.data()+1, "floor", 5) != 0)
        break;
      return "\xe2\x8c\x8b";	 // "rfloor"
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "igmaf", 5) != 0)
        break;
      return "\xcf\x82";	 // "sigmaf"
    case 'u':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cute", 4) != 0)
          break;
        return "\xc3\xba";	 // "uacute"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rave", 4) != 0)
          break;
        return "\xc3\xb9";	 // "ugrave"
      }
      break;
    case 'y':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "cute", 4) != 0)
          break;
        return "\xc3\xbd";	 // "yacute"
      case 'g':	 // 1 string to match.
        if (memcmp(Name.data()+2, "rave", 4) != 0)
          break;
        return "\xe1\xbb\xb3";	 // "ygrave"
      }
      break;
    }
    break;
  case 7:	 // 3 strings to match.
    switch (Name[0]) {
    default: break;
    case 'U':	 // 1 string to match.
      if (memcmp(Name.data()+1, "psilon", 6) != 0)
        break;
      return "\xce\xa5";	 // "Upsilon"
    case 'e':	 // 1 string to match.
      if (memcmp(Name.data()+1, "psilon", 6) != 0)
        break;
      return "\xce\xb5";	 // "epsilon"
    case 'u':	 // 1 string to match.
      if (memcmp(Name.data()+1, "psilon", 6) != 0)
        break;
      return "\xcf\x85";	 // "upsilon"
    }
    break;
  }
  return StringRef();
}

