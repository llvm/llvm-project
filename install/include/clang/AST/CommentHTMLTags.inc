/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* HTML tag name matcher                                                      *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: CommentHTMLTags.td                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

bool isHTMLTagName(StringRef Name) {
  switch (Name.size()) {
  default: break;
  case 1:	 // 6 strings to match.
    switch (Name[0]) {
    default: break;
    case 'a':	 // 1 string to match.
      return true;	 // "a"
    case 'b':	 // 1 string to match.
      return true;	 // "b"
    case 'i':	 // 1 string to match.
      return true;	 // "i"
    case 'p':	 // 1 string to match.
      return true;	 // "p"
    case 's':	 // 1 string to match.
      return true;	 // "s"
    case 'u':	 // 1 string to match.
      return true;	 // "u"
    }
    break;
  case 2:	 // 19 strings to match.
    switch (Name[0]) {
    default: break;
    case 'b':	 // 1 string to match.
      if (Name[1] != 'r')
        break;
      return true;	 // "br"
    case 'd':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'd':	 // 1 string to match.
        return true;	 // "dd"
      case 'l':	 // 1 string to match.
        return true;	 // "dl"
      case 't':	 // 1 string to match.
        return true;	 // "dt"
      }
      break;
    case 'e':	 // 1 string to match.
      if (Name[1] != 'm')
        break;
      return true;	 // "em"
    case 'h':	 // 7 strings to match.
      switch (Name[1]) {
      default: break;
      case '1':	 // 1 string to match.
        return true;	 // "h1"
      case '2':	 // 1 string to match.
        return true;	 // "h2"
      case '3':	 // 1 string to match.
        return true;	 // "h3"
      case '4':	 // 1 string to match.
        return true;	 // "h4"
      case '5':	 // 1 string to match.
        return true;	 // "h5"
      case '6':	 // 1 string to match.
        return true;	 // "h6"
      case 'r':	 // 1 string to match.
        return true;	 // "hr"
      }
      break;
    case 'l':	 // 1 string to match.
      if (Name[1] != 'i')
        break;
      return true;	 // "li"
    case 'o':	 // 1 string to match.
      if (Name[1] != 'l')
        break;
      return true;	 // "ol"
    case 't':	 // 4 strings to match.
      switch (Name[1]) {
      default: break;
      case 'd':	 // 1 string to match.
        return true;	 // "td"
      case 'h':	 // 1 string to match.
        return true;	 // "th"
      case 'r':	 // 1 string to match.
        return true;	 // "tr"
      case 't':	 // 1 string to match.
        return true;	 // "tt"
      }
      break;
    case 'u':	 // 1 string to match.
      if (Name[1] != 'l')
        break;
      return true;	 // "ul"
    }
    break;
  case 3:	 // 9 strings to match.
    switch (Name[0]) {
    default: break;
    case 'b':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ig", 2) != 0)
        break;
      return true;	 // "big"
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ol", 2) != 0)
        break;
      return true;	 // "col"
    case 'd':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'e':	 // 1 string to match.
        if (Name[2] != 'l')
          break;
        return true;	 // "del"
      case 'i':	 // 1 string to match.
        if (Name[2] != 'v')
          break;
        return true;	 // "div"
      }
      break;
    case 'i':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'm':	 // 1 string to match.
        if (Name[2] != 'g')
          break;
        return true;	 // "img"
      case 'n':	 // 1 string to match.
        if (Name[2] != 's')
          break;
        return true;	 // "ins"
      }
      break;
    case 'p':	 // 1 string to match.
      if (memcmp(Name.data()+1, "re", 2) != 0)
        break;
      return true;	 // "pre"
    case 's':	 // 2 strings to match.
      if (Name[1] != 'u')
        break;
      switch (Name[2]) {
      default: break;
      case 'b':	 // 1 string to match.
        return true;	 // "sub"
      case 'p':	 // 1 string to match.
        return true;	 // "sup"
      }
      break;
    }
    break;
  case 4:	 // 3 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ode", 3) != 0)
        break;
      return true;	 // "code"
    case 'f':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ont", 3) != 0)
        break;
      return true;	 // "font"
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "pan", 3) != 0)
        break;
      return true;	 // "span"
    }
    break;
  case 5:	 // 5 strings to match.
    switch (Name[0]) {
    default: break;
    case 's':	 // 1 string to match.
      if (memcmp(Name.data()+1, "mall", 4) != 0)
        break;
      return true;	 // "small"
    case 't':	 // 4 strings to match.
      switch (Name[1]) {
      default: break;
      case 'a':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ble", 3) != 0)
          break;
        return true;	 // "table"
      case 'b':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ody", 3) != 0)
          break;
        return true;	 // "tbody"
      case 'f':	 // 1 string to match.
        if (memcmp(Name.data()+2, "oot", 3) != 0)
          break;
        return true;	 // "tfoot"
      case 'h':	 // 1 string to match.
        if (memcmp(Name.data()+2, "ead", 3) != 0)
          break;
        return true;	 // "thead"
      }
      break;
    }
    break;
  case 6:	 // 2 strings to match.
    if (memcmp(Name.data()+0, "str", 3) != 0)
      break;
    switch (Name[3]) {
    default: break;
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+4, "ke", 2) != 0)
        break;
      return true;	 // "strike"
    case 'o':	 // 1 string to match.
      if (memcmp(Name.data()+4, "ng", 2) != 0)
        break;
      return true;	 // "strong"
    }
    break;
  case 7:	 // 1 string to match.
    if (memcmp(Name.data()+0, "caption", 7) != 0)
      break;
    return true;	 // "caption"
  case 8:	 // 1 string to match.
    if (memcmp(Name.data()+0, "colgroup", 8) != 0)
      break;
    return true;	 // "colgroup"
  case 10:	 // 1 string to match.
    if (memcmp(Name.data()+0, "blockquote", 10) != 0)
      break;
    return true;	 // "blockquote"
  }
  return false;
}

