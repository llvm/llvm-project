/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* HTML tag properties                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: CommentHTMLTags.td                                                   *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

bool isHTMLEndTagOptional(StringRef Name) {
  switch (Name.size()) {
  default: break;
  case 1:	 // 1 string to match.
    if (Name[0] != 'p')
      break;
    return true;	 // "p"
  case 2:	 // 6 strings to match.
    switch (Name[0]) {
    default: break;
    case 'd':	 // 2 strings to match.
      switch (Name[1]) {
      default: break;
      case 'd':	 // 1 string to match.
        return true;	 // "dd"
      case 't':	 // 1 string to match.
        return true;	 // "dt"
      }
      break;
    case 'l':	 // 1 string to match.
      if (Name[1] != 'i')
        break;
      return true;	 // "li"
    case 't':	 // 3 strings to match.
      switch (Name[1]) {
      default: break;
      case 'd':	 // 1 string to match.
        return true;	 // "td"
      case 'h':	 // 1 string to match.
        return true;	 // "th"
      case 'r':	 // 1 string to match.
        return true;	 // "tr"
      }
      break;
    }
    break;
  case 5:	 // 3 strings to match.
    if (Name[0] != 't')
      break;
    switch (Name[1]) {
    default: break;
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
  case 8:	 // 1 string to match.
    if (memcmp(Name.data()+0, "colgroup", 8) != 0)
      break;
    return true;	 // "colgroup"
  }
  return false;
}

bool isHTMLEndTagForbidden(StringRef Name) {
  switch (Name.size()) {
  default: break;
  case 2:	 // 2 strings to match.
    switch (Name[0]) {
    default: break;
    case 'b':	 // 1 string to match.
      if (Name[1] != 'r')
        break;
      return true;	 // "br"
    case 'h':	 // 1 string to match.
      if (Name[1] != 'r')
        break;
      return true;	 // "hr"
    }
    break;
  case 3:	 // 2 strings to match.
    switch (Name[0]) {
    default: break;
    case 'c':	 // 1 string to match.
      if (memcmp(Name.data()+1, "ol", 2) != 0)
        break;
      return true;	 // "col"
    case 'i':	 // 1 string to match.
      if (memcmp(Name.data()+1, "mg", 2) != 0)
        break;
      return true;	 // "img"
    }
    break;
  }
  return false;
}

