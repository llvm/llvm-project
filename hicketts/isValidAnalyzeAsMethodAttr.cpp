static bool isValidAnalyzeAsMethodAttr(Decl *D, StringRef Tag) {
  if (Tag.empty())
    return false;

  // Bare name with no parenthesised signature — accept as-is.
  size_t OpenParen = Tag.find('(');
  if (OpenParen == StringRef::npos)
    return true;

  // Method name before '(' must be non-empty.
  if (OpenParen == 0)
    return false;

  // Must end with ')' — no trailing junk.
  if (Tag.back() != ')')
    return false;

  // Extract the contents between the outer '(' and ')'.
  StringRef Params = Tag.slice(OpenParen + 1, Tag.size() - 1);

  // Empty param list "name()" is valid.
  if (Params.empty())
    return true;

  // Walk the param list checking balanced/non-interleaved () and <>
  // and that comma-separated segments are non-empty.
  SmallVector<char, 4> Stack;
  size_t SegStart = 0;

  for (size_t I = 0, E = Params.size(); I != E; ++I) {
    char C = Params[I];
    if (C == '(' || C == '<') {
      Stack.push_back(C);
    } else if (C == ')') {
      if (Stack.empty() || Stack.back() != '(')
        return false;
      Stack.pop_back();
    } else if (C == '>') {
      if (Stack.empty() || Stack.back() != '<')
        return false;
      Stack.pop_back();
    } else if (C == ',' && Stack.empty()) {
      if (Params.slice(SegStart, I).trim().empty())
        return false;
      SegStart = I + 1;
    }
  }

  if (!Stack.empty())
    return false;

  return !Params.slice(SegStart, Params.size()).trim().empty();
}
