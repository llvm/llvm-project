// RUN: %clang_cc1 %s -emit-html -o -

#line 42 "foo.c"

// PR3635
#define F(fmt, ...) fmt, ## __VA_ARGS__
int main(int argc, char **argv) {
  return F(argc, 1);
}

// PR3798
#define FOR_ALL_FILES(f,i) i

#if 0
  FOR_ALL_FILES(f) { }
#endif

// -emit-html filters out # directives, but not _Pragma (or MS __pragma)
// Diagnostic push/pop is stateful, so re-lexing a file can cause problems
// if these pragmas are interpreted normally.
_Pragma("clang diagnostic push")
_Pragma("clang diagnostic ignored \"-Wformat-extra-args\"")
_Pragma("clang diagnostic pop")

