// RUN: not clang-tblgen -gen-clang-comment-html-named-character-references %s -o /dev/null 2>&1 | FileCheck %s

class NCR<string spelling, int codePoint> {
  string Spelling = spelling;
  int CodePoint = codePoint;
}

def : NCR<"valid", 0x41>;
def : NCR<"invalid", 0x110000>;
// CHECK: html-named-character-references-errors.td:[[@LINE-1]]:1: error: invalid code point
// CHECK: 1 errors.
