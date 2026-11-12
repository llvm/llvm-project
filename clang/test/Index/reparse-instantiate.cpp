int main()
{
  return 0;
}

// RUN: c-index-test -write-pch %t.pch -fno-delayed-template-parsing -x c++-header %S/Inputs/reparse-instantiate.h
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 5 local -fno-delayed-template-parsing -I %S/Inputs -include %t %s
