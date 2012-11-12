// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core -analyzer-store=region -analyzer-constraints=range -verify -fblocks %s

// <rdar://problem/6776949>
// main's 'argc' argument is always > 0
int main(int argc, char* argv[]) {
  int *p = 0;

  if (argc == 0)
    *p = 1;

  if (argc == 1)
    return 1;

  int x = 1;
  int i;
  
  for(i=1;i<argc;i++){
    p = &x;
  }

  return *p; // no-warning
}

// PR 5969: the comparison of argc < 3 || argc > 4 should constraint the switch
//  statement from having the 'default' branch taken.  This previously reported a false
//  positive with the use of 'v'.

int pr5969(int argc, char *argv[]) {

  int v;

  if ((argc < 3) || (argc > 4)) return 0;

  switch(argc) {
    case 3:
      v = 33;
      break;
    case 4:
      v = 44;
      break;
  }

  return v; // no-warning
}

int pr5969_positive(int argc, char *argv[]) {

  int v;

  if ((argc < 3) || (argc > 4)) return 0;

  switch(argc) {
    case 3:
      v = 33;
      break;
  }

  return v; // expected-warning{{Undefined or garbage value returned to caller}}
}
