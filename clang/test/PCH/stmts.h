// Header for PCH test stmts.c

void f0(int x) { 
  // NullStmt
  ;
  // IfStmt
  if (x) {
  } else if (x + 1) {
  }

  switch (x) {
  case 0:
    x = 17;
    break;

  case 1:
    break;

  default:
    switch (x >> 1) {
    case 7:
      // fall through
    case 9:
      break;
    }
    x += 2;
    break;
  }

  while (x > 20) {
    if (x > 30) {
      --x;
      continue;
    } else if (x < 5)
      break;
    else
      goto done;
  }

  do {
    x++;
  } while (x < 10);

 almost_done:
  for (int y = x; y < 20; ++y) {
    if (x + y == 12)
      return;
    else if (x - y == 7)
      goto almost_done;
  }

 done:
  x = x + 2;

  int z = x, *y, j = 5;
}

int f1(int x) {
  switch (x) {
  case 17:
    return 12;

  default:
    break;
  }

  // variable-length array
  int array[x * 17 + 3];

  return x*2;
}

const char* what_is_my_name(void) { return __func__; }

int computed_goto(int x) {
 start:
  x = x << 1;
  void *location = &&start;

  if (x > 17)
    location = &&done;

  while (x > 12) {
    --x;
    if (x == 15)
      goto *location;
  }

  done:
  return 5;
}

#define maxint(a,b) ({int _a = (a), _b = (b); _a > _b ? _a : _b; })
int weird_max(int x, int y) {
  return maxint(++x, --y);
}
