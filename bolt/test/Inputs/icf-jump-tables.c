int inc(int x) {
  switch (x) {
    case 0: puts("0"); return 1;
    case 1: puts("1"); return 2;
    case 2: puts("2"); return 3;
    case 3: puts("3"); return 4;
    case 4: puts("4"); return 5;
    case 5: puts("5"); return 6;
    default: return x + 1;
  }
}

int inc_dup(int x) {
  switch (x) {
    case 0: puts("0"); return 1;
    case 1: puts("1"); return 2;
    case 2: puts("2"); return 3;
    case 3: puts("3"); return 4;
    case 4: puts("4"); return 5;
    case 5: puts("5"); return 6;
    default: return x + 1;
  }
}

int main() {
  return inc(5) - 2*inc_dup(2);
}
