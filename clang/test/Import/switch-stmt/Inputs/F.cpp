void f() {
  switch (1) {
  case 1:
  case 2:
    break;
  case 3 ... 4:
  case 5 ... 5:
    break;
  }
  switch (int varname; 1) {
  case 1:
    break;
  case 2:
    break;
  case 3 ... 5:
    break;
  }
  switch (1)
  default:
    break;
  switch (0)
    ;
}
