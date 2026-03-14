void f() {
  if (true)
    return;

  if (int j = 3)
    return;

  if (int j; true)
    return;

  if (true)
    return;
  else
    return;

  if (true) {
    return;
  } else {
    return;
  }
}
