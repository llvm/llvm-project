unsigned char determine_value_dyn(unsigned char c) {
  if (c < 0x80) {
    return c;
  } else {
    return -c;
  }
}
