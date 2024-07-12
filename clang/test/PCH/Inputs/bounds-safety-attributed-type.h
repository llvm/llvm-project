struct Test {
  int count;
  int fam[] __attribute__((counted_by(count)));
};
