// Test that this unreachable code warning is
// not reported because it is in a header.

void foo_unreachable_header(void) {
  return;
  foo_unreachable_header(); // no-warning
}