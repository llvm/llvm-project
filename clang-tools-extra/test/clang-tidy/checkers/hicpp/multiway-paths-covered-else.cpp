// RUN: %check_clang_tidy %s hicpp-multiway-paths-covered %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: hicpp-multiway-paths-covered.WarnOnMissingElse, value: true}]}'\
// RUN: --

enum OS { Mac,
          Windows,
          Linux };

void problematic_if(int i, enum OS os) {
  if (i > 0) {
    return;
  } else if (i < 0) {
    // CHECK-MESSAGES: [[@LINE-1]]:10: warning: potentially uncovered codepath; add an ending else statement
    return;
  }

  // Could be considered as false positive because all paths are covered logically.
  // I still think this is valid since the possibility of a final 'everything else'
  // codepath is expected from if-else if.
  if (i > 0) {
    return;
  } else if (i <= 0) {
    // CHECK-MESSAGES: [[@LINE-1]]:10: warning: potentially uncovered codepath; add an ending else statement
    return;
  }

  // Test if nesting of if-else chains does get caught as well.
  if (os == Mac) {
    return;
  } else if (os == Linux) {
    // These checks are kind of degenerated, but the check will not try to solve
    // if logically all paths are covered, which is more the area of the static analyzer.
    if (true) {
      return;
    } else if (false) {
      // CHECK-MESSAGES: [[@LINE-1]]:12: warning: potentially uncovered codepath; add an ending else statement
      return;
    }
    return;
  } else {
    /* unreachable */
    if (true) // check if the parent would match here as well
      return;
    // No warning for simple if statements, since it is common to just test one condition
    // and ignore the opposite.
  }

  // Ok, because all paths are covered
  if (i > 0) {
    return;
  } else if (i < 0) {
    return;
  } else {
    /* error, maybe precondition failed */
  }
}
