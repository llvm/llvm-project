// RUN: %check_clang_tidy %s google-readability-todo %t -- -config="{User: 'some user'}" --

//   TODOfix this1
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: missing username/bug in TODO
// CHECK-FIXES: // TODO(some user): fix this1

//   TODO fix this2
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: missing username/bug in TODO
// CHECK-FIXES: // TODO(some user): fix this2

// TODO fix this3
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: missing username/bug in TODO
// CHECK-FIXES: // TODO(some user): fix this3

// TODO: fix this4
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: missing username/bug in TODO
// CHECK-FIXES: // TODO(some user): fix this4

//   TODO(clang)fix this5

// TODO(foo):shave yaks
// TODO(bar):
// TODO(foo): paint bikeshed
// TODO(b/12345): find the holy grail
// TODO (b/12345): allow spaces before parentheses
// TODO(asdf) allow missing semicolon
// TODO: bug 12345678 - Remove this after the 2047q4 compatibility window expires.
// TODO: example.com/my-design-doc - Manually fix up this code the next time it's touched.
// TODO(bug 12345678): Update this list after the Foo service is turned down.
// TODO(John): Use a "\*" here for concatenation operator.
