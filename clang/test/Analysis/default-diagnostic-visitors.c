// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core -analyzer-store=region -analyzer-output=text -verify %s

// This file is for testing enhanced diagnostics produced by the default BugReporterVisitors.

int getPasswordAndItem(void)
{
  int err = 0;
  int *password; // expected-note {{'password' declared without an initial value}}
  if (password == 0) { // expected-warning {{The left operand of '==' is a garbage value}} // expected-note {{The left operand of '==' is a garbage value}}
    err = *password;
  }
  return err;
}
