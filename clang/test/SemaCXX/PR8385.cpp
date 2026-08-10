// RUN: not %clang_cc1 -fsyntax-only %s

// don't crash on this, but don't constrain our diagnostics here as they're
// currently rather poor (we even accept things like "template struct {}").
// Other, explicit tests, should verify the relevant behavior of template 
// instantiation.
struct{template struct{
