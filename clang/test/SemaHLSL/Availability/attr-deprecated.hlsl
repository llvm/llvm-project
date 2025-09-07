// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -std=hlsl202x -verify %s

[[deprecated("Woah!")]] // expected-note{{'myFn' has been explicitly marked deprecated here}}
void myFn() {}

[[deprecated("X is bad... chose Y instead")]] // expected-note{{'X' has been explicitly marked deprecated here}}
static const int X = 5;

enum States {
  Inactive,
  Active,
  Bored,
  Sleep [[deprecated("We don't allow sleep anymore!")]], // expected-note{{'Sleep' has been explicitly marked deprecated here}}
  Tired,
};

__attribute__((availability(shadermodel, introduced = 6.0, deprecated = 6.5)))
void fn65() {}

__attribute__((availability(shadermodel, introduced = 6.0, deprecated = 6.2)))
void fn62() {} // expected-note{{'fn62' has been explicitly marked deprecated here}}

void myOtherFn() {}

void otherFn() {
  myFn(); // expected-warning{{'myFn' is deprecated: Woah!}}

  int Y = X; // expected-warning{{'X' is deprecated: X is bad... chose Y instead}}

  States S = Bored;
  S = Sleep; // expected-warning{{'Sleep' is deprecated: We don't allow sleep anymore!}}
  S = Tired;

  fn65(); // No warning here because we're targeting 6.2 where this isn't yet deprecated.

  fn62(); // expected-warning{{'fn62' is deprecated: first deprecated in Shader Model 6.2}}
}
